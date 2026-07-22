//! Handler for `surtgis flow run` (spec surtgis-flow v1.0 §8).
//!
//! Writes one `h_t####.tif` frame every `--output-interval` seconds of
//! physical time (frame 0000 is the initial state), optional `u_t####.tif` /
//! `v_t####.tif` velocity frames, an optional arrival-time raster, and a
//! `manifest.json` describing the run — the contract consumed by
//! `geodeo-bridge`.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::time::Instant;

use surtgis_core::Raster;
use surtgis_core::io::{read_geotiff, write_geotiff};
use surtgis_flow::{EntrainmentParams, SimGrid, Simulation, SolverConfig, VoellmyParams};

use crate::commands::FlowCommands;
use crate::helpers::write_opts;

pub fn handle(command: FlowCommands, compress: bool) -> Result<()> {
    match command {
        FlowCommands::Run {
            dem,
            release,
            outdir,
            mu,
            xi,
            duration,
            output_interval,
            dump_velocity,
            arrival,
            erodible,
            entrainment_k,
            dump_erosion,
        } => run(RunArgs {
            dem_path: &dem,
            release_path: &release,
            outdir: &outdir,
            mu,
            xi,
            duration,
            output_interval,
            dump_velocity,
            arrival: arrival.as_deref(),
            erodible: erodible.as_deref(),
            entrainment_k,
            dump_erosion,
            compress,
        }),
    }
}

/// Arguments of one `flow run` invocation (bundled: the flag surface
/// outgrew a readable positional list).
struct RunArgs<'a> {
    dem_path: &'a Path,
    release_path: &'a Path,
    outdir: &'a Path,
    mu: f32,
    xi: f32,
    duration: f64,
    output_interval: f64,
    dump_velocity: bool,
    arrival: Option<&'a Path>,
    erodible: Option<&'a Path>,
    entrainment_k: f32,
    dump_erosion: bool,
    compress: bool,
}

fn run(args: RunArgs<'_>) -> Result<()> {
    let RunArgs {
        dem_path,
        release_path,
        outdir,
        mu,
        xi,
        duration,
        output_interval,
        dump_velocity,
        arrival,
        erodible,
        entrainment_k,
        dump_erosion,
        compress,
    } = args;
    anyhow::ensure!(
        duration > 0.0 && output_interval > 0.0,
        "duration and output-interval must be positive"
    );
    let start = Instant::now();

    let dem: Raster<f32> = read_geotiff(dem_path, None).context("Failed to read DEM")?;
    let release: Raster<f32> =
        read_geotiff(release_path, None).context("Failed to read release raster")?;
    println!(
        "DEM: {} x {} cells, cellsize {:.2} m",
        dem.cols(),
        dem.rows(),
        dem.cell_size()
    );

    let params = VoellmyParams {
        mu,
        xi,
        ..VoellmyParams::default()
    };
    let config = SolverConfig {
        // A render interval can span many CFL substeps on fine grids; the
        // guard only cuts truly pathological runs.
        max_substeps: 1_000_000,
        ..SolverConfig::default()
    };
    let mut sim = Simulation::new(&dem, &release, params, config)?;
    let mut ent_params: Option<EntrainmentParams> = None;
    if let Some(erodible_path) = erodible {
        let emax: Raster<f32> =
            read_geotiff(erodible_path, None).context("Failed to read erodible raster")?;
        let mut p = EntrainmentParams::default();
        p.k = entrainment_k;
        sim.set_erodible(&emax, p)?;
        ent_params = Some(p);
        println!(
            "Entrainment: K = {entrainment_k} /m, erodible raster {}",
            erodible_path.display()
        );
    }
    let mass0 = sim.total_mass();
    println!("Release volume: {mass0:.0} m³");

    std::fs::create_dir_all(outdir).context("Failed to create output directory")?;

    let n_outputs = (duration / output_interval).ceil() as usize;
    let n_frames = n_outputs + 1; // frame 0000 = initial state

    let crs = dem.crs().cloned();
    write_frames(
        &sim,
        crs.as_ref(),
        outdir,
        0,
        dump_velocity,
        dump_erosion,
        compress,
    )?;

    let pb = ProgressBar::new(n_outputs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:30.green} {pos}/{len} frames  t={msg}")
            .unwrap(),
    );
    let mut total_substeps: u64 = 0;
    for frame in 1..=n_outputs {
        let elapsed_target = (frame as f64) * output_interval;
        let dt = (elapsed_target.min(duration) - sim.time()).max(0.0);
        total_substeps += u64::from(sim.step(dt as f32)?);
        write_frames(
            &sim,
            crs.as_ref(),
            outdir,
            frame,
            dump_velocity,
            dump_erosion,
            compress,
        )?;
        pb.set_message(format!("{:.1} s", sim.time()));
        pb.inc(1);
    }
    pb.finish_and_clear();

    if let Some(arrival_path) = arrival {
        let raster = masked_raster(sim.grid(), dem.crs(), sim.arrival_times().to_vec());
        write_geotiff(&raster, arrival_path, Some(write_opts(compress)))
            .context("Failed to write arrival raster")?;
        println!("Arrival times saved to: {}", arrival_path.display());
    }

    write_manifest(
        outdir,
        &dem,
        output_interval,
        n_frames,
        mu,
        xi,
        ent_params.map(|p| (p, sim.total_eroded())),
    )?;

    let mass_end = sim.total_mass();
    if erodible.is_some() {
        println!("Eroded volume: {:.0} m³", sim.total_eroded());
    }
    println!(
        "Simulated {:.1} s in {} frames ({total_substeps} substeps) — wall time {:.2?}",
        sim.time(),
        n_frames,
        start.elapsed()
    );
    println!(
        "Mass: {mass0:.0} -> {mass_end:.0} m³ ({:+.2}% through open borders)",
        (mass_end - mass0) / mass0 * 100.0
    );
    println!("Frames saved to: {}", outdir.display());
    Ok(())
}

/// Write the `h` frame (and optionally `u`/`v`, `e`) for `frame`.
#[allow(clippy::fn_params_excessive_bools)]
fn write_frames(
    sim: &Simulation,
    crs: Option<&surtgis_core::CRS>,
    outdir: &Path,
    frame: usize,
    dump_velocity: bool,
    dump_erosion: bool,
    compress: bool,
) -> Result<()> {
    let grid = sim.grid();
    let state = sim.state();

    let h = masked_raster(grid, crs, state.h.clone());
    write_geotiff(
        &h,
        outdir.join(format!("h_t{frame:04}.tif")),
        Some(write_opts(compress)),
    )
    .with_context(|| format!("Failed to write h frame {frame}"))?;

    if dump_velocity {
        let n = state.h.len();
        let mut u = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        for i in 0..n {
            let hh = state.h[i];
            if hh >= 1e-3 {
                u[i] = state.hu[i] / hh;
                v[i] = state.hv[i] / hh;
            }
        }
        let u = masked_raster(grid, crs, u);
        let v = masked_raster(grid, crs, v);
        write_geotiff(
            &u,
            outdir.join(format!("u_t{frame:04}.tif")),
            Some(write_opts(compress)),
        )?;
        write_geotiff(
            &v,
            outdir.join(format!("v_t{frame:04}.tif")),
            Some(write_opts(compress)),
        )?;
    }
    if dump_erosion {
        let e = sim.eroded_depth();
        if !e.is_empty() {
            let e = masked_raster(grid, crs, e.to_vec());
            write_geotiff(
                &e,
                outdir.join(format!("e_t{frame:04}.tif")),
                Some(write_opts(compress)),
            )?;
        }
    }
    Ok(())
}

/// Build an output raster on the simulation grid with NaN over solid
/// (`NoData`) cells and NaN declared as the nodata value.
fn masked_raster(
    grid: &SimGrid,
    crs: Option<&surtgis_core::CRS>,
    mut values: Vec<f32>,
) -> Raster<f32> {
    let (rows, cols) = (grid.rows(), grid.cols());
    for r in 0..rows {
        for c in 0..cols {
            if grid.is_solid(r, c) {
                values[r * cols + c] = f32::NAN;
            }
        }
    }
    let mut raster = Raster::from_vec(values, rows, cols).expect("state length matches grid");
    raster.set_transform(*grid.transform());
    raster.set_crs(crs.cloned());
    raster.set_nodata(Some(f32::NAN));
    raster
}

/// `manifest.json` — the contract with `geodeo-bridge` (spec §8). The v1
/// field set is frozen with GEODEO's 2026-07-19 sign-off: the normative
/// {crs, origin, cellsize, dt_output, n_frames, mu, xi, units} plus
/// `duration` and `row0`. With entrainment active the manifest gains the
/// spec-v1.1 §5 block and declares `"manifest_version": 2`; without it the
/// output is byte-compatible with the frozen v1 (no version field), so
/// existing parsers are untouched.
fn write_manifest(
    outdir: &Path,
    dem: &Raster<f32>,
    dt_output: f64,
    n_frames: usize,
    mu: f32,
    xi: f32,
    entrainment: Option<(EntrainmentParams, f64)>,
) -> Result<()> {
    let t = dem.transform();
    let crs = dem.crs().map(std::string::ToString::to_string);
    // f32 params round-trip through f64 with decimal noise (0.15 ->
    // 0.15000000596...); round to 6 decimals for a clean contract file.
    let clean = |v: f32| (f64::from(v) * 1e6).round() / 1e6;
    let mut manifest = serde_json::json!({
        "crs": crs,
        "origin": [t.origin_x, t.origin_y],
        "cellsize": t.pixel_width,
        "dt_output": dt_output,
        "n_frames": n_frames,
        // GEODEO's definition, computed with their exact formula (not the
        // FP-accumulated simulation clock): the time span of the frame grid.
        "duration": dt_output * (n_frames as f64 - 1.0),
        // origin is the NW corner; row 0 of every frame is the northernmost.
        "row0": "north",
        "mu": clean(mu),
        "xi": clean(xi),
        "units": { "h": "m", "u": "m/s", "v": "m/s", "arrival": "s" },
    });
    if let Some((p, total_eroded)) = entrainment {
        manifest["manifest_version"] = serde_json::json!(2);
        manifest["units"]["e"] = serde_json::json!("m");
        manifest["entrainment"] = serde_json::json!({
            "k": f64::from(p.k),
            "rate_max": f64::from(p.rate_max),
            "v_entr_min": f64::from(p.v_entr_min),
            "f_max": f64::from(p.f_max),
            "total_eroded_m3": total_eroded.round(),
        });
    }
    let path = outdir.join("manifest.json");
    std::fs::write(&path, serde_json::to_string_pretty(&manifest)?)
        .context("Failed to write manifest.json")?;
    println!("Manifest saved to: {}", path.display());
    Ok(())
}
