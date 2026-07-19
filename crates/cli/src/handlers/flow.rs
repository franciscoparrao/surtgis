//! Handler for `surtgis flow run` (spec surtgis-flow v1.0 §8).
//!
//! Writes one `h_t####.tif` frame every `--output-interval` seconds of
//! physical time (frame 0000 is the initial state), optional `u_t####.tif` /
//! `v_t####.tif` velocity frames, an optional arrival-time raster, and a
//! `manifest.json` describing the run — the contract consumed by
//! `geodeo-bridge`.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::time::Instant;

use surtgis_core::Raster;
use surtgis_core::io::{read_geotiff, write_geotiff};
use surtgis_flow::{SimGrid, Simulation, SolverConfig, VoellmyParams};

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
        } => run(
            &dem,
            &release,
            &outdir,
            mu,
            xi,
            duration,
            output_interval,
            dump_velocity,
            arrival.as_deref(),
            compress,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    dem_path: &PathBuf,
    release_path: &PathBuf,
    outdir: &Path,
    mu: f32,
    xi: f32,
    duration: f64,
    output_interval: f64,
    dump_velocity: bool,
    arrival: Option<&Path>,
    compress: bool,
) -> Result<()> {
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
    let mass0 = sim.total_mass();
    println!("Release volume: {mass0:.0} m³");

    std::fs::create_dir_all(outdir).context("Failed to create output directory")?;

    let n_outputs = (duration / output_interval).ceil() as usize;
    let n_frames = n_outputs + 1; // frame 0000 = initial state

    let crs = dem.crs().cloned();
    write_frames(&sim, crs.as_ref(), outdir, 0, dump_velocity, compress)?;

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
        write_frames(&sim, crs.as_ref(), outdir, frame, dump_velocity, compress)?;
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

    write_manifest(outdir, &dem, output_interval, n_frames, mu, xi)?;

    let mass_end = sim.total_mass();
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

/// Write the `h` frame (and optionally `u`/`v`) for `frame`.
fn write_frames(
    sim: &Simulation,
    crs: Option<&surtgis_core::CRS>,
    outdir: &Path,
    frame: usize,
    dump_velocity: bool,
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

/// `manifest.json` — the contract with `geodeo-bridge` (spec §8). Field set
/// is normative: {crs, origin, cellsize, dt_output, n_frames, mu, xi, units}.
fn write_manifest(
    outdir: &Path,
    dem: &Raster<f32>,
    dt_output: f64,
    n_frames: usize,
    mu: f32,
    xi: f32,
) -> Result<()> {
    let t = dem.transform();
    let crs = dem.crs().map(std::string::ToString::to_string);
    // f32 params round-trip through f64 with decimal noise (0.15 ->
    // 0.15000000596...); round to 6 decimals for a clean contract file.
    let clean = |v: f32| (f64::from(v) * 1e6).round() / 1e6;
    let manifest = serde_json::json!({
        "crs": crs,
        "origin": [t.origin_x, t.origin_y],
        "cellsize": t.pixel_width,
        "dt_output": dt_output,
        "n_frames": n_frames,
        "mu": clean(mu),
        "xi": clean(xi),
        "units": { "h": "m", "u": "m/s", "v": "m/s", "arrival": "s" },
    });
    let path = outdir.join("manifest.json");
    std::fs::write(&path, serde_json::to_string_pretty(&manifest)?)
        .context("Failed to write manifest.json")?;
    println!("Manifest saved to: {}", path.display());
    Ok(())
}
