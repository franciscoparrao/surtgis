//! Handler for `surtgis fluvial <subcmd>` commands.
//!
//! Subcommand surface tracks docs/SPEC_morfometria_fluvial_tectonica.md.
//! Sprint 2 ships `chi` only; ksn / knickpoints / concavity /
//! divide-migration arrive in subsequent sprints.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use surtgis_algorithms::fluvial::{chi_transform, ChiParams};
use surtgis_core::io::{read_geotiff, write_geotiff};

use crate::commands::FluvialCommands;
use crate::helpers::write_opts;

pub fn handle(cmd: FluvialCommands, compress: bool) -> Result<()> {
    match cmd {
        FluvialCommands::Chi {
            stream,
            flow_dir,
            flow_acc,
            output,
            theta_ref,
            a_0_m2,
            cell_size_m,
        } => handle_chi(
            &stream,
            &flow_dir,
            &flow_acc,
            &output,
            theta_ref,
            a_0_m2,
            cell_size_m,
            compress,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_chi(
    stream_path: &Path,
    flow_dir_path: &Path,
    flow_acc_path: &Path,
    output_path: &Path,
    theta_ref: f64,
    a_0_m2: f64,
    cell_size_m_override: Option<f64>,
    compress: bool,
) -> Result<()> {
    let start = std::time::Instant::now();

    println!("SurtGIS Fluvial — χ (chi) transform");
    println!("====================================");
    println!("  Stream:        {}", stream_path.display());
    println!("  Flow dir:      {}", flow_dir_path.display());
    println!("  Flow acc:      {}", flow_acc_path.display());
    println!("  Output:        {}", output_path.display());
    println!("  θref:          {}", theta_ref);
    println!("  A0 (m²):       {:e}", a_0_m2);

    let stream: surtgis_core::Raster<u8> = read_geotiff(stream_path, None)
        .with_context(|| format!("read stream from {}", stream_path.display()))?;
    let flow_dir: surtgis_core::Raster<u8> = read_geotiff(flow_dir_path, None)
        .with_context(|| format!("read flow_dir from {}", flow_dir_path.display()))?;
    let flow_acc: surtgis_core::Raster<f64> = read_geotiff(flow_acc_path, None)
        .with_context(|| format!("read flow_acc from {}", flow_acc_path.display()))?;

    // Cell size resolution. Priority: explicit --cell-size-m > raster
    // transform pixel_width. Per spec §8 pitfall #8, the CRS must be
    // projected (units = metres). We can't introspect the CRS unit string
    // from here without proj4rs heavy lifting, so we use a heuristic: if
    // pixel_width is < 1.0 in absolute value, it's almost certainly
    // degrees → reject unless the user supplied --cell-size-m.
    let pixel_width = stream.transform().pixel_width.abs();
    let cell_size_m = match cell_size_m_override {
        Some(v) if v > 0.0 => v,
        Some(v) => return Err(anyhow!("--cell-size-m must be > 0, got {}", v)),
        None => {
            if pixel_width < 1.0 {
                return Err(anyhow!(
                    "Raster pixel size is {} (looks like degrees, not metres). \
                     Reproject your inputs to a projected CRS first \
                     (e.g. `surtgis reproject --to EPSG:32719 ...`) \
                     or pass `--cell-size-m <metres>` explicitly.",
                    pixel_width,
                ));
            }
            pixel_width
        }
    };
    println!("  Cell size (m): {:.2}", cell_size_m);

    // Cell-size > 30 m generates the warning from spec §8 pitfall #1: ksn
    // / chi sensitivity drops. Chi is less sensitive than ksn but we
    // surface the warning for consistency with the spec.
    if cell_size_m > 30.0 {
        eprintln!(
            "  WARNING: cell size {:.1} m > 30 m — χ and downstream ksn \
             will be noisier than the literature standard (~10–30 m).",
            cell_size_m,
        );
    }

    let params = ChiParams {
        theta_ref,
        a_0_m2,
        cell_size_m,
        base_outlets: None,
    };

    let chi = chi_transform(&stream, &flow_dir, &flow_acc, params)
        .context("χ transform failed")?;

    // Quick stats so the user knows the run produced something sensible.
    let stats = chi.statistics();
    println!();
    println!("  χ stats:");
    if let (Some(mn), Some(mx), Some(mean)) = (stats.min, stats.max, stats.mean) {
        println!("    min   {:.2}", mn);
        println!("    max   {:.2}", mx);
        println!("    mean  {:.2}", mean);
    } else {
        println!("    (all NaN — no stream cells reached an outlet?)");
    }
    println!("    valid {} / {} cells ({:.1}%)",
        stats.valid_count,
        chi.len(),
        100.0 * stats.valid_count as f64 / chi.len() as f64,
    );

    write_geotiff(&chi, output_path, Some(write_opts(compress)))
        .with_context(|| format!("write chi to {}", output_path.display()))?;

    println!();
    println!("Wrote {} in {:.1}s",
        output_path.display(),
        start.elapsed().as_secs_f64(),
    );
    Ok(())
}
