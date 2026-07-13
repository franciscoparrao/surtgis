//! Handler for `surtgis infer` — tiled ONNX model inference over one or
//! more aligned rasters.
//!
//! Scaffold per `SPEC_SURTGIS_ONNX_INFERENCE.md` (workspace root): this
//! handler does the real I/O and orchestration work (reads the feature
//! stack, calls `surtgis_algorithms::inference::run_tiled`), but
//! `OnnxModel::load` — the one piece that actually needs `tract-onnx` —
//! is not yet implemented. Running this command today reads the
//! inputs, then fails with a clear error at that point rather than
//! panicking or silently no-op'ing.
//!
//! Not the same as `surtgis ml predict` (smelt-ml): that applies a
//! tabular model pixel-wise, with no spatial context. This command is
//! for models with a spatial receptive field (CNNs, segmentation
//! networks).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use surtgis_algorithms::inference::onnx::OnnxModel;
use surtgis_algorithms::inference::run_tiled;
use surtgis_core::Raster;
use surtgis_core::io::{read_geotiff, write_geotiff, write_geotiff_multiband};

use crate::helpers::write_opts;

/// Run `surtgis infer`. See the module docs for the current scaffold
/// status: this always errors once it reaches `OnnxModel::load`.
pub fn run(
    model: &Path,
    features: &[PathBuf],
    output: &Path,
    tile_size: usize,
    halo: usize,
    argmax: bool,
    compress: bool,
) -> Result<()> {
    if features.is_empty() {
        bail!("surtgis infer: --features needs at least one input raster");
    }

    println!("SurtGIS Infer (scaffold — see SPEC_SURTGIS_ONNX_INFERENCE.md)");
    println!("================================================================");
    println!("  Model:     {}", model.display());
    println!("  Features:  {} raster(s)", features.len());
    println!("  Output:    {}", output.display());
    println!("  Tile size: {tile_size}");
    println!("  Halo:      {halo}");
    println!();

    let bands: Vec<Raster<f32>> = features
        .iter()
        .map(|p| {
            read_geotiff(p, None)
                .with_context(|| format!("Failed to read feature raster {}", p.display()))
        })
        .collect::<Result<_>>()?;

    // `OnnxModel::load` is the scaffold's one deliberate stub (always
    // returns Err). `out_bands` is hardcoded to 1 here because there's
    // no loaded graph yet to report its real output shape; a real
    // implementation reads it from the graph and validates `in_bands`
    // against `bands.len()` there instead of trusting the CLI's guess.
    let onnx_model = OnnxModel::load(model, bands.len(), 1, halo).context(
        "surtgis infer: ONNX model loading is not yet implemented (scaffold only) — \
         see SPEC_SURTGIS_ONNX_INFERENCE.md. The tiling/halo/stitching orchestration \
         this command uses (surtgis_algorithms::inference::run_tiled) is implemented \
         and tested independently of any ML runtime.",
    )?;

    let outputs = run_tiled(&onnx_model, &bands, tile_size)?;

    if argmax {
        bail!("surtgis infer: --argmax is not yet implemented (scaffold only)");
    }

    match outputs.as_slice() {
        [single] => {
            write_geotiff(single, output, Some(write_opts(compress)))
                .context("Failed to write output")?;
        }
        many => {
            let refs: Vec<&Raster<f32>> = many.iter().collect();
            write_geotiff_multiband(&refs, output, Some(write_opts(compress)))
                .context("Failed to write output")?;
        }
    }

    println!("Done: wrote {}", output.display());
    Ok(())
}
