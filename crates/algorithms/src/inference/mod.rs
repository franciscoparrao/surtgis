//! Tiled model inference: run a dense/convolutional model over one or
//! more aligned rasters, in halo-padded tiles, stitching only each
//! tile's *core* back into the output — bounded memory, seam-free.
//!
//! Design contract: see `SPEC_SURTGIS_ONNX_INFERENCE.md` at the
//! workspace root. This module implements the orchestration
//! ([`run_tiled`], built on [`surtgis_core::tiling::TileGrid`], whose
//! halo/core split already exists precisely for this use case) fully
//! and without any ML runtime dependency. The [`onnx`] submodule
//! (feature `onnx`, off by default) is the scaffolded model backend —
//! its `load`/`infer` are stubs; wiring up `tract-onnx` is deferred
//! work, not part of this scaffold.
//!
//! **Not this:** [`crate::sampling`]/`predict_raster` (in
//! `surtgis-python`) applies a model **pixel-wise** — one feature
//! vector per pixel, no spatial context. `surtgis ml predict`
//! (`smelt-ml`) is the same pixel-wise shape for tabular models
//! (random forest / GBM). This module is for models with a spatial
//! receptive field (CNNs, segmentation networks) that need to see a
//! neighbourhood — hence the halo.

use ndarray::{Array3, Axis, s};
use surtgis_core::raster::check_aligned;
use surtgis_core::tiling::TileGrid;
use surtgis_core::{Error, Raster, Result};

#[cfg(feature = "onnx")]
pub mod onnx;

/// One tile's input: `in_bands` aligned bands, halo-expanded to the
/// tile's *read* extent (see [`surtgis_core::tiling::Tile`]).
///
/// `bands` shape is `[bands, rows, cols]`; `rows`/`cols` equal the read
/// extent's `read_rows`/`read_cols`, which can be smaller than
/// `core_size + 2 * halo` at grid borders — `TileGrid` *clamps* the
/// halo there rather than zero-padding it, so the halo can be
/// asymmetric around the core. `core_offset_row`/`core_offset_col` /
/// `core_rows`/`core_cols` locate the core rectangle inside `bands`
/// (`bands[.., core_offset_row..core_offset_row+core_rows,
/// core_offset_col..core_offset_col+core_cols]` is the region the
/// model owns) — a model can't safely infer the core size from
/// `bands`' shape alone (e.g. `read_rows - 2 * halo`) precisely because
/// of that border clamping.
#[derive(Debug, Clone)]
pub struct TileInput {
    /// `[in_bands, read_rows, read_cols]`.
    pub bands: Array3<f32>,
    /// Row where the core starts inside `bands`.
    pub core_offset_row: usize,
    /// Column where the core starts inside `bands`.
    pub core_offset_col: usize,
    /// Core height — the expected [`TileOutput::bands`] row count.
    pub core_rows: usize,
    /// Core width — the expected [`TileOutput::bands`] column count.
    pub core_cols: usize,
}

/// One tile's output, sized to the tile's *core* extent — the part of
/// the read window that a [`TileModel`] actually owns after consuming
/// its halo (e.g. via valid-mode convolutions).
#[derive(Debug, Clone)]
pub struct TileOutput {
    /// `[out_bands, core_rows, core_cols]`. [`run_tiled`] rejects any
    /// other shape — a model that doesn't crop to the core exactly
    /// cannot be stitched seam-free.
    pub bands: Array3<f32>,
}

/// A model that turns a halo-padded input tile into a core-sized
/// output tile. Implementors own the model itself (weights, runtime
/// handle); [`run_tiled`] owns the tiling, halo extraction, and
/// stitching.
pub trait TileModel: Send + Sync {
    /// Number of input bands this model expects per tile.
    fn in_bands(&self) -> usize;

    /// Number of output bands this model produces per tile.
    fn out_bands(&self) -> usize;

    /// Halo size (cells) this model needs on every side of its core —
    /// typically `(receptive_field - 1) / 2`. [`run_tiled`] uses this
    /// as the `overlap` passed to `TileGrid`.
    fn required_halo(&self) -> usize;

    /// Run inference on one halo-padded tile. Implementations MUST
    /// return a [`TileOutput`] whose shape is exactly
    /// `[out_bands, tile.core_rows, tile.core_cols]` — see
    /// [`TileOutput`].
    fn infer(&self, tile: &TileInput) -> Result<TileOutput>;
}

/// Run `model` over `bands` (one [`Raster<f32>`] per input band, all on
/// the same grid) in `tile_size`-sized core tiles, with the halo
/// `model.required_halo()` requires, and return `model.out_bands()`
/// output rasters on that same grid.
///
/// Memory is bounded by `O(tile_size² · (in_bands + out_bands))` plus
/// whatever the model itself allocates per tile — the full rasters are
/// never materialised beyond what [`Raster`] already holds for I/O.
/// Output rasters inherit `bands[0]`'s transform, CRS, and nodata.
///
/// Tiles are processed sequentially: cores never overlap, so a future
/// parallel version can dispatch tiles across threads and write each
/// core independently without synchronisation — that parallelisation
/// is a natural follow-up, not implemented here.
///
/// # Errors
///
/// - [`Error::InvalidParameter`] if `bands` is empty or its length
///   doesn't match `model.in_bands()`.
/// - [`Error::ShapeMismatch`] / [`Error::Misaligned`] (via
///   [`check_aligned`]) if the input bands aren't on the same grid.
/// - [`Error::Algorithm`] if `model.infer` returns a tile whose shape
///   doesn't match `[out_bands, core_rows, core_cols]`.
pub fn run_tiled(
    model: &dyn TileModel,
    bands: &[Raster<f32>],
    tile_size: usize,
) -> Result<Vec<Raster<f32>>> {
    if bands.is_empty() {
        return Err(Error::InvalidParameter {
            name: "bands",
            value: "0".to_string(),
            reason: "run_tiled needs at least one input band".to_string(),
        });
    }
    if bands.len() != model.in_bands() {
        return Err(Error::InvalidParameter {
            name: "bands",
            value: bands.len().to_string(),
            reason: format!(
                "model expects {} input band(s), got {}",
                model.in_bands(),
                bands.len()
            ),
        });
    }

    let refs: Vec<&Raster<f32>> = bands.iter().collect();
    check_aligned(&refs)?;

    let (rows, cols) = bands[0].shape();
    let halo = model.required_halo();
    let out_bands = model.out_bands();

    let mut outputs: Vec<Raster<f32>> = (0..out_bands)
        .map(|_| {
            let mut r = bands[0].with_same_meta::<f32>(rows, cols);
            r.set_nodata(bands[0].nodata());
            r
        })
        .collect();

    for tile in TileGrid::new(rows, cols, tile_size, halo) {
        let mut input = Array3::<f32>::zeros((bands.len(), tile.read_rows, tile.read_cols));
        for (bi, band) in bands.iter().enumerate() {
            let view = band.view();
            let window = view.slice(s![
                tile.read_row..tile.read_row + tile.read_rows,
                tile.read_col..tile.read_col + tile.read_cols
            ]);
            input.index_axis_mut(Axis(0), bi).assign(&window);
        }

        let out = model.infer(&TileInput {
            bands: input,
            core_offset_row: tile.core_offset_row(),
            core_offset_col: tile.core_offset_col(),
            core_rows: tile.core_rows,
            core_cols: tile.core_cols,
        })?;
        let expected = [out_bands, tile.core_rows, tile.core_cols];
        if out.bands.shape() != expected {
            return Err(Error::Algorithm(format!(
                "TileModel::infer returned shape {:?} for tile {}, expected {:?}",
                out.bands.shape(),
                tile.index,
                expected
            )));
        }

        for (bi, output_raster) in outputs.iter_mut().enumerate() {
            let core = out.bands.index_axis(Axis(0), bi);
            output_raster
                .data_mut()
                .slice_mut(s![
                    tile.core_row..tile.core_row + tile.core_rows,
                    tile.core_col..tile.core_col + tile.core_cols
                ])
                .assign(&core);
        }
    }

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::GeoTransform;

    /// `in_bands = out_bands = 1`, `halo = 0` — the read extent equals
    /// the core extent, so this is the seam-free-stitching floor case:
    /// core tiles must reassemble the original raster bit-for-bit with
    /// no gaps, no overlap, no shift.
    struct IdentityModel;

    impl TileModel for IdentityModel {
        fn in_bands(&self) -> usize {
            1
        }
        fn out_bands(&self) -> usize {
            1
        }
        fn required_halo(&self) -> usize {
            0
        }
        fn infer(&self, tile: &TileInput) -> Result<TileOutput> {
            Ok(TileOutput {
                bands: tile.bands.clone(),
            })
        }
    }

    /// Computes a `(2·halo+1)²` box sum per core cell from the
    /// halo-padded input — the halo is only useful if it's actually
    /// read, so this model exercises that the orchestration hands it
    /// real neighbouring data, not zeros or garbage. Uses
    /// `core_offset_row/col` (not `read_rows - 2 * halo`) to locate the
    /// core, since `TileGrid` clamps (rather than zero-pads) the halo
    /// at grid borders, making it asymmetric there.
    struct BoxSumModel {
        halo: usize,
    }

    impl TileModel for BoxSumModel {
        fn in_bands(&self) -> usize {
            1
        }
        fn out_bands(&self) -> usize {
            1
        }
        fn required_halo(&self) -> usize {
            self.halo
        }
        fn infer(&self, tile: &TileInput) -> Result<TileOutput> {
            let read = tile.bands.index_axis(Axis(0), 0);
            let (read_rows, read_cols) = (read.shape()[0], read.shape()[1]);
            let halo = self.halo as isize;
            let mut out = Array3::<f32>::zeros((1, tile.core_rows, tile.core_cols));
            for cr in 0..tile.core_rows {
                for cc in 0..tile.core_cols {
                    let center_r = (tile.core_offset_row + cr) as isize;
                    let center_c = (tile.core_offset_col + cc) as isize;
                    let mut sum = 0.0f32;
                    for dr in -halo..=halo {
                        for dc in -halo..=halo {
                            let r = center_r + dr;
                            let c = center_c + dc;
                            if r >= 0
                                && c >= 0
                                && (r as usize) < read_rows
                                && (c as usize) < read_cols
                            {
                                sum += read[[r as usize, c as usize]];
                            }
                        }
                    }
                    out[[0, cr, cc]] = sum;
                }
            }
            Ok(TileOutput { bands: out })
        }
    }

    fn ramp_raster(rows: usize, cols: usize) -> Raster<f32> {
        let mut r: Raster<f32> = Raster::new(rows, cols);
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, (row * cols + col) as f32).unwrap();
            }
        }
        r.set_transform(GeoTransform::new(0.0, 0.0, 1.0, -1.0));
        r
    }

    #[test]
    fn identity_model_stitches_without_seams() {
        // 100x130 grid, tile_size=32 => ragged last row/col of tiles,
        // deliberately exercising TileGrid's border-clamping path too.
        let input = ramp_raster(100, 130);
        let out = run_tiled(&IdentityModel, std::slice::from_ref(&input), 32).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape(), input.shape());
        for row in 0..100 {
            for col in 0..130 {
                assert_eq!(
                    out[0].get(row, col).unwrap(),
                    input.get(row, col).unwrap(),
                    "mismatch at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn halo_is_read_and_used_correctly() {
        let halo = 2;
        // 64x71 with tile_size=16 forces ragged tiles AND border-clamped
        // (asymmetric) halos on every edge — the case that previously
        // broke a naive `read_rows - 2 * halo` core-size assumption.
        let input = ramp_raster(64, 71);
        let model = BoxSumModel { halo };
        let out = run_tiled(&model, std::slice::from_ref(&input), 16).unwrap();

        // Reference: the same box sum computed directly on the full
        // raster, summing only in-grid neighbours (out-of-grid ones
        // contribute nothing) — exactly what a clamped, not
        // zero-padded, halo should produce. Checked at EVERY cell,
        // including borders/corners, not just the interior: proves the
        // halo crossed tile boundaries correctly everywhere.
        let (rows, cols) = input.shape();
        for row in 0..rows {
            for col in 0..cols {
                let mut expected = 0.0f32;
                for dr in -(halo as isize)..=(halo as isize) {
                    for dc in -(halo as isize)..=(halo as isize) {
                        let r = row as isize + dr;
                        let c = col as isize + dc;
                        if r >= 0 && c >= 0 && (r as usize) < rows && (c as usize) < cols {
                            expected += input.get(r as usize, c as usize).unwrap();
                        }
                    }
                }
                assert_eq!(
                    out[0].get(row, col).unwrap(),
                    expected,
                    "box sum mismatch at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn wrong_band_count_is_rejected() {
        let input = ramp_raster(10, 10);
        let err = run_tiled(&BoxSumModel { halo: 1 }, &[input.clone(), input], 8).unwrap_err();
        assert!(matches!(err, Error::InvalidParameter { .. }));
    }

    #[test]
    fn misaligned_bands_are_rejected() {
        let a = ramp_raster(10, 10);
        let mut b = ramp_raster(10, 10);
        b.set_transform(GeoTransform::new(100.0, 100.0, 1.0, -1.0));
        struct TwoBandIdentity;
        impl TileModel for TwoBandIdentity {
            fn in_bands(&self) -> usize {
                2
            }
            fn out_bands(&self) -> usize {
                1
            }
            fn required_halo(&self) -> usize {
                0
            }
            fn infer(&self, tile: &TileInput) -> Result<TileOutput> {
                Ok(TileOutput {
                    bands: tile
                        .bands
                        .index_axis(Axis(0), 0)
                        .to_owned()
                        .insert_axis(Axis(0)),
                })
            }
        }
        let err = run_tiled(&TwoBandIdentity, &[a, b], 8).unwrap_err();
        assert!(matches!(err, Error::Misaligned { .. }));
    }

    #[test]
    fn model_returning_wrong_shape_is_rejected() {
        struct BrokenModel;
        impl TileModel for BrokenModel {
            fn in_bands(&self) -> usize {
                1
            }
            fn out_bands(&self) -> usize {
                1
            }
            fn required_halo(&self) -> usize {
                0
            }
            fn infer(&self, _tile: &TileInput) -> Result<TileOutput> {
                // Deliberately wrong: returns a 1x1 tile regardless of
                // the requested core size.
                Ok(TileOutput {
                    bands: Array3::<f32>::zeros((1, 1, 1)),
                })
            }
        }
        let input = ramp_raster(20, 20);
        let err = run_tiled(&BrokenModel, std::slice::from_ref(&input), 8).unwrap_err();
        assert!(matches!(err, Error::Algorithm(_)));
    }
}
