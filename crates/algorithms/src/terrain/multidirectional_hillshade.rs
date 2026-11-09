//! Multidirectional hillshade
//!
//! Computes a weighted blend of hillshade from multiple illumination azimuths,
//! reducing the directional bias inherent in single-azimuth hillshade.
//!
//! Uses the USGS method (Mark 1992): four illumination azimuths (225°,
//! 270°, 315°, 360°) blended with aspect-dependent weights, so each pixel
//! is dominated by the illumination direction that discriminates it most.
//! Matches `gdaldem hillshade -multidirectional`.
//!
//! Reference: Mark, R.K. (1992) "A multidirectional, oblique-weighted,
//! shaded-relief image of the Island of Hawaii" (USGS Open-File Report 92-422)

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for multidirectional hillshade
#[derive(Debug, Clone)]
pub struct MultiHillshadeParams {
    /// Sun altitude in degrees above horizon (0-90)
    pub altitude: f64,
    /// Vertical exaggeration applied to elevations before the gradient
    /// (GDAL/ArcGIS convention: `z' = z_factor * z`). Default 1.0.
    ///
    /// Rasters with a geographic CRS get automatic per-row metric cell
    /// sizes — do NOT use the legacy `111320` cell-size hack, which relied
    /// on the pre-0.17 reciprocal semantics.
    pub z_factor: f64,
    /// Output range: false = 0-255, true = 0.0-1.0
    pub normalized: bool,
}

impl Default for MultiHillshadeParams {
    fn default() -> Self {
        Self {
            altitude: 45.0,
            z_factor: 1.0,
            normalized: false,
        }
    }
}

/// The four illumination azimuths of Mark (1992), pre-converted to the
/// `(sin, cos)` pairs the shading kernel consumes.
///
/// Mark's method illuminates from **225°, 270°, 315° and 360°** — a
/// half-circle, not the full compass. That is the load-bearing detail: with
/// azimuths spread over the full circle the directional terms cancel in the
/// blend (`Σ sin az = Σ cos az = 0`), so every cell collapses to
/// `cos θz · 1/√(1+g²)` — a raster capped at `cos θz · 255 = 180` and
/// nearly constant. SurtGIS ≤ 1.2.4 used six full-circle azimuths and
/// produced exactly that degenerate output (reported from the WASM
/// bindings, 2026-08).
///
/// The aspect-dependent weights sum to exactly 2 over these four azimuths,
/// independent of aspect — the normalisation GDAL's
/// `gdaldem hillshade -multidirectional` relies on.
fn mark_azimuths_sin_cos() -> [(f64, f64); 4] {
    // Geographic azimuth -> the mathematical convention of the kernel below.
    let conv = |az_deg: f64| (360.0 - az_deg + 90.0).to_radians().sin_cos();
    [conv(225.0), conv(270.0), conv(315.0), conv(360.0)]
}

/// Mark (1992) aspect-dependent weight for one illumination azimuth, in the
/// exact algebra `gdaldem hillshade -multidirectional` uses:
/// `w270 = x²/g²`, `w360 = y²/g²`, `w225 = ½ − xy/g²`, `w315 = ½ + xy/g²`
/// (they sum to 2 for any aspect).
///
/// With `aspect = atan2(y, −x)` this equals `cos²(az − aspect)`: each cell is
/// dominated by the illumination direction most aligned with its aspect —
/// the direction that discriminates it most — which is what keeps the
/// blend's contrast. SurtGIS ≤ 1.2.4 used `1 + cross²/g²`: the complement,
/// plus a constant term that flattened the weighting towards a plain mean.
///
/// `g2` must be `dz_dx² + dz_dy²` and non-zero (flat cells are handled by
/// the caller).
#[inline]
fn mark_weight(dz_dx: f64, dz_dy: f64, g2: f64, sin_az: f64, cos_az: f64) -> f64 {
    let cross = dz_dx * sin_az + dz_dy * cos_az;
    (1.0 - (cross * cross) / g2).max(0.0)
}

/// Per-cell multidirectional hillshade kernel shared by the batch
/// (`multidirectional_hillshade`) and streaming
/// (`MultiHillshadeStreaming::process_row`) paths.
///
/// `a..i` is the validated (non-NaN) 3×3 neighborhood (see [`slope_kernel`
/// docs](super::slope) for the layout). `eight_dx`/`eight_dy` are
/// `8 * cell_size` for each axis, already resolved for the current row.
/// `az_sin_cos` are the four Mark (1992) azimuths from
/// [`mark_azimuths_sin_cos`].
///
/// Per azimuth and cell, with x = dz_dx, y = dz_dy, g² = x² + y²:
///
///   shade  = (cosθz + sinθz·(y·sin az − x·cos az)) / √(1+g²)
///   weight = cos²(az − aspect) = 1 − (x·sin az + y·cos az)² / g²
///            (aspect = atan2(y, −x); identical to GDAL's x²/g², y²/g²,
///             ½∓xy/g² weights for 270°, 360°, 225°/315°)
///
/// The weight is Mark's aspect-dependent weighting exactly as
/// `gdaldem hillshade -multidirectional` implements it: each cell is
/// dominated by the illumination direction most aligned with its aspect —
/// the direction that discriminates that cell most — which is what keeps
/// relief legible instead of averaging it away.
///
/// Flat cells (g² = 0): every azimuth shades to cosθz, so any equal
/// weighting yields the same blend — matching the old aspect=0 branch.
///
/// Returns the blended shade value in `[0, 1]` — the caller decides
/// whether to keep it normalized or rescale to `[0, 255]`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn multi_hillshade_shade(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    f: f64,
    g: f64,
    h: f64,
    i: f64,
    eight_dx: f64,
    eight_dy: f64,
    zf: f64,
    cos_zenith: f64,
    sin_zenith: f64,
    az_sin_cos: &[(f64, f64)],
) -> f64 {
    // Horn's method (z_factor scales z, per GDAL convention)
    let dz_dx = zf * ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / eight_dx;
    let dz_dy = zf * ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / eight_dy;

    let g2 = dz_dx * dz_dx + dz_dy * dz_dy;
    let inv_len = 1.0 / (1.0 + g2).sqrt();
    let flat = g2 < 1e-20;

    let mut weighted_sum = 0.0;
    let mut weight_total = 0.0;

    for &(sin_az, cos_az) in az_sin_cos {
        let shade =
            ((cos_zenith + sin_zenith * (dz_dy * sin_az - dz_dx * cos_az)) * inv_len).max(0.0);

        let w = if flat {
            1.0
        } else {
            mark_weight(dz_dx, dz_dy, g2, sin_az, cos_az)
        };

        weighted_sum += shade * w;
        weight_total += w;
    }

    if weight_total > 0.0 {
        (weighted_sum / weight_total).min(1.0)
    } else {
        0.0
    }
}

/// Calculate multidirectional hillshade
///
/// Blends hillshade from the four Mark (1992) azimuths (225°, 270°, 315°,
/// 360°) with GDAL's aspect-dependent weights. Output is comparable to
/// `gdaldem hillshade -multidirectional`.
///
/// Before 1.2.5 this blended six azimuths spread over the full compass,
/// whose directional terms cancel: the result was capped at
/// `cos θz · 255 = 180` and effectively constant. See
/// [`mark_azimuths_sin_cos`].
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Multidirectional hillshade parameters
///
/// # Returns
/// Raster with hillshade values (0-255 or 0.0-1.0)
pub fn multidirectional_hillshade(
    dem: &Raster<f64>,
    params: MultiHillshadeParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();

    // GDAL/ArcGIS semantics: z_factor scales the elevations. Geographic
    // rasters get automatic per-row metric cell sizes.
    let zf = params.z_factor;
    let cell_sizes = super::spheroidal_grid::CellSizes::for_dem(dem);

    let zenith_rad = (90.0 - params.altitude).to_radians();
    let cos_zenith = zenith_rad.cos();
    let sin_zenith = zenith_rad.sin();

    // Mark (1992): four azimuths over a half-circle. Everything per-cell is
    // algebraic — see `multi_hillshade_shade` for the derivation.
    let az_sin_cos = mark_azimuths_sin_cos();
    let normalized = params.normalized;

    let data = dem
        .data()
        .as_slice()
        .ok_or_else(|| Error::Other("raster data must be contiguous".into()))?;

    let output_data = par_map_rows(rows, cols, |row, out_row| {
        if row == 0 || row == rows - 1 {
            return; // edge rows stay NaN
        }
        let top = &data[(row - 1) * cols..row * cols];
        let mid = &data[row * cols..(row + 1) * cols];
        let bot = &data[(row + 1) * cols..(row + 2) * cols];

        let (dx, dy) = cell_sizes.at_row(row);
        let (eight_dx, eight_dy) = (8.0 * dx, 8.0 * dy);

        for col in 1..cols - 1 {
            let e = mid[col];
            if e.is_nan() || nodata.is_some_and(|nd| e == nd) {
                continue; // stays NaN
            }

            let (a, b, c) = (top[col - 1], top[col], top[col + 1]);
            let (d, f) = (mid[col - 1], mid[col + 1]);
            let (g, h, i) = (bot[col - 1], bot[col], bot[col + 1]);

            if a.is_nan()
                || b.is_nan()
                || c.is_nan()
                || d.is_nan()
                || f.is_nan()
                || g.is_nan()
                || h.is_nan()
                || i.is_nan()
            {
                continue; // stays NaN
            }

            let shade_val = multi_hillshade_shade(
                a,
                b,
                c,
                d,
                f,
                g,
                h,
                i,
                eight_dx,
                eight_dy,
                zf,
                cos_zenith,
                sin_zenith,
                &az_sin_cos,
            );

            out_row[col] = if normalized {
                shade_val
            } else {
                (shade_val * 255.0).round()
            };
        }
    });

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    // NaN, not 0.0: a shade of 0 is a valid value (full shadow).
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = output_data;

    Ok(output)
}

// ─── Streaming implementation ──────────────────────────────────────────

/// Streaming multi-directional hillshade calculator implementing `WindowAlgorithm`.
///
/// Processes a DEM strip-by-strip with bounded memory.
/// Uses the same USGS Mark (1992) method as `multidirectional_hillshade()`:
/// weighted blend of hillshade from the four half-circle azimuths.
#[derive(Debug, Clone)]
pub struct MultiHillshadeStreaming {
    /// Sun altitude above the horizon, in degrees.
    pub altitude: f64,
    /// Vertical exaggeration applied to elevations before shading.
    pub z_factor: f64,
    /// Whether to rescale the blended output to the full `[0, 255]` range.
    pub normalized: bool,
}

impl Default for MultiHillshadeStreaming {
    fn default() -> Self {
        Self {
            altitude: 45.0,
            z_factor: 1.0,
            normalized: false,
        }
    }
}

impl MultiHillshadeStreaming {
    /// Process a single output row given its already-resolved cell sizes
    /// (`eight_dx`/`eight_dy`) and pre-computed illumination geometry.
    /// Shared by the constant-cell-size path (`process_chunk`) and the
    /// per-row geographic-correction path (`process_chunk_geo`).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn process_row(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        r: usize,
        ir: usize,
        in_rows: usize,
        cols: usize,
        eight_dx: f64,
        eight_dy: f64,
        cos_zenith: f64,
        sin_zenith: f64,
        az_sin_cos: &[(f64, f64)],
    ) {
        if ir == 0 || ir >= in_rows - 1 {
            for c in 0..cols {
                output[[r, c]] = f64::NAN;
            }
            return;
        }

        // z_factor scales z (GDAL convention); dx and dy used separately.
        let zf = self.z_factor;

        for c in 0..cols {
            if c == 0 || c >= cols - 1 {
                output[[r, c]] = f64::NAN;
                continue;
            }

            let e = input[[ir, c]];
            if e.is_nan() || nodata.is_some_and(|nd| e == nd) {
                output[[r, c]] = f64::NAN;
                continue;
            }

            let a = input[[ir - 1, c - 1]];
            let b = input[[ir - 1, c]];
            let cv = input[[ir - 1, c + 1]];
            let d = input[[ir, c - 1]];
            let f = input[[ir, c + 1]];
            let g = input[[ir + 1, c - 1]];
            let h = input[[ir + 1, c]];
            let i = input[[ir + 1, c + 1]];

            if [a, b, cv, d, f, g, h, i].iter().any(|v| v.is_nan()) {
                output[[r, c]] = f64::NAN;
                continue;
            }

            let shade_val = multi_hillshade_shade(
                a, b, cv, d, f, g, h, i, eight_dx, eight_dy, zf, cos_zenith, sin_zenith, az_sin_cos,
            );

            output[[r, c]] = if self.normalized {
                shade_val
            } else {
                (shade_val * 255.0).round()
            };
        }
    }
}

impl surtgis_core::WindowAlgorithm for MultiHillshadeStreaming {
    fn kernel_radius(&self) -> usize {
        1 // 3x3 Horn kernel
    }

    fn process_chunk(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        cell_size_x: f64,
        cell_size_y: f64,
    ) {
        let (in_rows, cols) = input.dim();
        let out_rows = output.nrows();
        let radius = 1;

        let eight_dx = 8.0 * cell_size_x;
        let eight_dy = 8.0 * cell_size_y.abs();

        let zenith_rad = (90.0 - self.altitude).to_radians();
        let cos_zenith = zenith_rad.cos();
        let sin_zenith = zenith_rad.sin();

        // 6 equally-spaced azimuths converted to the mathematical convention
        let az_sin_cos = mark_azimuths_sin_cos();

        for r in 0..out_rows {
            let ir = r + radius; // input row corresponding to output row r
            self.process_row(
                input,
                output,
                nodata,
                r,
                ir,
                in_rows,
                cols,
                eight_dx,
                eight_dy,
                cos_zenith,
                sin_zenith,
                &az_sin_cos,
            );
        }
    }

    /// REG-1 fix: see `SlopeStreaming::process_chunk_geo` — same mechanism,
    /// applied to the 6-azimuth blend so the multidirectional hillshade is
    /// no longer silently wrong on geographic CRSs in streaming mode.
    fn process_chunk_geo(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        cell_size_x: f64,
        cell_size_y: f64,
        geo_ctx: Option<surtgis_core::GeoRowContext>,
    ) {
        let Some(ctx) = geo_ctx else {
            return self.process_chunk(input, output, nodata, cell_size_x, cell_size_y);
        };

        let (in_rows, cols) = input.dim();
        let out_rows = output.nrows();
        let radius = 1;

        let zenith_rad = (90.0 - self.altitude).to_radians();
        let cos_zenith = zenith_rad.cos();
        let sin_zenith = zenith_rad.sin();
        let az_sin_cos = mark_azimuths_sin_cos();

        for r in 0..out_rows {
            let ir = r + radius;
            let abs_row = ctx.row_offset + r;
            let lat = ctx.origin_y + (abs_row as f64 + 0.5) * ctx.pixel_height;
            let dims = super::spheroidal_grid::cell_dimensions(
                lat,
                cell_size_x,
                cell_size_y,
                &super::spheroidal_grid::SpheroidalParams::default(),
            );
            let eight_dx = 8.0 * dims.dx;
            let eight_dy = 8.0 * dims.dy;
            self.process_row(
                input,
                output,
                nodata,
                r,
                ir,
                in_rows,
                cols,
                eight_dx,
                eight_dy,
                cos_zenith,
                sin_zenith,
                &az_sin_cos,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn test_dem() -> Raster<f64> {
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, (row + col) as f64 * 10.0).unwrap();
            }
        }
        dem
    }

    /// The algebraic kernel (shade and weight both trig-free) must match
    /// the classic Mark (1992) formulation: shade = cosθz·cos s +
    /// sinθz·sin s·cos(az − aspect), w = 1 + cos²(az − aspect + π/2).
    #[test]
    fn test_algebraic_matches_trig_formulation() {
        use std::f64::consts::PI;
        let n = 30;
        let mut dem = Raster::new(n, n);
        dem.set_transform(GeoTransform::new(0.0, n as f64, 1.0, -1.0));
        for r in 0..n {
            for c in 0..n {
                let z = (r as f64 * 0.41).sin() * 30.0
                    + (c as f64 * 0.29).cos() * 20.0
                    + r as f64 * 0.9;
                dem.set(r, c, z).unwrap();
            }
        }

        let result = multidirectional_hillshade(
            &dem,
            MultiHillshadeParams {
                normalized: true,
                ..Default::default()
            },
        )
        .unwrap();

        let zenith_rad = (90.0f64 - 45.0).to_radians();
        let azimuths_rad: Vec<f64> = [225.0f64, 270.0, 315.0, 360.0]
            .iter()
            .map(|az| (360.0 - az + 90.0).to_radians())
            .collect();
        let data = dem.data();
        for r in 1..n - 1 {
            for c in 1..n - 1 {
                let dz_dx =
                    ((data[[r - 1, c + 1]] + 2.0 * data[[r, c + 1]] + data[[r + 1, c + 1]])
                        - (data[[r - 1, c - 1]] + 2.0 * data[[r, c - 1]] + data[[r + 1, c - 1]]))
                        / 8.0;
                let dz_dy =
                    ((data[[r + 1, c - 1]] + 2.0 * data[[r + 1, c]] + data[[r + 1, c + 1]])
                        - (data[[r - 1, c - 1]] + 2.0 * data[[r - 1, c]] + data[[r - 1, c + 1]]))
                        / 8.0;
                let slope_rad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();
                let aspect_rad = dz_dy.atan2(-dz_dx);
                let (mut ws, mut wt) = (0.0, 0.0);
                for az in &azimuths_rad {
                    let shade = (zenith_rad.cos() * slope_rad.cos()
                        + zenith_rad.sin() * slope_rad.sin() * (az - aspect_rad).cos())
                    .max(0.0);
                    // Mark/GDAL weight in this aspect convention
                    // (aspect = atan2(dz_dy, −dz_dx)): cos²(az − aspect).
                    let w = (az - aspect_rad).cos().powi(2);
                    ws += shade * w;
                    wt += w;
                }
                let expected = (ws / wt).min(1.0);
                let got = result.get(r, c).unwrap();
                assert!(
                    (got - expected).abs() < 1e-12,
                    "at ({},{}): algebraic {} vs trig {}",
                    r,
                    c,
                    got,
                    expected
                );
            }
        }
    }

    #[test]
    fn weights_match_gdal_multidirectional_algebra() {
        // GDAL gdaldem_lib.cpp (GDALHillshadeMultiDirectionalAlg) normalises
        // these by xx_plus_yy: w225 = 0.5*g² − x*y, w270 = x², w315 = g² −
        // w225, w360 = y². Pinning them here means the blend stays
        // GDAL-compatible even if the kernel is refactored again.
        let az = mark_azimuths_sin_cos(); // [225, 270, 315, 360]
        for &(x, y) in &[
            (1.0, 0.0),
            (0.0, 1.0),
            (0.7, -0.3),
            (-1.4, 2.1),
            (3.0, 3.0),
            (-0.2, -0.9),
        ] {
            let g2 = x * x + y * y;
            let w: Vec<f64> = az
                .iter()
                .map(|&(s, c)| mark_weight(x, y, g2, s, c))
                .collect();
            let gdal = [
                (0.5 * g2 - x * y) / g2,
                (x * x) / g2,
                (g2 - (0.5 * g2 - x * y)) / g2,
                (y * y) / g2,
            ];
            for k in 0..4 {
                assert!(
                    (w[k] - gdal[k]).abs() < 1e-12,
                    "weight {k} for (x={x}, y={y}): {} vs GDAL {}",
                    w[k],
                    gdal[k]
                );
            }
            // The normalisation the whole method rests on.
            let sum: f64 = w.iter().sum();
            assert!(
                (sum - 2.0).abs() < 1e-12,
                "weights must sum to 2, got {sum}"
            );
        }
    }

    #[test]
    fn multidirectional_is_not_capped_at_the_flat_value() {
        // Regression (reported from the WASM bindings, 2026-08): blending six
        // azimuths spread over the full compass makes the directional terms
        // cancel (Σ sin az = Σ cos az = 0), so every cell collapsed to
        // cos θz · 1/√(1+g²) — an output capped at cos(45°)·255 = 180.3 whose
        // percentiles were *all* exactly 180 on real DEMs.
        //
        // Sentinel: on terrain with varied aspects some cells must come out
        // BRIGHTER than the flat value (slopes tilted toward the illumination)
        // and others much darker. Both are unreachable under the old blend.
        let n = 80;
        let mut dem: Raster<f64> = Raster::new(n, n);
        dem.set_transform(GeoTransform::new(0.0, n as f64 * 30.0, 30.0, -30.0));
        for r in 0..n {
            for c in 0..n {
                // A cone: every aspect is present somewhere on the grid.
                let (dr, dc) = (r as f64 - 40.0, c as f64 - 40.0);
                dem.set(r, c, 400.0 - 6.0 * (dr * dr + dc * dc).sqrt())
                    .unwrap();
            }
        }

        let out = multidirectional_hillshade(&dem, MultiHillshadeParams::default()).unwrap();
        let mut v: Vec<f64> = out
            .data()
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let flat_value = (90.0f64 - 45.0).to_radians().cos() * 255.0; // 180.31

        assert!(
            *v.last().unwrap() > flat_value + 5.0,
            "no cell is brighter than the flat value {flat_value:.1} (max = {}); \
             the blend collapsed to the degenerate full-circle average",
            v.last().unwrap()
        );
        let p05 = v[v.len() / 20];
        let p95 = v[v.len() * 19 / 20];
        // This cone spreads p05..p95 over ~60 shading levels; the degenerate
        // blend spread ~0. 40 leaves margin on both sides.
        assert!(
            p95 - p05 > 40.0,
            "distribution is saturated: p05={p05:.0}, p95={p95:.0} (spread \
             {:.0} < 40); the pre-1.2.5 blend produced p25..p99 all equal",
            p95 - p05
        );
    }

    #[test]
    fn test_multihillshade_range() {
        let dem = test_dem();
        let result = multidirectional_hillshade(&dem, MultiHillshadeParams::default()).unwrap();

        for row in 0..result.rows() {
            for col in 0..result.cols() {
                let val = result.get(row, col).unwrap();
                if val.is_nan() {
                    continue; // edges / nodata
                }
                assert!(
                    val >= 0.0 && val <= 255.0,
                    "Value {} out of range at ({}, {})",
                    val,
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_multihillshade_flat() {
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = multidirectional_hillshade(&dem, MultiHillshadeParams::default()).unwrap();
        let val = result.get(5, 5).unwrap();

        // Flat surface at 45° altitude → shade ≈ cos(45°) ≈ 0.707 → ~180
        assert!(
            (val - 180.0).abs() < 20.0,
            "Expected ~180 for flat surface, got {}",
            val
        );
    }

    #[test]
    fn test_multihillshade_less_directional_bias() {
        // Multi-hillshade should produce more uniform shading than single
        let dem = test_dem();
        let multi = multidirectional_hillshade(
            &dem,
            MultiHillshadeParams {
                normalized: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Interior values should all be > 0 (no completely dark faces)
        let mut all_positive = true;
        for row in 1..9 {
            for col in 1..9 {
                let val = multi.get(row, col).unwrap();
                if val <= 0.0 {
                    all_positive = false;
                }
            }
        }
        assert!(
            all_positive,
            "Multi-hillshade should avoid fully dark pixels"
        );
    }
}
