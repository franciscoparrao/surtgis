//! Raster resampling: align a raster to a reference grid.
//!
//! Resamples a source raster to match the exact grid (origin, cell size,
//! dimensions) of a reference raster. Essential for stacking rasters from
//! different sources (e.g., Copernicus DEM at 30m + Sentinel-2 at 10m).

use ndarray::Array2;

use crate::error::Result;
use crate::raster::Raster;

/// Interpolation method for resampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMethod {
    /// Nearest neighbor — fast, preserves discrete values (classification, SCL)
    NearestNeighbor,
    /// Bilinear interpolation — smooth, best for continuous data (elevation, indices)
    Bilinear,
    /// Area-weighted average of all source pixels overlapping each output
    /// cell — the correct choice for *downsampling* continuous data (e.g.
    /// Sentinel-2 10m → 30m composite grid). Unlike bilinear, which only
    /// samples 4 neighbors around the output cell center, `Average`
    /// integrates the energy of every contributing source pixel, avoiding
    /// aliasing of high-frequency content.
    ///
    /// When an output cell is *smaller* than a source pixel (i.e. genuine
    /// upsampling), there is no source pixel to average over, so this falls
    /// back to bilinear interpolation for that cell.
    Average,
    /// Bicubic interpolation (4x4 = 16 neighbor window) using the Catmull-Rom
    /// kernel (`a = -0.5`, matching GDAL's default `cubic` resampling).
    /// Reproduces smooth continuous surfaces (e.g. elevation) with less
    /// blurring than bilinear, and is exact for polynomials up to degree 3.
    ///
    /// If any of the 16 neighbors is nodata/NaN, this falls back to
    /// bilinear for that cell (a full cubic reconstruction is more sensitive
    /// to missing data than bilinear, since it touches 4x more neighbors).
    Cubic,
}

/// Resample a raster to match the grid of a reference raster.
///
/// The output has the same dimensions, transform, and CRS as `reference`.
/// Each output pixel is computed by mapping its geographic coordinate back
/// to the source raster and interpolating.
///
/// # Arguments
/// * `source` - Raster to resample
/// * `reference` - Reference raster defining the target grid
/// * `method` - Interpolation method
///
/// # Example
/// ```ignore
/// // Resample Sentinel-2 band (10m) to match DEM grid (30m)
/// let aligned = resample_to_grid(&s2_band, &dem, ResampleMethod::Bilinear)?;
/// ```
pub fn resample_to_grid(
    source: &Raster<f64>,
    reference: &Raster<f64>,
    method: ResampleMethod,
) -> Result<Raster<f64>> {
    let (out_rows, out_cols) = reference.shape();
    let ref_gt = reference.transform();
    let src_gt = source.transform();
    let (src_rows, src_cols) = source.shape();
    let src_data = source.data();
    let nodata = source.nodata();

    let mut output = Array2::<f64>::from_elem((out_rows, out_cols), f64::NAN);

    for out_row in 0..out_rows {
        for out_col in 0..out_cols {
            // Geographic coordinate of output pixel center
            let geo_x = ref_gt.origin_x + (out_col as f64 + 0.5) * ref_gt.pixel_width;
            let geo_y = ref_gt.origin_y + (out_row as f64 + 0.5) * ref_gt.pixel_height;

            // Map to source pixel coordinates (continuous)
            let src_col_f = (geo_x - src_gt.origin_x) / src_gt.pixel_width - 0.5;
            let src_row_f = (geo_y - src_gt.origin_y) / src_gt.pixel_height - 0.5;

            // Check if source pixel is within bounds
            if src_col_f < -0.5
                || src_row_f < -0.5
                || src_col_f >= src_cols as f64 - 0.5
                || src_row_f >= src_rows as f64 - 0.5
            {
                continue; // NaN (outside source extent)
            }

            output[[out_row, out_col]] = match method {
                ResampleMethod::NearestNeighbor => {
                    let r = src_row_f.round().max(0.0) as usize;
                    let c = src_col_f.round().max(0.0) as usize;
                    let r = r.min(src_rows - 1);
                    let c = c.min(src_cols - 1);
                    let v = src_data[[r, c]];
                    if is_nodata(v, nodata) { f64::NAN } else { v }
                }
                ResampleMethod::Bilinear => {
                    bilinear_value(src_data, src_rows, src_cols, nodata, src_row_f, src_col_f)
                }
                ResampleMethod::Average => {
                    // Geographic bounds of the output cell (world coordinates).
                    let (gx0, gy0) = ref_gt.pixel_to_geo_corner(out_col, out_row);
                    let (gx1, gy1) = ref_gt.pixel_to_geo_corner(out_col + 1, out_row + 1);

                    // Map those corners into source pixel-index space (integer
                    // values = pixel boundaries, matching GDAL convention).
                    let (pc0, pr0) = src_gt.geo_to_pixel(gx0, gy0);
                    let (pc1, pr1) = src_gt.geo_to_pixel(gx1, gy1);

                    if pc0.is_nan() || pr0.is_nan() || pc1.is_nan() || pr1.is_nan() {
                        bilinear_value(src_data, src_rows, src_cols, nodata, src_row_f, src_col_f)
                    } else {
                        let col_min = pc0.min(pc1);
                        let col_max = pc0.max(pc1);
                        let row_min = pr0.min(pr1);
                        let row_max = pr0.max(pr1);

                        // If the output cell is smaller than a source pixel in
                        // either dimension (upsampling), there is no whole
                        // source pixel to average over — fall back to bilinear.
                        if (row_max - row_min) < 1.0 || (col_max - col_min) < 1.0 {
                            bilinear_value(
                                src_data, src_rows, src_cols, nodata, src_row_f, src_col_f,
                            )
                        } else {
                            average_value(
                                src_data, src_rows, src_cols, nodata, row_min, row_max, col_min,
                                col_max,
                            )
                        }
                    }
                }
                ResampleMethod::Cubic => {
                    cubic_value(src_data, src_rows, src_cols, nodata, src_row_f, src_col_f)
                }
            };
        }
    }

    let mut result = Raster::from_array(output);
    result.set_transform(ref_gt.clone());
    result.set_nodata(Some(f64::NAN));
    if let Some(crs) = reference.crs() {
        result.set_crs(Some(crs.clone()));
    }
    Ok(result)
}

fn is_nodata(v: f64, nodata: Option<f64>) -> bool {
    if !v.is_finite() {
        return true;
    }
    if let Some(nd) = nodata {
        if nd.is_finite() && (v - nd).abs() < 1e-10 {
            return true;
        }
    }
    false
}

/// NaN-tolerant bilinear interpolation at continuous source pixel coordinates
/// `(src_row_f, src_col_f)`. Shared by `ResampleMethod::Bilinear` and as the
/// fallback for `Average` (upsampling case) and `Cubic` (nodata in window).
#[allow(clippy::too_many_arguments)]
fn bilinear_value(
    src_data: &Array2<f64>,
    src_rows: usize,
    src_cols: usize,
    nodata: Option<f64>,
    src_row_f: f64,
    src_col_f: f64,
) -> f64 {
    let r0 = src_row_f.floor().max(0.0) as usize;
    let c0 = src_col_f.floor().max(0.0) as usize;
    let r0 = r0.min(src_rows - 1);
    let c0 = c0.min(src_cols - 1);
    let r1 = (r0 + 1).min(src_rows - 1);
    let c1 = (c0 + 1).min(src_cols - 1);

    let dr = src_row_f - r0 as f64;
    let dc = src_col_f - c0 as f64;
    let dr = dr.clamp(0.0, 1.0);
    let dc = dc.clamp(0.0, 1.0);

    let v00 = src_data[[r0, c0]];
    let v01 = src_data[[r0, c1]];
    let v10 = src_data[[r1, c0]];
    let v11 = src_data[[r1, c1]];

    // Weighted average of valid neighbors (NaN-tolerant bilinear).
    // This prevents NaN borders at internal COG tile edges from
    // producing stripe artifacts during resampling.
    let neighbors = [
        (v00, (1.0 - dc) * (1.0 - dr)),
        (v01, dc * (1.0 - dr)),
        (v10, (1.0 - dc) * dr),
        (v11, dc * dr),
    ];
    let mut wsum = 0.0;
    let mut wtotal = 0.0;
    for &(v, w) in &neighbors {
        if !is_nodata(v, nodata) && v.is_finite() {
            wsum += v * w;
            wtotal += w;
        }
    }
    if wtotal > 0.0 {
        wsum / wtotal
    } else {
        f64::NAN
    }
}

/// Area-weighted average of every source pixel overlapping the rectangle
/// `[row_min, row_max) x [col_min, col_max)` in source pixel-index space
/// (integer values = pixel boundaries, fractional = partial overlap).
///
/// Each contributing pixel is weighted by the fraction of its area that
/// falls inside the rectangle, so partially-covered pixels at the edges of
/// the output cell contribute proportionally rather than all-or-nothing.
/// Nodata/NaN pixels are excluded from both the weighted sum and the total
/// weight; if no valid pixel overlaps, the result is NaN.
#[allow(clippy::too_many_arguments)]
fn average_value(
    src_data: &Array2<f64>,
    src_rows: usize,
    src_cols: usize,
    nodata: Option<f64>,
    row_min: f64,
    row_max: f64,
    col_min: f64,
    col_max: f64,
) -> f64 {
    let r_start = row_min.floor().max(0.0) as isize;
    let r_end = row_max.ceil().min(src_rows as f64) as isize;
    let c_start = col_min.floor().max(0.0) as isize;
    let c_end = col_max.ceil().min(src_cols as f64) as isize;

    let mut wsum = 0.0;
    let mut wtotal = 0.0;

    for r in r_start..r_end {
        if r < 0 || r >= src_rows as isize {
            continue;
        }
        let r_lo = r as f64;
        let r_hi = r_lo + 1.0;
        let overlap_r = (r_hi.min(row_max) - r_lo.max(row_min)).max(0.0);
        if overlap_r <= 0.0 {
            continue;
        }
        for c in c_start..c_end {
            if c < 0 || c >= src_cols as isize {
                continue;
            }
            let c_lo = c as f64;
            let c_hi = c_lo + 1.0;
            let overlap_c = (c_hi.min(col_max) - c_lo.max(col_min)).max(0.0);
            if overlap_c <= 0.0 {
                continue;
            }

            let w = overlap_r * overlap_c;
            let v = src_data[[r as usize, c as usize]];
            if !is_nodata(v, nodata) && v.is_finite() {
                wsum += v * w;
                wtotal += w;
            }
        }
    }

    if wtotal > 0.0 {
        wsum / wtotal
    } else {
        f64::NAN
    }
}

/// 1D cubic convolution kernel (Catmull-Rom, `a = -0.5`), matching GDAL's
/// default `cubic` resampling kernel.
///
/// `w(x) = (a+2)|x|^3 - (a+3)|x|^2 + 1`        for `|x| <= 1`
/// `w(x) = a|x|^3 - 5a|x|^2 + 8a|x| - 4a`       for `1 < |x| <= 2`
/// `w(x) = 0`                                   otherwise
fn cubic_kernel(x: f64) -> f64 {
    const A: f64 = -0.5;
    let ax = x.abs();
    if ax <= 1.0 {
        (A + 2.0) * ax.powi(3) - (A + 3.0) * ax.powi(2) + 1.0
    } else if ax <= 2.0 {
        A * ax.powi(3) - 5.0 * A * ax.powi(2) + 8.0 * A * ax - 4.0 * A
    } else {
        0.0
    }
}

/// Bicubic interpolation over a 4x4 neighborhood (separable Catmull-Rom
/// kernel applied along rows then columns). Out-of-range indices are
/// clamped to the raster bounds (edge replication). If any of the 16
/// neighbors is nodata/NaN, falls back to bilinear interpolation to avoid
/// propagating NaN through the wider cubic window.
#[allow(clippy::too_many_arguments)]
fn cubic_value(
    src_data: &Array2<f64>,
    src_rows: usize,
    src_cols: usize,
    nodata: Option<f64>,
    src_row_f: f64,
    src_col_f: f64,
) -> f64 {
    let r0 = src_row_f.floor() as isize;
    let c0 = src_col_f.floor() as isize;
    let t = (src_row_f - r0 as f64).clamp(0.0, 1.0);
    let s = (src_col_f - c0 as f64).clamp(0.0, 1.0);

    let clamp_idx = |i: isize, len: usize| -> usize { i.clamp(0, len as isize - 1) as usize };

    // Gather the 4x4 window, clamping indices at the borders (edge replication).
    let mut window = [[0.0f64; 4]; 4];
    let mut has_nodata = false;
    for (wi, ki) in (-1..=2).enumerate() {
        let r = clamp_idx(r0 + ki, src_rows);
        for (wj, kj) in (-1..=2).enumerate() {
            let c = clamp_idx(c0 + kj, src_cols);
            let v = src_data[[r, c]];
            if is_nodata(v, nodata) || !v.is_finite() {
                has_nodata = true;
            }
            window[wi][wj] = v;
        }
    }

    if has_nodata {
        return bilinear_value(src_data, src_rows, src_cols, nodata, src_row_f, src_col_f);
    }

    // Row weights (for the 4 sample rows r0-1..=r0+2) and column weights,
    // using the Catmull-Rom kernel evaluated at the signed distance to each
    // sample position.
    let wy: [f64; 4] = [
        cubic_kernel(t + 1.0),
        cubic_kernel(t),
        cubic_kernel(t - 1.0),
        cubic_kernel(t - 2.0),
    ];
    let wx: [f64; 4] = [
        cubic_kernel(s + 1.0),
        cubic_kernel(s),
        cubic_kernel(s - 1.0),
        cubic_kernel(s - 2.0),
    ];

    let mut result = 0.0;
    for i in 0..4 {
        let mut row_val = 0.0;
        for j in 0..4 {
            row_val += window[i][j] * wx[j];
        }
        result += row_val * wy[i];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::GeoTransform;

    fn make_raster(rows: usize, cols: usize, value: f64, gt: GeoTransform) -> Raster<f64> {
        let arr = Array2::from_elem((rows, cols), value);
        let mut r = Raster::from_array(arr);
        r.set_transform(gt);
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_resample_same_grid() {
        // Same grid → output equals input
        let gt = GeoTransform::new(0.0, 100.0, 10.0, -10.0);
        let src = make_raster(10, 10, 42.0, gt);
        let reference = make_raster(10, 10, 0.0, gt);

        let result = resample_to_grid(&src, &reference, ResampleMethod::NearestNeighbor).unwrap();
        assert!((result.data()[[5, 5]] - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_resample_downsample() {
        // Source 10m → reference 30m (3x downsample)
        let src_gt = GeoTransform::new(0.0, 300.0, 10.0, -10.0);
        let ref_gt = GeoTransform::new(0.0, 300.0, 30.0, -30.0);

        // Source: 30x30 at 10m, all value 100.0
        let src = make_raster(30, 30, 100.0, src_gt);
        // Reference: 10x10 at 30m
        let reference = make_raster(10, 10, 0.0, ref_gt);

        let result = resample_to_grid(&src, &reference, ResampleMethod::Bilinear).unwrap();
        assert_eq!(result.shape(), (10, 10));
        // All values should be 100.0 (uniform source)
        assert!((result.data()[[5, 5]] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_resample_with_offset() {
        // Source and reference have different origins (the real-world case)
        let src_gt = GeoTransform::new(100.0, 500.0, 10.0, -10.0);
        let ref_gt = GeoTransform::new(115.0, 485.0, 30.0, -30.0); // offset by 15m,15m

        // Source: gradient where value = col index
        let mut src_data = Array2::<f64>::zeros((40, 40));
        for r in 0..40 {
            for c in 0..40 {
                src_data[[r, c]] = c as f64;
            }
        }
        let mut src = Raster::from_array(src_data);
        src.set_transform(src_gt);
        src.set_nodata(Some(f64::NAN));

        let reference = make_raster(10, 10, 0.0, ref_gt);

        let result = resample_to_grid(&src, &reference, ResampleMethod::Bilinear).unwrap();
        assert_eq!(result.shape(), (10, 10));
        // First output pixel center is at (115+15, 485-15) = (130, 470)
        // In source: col = (130 - 100) / 10 - 0.5 = 2.5 → bilinear between col 2 and 3
        let v = result.data()[[0, 0]];
        assert!(v.is_finite());
        assert!((v - 2.5).abs() < 0.5, "Expected ~2.5, got {}", v);
    }

    #[test]
    fn test_resample_outside_bounds() {
        // Reference extends beyond source → NaN for out-of-bounds pixels
        let src_gt = GeoTransform::new(100.0, 200.0, 10.0, -10.0);
        let ref_gt = GeoTransform::new(0.0, 300.0, 10.0, -10.0); // extends far beyond source

        let src = make_raster(10, 10, 42.0, src_gt);
        let reference = make_raster(30, 30, 0.0, ref_gt);

        let result = resample_to_grid(&src, &reference, ResampleMethod::NearestNeighbor).unwrap();
        // Pixel at (0,0) maps to geo (5, 295), source origin is (100, 200) → outside
        assert!(result.data()[[0, 0]].is_nan());
        // Pixel near source area should have value
        // Source covers x=[100,200], y=[100,200]. ref pixel at col=10,row=10 → geo(105,195)
        let v = result.data()[[10, 10]];
        assert!(v.is_finite());
    }

    // ------------------------------------------------------------------
    // Average
    // ------------------------------------------------------------------

    #[test]
    fn test_average_uniform_raster_is_trivial() {
        // Average over a uniform source must reproduce that same value,
        // regardless of how many source pixels contribute.
        let src_gt = GeoTransform::new(0.0, 300.0, 10.0, -10.0);
        let ref_gt = GeoTransform::new(0.0, 300.0, 30.0, -30.0);

        let src = make_raster(30, 30, 100.0, src_gt);
        let reference = make_raster(10, 10, 0.0, ref_gt);

        let result = resample_to_grid(&src, &reference, ResampleMethod::Average).unwrap();
        assert_eq!(result.shape(), (10, 10));
        for r in 0..10 {
            for c in 0..10 {
                assert!(
                    (result.data()[[r, c]] - 100.0).abs() < 1e-9,
                    "cell ({r},{c}) = {}",
                    result.data()[[r, c]]
                );
            }
        }
    }

    /// The core anti-aliasing claim: on a high-frequency checkerboard pattern
    /// downsampled 4:1, `Average` integrates the full contributing area and
    /// lands very close to the true mean (0.5), while `Bilinear` only samples
    /// a 2x2 neighborhood around the output cell center and — depending on
    /// the sub-pixel phase between source and reference grids (the common
    /// real-world case of misaligned tile origins) — can land far from 0.5.
    #[test]
    fn test_average_avoids_aliasing_vs_bilinear_on_checkerboard() {
        // 16x16 checkerboard, value = (row+col) % 2, at 10m resolution.
        let src_gt = GeoTransform::new(0.0, 160.0, 10.0, -10.0);
        let mut src_data = Array2::<f64>::zeros((16, 16));
        for r in 0..16 {
            for c in 0..16 {
                src_data[[r, c]] = ((r + c) % 2) as f64;
            }
        }
        let mut src = Raster::from_array(src_data);
        src.set_transform(src_gt);
        src.set_nodata(Some(f64::NAN));

        // Reference grid: single 40m cell, deliberately offset (not an exact
        // multiple of the source grid) so bilinear's sample point lands at a
        // biased sub-pixel phase (dr = dc = 0.1) instead of the symmetric
        // dr = dc = 0.5 case, mirroring real STAC composites where tiles
        // from different sources rarely share a pixel-aligned origin.
        let ref_gt = GeoTransform::new(46.0, 114.0, 40.0, -40.0);
        let reference = make_raster(1, 1, 0.0, ref_gt);

        let avg = resample_to_grid(&src, &reference, ResampleMethod::Average).unwrap();
        let bil = resample_to_grid(&src, &reference, ResampleMethod::Bilinear).unwrap();

        let avg_v = avg.data()[[0, 0]];
        let bil_v = bil.data()[[0, 0]];

        assert!(avg_v.is_finite());
        assert!(bil_v.is_finite());

        // Average integrates the whole 4x4 contributing footprint → ~0.5.
        assert!(
            (avg_v - 0.5).abs() < 0.05,
            "Average should be close to the true mean 0.5, got {avg_v}"
        );

        // Bilinear only samples the 2x2 neighborhood around the (biased)
        // output cell center → far from 0.5 (aliased towards one value).
        assert!(
            (bil_v - 0.5).abs() > 0.2,
            "Expected Bilinear to alias away from 0.5 on this checkerboard \
             (demonstrating why Average exists for downsampling), got {bil_v}"
        );
    }

    #[test]
    fn test_average_upsampling_falls_back_to_bilinear() {
        // Output cell smaller than a source pixel in both dimensions
        // (upsampling) → Average has no whole pixel to integrate over and
        // must fall back to bilinear, producing an identical result.
        let src_gt = GeoTransform::new(0.0, 400.0, 100.0, -100.0);
        let ref_gt = GeoTransform::new(0.0, 400.0, 25.0, -25.0);

        let mut src_data = Array2::<f64>::zeros((4, 4));
        for r in 0..4 {
            for c in 0..4 {
                src_data[[r, c]] = (r * 4 + c) as f64;
            }
        }
        let mut src = Raster::from_array(src_data);
        src.set_transform(src_gt);
        src.set_nodata(Some(f64::NAN));

        let reference = make_raster(16, 16, 0.0, ref_gt);

        let avg = resample_to_grid(&src, &reference, ResampleMethod::Average).unwrap();
        let bil = resample_to_grid(&src, &reference, ResampleMethod::Bilinear).unwrap();

        for r in 0..16 {
            for c in 0..16 {
                let a = avg.data()[[r, c]];
                let b = bil.data()[[r, c]];
                if a.is_nan() && b.is_nan() {
                    continue;
                }
                assert!(
                    (a - b).abs() < 1e-9,
                    "cell ({r},{c}): Average={a} should equal Bilinear fallback={b}"
                );
            }
        }
    }

    /// Cross-check against GDAL's own `-r average` resampler. Skips (rather
    /// than fails) if `gdal_translate` isn't installed on this machine, so
    /// the test suite stays portable. Not expected to be bit-exact (area
    /// weighting details differ) but both should land close to the same
    /// integrated mean, unlike bilinear.
    #[test]
    fn test_average_matches_gdal_translate_order_of_magnitude() {
        use std::process::Command;

        if Command::new("gdal_translate")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("skipping: gdal_translate not found on PATH");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let src_path = dir.path().join("checkerboard.tif");
        let out_path = dir.path().join("checkerboard_avg.tif");

        // 32x32 high-frequency checkerboard at 10m, true mean = 0.5.
        let src_gt = GeoTransform::new(0.0, 320.0, 10.0, -10.0);
        let mut src_data = Array2::<f64>::zeros((32, 32));
        for r in 0..32 {
            for c in 0..32 {
                src_data[[r, c]] = ((r + c) % 2) as f64;
            }
        }
        let mut src = Raster::from_array(src_data);
        src.set_transform(src_gt);
        src.set_nodata(Some(f64::NAN));
        src.set_crs(Some(crate::crs::CRS::from_epsg(32719)));

        crate::io::write_geotiff(&src, &src_path, None).unwrap();

        // Downsample 4:1 to 40m using our Average.
        let ref_gt = GeoTransform::new(0.0, 320.0, 40.0, -40.0);
        let reference = make_raster(8, 8, 0.0, ref_gt);
        let ours = resample_to_grid(&src, &reference, ResampleMethod::Average).unwrap();

        // Downsample 4:1 to 40m using gdal_translate -r average.
        let status = Command::new("gdal_translate")
            .args([
                "-r",
                "average",
                "-outsize",
                "8",
                "8",
                src_path.to_str().unwrap(),
                out_path.to_str().unwrap(),
            ])
            .status()
            .expect("failed to invoke gdal_translate");
        assert!(status.success(), "gdal_translate exited with failure");

        let gdal_result: Raster<f64> = crate::io::read_geotiff(&out_path, None).unwrap();

        let our_mean: f64 = ours.data().iter().sum::<f64>() / (ours.data().len() as f64);
        let gdal_mean: f64 =
            gdal_result.data().iter().sum::<f64>() / (gdal_result.data().len() as f64);

        eprintln!(
            "Average interop: ours mean = {our_mean:.6}, gdal_translate -r average mean = {gdal_mean:.6}"
        );

        // Both should integrate the checkerboard's true mean (0.5); neither
        // should show the aliasing bias a naive point-sample method would.
        assert!(
            (our_mean - 0.5).abs() < 0.05,
            "our Average mean {our_mean} too far from 0.5"
        );
        assert!(
            (gdal_mean - 0.5).abs() < 0.05,
            "gdal_translate -r average mean {gdal_mean} too far from 0.5"
        );
        assert!(
            (our_mean - gdal_mean).abs() < 0.1,
            "our Average ({our_mean}) and gdal_translate -r average ({gdal_mean}) disagree by more than 0.1"
        );
    }

    // ------------------------------------------------------------------
    // Cubic
    // ------------------------------------------------------------------

    #[test]
    fn test_cubic_reproduces_plane_exactly() {
        // Cubic convolution (Catmull-Rom, a=-0.5) is exact for polynomials up
        // to degree 3; a plane z = a*x + b*y + c0 is degree 1, so it must be
        // reproduced exactly (away from raster borders, where clamping would
        // introduce edge effects).
        let a = 2.0;
        let b = -3.0;
        let c0 = 5.0;

        let src_gt = GeoTransform::new(0.0, 400.0, 10.0, -10.0);
        let mut src_data = Array2::<f64>::zeros((40, 40));
        for r in 0..40 {
            for c in 0..40 {
                let (x, y) = src_gt.pixel_to_geo(c, r);
                src_data[[r, c]] = a * x + b * y + c0;
            }
        }
        let mut src = Raster::from_array(src_data);
        src.set_transform(src_gt);
        src.set_nodata(Some(f64::NAN));

        // Reference grid well within the interior (needs a 4x4 neighborhood
        // margin on every side), non-pixel-aligned origin.
        let ref_gt = GeoTransform::new(153.0, 247.0, 10.0, -10.0);
        let reference = make_raster(5, 5, 0.0, ref_gt);

        let result = resample_to_grid(&src, &reference, ResampleMethod::Cubic).unwrap();

        for r in 0..5 {
            for c in 0..5 {
                let (x, y) = ref_gt.pixel_to_geo(c, r);
                let expected = a * x + b * y + c0;
                let got = result.data()[[r, c]];
                assert!(
                    (got - expected).abs() < 1e-9,
                    "cell ({r},{c}): expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn test_cubic_nan_neighbor_falls_back_to_bilinear_without_propagating_nan() {
        // A single NaN anywhere in the 4x4 cubic window must not turn the
        // whole cell into NaN — it should fall back to (NaN-tolerant)
        // bilinear instead.
        let mut data = Array2::<f64>::zeros((6, 6));
        for r in 0..6 {
            for c in 0..6 {
                data[[r, c]] = (r * 6 + c) as f64;
            }
        }
        // Poison one neighbor inside the 4x4 window used for (src_row_f,
        // src_col_f) = (2.4, 2.4) → window rows/cols 1..=4.
        data[[1, 4]] = f64::NAN;

        let src_row_f = 2.4;
        let src_col_f = 2.4;

        let cubic = cubic_value(&data, 6, 6, Some(f64::NAN), src_row_f, src_col_f);
        let bilinear = bilinear_value(&data, 6, 6, Some(f64::NAN), src_row_f, src_col_f);

        assert!(!cubic.is_nan(), "cubic fallback must not propagate NaN");
        assert!(
            (cubic - bilinear).abs() < 1e-12,
            "cubic should fall back to exactly the bilinear value: cubic={cubic}, bilinear={bilinear}"
        );
    }
}
