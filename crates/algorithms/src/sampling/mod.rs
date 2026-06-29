//! Raster sampling primitives: point extraction, CNN patches, and
//! polygon grid sampling.
//!
//! These are the library-level building blocks behind the
//! `surtgis extract` and `surtgis extract-patches` CLI commands,
//! exposed so sibling crates can sample rasters without shelling out:
//!
//! - [`sample_at_points`] — multi-raster values at world-coordinate
//!   points (feature matrix for ML training / prediction).
//! - [`extract_patches`] — fixed-size multi-band windows centred on
//!   points, as a flat `[n, bands, size, size]` f32 tensor.
//! - [`grid_points_in_polygon`] — regular grid of cell-centre points
//!   inside a polygon (patch centres for segmentation training).
//!
//! All functions take world coordinates in the reference raster's CRS
//! and validate that the input rasters share one grid.

use geo::Contains;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

fn check_same_shape(rasters: &[&Raster<f64>]) -> Result<(usize, usize)> {
    let Some(first) = rasters.first() else {
        return Err(Error::Algorithm("sampling: empty raster list".into()));
    };
    let shape = first.shape();
    for (i, r) in rasters.iter().enumerate().skip(1) {
        if r.shape() != shape {
            return Err(Error::Algorithm(format!(
                "sampling: raster {} shape {:?} does not match raster 0 shape {:?}",
                i,
                r.shape(),
                shape
            )));
        }
    }
    Ok(shape)
}

/// Sample every raster at each world-coordinate point.
///
/// Coordinates are interpreted in the grid of `rasters[0]` (all
/// rasters must share shape and are assumed grid-aligned). For each
/// point the result is `Some(values)` — one value per raster, in
/// input order — or `None` when the point falls outside the grid or
/// any raster has a non-finite value there.
///
/// # Example
/// ```no_run
/// # use surtgis_algorithms::sampling::sample_at_points;
/// # let slope: surtgis_core::Raster<f64> = unimplemented!();
/// # let aspect: surtgis_core::Raster<f64> = unimplemented!();
/// let rows = sample_at_points(&[&slope, &aspect], &[(350_000.0, 6_300_000.0)])?;
/// # Ok::<(), surtgis_core::Error>(())
/// ```
pub fn sample_at_points(
    rasters: &[&Raster<f64>],
    points: &[(f64, f64)],
) -> Result<Vec<Option<Vec<f64>>>> {
    let (rows, cols) = check_same_shape(rasters)?;
    let reference = rasters[0];

    let mut out = Vec::with_capacity(points.len());
    for &(x, y) in points {
        let (col_f, row_f) = reference.geo_to_pixel(x, y);
        let col = col_f.floor() as isize;
        let row = row_f.floor() as isize;
        if row < 0 || col < 0 || row as usize >= rows || col as usize >= cols {
            out.push(None);
            continue;
        }
        let (row, col) = (row as usize, col as usize);

        let mut values = Vec::with_capacity(rasters.len());
        let mut valid = true;
        for raster in rasters {
            let v = unsafe { raster.get_unchecked(row, col) };
            if !v.is_finite() {
                valid = false;
                break;
            }
            values.push(v);
        }
        out.push(if valid { Some(values) } else { None });
    }
    Ok(out)
}

/// Parameters for [`extract_patches`].
#[derive(Debug, Clone, Copy)]
pub struct PatchParams {
    /// Patch side length in pixels.
    pub size: usize,
    /// Maximum tolerated fraction of non-finite cells per patch
    /// (over all bands). Patches above the threshold are skipped.
    /// Non-finite cells in kept patches are emitted as f32 NaN.
    pub max_nan_fraction: f64,
}

impl Default for PatchParams {
    fn default() -> Self {
        Self {
            size: 64,
            max_nan_fraction: 0.0,
        }
    }
}

/// CNN-ready patch tensor produced by [`extract_patches`].
pub struct Patches {
    /// Row-major `[n, bands, size, size]` f32 tensor.
    pub data: Vec<f32>,
    /// Number of patches kept.
    pub n: usize,
    /// Number of bands per patch.
    pub n_bands: usize,
    /// Patch side length in pixels.
    pub size: usize,
    /// For each kept patch, the index of its center in the input
    /// `centers` slice (skipped centers leave gaps).
    pub kept: Vec<usize>,
}

/// Extract fixed-size multi-band patches centred on world-coordinate
/// points.
///
/// The patch window for a center at pixel `(r, c)` spans
/// `[r - size/2, r - size/2 + size)` × the same in columns — matching
/// the `surtgis extract-patches` CLI. Centers whose window falls
/// partially outside the grid, or whose non-finite fraction exceeds
/// [`PatchParams::max_nan_fraction`], are skipped; `kept` records the
/// surviving input indices so callers can align labels.
pub fn extract_patches(
    bands: &[&Raster<f64>],
    centers: &[(f64, f64)],
    params: PatchParams,
) -> Result<Patches> {
    if params.size == 0 {
        return Err(Error::Algorithm("extract_patches: size must be > 0".into()));
    }
    let (rows, cols) = check_same_shape(bands)?;
    let reference = bands[0];
    let size = params.size;
    let half = size / 2;
    let n_bands = bands.len();
    let cells_per_patch = n_bands * size * size;

    let mut data = Vec::new();
    let mut kept = Vec::new();

    for (idx, &(x, y)) in centers.iter().enumerate() {
        let (col_f, row_f) = reference.geo_to_pixel(x, y);
        let row = row_f.floor() as isize;
        let col = col_f.floor() as isize;

        if row < half as isize || col < half as isize {
            continue;
        }
        let (row, col) = (row as usize, col as usize);
        if row + (size - half) > rows || col + (size - half) > cols {
            continue;
        }
        let (r0, c0) = (row - half, col - half);

        let start = data.len();
        let mut nan_count = 0usize;
        for band in bands {
            for dr in 0..size {
                for dc in 0..size {
                    let v = unsafe { band.get_unchecked(r0 + dr, c0 + dc) };
                    if !v.is_finite() {
                        nan_count += 1;
                        data.push(f32::NAN);
                    } else {
                        data.push(v as f32);
                    }
                }
            }
        }

        let nan_frac = nan_count as f64 / cells_per_patch as f64;
        if nan_frac > params.max_nan_fraction {
            data.truncate(start);
            continue;
        }
        kept.push(idx);
    }

    Ok(Patches {
        n: kept.len(),
        n_bands,
        size,
        data,
        kept,
    })
}

/// Regular grid of cell-centre world coordinates inside a polygon.
///
/// Walks the polygon's bounding box on the reference grid every
/// `stride` pixels and keeps cell centres that the polygon contains.
/// `margin` shrinks the candidate area by that many pixels from every
/// raster edge — pass `size / 2` to guarantee the returned points are
/// valid patch centres for [`extract_patches`].
pub fn grid_points_in_polygon(
    polygon: &geo::Polygon<f64>,
    reference: &Raster<f64>,
    stride: usize,
    margin: usize,
) -> Result<Vec<(f64, f64)>> {
    if stride == 0 {
        return Err(Error::Algorithm(
            "grid_points_in_polygon: stride must be > 0".into(),
        ));
    }
    let (rows, cols) = reference.shape();
    if 2 * margin >= rows || 2 * margin >= cols {
        return Ok(Vec::new());
    }

    use geo::BoundingRect;
    let Some(bbox) = polygon.bounding_rect() else {
        return Ok(Vec::new());
    };
    let (cx0, ry0) = reference.geo_to_pixel(bbox.min().x, bbox.max().y);
    let (cx1, ry1) = reference.geo_to_pixel(bbox.max().x, bbox.min().y);

    let row_min = (ry0.floor() as isize).max(margin as isize) as usize;
    let row_max = (ry1.ceil() as isize).min((rows - margin) as isize - 1);
    let col_min = (cx0.floor() as isize).max(margin as isize) as usize;
    let col_max = (cx1.ceil() as isize).min((cols - margin) as isize - 1);
    if row_max < 0 || col_max < 0 {
        return Ok(Vec::new());
    }
    let (row_max, col_max) = (row_max as usize, col_max as usize);

    let mut points = Vec::new();
    let mut row = row_min;
    while row <= row_max {
        let mut col = col_min;
        while col <= col_max {
            let (x, y) = reference.pixel_to_geo(col, row);
            if polygon.contains(&geo::Point::new(x, y)) {
                points.push((x, y));
            }
            col += stride;
        }
        row += stride;
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn ramp(rows: usize, cols: usize) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        // Origin (1000, 2000), 10 m cells, north-up
        r.set_transform(GeoTransform::new(1000.0, 2000.0, 10.0, -10.0));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, (row * cols + col) as f64).unwrap();
            }
        }
        r
    }

    #[test]
    fn sample_at_points_basic() {
        let a = ramp(10, 10);
        let mut b = ramp(10, 10);
        b.set(2, 3, f64::NAN).unwrap();

        // Center of pixel (row=2, col=3): x = 1000 + 3.5*10, y = 2000 - 2.5*10
        let inside = (1035.0, 1975.0);
        let outside = (0.0, 0.0);

        let out = sample_at_points(&[&a, &b], &[inside, outside]).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_none(), "NaN in raster b should drop the point");
        assert!(out[1].is_none(), "out-of-bounds point should be None");

        let out = sample_at_points(&[&a], &[inside]).unwrap();
        assert_eq!(out[0], Some(vec![23.0]));
    }

    #[test]
    fn sample_at_points_rejects_mismatched_shapes() {
        let a = ramp(10, 10);
        let b = ramp(8, 10);
        assert!(sample_at_points(&[&a, &b], &[(1035.0, 1975.0)]).is_err());
    }

    #[test]
    fn extract_patches_window_and_skip() {
        let a = ramp(20, 20);
        let b = ramp(20, 20);

        // Pixel (10, 10) center; (0, 0) too close to the edge for size 5
        let center_ok = (1105.0, 1895.0);
        let center_edge = (1005.0, 1995.0);

        let patches = extract_patches(
            &[&a, &b],
            &[center_edge, center_ok],
            PatchParams {
                size: 5,
                max_nan_fraction: 0.0,
            },
        )
        .unwrap();

        assert_eq!(patches.n, 1);
        assert_eq!(patches.kept, vec![1]);
        assert_eq!(patches.n_bands, 2);
        assert_eq!(patches.data.len(), 2 * 5 * 5);
        // Window starts at (10-2, 10-2) = (8, 8); first value = 8*20+8
        assert_eq!(patches.data[0], 168.0);
        // Band 1 repeats band 0 (same ramp)
        assert_eq!(patches.data[25], 168.0);
    }

    #[test]
    fn extract_patches_nan_threshold() {
        let mut a = ramp(20, 20);
        a.set(10, 10, f64::NAN).unwrap();
        let center = (1105.0, 1895.0);

        let strict = extract_patches(
            &[&a],
            &[center],
            PatchParams {
                size: 5,
                max_nan_fraction: 0.0,
            },
        )
        .unwrap();
        assert_eq!(strict.n, 0);

        let tolerant = extract_patches(
            &[&a],
            &[center],
            PatchParams {
                size: 5,
                max_nan_fraction: 0.1,
            },
        )
        .unwrap();
        assert_eq!(tolerant.n, 1);
        assert!(tolerant.data[2 * 5 + 2].is_nan());
    }

    #[test]
    fn grid_points_in_polygon_counts() {
        let r = ramp(20, 20);
        // Polygon covering pixels rows 5..15, cols 5..15 in world coords:
        // x in [1050, 1150], y in [1850, 1950]
        let poly = geo::Polygon::new(
            geo::LineString::from(vec![
                (1050.0, 1850.0),
                (1150.0, 1850.0),
                (1150.0, 1950.0),
                (1050.0, 1950.0),
                (1050.0, 1850.0),
            ]),
            vec![],
        );

        let pts = grid_points_in_polygon(&poly, &r, 1, 0).unwrap();
        // 10x10 pixel centers strictly inside
        assert_eq!(pts.len(), 100);

        let strided = grid_points_in_polygon(&poly, &r, 5, 0).unwrap();
        assert_eq!(strided.len(), 4);

        for &(x, y) in &pts {
            assert!((1050.0..=1150.0).contains(&x));
            assert!((1850.0..=1950.0).contains(&y));
        }
    }

    #[test]
    fn grid_points_then_patches_roundtrip() {
        let r = ramp(40, 40);
        let poly = geo::Polygon::new(
            geo::LineString::from(vec![
                (1100.0, 1700.0),
                (1300.0, 1700.0),
                (1300.0, 1900.0),
                (1100.0, 1900.0),
                (1100.0, 1700.0),
            ]),
            vec![],
        );
        let size = 8;
        let centers = grid_points_in_polygon(&poly, &r, 4, size / 2).unwrap();
        assert!(!centers.is_empty());

        let patches = extract_patches(
            &[&r],
            &centers,
            PatchParams {
                size,
                max_nan_fraction: 0.0,
            },
        )
        .unwrap();
        // Every center produced by grid_points_in_polygon with
        // margin = size/2 must be a valid patch center.
        assert_eq!(patches.n, centers.len());
    }
}
