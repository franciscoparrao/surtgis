//! Raster warping between coordinate reference systems (feature `projections`).
//!
//! Inverse mapping: for every cell of a **target grid** the centre is
//! transformed back into the source CRS and the source bands are sampled
//! there (nearest or bilinear). proj4rs does the coordinate transform in
//! pure Rust, so this runs natively, in WASM and in a server alike.
//!
//! Two callers with different needs share the same kernel:
//!
//! - `surtgis reproject` warps a whole raster: [`grid_for`] derives the
//!   target grid (extent from the transformed corners and edge midpoints,
//!   pixel size preserved or inferred) and [`warp`] fills it.
//! - A tile server warps a *window*: the target grid is the tile (plus the
//!   algorithm's gutter), [`source_window`] says which source bounds to
//!   read for it, and [`warp`] resamples that window onto the tile.
//!
//! Only the affine, north-up part of a [`GeoTransform`] is honoured
//! (rotation terms are ignored), which is what every reader in this
//! workspace produces.

use ndarray::Array2;
#[cfg(feature = "parallel")]
use ndarray::parallel::prelude::*;
use proj4rs::Proj;

use crate::crs::CRS;
use crate::error::{Error, Result};
use crate::raster::{GeoTransform, Raster};

/// Interpolation used when sampling the source at a back-projected point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Resampling {
    /// Value of the nearest source cell.
    Nearest,
    /// Bilinear blend of the four surrounding cells; `None` (NaN) if any of
    /// them is missing or the point falls outside the source footprint.
    Bilinear,
}

impl Resampling {
    /// Parse `"nearest"`/`"nn"` or `"bilinear"`/`"bl"` (case-insensitive).
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "nearest" | "nn" => Ok(Resampling::Nearest),
            "bilinear" | "bl" => Ok(Resampling::Bilinear),
            other => Err(Error::Other(format!(
                "unknown resampling method: '{other}'. Supported: nearest, bilinear"
            ))),
        }
    }
}

/// Axis-aligned bounds in some CRS (`min_x ≤ max_x`, `min_y ≤ max_y`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// Western edge.
    pub min_x: f64,
    /// Southern edge.
    pub min_y: f64,
    /// Eastern edge.
    pub max_x: f64,
    /// Northern edge.
    pub max_y: f64,
}

impl Bounds {
    /// Width in CRS units.
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Height in CRS units.
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// Grow by `margin` on every side.
    pub fn expanded(&self, margin: f64) -> Bounds {
        Bounds {
            min_x: self.min_x - margin,
            min_y: self.min_y - margin,
            max_x: self.max_x + margin,
            max_y: self.max_y + margin,
        }
    }
}

/// A north-up target grid: georeferencing, size and CRS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSpec {
    /// Upper-left origin and pixel size (`pixel_height` negative).
    pub transform: GeoTransform,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// EPSG code of the grid's CRS.
    pub epsg: u32,
}

impl GridSpec {
    /// Grid covering `bounds` with square pixels of `pixel_size`, its
    /// origin at the north-west corner; the size is rounded up so the
    /// bounds are fully covered.
    pub fn covering(bounds: &Bounds, pixel_size: f64, epsg: u32) -> Result<Self> {
        if pixel_size <= 0.0 || !pixel_size.is_finite() {
            return Err(Error::Other(format!(
                "pixel size must be positive and finite, got {pixel_size}"
            )));
        }
        let cols = (bounds.width() / pixel_size).ceil() as usize;
        let rows = (bounds.height() / pixel_size).ceil() as usize;
        if rows == 0 || cols == 0 {
            return Err(Error::Other(format!(
                "output dimensions are zero (extent: {:.3}..{:.3}, {:.3}..{:.3}; pixel {})",
                bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y, pixel_size
            )));
        }
        Ok(GridSpec {
            transform: GeoTransform::new(bounds.min_x, bounds.max_y, pixel_size, -pixel_size),
            rows,
            cols,
            epsg,
        })
    }

    /// Outer bounds of the grid.
    pub fn bounds(&self) -> Bounds {
        let t = &self.transform;
        let x1 = t.origin_x + self.cols as f64 * t.pixel_width;
        let y1 = t.origin_y + self.rows as f64 * t.pixel_height;
        Bounds {
            min_x: t.origin_x.min(x1),
            min_y: t.origin_y.min(y1),
            max_x: t.origin_x.max(x1),
            max_y: t.origin_y.max(y1),
        }
    }

    /// Centre of cell `(row, col)` in grid coordinates.
    pub fn cell_centre(&self, row: usize, col: usize) -> (f64, f64) {
        let t = &self.transform;
        (
            t.origin_x + (col as f64 + 0.5) * t.pixel_width,
            t.origin_y + (row as f64 + 0.5) * t.pixel_height,
        )
    }
}

/// Coordinate transform between two EPSG CRSs, with the degrees/radians
/// convention of proj4rs handled internally: callers always pass and
/// receive CRS-native units (degrees for geographic CRSs, metres for
/// projected ones).
#[derive(Debug)]
pub struct Transformer {
    src: Proj,
    dst: Proj,
    src_epsg: u32,
    dst_epsg: u32,
}

impl Transformer {
    /// Build the `src_epsg → dst_epsg` transform.
    pub fn new(src_epsg: u32, dst_epsg: u32) -> Result<Self> {
        let load = |code: u32| {
            Proj::from_epsg_code(code as u16)
                .map_err(|e| Error::Other(format!("proj4rs failed to load EPSG:{code}: {e:?}")))
        };
        Ok(Transformer {
            src: load(src_epsg)?,
            dst: load(dst_epsg)?,
            src_epsg,
            dst_epsg,
        })
    }

    /// Source EPSG code.
    pub fn src_epsg(&self) -> u32 {
        self.src_epsg
    }

    /// Target EPSG code.
    pub fn dst_epsg(&self) -> u32 {
        self.dst_epsg
    }

    /// Whether source and target are the same CRS.
    pub fn is_identity(&self) -> bool {
        self.src_epsg == self.dst_epsg
    }

    /// Whether the source CRS is geographic (degrees).
    pub fn src_is_geographic(&self) -> bool {
        self.src.is_latlong()
    }

    /// Whether the target CRS is geographic (degrees).
    pub fn dst_is_geographic(&self) -> bool {
        self.dst.is_latlong()
    }

    /// Source → target. `None` where proj4rs cannot transform the point.
    pub fn forward(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        transform(&self.src, &self.dst, x, y)
    }

    /// Target → source. `None` where proj4rs cannot transform the point.
    pub fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        transform(&self.dst, &self.src, x, y)
    }
}

fn transform(from: &Proj, to: &Proj, x: f64, y: f64) -> Option<(f64, f64)> {
    let (ix, iy) = if from.is_latlong() {
        (x.to_radians(), y.to_radians())
    } else {
        (x, y)
    };
    let (ox, oy) = proj4rs::adaptors::transform_xy(from, to, ix, iy).ok()?;
    let out = if to.is_latlong() {
        (ox.to_degrees(), oy.to_degrees())
    } else {
        (ox, oy)
    };
    (out.0.is_finite() && out.1.is_finite()).then_some(out)
}

/// The 3×3 sample points (corners, edge midpoints, centre) of a raster's
/// outer bounds — enough to bound the target extent of projections that
/// bow the edges.
fn sample_points(gt: &GeoTransform, rows: usize, cols: usize) -> Vec<(f64, f64)> {
    let mut pts = Vec::with_capacity(9);
    for &r in &[0usize, rows / 2, rows] {
        for &c in &[0usize, cols / 2, cols] {
            pts.push((
                gt.origin_x + c as f64 * gt.pixel_width,
                gt.origin_y + r as f64 * gt.pixel_height,
            ));
        }
    }
    pts
}

fn bounds_of(points: impl IntoIterator<Item = (f64, f64)>) -> Option<Bounds> {
    let mut b = Bounds {
        min_x: f64::INFINITY,
        min_y: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        max_y: f64::NEG_INFINITY,
    };
    let mut any = false;
    for (x, y) in points {
        any = true;
        b.min_x = b.min_x.min(x);
        b.min_y = b.min_y.min(y);
        b.max_x = b.max_x.max(x);
        b.max_y = b.max_y.max(y);
    }
    any.then_some(b)
}

/// Bounds of a source raster once transformed to the target CRS.
///
/// Transforms the corners, edge midpoints and centre of the source's outer
/// bounds and takes their envelope; fails if any of them cannot be
/// transformed.
pub fn target_bounds(
    src_transform: &GeoTransform,
    src_rows: usize,
    src_cols: usize,
    tf: &Transformer,
) -> Result<Bounds> {
    let pts = sample_points(src_transform, src_rows, src_cols);
    let mut out = Vec::with_capacity(pts.len());
    for (x, y) in pts {
        let p = tf.forward(x, y).ok_or_else(|| {
            Error::Other(format!(
                "proj4rs transform failed at ({x}, {y}) (EPSG:{} → EPSG:{})",
                tf.src_epsg, tf.dst_epsg
            ))
        })?;
        out.push(p);
    }
    bounds_of(out).ok_or_else(|| Error::Other("empty source extent".into()))
}

/// Default target pixel size for a whole-raster warp.
///
/// When both CRSs use the same kind of unit (both metric or both
/// geographic) the source pixel width is kept. Otherwise the size that
/// roughly preserves the column count is used: target width / source width
/// × source pixel width.
pub fn default_pixel_size(
    src_transform: &GeoTransform,
    src_rows: usize,
    src_cols: usize,
    tf: &Transformer,
    target: &Bounds,
) -> Result<f64> {
    let src_px = src_transform.pixel_width.abs();
    if tf.src_is_geographic() == tf.dst_is_geographic() {
        return Ok(src_px);
    }
    let src = bounds_of(sample_points(src_transform, src_rows, src_cols))
        .ok_or_else(|| Error::Other("empty source extent".into()))?;
    if src.width() <= 0.0 {
        return Err(Error::Other(
            "could not infer pixel size from a zero-width source extent".into(),
        ));
    }
    Ok(target.width() / src.width() * src_px)
}

/// Target grid for warping a whole source raster: its transformed bounds
/// at `pixel_size` (or [`default_pixel_size`] when `None`).
pub fn grid_for(
    src_transform: &GeoTransform,
    src_rows: usize,
    src_cols: usize,
    tf: &Transformer,
    pixel_size: Option<f64>,
) -> Result<GridSpec> {
    let bounds = target_bounds(src_transform, src_rows, src_cols, tf)?;
    let px = match pixel_size {
        Some(p) => p,
        None => default_pixel_size(src_transform, src_rows, src_cols, tf, &bounds)?,
    };
    GridSpec::covering(&bounds, px, tf.dst_epsg)
}

/// Source-CRS bounds that must be read to fill `grid`, grown by
/// `margin_cells` target cells on every side (interpolation support plus
/// any focal-kernel gutter the caller needs).
///
/// Uses the same 3×3 sampling as [`target_bounds`], in the inverse
/// direction. Fails if the grid cannot be back-projected at all.
pub fn source_window(grid: &GridSpec, tf: &Transformer, margin_cells: usize) -> Result<Bounds> {
    let margin = margin_cells as f64 * grid.transform.pixel_width.abs();
    let outer = grid.bounds().expanded(margin);
    let gt = GeoTransform::new(outer.min_x, outer.max_y, outer.width(), -outer.height());
    let pts = sample_points(&gt, 1, 1);
    let back: Vec<(f64, f64)> = pts
        .into_iter()
        .filter_map(|(x, y)| tf.inverse(x, y))
        .collect();
    bounds_of(back).ok_or_else(|| {
        Error::Other(format!(
            "grid cannot be back-projected from EPSG:{} to EPSG:{}",
            tf.dst_epsg, tf.src_epsg
        ))
    })
}

/// Warp co-registered source bands onto `grid` by inverse mapping.
///
/// Every target cell centre is transformed into the source CRS once and
/// that position samples every band. Cells outside the source footprint
/// are NaN. The outputs carry `grid`'s transform, its CRS and each input's
/// nodata. Rows are processed in parallel with the `parallel` feature.
#[allow(clippy::needless_range_loop)]
pub fn warp(
    src_bands: &[Raster<f64>],
    src_transform: &GeoTransform,
    tf: &Transformer,
    grid: &GridSpec,
    method: Resampling,
) -> Result<Vec<Raster<f64>>> {
    let Some(first) = src_bands.first() else {
        return Err(Error::Other("no bands to warp".into()));
    };
    let (src_rows, src_cols) = first.shape();
    for (i, b) in src_bands.iter().enumerate().skip(1) {
        if b.shape() != first.shape() {
            return Err(Error::Other(format!(
                "band {} has shape {:?}, band 1 has {:?}; bands must be co-registered",
                i + 1,
                b.shape(),
                first.shape()
            )));
        }
    }
    let n_bands = src_bands.len();
    let datas: Vec<&Array2<f64>> = src_bands.iter().map(|b| b.data()).collect();

    let fill_row = |out_r: usize| -> Vec<Vec<f64>> {
        let mut rows: Vec<Vec<f64>> = (0..n_bands).map(|_| vec![f64::NAN; grid.cols]).collect();
        for out_c in 0..grid.cols {
            let (dst_x, dst_y) = grid.cell_centre(out_r, out_c);
            let Some((src_x, src_y)) = tf.inverse(dst_x, dst_y) else {
                continue;
            };
            let src_c_f = (src_x - src_transform.origin_x) / src_transform.pixel_width - 0.5;
            let src_r_f = (src_y - src_transform.origin_y) / src_transform.pixel_height - 0.5;
            for (band, data) in datas.iter().enumerate() {
                let val = match method {
                    Resampling::Nearest => {
                        sample_nearest(data, src_rows, src_cols, src_r_f, src_c_f)
                    }
                    Resampling::Bilinear => {
                        sample_bilinear(data, src_rows, src_cols, src_r_f, src_c_f)
                    }
                };
                if let Some(v) = val {
                    rows[band][out_c] = v;
                }
            }
        }
        rows
    };

    #[cfg(feature = "parallel")]
    let row_results: Vec<Vec<Vec<f64>>> = (0..grid.rows).into_par_iter().map(fill_row).collect();
    #[cfg(not(feature = "parallel"))]
    let row_results: Vec<Vec<Vec<f64>>> = (0..grid.rows).map(fill_row).collect();

    let mut outputs: Vec<Raster<f64>> = (0..n_bands)
        .map(|band| {
            let mut out = Raster::<f64>::new(grid.rows, grid.cols);
            out.set_transform(grid.transform);
            out.set_crs(Some(CRS::from_epsg(grid.epsg)));
            if let Some(nd) = src_bands[band].nodata() {
                out.set_nodata(Some(nd));
            }
            out
        })
        .collect();

    for (out_r, rows) in row_results.into_iter().enumerate() {
        for (band, row) in rows.into_iter().enumerate() {
            for (out_c, v) in row.into_iter().enumerate() {
                outputs[band].data_mut()[[out_r, out_c]] = v;
            }
        }
    }
    Ok(outputs)
}

fn sample_nearest(
    data: &Array2<f64>,
    rows: usize,
    cols: usize,
    src_r_f: f64,
    src_c_f: f64,
) -> Option<f64> {
    let r = src_r_f.round();
    let c = src_c_f.round();
    if r < 0.0 || c < 0.0 || r >= rows as f64 || c >= cols as f64 {
        return None;
    }
    let v = data[[r as usize, c as usize]];
    v.is_finite().then_some(v)
}

fn sample_bilinear(
    data: &Array2<f64>,
    rows: usize,
    cols: usize,
    src_r_f: f64,
    src_c_f: f64,
) -> Option<f64> {
    let c0 = src_c_f.floor() as isize;
    let r0 = src_r_f.floor() as isize;
    let fc = src_c_f - c0 as f64;
    let fr = src_r_f - r0 as f64;

    if c0 < 0 || r0 < 0 || (c0 + 1) >= cols as isize || (r0 + 1) >= rows as isize {
        return None;
    }

    let (r0u, c0u) = (r0 as usize, c0 as usize);
    let v00 = data[[r0u, c0u]];
    let v01 = data[[r0u, c0u + 1]];
    let v10 = data[[r0u + 1, c0u]];
    let v11 = data[[r0u + 1, c0u + 1]];

    if !(v00.is_finite() && v01.is_finite() && v10.is_finite() && v11.is_finite()) {
        return None;
    }

    Some(
        v00 * (1.0 - fc) * (1.0 - fr)
            + v01 * fc * (1.0 - fr)
            + v10 * (1.0 - fc) * fr
            + v11 * fc * fr,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(rows: usize, cols: usize, gt: GeoTransform, epsg: u32) -> Raster<f64> {
        let mut r = Raster::<f64>::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                r.data_mut()[[i, j]] = (i * cols + j) as f64;
            }
        }
        r.set_transform(gt);
        r.set_crs(Some(CRS::from_epsg(epsg)));
        r
    }

    /// WGS84 → Web Mercator has a closed form; proj4rs must match it.
    #[test]
    fn transformer_matches_web_mercator_closed_form() {
        let tf = Transformer::new(4326, 3857).unwrap();
        let (lon, lat) = (-70.6693, -33.4489); // Santiago
        let (x, y) = tf.forward(lon, lat).unwrap();
        let r = 6_378_137.0_f64;
        let ex = r * lon.to_radians();
        let ey = r
            * (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0)
                .tan()
                .ln();
        assert!((x - ex).abs() < 1e-3, "{x} vs {ex}");
        assert!((y - ey).abs() < 1e-3, "{y} vs {ey}");
        let (lon2, lat2) = tf.inverse(x, y).unwrap();
        assert!((lon2 - lon).abs() < 1e-9 && (lat2 - lat).abs() < 1e-9);
        assert!(tf.src_is_geographic() && !tf.dst_is_geographic());
    }

    /// Warping onto the source's own grid is the identity for both
    /// resamplings (cell centres map exactly onto cell centres).
    #[test]
    fn identity_grid_reproduces_source() {
        let gt = GeoTransform::new(500_000.0, 6_300_000.0, 30.0, -30.0);
        let src = ramp(8, 6, gt, 32719);
        let tf = Transformer::new(32719, 32719).unwrap();
        let grid = GridSpec {
            transform: gt,
            rows: 8,
            cols: 6,
            epsg: 32719,
        };
        for m in [Resampling::Nearest, Resampling::Bilinear] {
            let out = warp(std::slice::from_ref(&src), &gt, &tf, &grid, m).unwrap();
            let (rows, cols) = out[0].shape();
            for i in 0..rows {
                for j in 0..cols {
                    let got = out[0].data()[[i, j]];
                    let want = src.data()[[i, j]];
                    // Bilinear needs all four neighbours: the outer ring can
                    // legitimately be NaN (the back-projected centre lands
                    // a rounding error outside the last cell).
                    let border = i == 0 || j == 0 || i == rows - 1 || j == cols - 1;
                    if m == Resampling::Bilinear && border && got.is_nan() {
                        continue;
                    } else {
                        assert!(
                            (got - want).abs() < 1e-9,
                            "{m:?} ({i},{j}): {got} vs {want}"
                        );
                    }
                }
            }
            assert_eq!(out[0].crs().and_then(|c| c.epsg()), Some(32719));
        }
    }

    /// The window read for a target grid contains the back-projected grid
    /// and grows with the margin.
    #[test]
    fn source_window_contains_back_projected_grid() {
        let tf = Transformer::new(32719, 3857).unwrap();
        // A 256×256 Web Mercator tile-ish grid over central Chile.
        let (x, y) = tf.forward(400_000.0, 6_300_000.0).unwrap();
        let grid = GridSpec {
            transform: GeoTransform::new(x, y, 10.0, -10.0),
            rows: 256,
            cols: 256,
            epsg: 3857,
        };
        let w0 = source_window(&grid, &tf, 0).unwrap();
        let w2 = source_window(&grid, &tf, 2).unwrap();
        let (sx, sy) = tf.inverse(x + 1280.0, y - 1280.0).unwrap();
        assert!(w0.min_x <= sx && sx <= w0.max_x && w0.min_y <= sy && sy <= w0.max_y);
        assert!(
            w2.min_x < w0.min_x
                && w2.max_x > w0.max_x
                && w2.min_y < w0.min_y
                && w2.max_y > w0.max_y
        );
    }

    /// Whole-raster grid: metric → metric keeps the pixel size; metric →
    /// geographic infers one that keeps the column count.
    #[test]
    fn grid_for_keeps_or_infers_pixel_size() {
        let gt = GeoTransform::new(400_000.0, 6_300_000.0, 30.0, -30.0);
        let tf = Transformer::new(32719, 3857).unwrap();
        let g = grid_for(&gt, 100, 200, &tf, None).unwrap();
        assert_eq!(g.transform.pixel_width, 30.0);
        assert_eq!(g.epsg, 3857);
        let tf = Transformer::new(32719, 4326).unwrap();
        let g = grid_for(&gt, 100, 200, &tf, None).unwrap();
        assert!((g.cols as i64 - 200).abs() <= 1, "cols {}", g.cols);
        assert!(g.transform.pixel_width < 1e-2);
        assert!(GridSpec::covering(&g.bounds(), 0.0, 4326).is_err());
    }

    #[test]
    fn resampling_parse() {
        assert_eq!(Resampling::parse("NN").unwrap(), Resampling::Nearest);
        assert_eq!(Resampling::parse("bilinear").unwrap(), Resampling::Bilinear);
        assert!(Resampling::parse("cubic").is_err());
    }
}
