//! Raster sources: what `?url=` may point at, and how a window of it is read.
//!
//! Two kinds in M0:
//!
//! - **HTTP(S) COG**, read through `surtgis_cloud::CogReader` at the
//!   overview closest to the requested resolution. Single band (the reader
//!   is single-band today).
//! - **Local GeoTIFF** under the configured root, read once in full and
//!   kept in memory (every band), then cropped per request. Fine for DEMs
//!   and demo-sized imagery; windowed local reads are an M1 item.
//! - **Local ECW** under the root (feature `ecw`), decoded per request
//!   through `surtgis_ecw` at the pyramid level that matches the tile —
//!   gigapixel orthomosaics without the vendor SDK.
//!
//! `?url=` is a server-side request forgery vector, so a source is only
//! accepted when it matches the allowlist (HTTP prefixes) or lives under
//! the root (local paths). Without an allowlist no remote source is
//! accepted at all.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::s;
use surtgis_cloud::BBox;

use crate::pool::ReaderPool;
use surtgis_core::Raster;
use surtgis_core::io::read_geotiff_bands;
use surtgis_core::raster::GeoTransform;
use surtgis_core::warp::Bounds;

use crate::error::ServeError;

/// Which sources a deployment accepts.
#[derive(Debug, Clone, Default)]
pub struct SourceConfig {
    /// URL prefixes that remote sources may start with.
    pub allow: Vec<String>,
    /// Directory local sources must live under (canonicalised).
    pub root: Option<PathBuf>,
}

/// A validated source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Source {
    /// Remote Cloud Optimized GeoTIFF.
    Http(String),
    /// Local GeoTIFF under the root.
    Local(PathBuf),
    /// Local ECW orthomosaic under the root (feature `ecw`).
    Ecw(PathBuf),
}

impl Source {
    /// Validate `url` against the configuration.
    pub fn resolve(url: &str, cfg: &SourceConfig) -> Result<Source, ServeError> {
        let url = url.trim();
        if url.is_empty() {
            return Err(ServeError::BadRequest("missing url".into()));
        }
        if url.starts_with("http://") || url.starts_with("https://") {
            if cfg.allow.iter().any(|p| url.starts_with(p.as_str())) {
                return Ok(Source::Http(url.to_string()));
            }
            return Err(ServeError::Forbidden(
                "remote source not in the allowlist (--allow <prefix>)".into(),
            ));
        }
        if url.contains("://") {
            return Err(ServeError::BadRequest(format!(
                "unsupported source scheme in '{url}' (http(s) URLs and local paths only)"
            )));
        }
        let Some(root) = &cfg.root else {
            return Err(ServeError::Forbidden(
                "local sources need --root <dir>".into(),
            ));
        };
        let candidate = if Path::new(url).is_absolute() {
            PathBuf::from(url)
        } else {
            root.join(url)
        };
        let canonical = candidate
            .canonicalize()
            .map_err(|_| ServeError::NotFound(format!("source not found: {url}")))?;
        if !canonical.starts_with(root) {
            return Err(ServeError::Forbidden("local source outside --root".into()));
        }
        let is_ecw = canonical
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("ecw"));
        if is_ecw {
            if cfg!(feature = "ecw") {
                return Ok(Source::Ecw(canonical));
            }
            return Err(ServeError::BadRequest(
                "ECW sources need a build with the `ecw` feature".into(),
            ));
        }
        Ok(Source::Local(canonical))
    }

    /// Stable identifier for cache keys and ETags.
    pub fn key(&self) -> String {
        match self {
            Source::Http(u) => u.clone(),
            Source::Local(p) | Source::Ecw(p) => p.display().to_string(),
        }
    }
}

/// What a client needs to know about a source before asking for tiles.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SourceInfo {
    /// EPSG code of the source CRS.
    pub epsg: u32,
    /// Outer bounds in the source CRS: `[min_x, min_y, max_x, max_y]`.
    pub bounds: [f64; 4],
    /// Full-resolution size.
    pub width: usize,
    /// Full-resolution size.
    pub height: usize,
    /// Pixel size in source units.
    pub pixel_size: f64,
    /// Number of bands the server can address.
    pub bands: usize,
    /// Reduced-resolution levels available (0 for in-memory sources).
    pub overviews: usize,
    /// Declared nodata, if any.
    pub nodata: Option<f64>,
}

/// A window of a source: co-registered f64 bands with nodata already
/// mapped to NaN, plus the CRS they are in.
pub struct Window {
    /// Bands, nodata as NaN.
    pub bands: Vec<Raster<f64>>,
    /// EPSG code of the bands' CRS.
    pub epsg: u32,
}

/// In-memory copies of local sources, bounded by a byte budget (LRU).
pub struct LocalCache {
    entries: crate::cache::ByteLru<PathBuf, Arc<Vec<Raster<f64>>>>,
}

fn bands_bytes(b: &Arc<Vec<Raster<f64>>>) -> usize {
    b.iter()
        .map(|r| {
            let (rows, cols) = r.shape();
            rows * cols * std::mem::size_of::<f64>()
        })
        .sum()
}

impl LocalCache {
    /// Cache holding at most `budget` bytes of decoded rasters.
    pub fn new(budget: usize) -> Self {
        Self {
            entries: crate::cache::ByteLru::new(budget, bands_bytes),
        }
    }

    /// `(files, bytes, hits, misses)`.
    pub fn stats(&self) -> (usize, usize, u64, u64) {
        self.entries.stats()
    }

    fn get_or_load(&self, path: &Path) -> Result<Arc<Vec<Raster<f64>>>, ServeError> {
        if let Some(b) = self.entries.get(&path.to_path_buf()) {
            return Ok(b);
        }
        let bands: Vec<Raster<f64>> = read_geotiff_bands(path)
            .map_err(|e| ServeError::Source(format!("failed to read {}: {e}", path.display())))?;
        if bands.is_empty() {
            return Err(ServeError::Source(format!(
                "{} has no bands",
                path.display()
            )));
        }
        let bands: Vec<Raster<f64>> = bands.into_iter().map(nan_nodata).collect();
        let arc = Arc::new(bands);
        if bands_bytes(&arc) > self.entries.budget() {
            return Err(ServeError::Source(format!(
                "{} needs {} MiB in memory, over the --local-cache-mb budget of {} MiB",
                path.display(),
                bands_bytes(&arc) >> 20,
                self.entries.budget() >> 20
            )));
        }
        self.entries.put(path.to_path_buf(), arc.clone());
        Ok(arc)
    }
}

fn epsg_of(crs: Option<&surtgis_core::CRS>, what: &str) -> Result<u32, ServeError> {
    crs.and_then(|c| c.epsg())
        .ok_or_else(|| ServeError::Source(format!("{what}: source CRS has no EPSG code")))
}

/// Replace a finite nodata sentinel with NaN so resampling and the
/// algorithms never blend it with real values.
fn nan_nodata(mut r: Raster<f64>) -> Raster<f64> {
    if let Some(nd) = r.nodata().filter(|nd| nd.is_finite()) {
        r.data_mut()
            .mapv_inplace(|v| if v == nd { f64::NAN } else { v });
        r.set_nodata(Some(f64::NAN));
    }
    r
}

fn bounds_of(gt: &GeoTransform, rows: usize, cols: usize) -> Bounds {
    let x1 = gt.origin_x + cols as f64 * gt.pixel_width;
    let y1 = gt.origin_y + rows as f64 * gt.pixel_height;
    Bounds {
        min_x: gt.origin_x.min(x1),
        min_y: gt.origin_y.min(y1),
        max_x: gt.origin_x.max(x1),
        max_y: gt.origin_y.max(y1),
    }
}

/// Crop `bands` to the cells intersecting `bounds` (north-up transform).
fn crop(bands: &[Raster<f64>], bounds: &Bounds) -> Result<Vec<Raster<f64>>, ServeError> {
    let first = &bands[0];
    let gt = *first.transform();
    let (rows, cols) = first.shape();
    let pw = gt.pixel_width;
    let ph = gt.pixel_height.abs();
    let c0 = ((bounds.min_x - gt.origin_x) / pw).floor().max(0.0) as usize;
    let c1 = (((bounds.max_x - gt.origin_x) / pw).ceil().max(0.0) as usize).min(cols);
    let r0 = ((gt.origin_y - bounds.max_y) / ph).floor().max(0.0) as usize;
    let r1 = (((gt.origin_y - bounds.min_y) / ph).ceil().max(0.0) as usize).min(rows);
    if c0 >= c1 || r0 >= r1 {
        return Err(ServeError::Outside);
    }
    let sub_gt = GeoTransform::new(
        gt.origin_x + c0 as f64 * pw,
        gt.origin_y + r0 as f64 * gt.pixel_height,
        pw,
        gt.pixel_height,
    );
    Ok(bands
        .iter()
        .map(|b| {
            let mut out = Raster::from_array(b.data().slice(s![r0..r1, c0..c1]).to_owned());
            out.set_transform(sub_gt);
            out.set_crs(b.crs().cloned());
            out.set_nodata(b.nodata());
            out
        })
        .collect())
}

impl Source {
    /// Describe the source.
    pub async fn info(
        &self,
        cache: &LocalCache,
        pool: &ReaderPool,
    ) -> Result<SourceInfo, ServeError> {
        match self {
            Source::Local(path) => {
                let bands = cache.get_or_load(path)?;
                let b = &bands[0];
                let (rows, cols) = b.shape();
                let gt = b.transform();
                let bb = bounds_of(gt, rows, cols);
                Ok(SourceInfo {
                    epsg: epsg_of(b.crs(), "local")?,
                    bounds: [bb.min_x, bb.min_y, bb.max_x, bb.max_y],
                    width: cols,
                    height: rows,
                    pixel_size: gt.pixel_width.abs(),
                    bands: bands.len(),
                    overviews: 0,
                    nodata: b.nodata(),
                })
            }
            #[cfg(feature = "ecw")]
            Source::Ecw(path) => ecw::info(path).await,
            #[cfg(not(feature = "ecw"))]
            Source::Ecw(_) => Err(ServeError::BadRequest(
                "ECW sources need a build with the `ecw` feature".into(),
            )),
            Source::Http(url) => {
                let reader = pool.acquire(url).await?;
                let m = reader.metadata();
                let bb = bounds_of(&m.geo_transform, m.height as usize, m.width as usize);
                Ok(SourceInfo {
                    epsg: epsg_of(m.crs.as_ref(), "cog")?,
                    bounds: [bb.min_x, bb.min_y, bb.max_x, bb.max_y],
                    width: m.width as usize,
                    height: m.height as usize,
                    pixel_size: m.geo_transform.pixel_width.abs(),
                    bands: 1,
                    overviews: m.num_overviews,
                    nodata: m.nodata,
                })
            }
        }
    }

    /// Read the source cells covering `bounds` (source CRS), at a
    /// resolution suited to `target_px` output pixels across the window.
    pub async fn read_window(
        &self,
        cache: &LocalCache,
        pool: &ReaderPool,
        bounds: &Bounds,
        target_px: usize,
    ) -> Result<Window, ServeError> {
        match self {
            Source::Local(path) => {
                let bands = cache.get_or_load(path)?;
                let epsg = epsg_of(bands[0].crs(), "local")?;
                let bands = crop(&bands, bounds)?;
                Ok(Window { bands, epsg })
            }
            #[cfg(feature = "ecw")]
            Source::Ecw(path) => ecw::read_window(path, bounds, target_px).await,
            #[cfg(not(feature = "ecw"))]
            Source::Ecw(_) => Err(ServeError::BadRequest(
                "ECW sources need a build with the `ecw` feature".into(),
            )),
            Source::Http(url) => {
                let mut reader = pool.acquire(url).await?;
                let m = reader.metadata();
                let epsg = epsg_of(m.crs.as_ref(), "cog")?;
                // Pick the coarsest overview that still gives at least one
                // source pixel per output pixel.
                let px = m.geo_transform.pixel_width.abs();
                let wanted = bounds.width() / target_px.max(1) as f64;
                let mut level = 0usize;
                while level < m.num_overviews && px * (1u64 << (level + 1)) as f64 <= wanted {
                    level += 1;
                }
                let bbox = BBox {
                    min_x: bounds.min_x,
                    min_y: bounds.min_y,
                    max_x: bounds.max_x,
                    max_y: bounds.max_y,
                };
                let raster: Raster<f64> = match reader
                    .read_bbox::<f64>(&bbox, if level == 0 { None } else { Some(level) })
                    .await
                {
                    Ok(r) => r,
                    Err(surtgis_cloud::CloudError::BBoxOutside) => return Err(ServeError::Outside),
                    Err(e) => return Err(ServeError::Source(format!("read failed: {e}"))),
                };
                let mut raster = raster;
                if raster.nodata().is_none() {
                    raster.set_nodata(m.nodata);
                }
                Ok(Window {
                    bands: vec![nan_nodata(raster)],
                    epsg,
                })
            }
        }
    }
}

/// `((first_col, last_col), (first_row, last_row), (out_cols, out_rows))`.
pub type CellWindow = ((u32, u32), (u32, u32), (u32, u32));

/// Cell window of a north-up grid (`origin`, `inc`, `size`) intersecting
/// `bounds`, as inclusive `(first, last)` columns and rows, plus the output
/// size that keeps at most `target_px` cells along the longer side.
/// `None` when the window is empty.
#[allow(clippy::too_many_arguments)]
pub fn cell_window(
    origin_x: f64,
    origin_y: f64,
    inc_x: f64,
    inc_y: f64,
    x_size: u32,
    y_size: u32,
    bounds: &Bounds,
    target_px: usize,
) -> Option<CellWindow> {
    if x_size == 0 || y_size == 0 || inc_x <= 0.0 || inc_y >= 0.0 {
        return None;
    }
    let inc_y = inc_y.abs();
    let c0 = ((bounds.min_x - origin_x) / inc_x).floor();
    let c1 = ((bounds.max_x - origin_x) / inc_x).ceil() - 1.0;
    let r0 = ((origin_y - bounds.max_y) / inc_y).floor();
    let r1 = ((origin_y - bounds.min_y) / inc_y).ceil() - 1.0;
    let c0 = c0.max(0.0) as u32;
    let r0 = r0.max(0.0) as u32;
    if c1 < 0.0 || r1 < 0.0 {
        return None;
    }
    let c1 = (c1 as u32).min(x_size - 1);
    let r1 = (r1 as u32).min(y_size - 1);
    if c0 > c1 || r0 > r1 {
        return None;
    }
    let (w, h) = (c1 - c0 + 1, r1 - r0 + 1);
    let target = target_px.max(1) as u32;
    let scale = (w.max(h) as f64 / target as f64).max(1.0);
    let nx = ((w as f64 / scale).round() as u32).clamp(1, w);
    let ny = ((h as f64 / scale).round() as u32).clamp(1, h);
    Some(((c0, c1), (r0, r1), (nx, ny)))
}

#[cfg(feature = "ecw")]
mod ecw {
    use std::path::Path;

    use surtgis_core::Raster;
    use surtgis_core::warp::Bounds;
    use surtgis_ecw::{EcwReader, RegionParams};

    use super::{SourceInfo, Window, cell_window};
    use crate::error::ServeError;

    fn open(path: &Path) -> Result<EcwReader, ServeError> {
        EcwReader::open(path)
            .map_err(|e| ServeError::Source(format!("failed to open {}: {e}", path.display())))
    }

    pub async fn info(path: &Path) -> Result<SourceInfo, ServeError> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let reader = open(&path)?;
            let h = reader.header();
            let epsg = h.epsg().ok_or_else(|| {
                ServeError::Source(format!(
                    "{}: ECW datum/projection '{}'/'{}' has no EPSG mapping",
                    path.display(),
                    h.datum,
                    h.projection
                ))
            })?;
            let (w, hh) = (f64::from(h.x_size), f64::from(h.y_size));
            let x1 = h.origin_x + w * h.cell_increment_x;
            let y1 = h.origin_y + hh * h.cell_increment_y;
            Ok(SourceInfo {
                epsg,
                bounds: [
                    h.origin_x.min(x1),
                    h.origin_y.min(y1),
                    h.origin_x.max(x1),
                    h.origin_y.max(y1),
                ],
                width: h.x_size as usize,
                height: h.y_size as usize,
                pixel_size: h.cell_increment_x.abs(),
                bands: h.nr_bands as usize,
                overviews: h.num_levels as usize,
                nodata: None,
            })
        })
        .await
        .map_err(|e| ServeError::Source(format!("ecw task failed: {e}")))?
    }

    pub async fn read_window(
        path: &Path,
        bounds: &Bounds,
        target_px: usize,
    ) -> Result<Window, ServeError> {
        let (path, bounds) = (path.to_path_buf(), *bounds);
        tokio::task::spawn_blocking(move || {
            let mut reader = open(&path)?;
            let h = reader.header();
            let epsg = h
                .epsg()
                .ok_or_else(|| ServeError::Source("ECW CRS has no EPSG mapping".into()))?;
            let Some(((c0, c1), (r0, r1), (nx, ny))) = cell_window(
                h.origin_x,
                h.origin_y,
                h.cell_increment_x,
                h.cell_increment_y,
                h.x_size,
                h.y_size,
                &bounds,
                target_px,
            ) else {
                return Err(ServeError::Outside);
            };
            let planes = reader
                .read_region(RegionParams {
                    start_x: c0,
                    start_y: r0,
                    end_x: c1,
                    end_y: r1,
                    number_x: nx,
                    number_y: ny,
                })
                .map_err(|e| ServeError::Source(format!("ECW read failed: {e}")))?;
            let bands: Vec<Raster<f64>> = planes
                .into_iter()
                .map(|p| {
                    let mut out = Raster::from_array(p.data().mapv(f64::from));
                    out.set_transform(*p.transform());
                    out.set_crs(p.crs().cloned());
                    out
                })
                .collect();
            if bands.is_empty() {
                return Err(ServeError::Source("ECW read returned no bands".into()));
            }
            Ok(Window { bands, epsg })
        })
        .await
        .map_err(|e| ServeError::Source(format!("ecw task failed: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_and_root_gate_sources() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::write(root.join("dem.tif"), b"x").unwrap();
        let cfg = SourceConfig {
            allow: vec!["https://example.org/cogs/".into()],
            root: Some(root.clone()),
        };
        assert!(matches!(
            Source::resolve("https://example.org/cogs/a.tif", &cfg),
            Ok(Source::Http(_))
        ));
        assert!(matches!(
            Source::resolve("https://evil.example/a.tif", &cfg),
            Err(ServeError::Forbidden(_))
        ));
        assert!(matches!(
            Source::resolve("dem.tif", &cfg),
            Ok(Source::Local(_))
        ));
        assert!(matches!(
            Source::resolve("../../etc/passwd", &cfg),
            Err(ServeError::NotFound(_)) | Err(ServeError::Forbidden(_))
        ));
        assert!(matches!(
            Source::resolve("s3://bucket/key", &cfg),
            Err(ServeError::BadRequest(_))
        ));
        let no_root = SourceConfig::default();
        assert!(matches!(
            Source::resolve("dem.tif", &no_root),
            Err(ServeError::Forbidden(_))
        ));
        assert!(matches!(
            Source::resolve("https://example.org/cogs/a.tif", &no_root),
            Err(ServeError::Forbidden(_))
        ));
    }

    #[test]
    fn crop_selects_the_intersecting_cells() {
        let mut r = Raster::<f64>::new(10, 10);
        for i in 0..10 {
            for j in 0..10 {
                r.data_mut()[[i, j]] = (i * 10 + j) as f64;
            }
        }
        r.set_transform(GeoTransform::new(100.0, 200.0, 10.0, -10.0));
        let sub = crop(
            std::slice::from_ref(&r),
            &Bounds {
                min_x: 125.0,
                min_y: 155.0,
                max_x: 145.0,
                max_y: 175.0,
            },
        )
        .unwrap();
        // x 125..145 → cols 2..5, y 155..175 → rows 2..5
        assert_eq!(sub[0].shape(), (3, 3));
        assert_eq!(sub[0].data()[[0, 0]], 22.0);
        assert_eq!(sub[0].transform().origin_x, 120.0);
        assert_eq!(sub[0].transform().origin_y, 180.0);
        assert!(matches!(
            crop(
                std::slice::from_ref(&r),
                &Bounds {
                    min_x: 500.0,
                    min_y: 500.0,
                    max_x: 600.0,
                    max_y: 600.0
                }
            ),
            Err(ServeError::Outside)
        ));
    }

    #[test]
    fn nodata_becomes_nan() {
        let mut r = Raster::<f64>::new(2, 2);
        r.data_mut()[[0, 0]] = -9999.0;
        r.data_mut()[[1, 1]] = 5.0;
        r.set_nodata(Some(-9999.0));
        let r = nan_nodata(r);
        assert!(r.data()[[0, 0]].is_nan());
        assert_eq!(r.data()[[1, 1]], 5.0);
        assert!(r.nodata().unwrap().is_nan());
    }

    /// ECW window arithmetic: a bounds box maps to inclusive cells, is
    /// clamped to the image, keeps aspect when reduced, and is `None`
    /// outside.
    #[test]
    fn cell_window_clamps_and_reduces() {
        // 1000×800 cells of 0.04 m from (300000, 6300000), north-up.
        let b = Bounds {
            min_x: 300_004.0,
            min_y: 6_299_990.0,
            max_x: 300_012.0,
            max_y: 6_299_998.0,
        };
        let ((c0, c1), (r0, r1), (nx, ny)) =
            cell_window(300_000.0, 6_300_000.0, 0.04, -0.04, 1000, 800, &b, 256).unwrap();
        assert_eq!((c0, c1), (100, 299));
        assert_eq!((r0, r1), (50, 249));
        assert_eq!((nx, ny), (200, 200)); // 200 cells ≤ 256: full resolution
        // Ask for 50 px across a 200-cell window → reduction 4.
        let (_, _, (nx, ny)) =
            cell_window(300_000.0, 6_300_000.0, 0.04, -0.04, 1000, 800, &b, 50).unwrap();
        assert_eq!((nx, ny), (50, 50));
        // Partly outside: clamped to the image.
        let b2 = Bounds {
            min_x: 299_990.0,
            min_y: 6_299_999.0,
            max_x: 300_001.0,
            max_y: 6_300_010.0,
        };
        let ((c0, c1), (r0, r1), _) =
            cell_window(300_000.0, 6_300_000.0, 0.04, -0.04, 1000, 800, &b2, 256).unwrap();
        assert_eq!((c0, r0), (0, 0));
        assert_eq!((c1, r1), (24, 24));
        // Fully outside.
        let b3 = Bounds {
            min_x: 400_000.0,
            min_y: 6_299_990.0,
            max_x: 400_010.0,
            max_y: 6_299_998.0,
        };
        assert!(cell_window(300_000.0, 6_300_000.0, 0.04, -0.04, 1000, 800, &b3, 256).is_none());
    }
}
