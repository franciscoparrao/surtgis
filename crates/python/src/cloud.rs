//! Python bindings for the cloud module: remote COG reading and STAC search.
//!
//! These wrap the blocking (synchronous) cloud API from `surtgis-cloud`'s
//! `sync_api` (which drives an internal Tokio runtime), so Python callers get
//! ordinary blocking functions. PyO3 releases the GIL during the FFI call, so
//! the network I/O does not block other Python threads.
//!
//! Georeferencing is returned alongside the pixel array as a metadata dict
//! whose `transform` is in GDAL affine order — pass it straight to
//! `rasterio.transform.Affine.from_gdal(*meta["transform"])`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use surtgis_cloud::cog_reader::{CogMetadata, CogReaderOptions};
use surtgis_cloud::stac_client::{StacCatalog, StacClientOptions};
use surtgis_cloud::stac_models::StacSearchParams;
use surtgis_cloud::sync_api::{CogReaderBlocking, StacClientBlocking};
use surtgis_cloud::tile_index::BBox;

use surtgis_core::raster::Raster;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Build a metadata dict from a fetched raster window.
///
/// `transform` is a 6-tuple in GDAL order:
/// `(origin_x, pixel_width, row_rotation, origin_y, col_rotation, pixel_height)`.
fn raster_meta_dict<'py>(py: Python<'py>, raster: &Raster<f64>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let (rows, cols) = raster.shape();
    let t = raster.transform();
    d.set_item(
        "transform",
        (
            t.origin_x,
            t.pixel_width,
            t.row_rotation,
            t.origin_y,
            t.col_rotation,
            t.pixel_height,
        ),
    )?;
    match raster.crs().and_then(|c| c.epsg()) {
        Some(code) => d.set_item("crs", format!("EPSG:{code}"))?,
        None => d.set_item("crs", py.None())?,
    }
    match raster.nodata() {
        Some(nd) => d.set_item("nodata", nd)?,
        None => d.set_item("nodata", py.None())?,
    }
    d.set_item("width", cols)?;
    d.set_item("height", rows)?;
    Ok(d)
}

/// Build a metadata dict from COG header metadata (full-resolution extent).
fn cog_meta_dict<'py>(py: Python<'py>, m: &CogMetadata) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let t = &m.geo_transform;
    d.set_item(
        "transform",
        (
            t.origin_x,
            t.pixel_width,
            t.row_rotation,
            t.origin_y,
            t.col_rotation,
            t.pixel_height,
        ),
    )?;
    match m.crs.as_ref().and_then(|c| c.epsg()) {
        Some(code) => d.set_item("crs", format!("EPSG:{code}"))?,
        None => d.set_item("crs", py.None())?,
    }
    match m.nodata {
        Some(nd) => d.set_item("nodata", nd)?,
        None => d.set_item("nodata", py.None())?,
    }
    d.set_item("width", m.width)?;
    d.set_item("height", m.height)?;
    d.set_item("num_overviews", m.num_overviews)?;
    d.set_item("bits_per_sample", m.bits_per_sample)?;
    Ok(d)
}

/// Fetch a bounding-box window from a remote Cloud-Optimized GeoTIFF.
///
/// Args:
///     url: HTTP(S) or s3:// URL of the COG.
///     bbox: (min_x, min_y, max_x, max_y) in the COG's CRS.
///     overview: Optional overview level (0 = full resolution).
///
/// Returns:
///     (array, meta): a 2D float64 numpy array and a metadata dict
///     (transform, crs, nodata, width, height).
#[pyfunction]
#[pyo3(signature = (url, bbox, overview=None))]
fn cog_fetch<'py>(
    py: Python<'py>,
    url: &str,
    bbox: (f64, f64, f64, f64),
    overview: Option<usize>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let bb = BBox::new(bbox.0, bbox.1, bbox.2, bbox.3);
    let raster: Raster<f64> = py
        .allow_threads(|| {
            let mut reader = CogReaderBlocking::open(url, CogReaderOptions::default())?;
            reader.read_bbox(&bb, overview)
        })
        .map_err(err)?;
    let meta = raster_meta_dict(py, &raster)?;
    let arr: Array2<f64> = raster.data().clone();
    Ok((arr.into_pyarray(py), meta))
}

/// Fetch the full extent of a remote COG (optionally at an overview level).
///
/// Returns (array, meta) like [`cog_fetch`]. Use an `overview` > 0 for large
/// scenes to avoid downloading every full-resolution tile.
#[pyfunction]
#[pyo3(signature = (url, overview=None))]
fn cog_fetch_full<'py>(
    py: Python<'py>,
    url: &str,
    overview: Option<usize>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let raster: Raster<f64> = py
        .allow_threads(|| {
            let mut reader = CogReaderBlocking::open(url, CogReaderOptions::default())?;
            reader.read_full(overview)
        })
        .map_err(err)?;
    let meta = raster_meta_dict(py, &raster)?;
    let arr: Array2<f64> = raster.data().clone();
    Ok((arr.into_pyarray(py), meta))
}

/// Read a remote COG's header metadata without downloading pixel data.
///
/// Returns a dict: transform, crs, nodata, width, height, num_overviews,
/// bits_per_sample.
#[pyfunction]
fn cog_info<'py>(py: Python<'py>, url: &str) -> PyResult<Bound<'py, PyDict>> {
    let meta: CogMetadata = py
        .allow_threads(|| {
            let reader = CogReaderBlocking::open(url, CogReaderOptions::default())?;
            Ok::<_, surtgis_cloud::error::CloudError>(reader.metadata())
        })
        .map_err(err)?;
    cog_meta_dict(py, &meta)
}

/// Search a STAC catalog and return the matching items.
///
/// Args:
///     catalog: "pc" (Planetary Computer), "es"/"earth-search" (Earth Search),
///         or a STAC API root URL.
///     bbox: optional (min_x, min_y, max_x, max_y) in WGS84 lon/lat.
///     datetime: optional ISO range, e.g. "2023-01-01/2023-12-31".
///     collections: optional list of collection ids.
///     limit: optional per-page item cap.
///
/// Returns:
///     A list of dicts, one per item: {id, collection, datetime, bbox,
///     assets: {key: href}}. Hrefs from Planetary Computer must be signed with
///     [`stac_sign_href`] before passing them to [`cog_fetch`].
#[pyfunction]
#[pyo3(signature = (catalog, bbox=None, datetime=None, collections=None, limit=None))]
fn stac_search<'py>(
    py: Python<'py>,
    catalog: &str,
    bbox: Option<(f64, f64, f64, f64)>,
    datetime: Option<String>,
    collections: Option<Vec<String>>,
    limit: Option<u32>,
) -> PyResult<Bound<'py, PyList>> {
    let cat = StacCatalog::from_str_or_url(catalog);
    // Build via the builder so datetime is normalised to RFC 3339 (bare
    // YYYY-MM-DD dates are rejected by strict catalogs like Earth Search).
    let mut params = StacSearchParams::new();
    if let Some(b) = bbox {
        params = params.bbox(b.0, b.1, b.2, b.3);
    }
    if let Some(dt) = datetime.as_deref() {
        params = params.datetime(dt);
    }
    if let Some(cols) = collections.as_ref() {
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        params = params.collections(&refs);
    }
    if let Some(n) = limit {
        params = params.limit(n);
    }
    let items = py
        .allow_threads(|| {
            let client = StacClientBlocking::new(cat, StacClientOptions::default())?;
            client.search_all(&params)
        })
        .map_err(err)?;

    let out = PyList::empty(py);
    for item in &items {
        let d = PyDict::new(py);
        d.set_item("id", &item.id)?;
        match &item.collection {
            Some(c) => d.set_item("collection", c)?,
            None => d.set_item("collection", py.None())?,
        }
        match &item.properties.datetime {
            Some(dt) => d.set_item("datetime", dt)?,
            None => d.set_item("datetime", py.None())?,
        }
        match &item.bbox {
            Some(b) => d.set_item("bbox", b.clone())?,
            None => d.set_item("bbox", py.None())?,
        }
        let assets = PyDict::new(py);
        for (key, asset) in &item.assets {
            assets.set_item(key, &asset.href)?;
        }
        d.set_item("assets", assets)?;
        out.append(d)?;
    }
    Ok(out)
}

/// Sign a Planetary Computer asset href so it can be fetched.
///
/// PC asset hrefs are private blob-store URLs that require a short-lived SAS
/// token. Pass the unsigned href from [`stac_search`] and the item's collection
/// id; the returned URL can be handed directly to [`cog_fetch`].
///
/// Args:
///     href: the unsigned asset href.
///     collection: the STAC collection id (e.g. "sentinel-2-l2a").
///     catalog: catalog to sign against (default "pc").
#[pyfunction]
#[pyo3(signature = (href, collection, catalog="pc"))]
fn stac_sign_href(py: Python<'_>, href: &str, collection: &str, catalog: &str) -> PyResult<String> {
    let cat = StacCatalog::from_str_or_url(catalog);
    py.allow_threads(|| {
        let client = StacClientBlocking::new(cat, StacClientOptions::default())?;
        client.sign_asset_href(href, collection)
    })
    .map_err(err)
}

// ── STAC multiband composite ────────────────────────────────────────────

use std::sync::Arc;
use surtgis_algorithms::imagery::{CloudMaskStrategy, LandsatQaMask, NoCloudMask, S2SclMask};
use surtgis_cloud::composite::{
    CompositeEngine, CompositeSpec, DefaultAssetResolver, MaskApplier, NoProgress, OutputGrid,
    StripSink,
};

/// Wraps a cloud-mask strategy as a composite `MaskApplier`.
struct StrategyMask(Arc<dyn CloudMaskStrategy>);
impl MaskApplier for StrategyMask {
    fn apply(&self, data: &Raster<f64>, mask: &Raster<f64>) -> Raster<f64> {
        self.0.mask(data, mask).unwrap_or_else(|_| data.clone())
    }
}

/// Collects composited band strips into per-band row-major buffers.
struct BufferSink {
    n_bands: usize,
    cols: usize,
    buffers: Vec<Vec<f64>>,
}
impl StripSink for BufferSink {
    fn begin(&mut self, grid: &OutputGrid) -> surtgis_cloud::Result<()> {
        self.cols = grid.cols;
        self.buffers = (0..self.n_bands)
            .map(|_| vec![f64::NAN; grid.cols * grid.rows])
            .collect();
        Ok(())
    }
    fn accept(
        &mut self,
        band_idx: usize,
        row_start: usize,
        strip: Array2<f64>,
    ) -> surtgis_cloud::Result<()> {
        let buf = &mut self.buffers[band_idx];
        let offset = row_start * self.cols;
        if let Some(slice) = strip.as_slice() {
            buf[offset..offset + slice.len()].copy_from_slice(slice);
        }
        Ok(())
    }
}

/// Map a collection id to (mask asset key, cloud-mask strategy), mirroring the
/// CLI's `CollectionProfile`. Unknown collections get no masking.
fn mask_for_collection(collection: &str) -> (Option<String>, Arc<dyn CloudMaskStrategy>) {
    match collection {
        "sentinel-2-l2a" => (Some("scl".into()), Arc::new(S2SclMask::new())),
        "landsat-c2-l2" => (Some("QA_PIXEL".into()), Arc::new(LandsatQaMask::new())),
        _ => (None, Arc::new(NoCloudMask)),
    }
}

/// Build a cloud-free multiband composite from a STAC catalog.
///
/// Searches the catalog, downloads and mosaics each band across the selected
/// scene dates, applies cloud masking, and reduces per pixel to a median
/// composite (with coverage- and neighbour-based gap filling). Experimental
/// (mirrors `surtgis stac composite`); the API may change in a minor release.
///
/// Args:
///     catalog: "pc", "es", or a full STAC API URL.
///     collection: collection id (e.g. "sentinel-2-l2a").
///     bbox: (min_x, min_y, max_x, max_y) in WGS84 degrees.
///     bands: band/asset keys to composite (e.g. ["red", "nir"]).
///     datetime: STAC datetime query (instant or "start/end").
///     max_scenes: max scene dates to composite (default 6).
///     band_chunk_size: bands downloaded together per chunk (default 1 = min RAM).
///     budget_gb: RAM budget for strip sizing (default 16.0).
///     max_tile_failures: abort after this many failed tiles (0 = never).
///     use_cache: cache decoded tiles on disk between strips (default False).
///
/// Returns:
///     (bands, meta): `bands` is a dict of band key → 2D float64 numpy array
///     (NaN = no data); `meta` is a dict with `transform` (GDAL 6-tuple),
///     `crs`, `width`, `height` shared by every band, plus a quality report
///     (`scenes_used`, `total_tiles`, `failed_tiles`, `failed_dates`,
///     `last_error`) so degraded output from failed tiles is visible rather
///     than silently gap-filled.
#[pyfunction]
#[pyo3(signature = (
    catalog, collection, bbox, bands, datetime, max_scenes=6,
    band_chunk_size=1, budget_gb=16.0, max_tile_failures=0, use_cache=false
))]
#[allow(clippy::too_many_arguments)]
fn composite<'py>(
    py: Python<'py>,
    catalog: &str,
    collection: &str,
    bbox: (f64, f64, f64, f64),
    bands: Vec<String>,
    datetime: &str,
    max_scenes: usize,
    band_chunk_size: usize,
    budget_gb: f64,
    max_tile_failures: usize,
    use_cache: bool,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    if bands.is_empty() {
        return Err(PyValueError::new_err("bands must not be empty"));
    }
    let (mask_key, strategy) = mask_for_collection(collection);
    let spec = CompositeSpec {
        catalog: catalog.to_string(),
        collection: collection.to_string(),
        bbox_wgs84: BBox::new(bbox.0, bbox.1, bbox.2, bbox.3),
        band_keys: bands.clone(),
        mask_key,
        datetime: datetime.to_string(),
        max_scenes,
        align_grid: None,
        strip_rows: 512,
        band_chunk_size,
        budget_gb,
        max_tile_failures,
        use_cache,
    };

    let n_bands = bands.len();
    let mask = StrategyMask(strategy);
    let mut sink = BufferSink {
        n_bands,
        cols: 0,
        buffers: Vec::new(),
    };

    let report = py
        .allow_threads(|| {
            let mut engine = CompositeEngine::new(spec)?;
            engine.run(&DefaultAssetResolver, &mask, &mut sink, &mut NoProgress)
        })
        .map_err(err)?;
    let grid = &report.grid;

    // Assemble the per-band numpy arrays.
    let bands_out = PyDict::new(py);
    for (bi, name) in bands.iter().enumerate() {
        let buf = std::mem::take(&mut sink.buffers[bi]);
        let arr = Array2::from_shape_vec((grid.rows, grid.cols), buf).map_err(err)?;
        bands_out.set_item(name, arr.into_pyarray(py))?;
    }

    let meta = PyDict::new(py);
    let t = &grid.transform;
    meta.set_item(
        "transform",
        (
            t.origin_x,
            t.pixel_width,
            t.row_rotation,
            t.origin_y,
            t.col_rotation,
            t.pixel_height,
        ),
    )?;
    match grid.epsg() {
        Some(code) => meta.set_item("crs", format!("EPSG:{code}"))?,
        None => meta.set_item("crs", py.None())?,
    }
    meta.set_item("width", grid.cols)?;
    meta.set_item("height", grid.rows)?;
    // Quality report: with the default `max_tile_failures=0` the composite
    // does not abort on failed tiles, so surface the counts instead of
    // silently returning gap-filled data as if it were complete.
    meta.set_item("scenes_used", report.scenes_used)?;
    meta.set_item("total_tiles", report.total_tiles)?;
    meta.set_item("failed_tiles", report.failed_tiles)?;
    meta.set_item("failed_dates", report.failed_dates)?;
    match &report.last_error {
        Some(e) => meta.set_item("last_error", e)?,
        None => meta.set_item("last_error", py.None())?,
    }
    Ok((bands_out, meta))
}

/// Register the cloud functions on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cog_fetch, m)?)?;
    m.add_function(wrap_pyfunction!(cog_fetch_full, m)?)?;
    m.add_function(wrap_pyfunction!(cog_info, m)?)?;
    m.add_function(wrap_pyfunction!(stac_search, m)?)?;
    m.add_function(wrap_pyfunction!(stac_sign_href, m)?)?;
    m.add_function(wrap_pyfunction!(composite, m)?)?;
    Ok(())
}
