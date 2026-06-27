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

/// Register the cloud functions on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cog_fetch, m)?)?;
    m.add_function(wrap_pyfunction!(cog_fetch_full, m)?)?;
    m.add_function(wrap_pyfunction!(cog_info, m)?)?;
    m.add_function(wrap_pyfunction!(stac_search, m)?)?;
    m.add_function(wrap_pyfunction!(stac_sign_href, m)?)?;
    Ok(())
}
