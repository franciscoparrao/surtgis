//! HTTP handlers.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use surtgis_colormap::{
    ColorScheme, ColormapParams, auto_params, raster_to_rgba, rgba_to_png_bytes,
};
use surtgis_core::Raster;
use surtgis_core::warp::{self, Resampling, Transformer};

use crate::AppState;
use crate::algorithms::{self, Op};
use crate::error::ServeError;
use crate::source::Source;
use crate::tiles::{self, TILE_EPSG, TILE_SIZE};

/// Query string of `/tiles` and `/tilejson`.
#[derive(Debug, Clone, Deserialize)]
pub struct TileQuery {
    /// Source: COG URL or local path.
    pub url: String,
    /// Operator name (`hillshade`, `slope`, `value`).
    pub alg: Option<String>,
    /// Band formula (ASI grammar); exclusive with `alg`.
    pub formula: Option<String>,
    /// Band mapping `NAME:INDEX,...` (formula) or a single index (`value`).
    pub bands: Option<String>,
    /// Colormap scheme name (see `surtgis_colormap::ColorScheme::ALL`).
    pub cmap: Option<String>,
    /// Colormap domain `min,max`; defaults to the operator's range or the
    /// tile's own min/max.
    pub rescale: Option<String>,
    /// Operator parameters `key:value,...`.
    pub params: Option<String>,
}

/// Query string of `/info`.
#[derive(Debug, Clone, Deserialize)]
pub struct InfoQuery {
    /// Source: COG URL or local path.
    pub url: String,
}

fn parse_cmap(name: Option<&str>) -> Result<ColorScheme, ServeError> {
    let name = name.unwrap_or("grayscale");
    ColorScheme::ALL
        .iter()
        .copied()
        .find(|s| {
            s.name().eq_ignore_ascii_case(name)
                || s.name().replace(' ', "-").eq_ignore_ascii_case(name)
        })
        .ok_or_else(|| {
            ServeError::BadRequest(format!(
                "unknown cmap '{name}'; one of: {}",
                ColorScheme::ALL
                    .iter()
                    .map(|s| s.name().to_ascii_lowercase())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

fn parse_rescale(s: Option<&str>) -> Result<Option<(f64, f64)>, ServeError> {
    let Some(s) = s else { return Ok(None) };
    let (a, b) = s
        .split_once(',')
        .ok_or_else(|| ServeError::BadRequest("rescale must be min,max".into()))?;
    let (min, max) = (a.trim().parse::<f64>(), b.trim().parse::<f64>());
    match (min, max) {
        (Ok(min), Ok(max)) if min < max => Ok(Some((min, max))),
        _ => Err(ServeError::BadRequest(
            "rescale must be two numbers with min < max".into(),
        )),
    }
}

fn crop_gutter(r: &Raster<f64>, gutter: usize) -> Raster<f64> {
    if gutter == 0 {
        return r.clone();
    }
    let (rows, cols) = r.shape();
    let view = r
        .data()
        .slice(ndarray::s![gutter..rows - gutter, gutter..cols - gutter])
        .to_owned();
    let mut out = Raster::from_array(view);
    let t = r.transform();
    out.set_transform(surtgis_core::raster::GeoTransform::new(
        t.origin_x + gutter as f64 * t.pixel_width,
        t.origin_y + gutter as f64 * t.pixel_height,
        t.pixel_width,
        t.pixel_height,
    ));
    out.set_crs(r.crs().cloned());
    out.set_nodata(r.nodata());
    out
}

fn transparent_png() -> Bytes {
    let rgba = vec![0u8; TILE_SIZE * TILE_SIZE * 4];
    Bytes::from(rgba_to_png_bytes(TILE_SIZE as u32, TILE_SIZE as u32, &rgba).unwrap_or_default())
}

fn png_response(bytes: Bytes, etag: &str, max_age: u32) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_str(&format!("public, max-age={max_age}")).unwrap(),
    );
    if let Ok(v) = HeaderValue::from_str(&format!("\"{etag}\"")) {
        headers.insert(header::ETAG, v);
    }
    (StatusCode::OK, headers, bytes).into_response()
}

/// Render tile `(z, x, y)` for the query. This is the whole pipeline of the
/// design document §4.2: validate the source, tile → grid + gutter, read
/// the source window, warp onto the grid, run the operator, crop, colour,
/// encode.
pub async fn render_tile(
    state: &AppState,
    z: u8,
    x: u32,
    y: u32,
    q: &TileQuery,
) -> Result<Bytes, ServeError> {
    if !tiles::is_valid(z, x, y) {
        return Err(ServeError::BadRequest(format!(
            "tile {z}/{x}/{y} does not exist"
        )));
    }
    let source = Source::resolve(&q.url, &state.sources)?;
    let op = Op::parse(
        q.alg.as_deref(),
        q.formula.as_deref(),
        q.bands.as_deref(),
        q.params.as_deref(),
    )?;
    let scheme = parse_cmap(q.cmap.as_deref())?;
    let rescale = parse_rescale(q.rescale.as_deref())?;

    let gutter = op.gutter();
    let grid = tiles::grid(z, x, y, gutter);

    // Source CRS → the window to read (2 extra target cells of bilinear support).
    let info_epsg = match &source {
        Source::Local(_) | Source::Http(_) => source.info(&state.local).await?.epsg,
    };
    let tf =
        Transformer::new(info_epsg, TILE_EPSG).map_err(|e| ServeError::Source(e.to_string()))?;
    let src_bounds =
        warp::source_window(&grid, &tf, 2).map_err(|e| ServeError::Source(e.to_string()))?;
    let window = source
        .read_window(&state.local, &src_bounds, grid.cols)
        .await?;
    if window.epsg != info_epsg {
        return Err(ServeError::Source(
            "source CRS changed between reads".into(),
        ));
    }

    // CPU work off the async runtime.
    let max_age = state.max_age;
    let png = tokio::task::spawn_blocking(move || -> Result<Bytes, ServeError> {
        let src_gt = *window.bands[0].transform();
        let warped = warp::warp(&window.bands, &src_gt, &tf, &grid, Resampling::Bilinear)
            .map_err(|e| ServeError::Compute(e.to_string()))?;
        let result = op.run(&warped)?;
        let tile = crop_gutter(&result, gutter);
        let params = match rescale.or(op.default_range()) {
            Some((min, max)) => ColormapParams::with_range(scheme, min, max),
            None => auto_params(&tile, scheme),
        };
        let rgba = raster_to_rgba(&tile, &params);
        rgba_to_png_bytes(TILE_SIZE as u32, TILE_SIZE as u32, &rgba)
            .map(Bytes::from)
            .map_err(|e| ServeError::Compute(e.to_string()))
    })
    .await
    .map_err(|e| ServeError::Compute(format!("render task failed: {e}")))??;
    let _ = max_age;
    Ok(png)
}

fn etag_for(z: u8, x: u32, y: u32, q: &TileQuery) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (
        z, x, y, &q.url, &q.alg, &q.formula, &q.bands, &q.cmap, &q.rescale, &q.params,
    )
        .hash(&mut h);
    format!("{:016x}", h.finish())
}

/// `GET /tiles/{z}/{x}/{y}.png?url=…`
pub async fn tile(
    State(state): State<Arc<AppState>>,
    Path((z, x, y_ext)): Path<(u8, u32, String)>,
    Query(q): Query<TileQuery>,
) -> Response {
    let y_str = y_ext.strip_suffix(".png").unwrap_or(&y_ext);
    let Ok(y) = y_str.parse::<u32>() else {
        return ServeError::BadRequest(format!("bad tile row '{y_ext}'")).into_response();
    };
    let etag = etag_for(z, x, y, &q);
    match render_tile(&state, z, x, y, &q).await {
        Ok(png) => png_response(png, &etag, state.max_age),
        Err(ServeError::Outside) => png_response(transparent_png(), &etag, state.max_age),
        Err(e) => {
            tracing::warn!(tile = %format!("{z}/{x}/{y}"), url = %q.url, error = %e, "tile failed");
            e.into_response()
        }
    }
}

/// `GET /tilejson?url=…` — TileJSON 3.0 for the same query.
pub async fn tilejson(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<TileQuery>,
    raw: axum::extract::RawQuery,
) -> Result<axum::Json<serde_json::Value>, ServeError> {
    let source = Source::resolve(&q.url, &state.sources)?;
    Op::parse(
        q.alg.as_deref(),
        q.formula.as_deref(),
        q.bands.as_deref(),
        q.params.as_deref(),
    )?;
    let info = source.info(&state.local).await?;
    let tf = Transformer::new(info.epsg, 4326).map_err(|e| ServeError::Source(e.to_string()))?;
    let corners = [
        (info.bounds[0], info.bounds[1]),
        (info.bounds[2], info.bounds[1]),
        (info.bounds[2], info.bounds[3]),
        (info.bounds[0], info.bounds[3]),
    ];
    let mut b = [180.0f64, 90.0, -180.0, -90.0];
    for (x, y) in corners {
        if let Some((lon, lat)) = tf.forward(x, y) {
            b[0] = b[0].min(lon);
            b[1] = b[1].min(lat);
            b[2] = b[2].max(lon);
            b[3] = b[3].max(lat);
        }
    }
    let host = headers
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("localhost");
    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("http");
    let query = raw.0.unwrap_or_default();
    // Native pixel size → the zoom whose resolution is about as fine.
    let native_res_m = if tf.src_is_geographic() {
        info.pixel_size * 111_320.0
    } else {
        info.pixel_size
    };
    let maxzoom = (0..=24u8)
        .find(|&z| tiles::resolution(z) <= native_res_m)
        .unwrap_or(24)
        .min(24);
    Ok(axum::Json(serde_json::json!({
        "tilejson": "3.0.0",
        "name": q.alg.clone().or(q.formula.clone()).unwrap_or_else(|| "value".into()),
        "tiles": [format!("{scheme}://{host}/tiles/{{z}}/{{x}}/{{y}}.png?{query}")],
        "bounds": b,
        "minzoom": 0,
        "maxzoom": maxzoom,
        "scheme": "xyz",
        "attribution": "SurtGIS Server",
    })))
}

/// `GET /info?url=…`
pub async fn info(
    State(state): State<Arc<AppState>>,
    Query(q): Query<InfoQuery>,
) -> Result<axum::Json<crate::source::SourceInfo>, ServeError> {
    let source = Source::resolve(&q.url, &state.sources)?;
    Ok(axum::Json(source.info(&state.local).await?))
}

/// `GET /algorithms`
pub async fn algorithms_catalog() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "algorithms": algorithms::catalog(),
        "colormaps": ColorScheme::ALL.iter().map(|s| s.name().to_ascii_lowercase()).collect::<Vec<_>>(),
        "tile_matrix_set": "WebMercatorQuad",
        "tile_size": TILE_SIZE,
    }))
}

/// `GET /healthz`
pub async fn healthz() -> &'static str {
    "ok"
}

/// `GET /` — the MapLibre demo page.
pub async fn demo() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../demo/index.html"))
}
