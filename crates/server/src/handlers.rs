//! HTTP handlers.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use surtgis_colormap::{
    ColorScheme, ColormapParams, auto_params, raster_to_rgba, rgba_to_png_bytes,
};
use surtgis_core::Raster;
use surtgis_core::warp::{self, Bounds, GridSpec, Resampling, Transformer};

use crate::AppState;
use crate::algorithms::{self, Op};
use crate::error::ServeError;
use crate::source::Source;
use crate::tiles::{self, TILE_EPSG, TILE_SIZE};

/// Query string of `/tiles`, `/tilejson` and `/statistics`.
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
    /// `/statistics` only: side of the sampling grid in pixels (default 512).
    pub size: Option<usize>,
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

fn png_response(bytes: Bytes, etag: &str, max_age: u32, cached: bool) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_str(&format!("public, max-age={max_age}")).unwrap(),
    );
    if let Ok(v) = HeaderValue::from_str(&format!("\"{etag}\"")) {
        headers.insert(header::ETAG, v);
    }
    headers.insert(
        axum::http::HeaderName::from_static("x-cache"),
        HeaderValue::from_static(if cached { "hit" } else { "miss" }),
    );
    (StatusCode::OK, headers, bytes).into_response()
}

/// Run `op` over `grid`: read the source window it needs, warp it onto
/// the grid and apply the operator. Shared by tiles and `/statistics`.
pub async fn compute_grid(
    state: &AppState,
    source: &Source,
    op: &Op,
    grid: GridSpec,
) -> Result<Raster<f64>, ServeError> {
    let info = source.info(&state.local, &state.pool).await?;
    let tf =
        Transformer::new(info.epsg, grid.epsg).map_err(|e| ServeError::Source(e.to_string()))?;
    // 2 extra target cells of bilinear support around the grid.
    let src_bounds =
        warp::source_window(&grid, &tf, 2).map_err(|e| ServeError::Source(e.to_string()))?;
    let window = source
        .read_window(&state.local, &state.pool, &src_bounds, grid.cols)
        .await?;
    if window.epsg != info.epsg {
        return Err(ServeError::Source(
            "source CRS changed between reads".into(),
        ));
    }
    let op = op.clone();
    tokio::task::spawn_blocking(move || -> Result<Raster<f64>, ServeError> {
        let warped = warp_window(&window.bands, &tf, &grid)?;
        op.run(&warped)
    })
    .await
    .map_err(|e| ServeError::Compute(format!("compute task failed: {e}")))?
}

fn warp_window(
    bands: &[Raster<f64>],
    tf: &Transformer,
    grid: &GridSpec,
) -> Result<Vec<Raster<f64>>, ServeError> {
    let src_gt = *bands[0].transform();
    warp::warp(bands, &src_gt, tf, grid, Resampling::Bilinear)
        .map_err(|e| ServeError::Compute(e.to_string()))
}

/// True-colour tile: warp the three requested bands and pack them as RGBA
/// (alpha 0 where any band is NaN), scaling `rescale` (default 0–255) to
/// 0–255.
async fn render_rgb(
    state: &AppState,
    req: &Request,
    grid: GridSpec,
    bands: [usize; 3],
) -> Result<Bytes, ServeError> {
    let info = req.source.info(&state.local, &state.pool).await?;
    let tf =
        Transformer::new(info.epsg, grid.epsg).map_err(|e| ServeError::Source(e.to_string()))?;
    let src_bounds =
        warp::source_window(&grid, &tf, 2).map_err(|e| ServeError::Source(e.to_string()))?;
    let window = req
        .source
        .read_window(&state.local, &state.pool, &src_bounds, grid.cols)
        .await?;
    let need = *bands.iter().max().unwrap();
    if window.bands.len() < need {
        return Err(ServeError::BadRequest(format!(
            "source has {} band(s), rgb needs band {need}",
            window.bands.len()
        )));
    }
    let (lo, hi) = req.rescale.unwrap_or((0.0, 255.0));
    tokio::task::spawn_blocking(move || -> Result<Bytes, ServeError> {
        let picked: Vec<Raster<f64>> = bands.iter().map(|&i| window.bands[i - 1].clone()).collect();
        let warped = warp_window(&picked, &tf, &grid)?;
        let n = TILE_SIZE * TILE_SIZE;
        let mut rgba = vec![0u8; n * 4];
        let scale = 255.0 / (hi - lo);
        for r in 0..TILE_SIZE {
            for c in 0..TILE_SIZE {
                let px = (r * TILE_SIZE + c) * 4;
                let v = [
                    warped[0].data()[[r, c]],
                    warped[1].data()[[r, c]],
                    warped[2].data()[[r, c]],
                ];
                if v.iter().any(|x| !x.is_finite()) {
                    continue;
                }
                for k in 0..3 {
                    rgba[px + k] = ((v[k] - lo) * scale).round().clamp(0.0, 255.0) as u8;
                }
                rgba[px + 3] = 255;
            }
        }
        rgba_to_png_bytes(TILE_SIZE as u32, TILE_SIZE as u32, &rgba)
            .map(Bytes::from)
            .map_err(|e| ServeError::Compute(e.to_string()))
    })
    .await
    .map_err(|e| ServeError::Compute(format!("render task failed: {e}")))?
}

/// Parsed request pieces shared by the endpoints.
struct Request {
    source: Source,
    op: Op,
    scheme: ColorScheme,
    rescale: Option<(f64, f64)>,
}

fn parse_request(state: &AppState, q: &TileQuery) -> Result<Request, ServeError> {
    Ok(Request {
        source: Source::resolve(&q.url, &state.sources)?,
        op: Op::parse(
            q.alg.as_deref(),
            q.formula.as_deref(),
            q.bands.as_deref(),
            q.params.as_deref(),
        )?,
        scheme: parse_cmap(q.cmap.as_deref())?,
        rescale: parse_rescale(q.rescale.as_deref())?,
    })
}

/// Render tile `(z, x, y)` for the query: the pipeline of the design
/// document §4.2 — validate, tile → grid + gutter, read, warp, operator,
/// crop, colour, encode.
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
    let req = parse_request(state, q)?;
    let gutter = req.op.gutter();
    let grid = tiles::grid(z, x, y, gutter);
    if let Some(bands) = req.op.rgb_bands() {
        return render_rgb(state, &req, grid, bands).await;
    }
    let result = compute_grid(state, &req.source, &req.op, grid).await?;
    let (scheme, rescale, default_range) = (req.scheme, req.rescale, req.op.default_range());
    tokio::task::spawn_blocking(move || -> Result<Bytes, ServeError> {
        let tile = crop_gutter(&result, gutter);
        let params = match rescale.or(default_range) {
            Some((min, max)) => ColormapParams::with_range(scheme, min, max),
            None => auto_params(&tile, scheme),
        };
        let rgba = raster_to_rgba(&tile, &params);
        rgba_to_png_bytes(TILE_SIZE as u32, TILE_SIZE as u32, &rgba)
            .map(Bytes::from)
            .map_err(|e| ServeError::Compute(e.to_string()))
    })
    .await
    .map_err(|e| ServeError::Compute(format!("render task failed: {e}")))?
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

fn alg_label(q: &TileQuery) -> String {
    if q.formula.is_some() {
        "formula".into()
    } else {
        q.alg.clone().unwrap_or_else(|| "value".into())
    }
}

/// `GET /tiles/{z}/{x}/{y}.png?url=…`
pub async fn tile(
    State(state): State<Arc<AppState>>,
    Path((z, x, y_ext)): Path<(u8, u32, String)>,
    headers: HeaderMap,
    Query(q): Query<TileQuery>,
) -> Response {
    let y_str = y_ext.strip_suffix(".png").unwrap_or(&y_ext);
    let Ok(y) = y_str.parse::<u32>() else {
        return ServeError::BadRequest(format!("bad tile row '{y_ext}'")).into_response();
    };
    let etag = etag_for(z, x, y, &q);
    let alg = alg_label(&q);

    // Conditional request: the ETag is a pure function of the query.
    if headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains(etag.as_str()))
    {
        state.metrics.tile(&alg, "not_modified");
        return StatusCode::NOT_MODIFIED.into_response();
    }
    if let Some(png) = state.tile_cache.get(&etag) {
        state.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
        state.metrics.tile(&alg, "cached");
        return png_response(png, &etag, state.max_age, true);
    }
    state.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);

    let Ok(_permit) = state.inflight.try_acquire() else {
        state.metrics.rejected.fetch_add(1, Ordering::Relaxed);
        state.metrics.tile(&alg, "rejected");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "1")],
            "too many tiles in flight",
        )
            .into_response();
    };

    let started = Instant::now();
    let outcome = tokio::time::timeout(state.timeout, render_tile(&state, z, x, y, &q)).await;
    let elapsed = started.elapsed().as_secs_f64();
    match outcome {
        Ok(Ok(png)) => {
            state.metrics.observe(&alg, elapsed);
            state.metrics.tile(&alg, "ok");
            state.tile_cache.put(etag.clone(), png.clone());
            png_response(png, &etag, state.max_age, false)
        }
        Ok(Err(ServeError::Outside)) => {
            state.metrics.tile(&alg, "empty");
            let png = transparent_png();
            state.tile_cache.put(etag.clone(), png.clone());
            png_response(png, &etag, state.max_age, false)
        }
        Ok(Err(e)) => {
            state.metrics.tile(&alg, "error");
            tracing::warn!(tile = %format!("{z}/{x}/{y}"), url = %q.url, error = %e, "tile failed");
            e.into_response()
        }
        Err(_) => {
            state.metrics.timeouts.fetch_add(1, Ordering::Relaxed);
            state.metrics.tile(&alg, "timeout");
            tracing::warn!(tile = %format!("{z}/{x}/{y}"), url = %q.url, "tile timed out");
            (StatusCode::GATEWAY_TIMEOUT, "tile timed out").into_response()
        }
    }
}

/// Source bounds in WGS84 as `[w, s, e, n]`.
fn bounds_wgs84(info: &crate::source::SourceInfo) -> Result<[f64; 4], ServeError> {
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
    Ok(b)
}

/// `GET /tilejson?url=…` — TileJSON 3.0 for the same query.
pub async fn tilejson(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<TileQuery>,
    raw: axum::extract::RawQuery,
) -> Result<axum::Json<serde_json::Value>, ServeError> {
    let req = parse_request(&state, &q)?;
    let info = req.source.info(&state.local, &state.pool).await?;
    let b = bounds_wgs84(&info)?;
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
    let native_res_m = if Transformer::new(info.epsg, 4326)
        .map(|t| t.src_is_geographic())
        .unwrap_or(false)
    {
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
        "name": alg_label(&q),
        "tiles": [format!("{scheme}://{host}/tiles/{{z}}/{{x}}/{{y}}.png?{query}")],
        "bounds": b,
        "minzoom": 0,
        "maxzoom": maxzoom,
        "scheme": "xyz",
        "attribution": "SurtGIS Server",
    })))
}

/// `GET /statistics?url=…&alg=…[&size=512]` — the operator evaluated on a
/// coarse grid covering the whole source (Web Mercator), summarised:
/// count, min, max, mean, std, 2nd and 98th percentiles, and the
/// `rescale` those percentiles suggest.
pub async fn statistics(
    State(state): State<Arc<AppState>>,
    Query(q): Query<TileQuery>,
) -> Result<axum::Json<serde_json::Value>, ServeError> {
    let req = parse_request(&state, &q)?;
    let size = q.size.unwrap_or(512).clamp(16, 2048);
    let info = req.source.info(&state.local, &state.pool).await?;
    let tf =
        Transformer::new(info.epsg, TILE_EPSG).map_err(|e| ServeError::Source(e.to_string()))?;
    let src_gt = surtgis_core::raster::GeoTransform::new(
        info.bounds[0],
        info.bounds[3],
        (info.bounds[2] - info.bounds[0]) / info.width as f64,
        -(info.bounds[3] - info.bounds[1]) / info.height as f64,
    );
    let merc: Bounds = warp::target_bounds(&src_gt, info.height, info.width, &tf)
        .map_err(|e| ServeError::Source(e.to_string()))?;
    let px = merc.width().max(merc.height()) / size as f64;
    let gutter = req.op.gutter();
    let inner =
        GridSpec::covering(&merc, px, TILE_EPSG).map_err(|e| ServeError::Source(e.to_string()))?;
    let grid = GridSpec {
        transform: surtgis_core::raster::GeoTransform::new(
            inner.transform.origin_x - gutter as f64 * px,
            inner.transform.origin_y + gutter as f64 * px,
            px,
            -px,
        ),
        rows: inner.rows + 2 * gutter,
        cols: inner.cols + 2 * gutter,
        epsg: TILE_EPSG,
    };
    let result = compute_grid(&state, &req.source, &req.op, grid).await?;
    let stats = tokio::task::spawn_blocking(move || {
        let tile = crop_gutter(&result, gutter);
        let mut v: Vec<f64> = tile
            .data()
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if v.is_empty() {
            return serde_json::json!({ "count": 0 });
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len();
        let mean = v.iter().sum::<f64>() / n as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let pct = |p: f64| v[((n - 1) as f64 * p).round() as usize];
        serde_json::json!({
            "count": n,
            "grid": [tile.shape().1, tile.shape().0],
            "min": v[0],
            "max": v[n - 1],
            "mean": mean,
            "std": var.sqrt(),
            "p2": pct(0.02),
            "p50": pct(0.5),
            "p98": pct(0.98),
            "rescale": format!("{},{}", pct(0.02), pct(0.98)),
        })
    })
    .await
    .map_err(|e| ServeError::Compute(format!("statistics task failed: {e}")))?;
    Ok(axum::Json(stats))
}

/// `GET /info?url=…`
pub async fn info(
    State(state): State<Arc<AppState>>,
    Query(q): Query<InfoQuery>,
) -> Result<axum::Json<crate::source::SourceInfo>, ServeError> {
    let source = Source::resolve(&q.url, &state.sources)?;
    Ok(axum::Json(source.info(&state.local, &state.pool).await?))
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

/// `GET /metrics` — Prometheus text exposition.
pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let (entries, bytes, _, _) = state.tile_cache.stats();
    let (files, local_bytes, _, _) = state.local.stats();
    let (urls, readers) = state.pool.stats();
    let body = state.metrics.render(&[
        (
            "surtgis_tile_cache_entries",
            "Rendered tiles held in the L1 cache.",
            entries as f64,
        ),
        (
            "surtgis_tile_cache_bytes",
            "Bytes held in the L1 cache.",
            bytes as f64,
        ),
        (
            "surtgis_local_sources",
            "Local sources held in memory.",
            files as f64,
        ),
        (
            "surtgis_local_bytes",
            "Bytes of local sources held in memory.",
            local_bytes as f64,
        ),
        (
            "surtgis_pool_urls",
            "Remote sources with open readers.",
            urls as f64,
        ),
        (
            "surtgis_pool_readers",
            "Open remote readers.",
            readers as f64,
        ),
        (
            "surtgis_inflight_available",
            "Render slots currently free.",
            state.inflight.available_permits() as f64,
        ),
    ]);
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

/// `GET /healthz`
pub async fn healthz() -> &'static str {
    "ok"
}

/// `GET /` — the MapLibre demo page.
pub async fn demo() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../demo/index.html"))
}
