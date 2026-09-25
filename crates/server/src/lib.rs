//! SurtGIS Server — an analysis-first dynamic tile server.
//!
//! A tile is not a stored byte but the result of a computation over a
//! remote or local raster, performed at request time: terrain operators,
//! spectral formulas and colormaps are parameters of a standard XYZ tile
//! URL. Design: `docs/surtgis_server_design.md`.
//!
//! ```text
//! GET /tiles/{z}/{x}/{y}.png?url=<COG|path>&alg=hillshade&cmap=terrain
//! GET /tilejson?url=…&alg=…        TileJSON 3.0 for a MapLibre/Leaflet source
//! GET /statistics?url=…&alg=…      min/max/mean/percentiles of the operator over the source
//! GET /info?url=…                  source metadata
//! GET /algorithms                  catalogue (name, class, gutter, params)
//! GET /metrics                     Prometheus text exposition
//! GET /healthz
//! GET /                            MapLibre demo page
//! ```
//!
//! Only local and focal operators tile on the fly; global ones (flow
//! accumulation, watersheds) are materialised offline and served as
//! ordinary sources.

#![deny(missing_docs)]

pub mod algorithms;
pub mod cache;
pub mod error;
pub mod handlers;
pub mod metrics;
pub mod pool;
pub mod source;
pub mod tiles;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::routing::get;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub use error::ServeError;
pub use source::{LocalCache, Source, SourceConfig, SourceInfo};

/// Deployment configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Socket to listen on.
    pub bind: SocketAddr,
    /// URL prefixes remote sources may start with (empty: no remote sources).
    pub allow: Vec<String>,
    /// Directory local sources must live under (none: no local sources).
    pub root: Option<PathBuf>,
    /// `Cache-Control: max-age` sent with tiles, in seconds.
    pub max_age: u32,
    /// L1 cache of rendered tiles, in MiB (0 disables).
    pub cache_mb: usize,
    /// Per-tile deadline, in milliseconds.
    pub timeout_ms: u64,
    /// Tiles rendered concurrently before requests are rejected with 503.
    pub max_inflight: usize,
    /// Open COG readers per URL.
    pub pool_per_url: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: ([127, 0, 0, 1], 8080).into(),
            allow: Vec::new(),
            root: None,
            max_age: 3600,
            cache_mb: 256,
            timeout_ms: 30_000,
            max_inflight: 64,
            pool_per_url: 4,
        }
    }
}

/// Shared state behind every handler.
pub struct AppState {
    /// Source policy.
    pub sources: SourceConfig,
    /// Metadata of local sources.
    pub local: LocalCache,
    /// Open remote readers.
    pub pool: pool::ReaderPool,
    /// Rendered tiles (L1), keyed by the request's ETag.
    pub tile_cache: cache::ByteLru<String, Bytes>,
    /// Request counters.
    pub metrics: metrics::Metrics,
    /// Concurrency limit on rendering.
    pub inflight: tokio::sync::Semaphore,
    /// Per-tile deadline.
    pub timeout: Duration,
    /// `Cache-Control: max-age` for tiles.
    pub max_age: u32,
}

impl AppState {
    /// Build the state from a configuration (canonicalising the root).
    pub fn new(cfg: &ServerConfig) -> anyhow::Result<Self> {
        let root = match &cfg.root {
            Some(r) => Some(
                r.canonicalize()
                    .map_err(|e| anyhow::anyhow!("--root {}: {e}", r.display()))?,
            ),
            None => None,
        };
        Ok(Self {
            sources: SourceConfig {
                allow: cfg.allow.clone(),
                root,
            },
            local: LocalCache::default(),
            pool: pool::ReaderPool::new(cfg.pool_per_url, Duration::from_secs(300)),
            tile_cache: cache::ByteLru::new(cfg.cache_mb << 20, |b: &Bytes| b.len()),
            metrics: metrics::Metrics::default(),
            inflight: tokio::sync::Semaphore::new(cfg.max_inflight.max(1)),
            timeout: Duration::from_millis(cfg.timeout_ms.max(100)),
            max_age: cfg.max_age,
        })
    }
}

/// The application router over `state`.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(handlers::demo))
        .route("/healthz", get(handlers::healthz))
        .route("/metrics", get(handlers::metrics))
        .route("/algorithms", get(handlers::algorithms_catalog))
        .route("/info", get(handlers::info))
        .route("/statistics", get(handlers::statistics))
        .route("/tilejson", get(handlers::tilejson))
        .route("/tiles/{z}/{x}/{y}", get(handlers::tile))
        .layer(CorsLayer::permissive())
        .layer(
            TraceLayer::new_for_http()
                .on_failure(tower_http::trace::DefaultOnFailure::new().level(tracing::Level::WARN)),
        )
        .with_state(state)
}

/// Serve until Ctrl-C.
pub async fn serve(cfg: ServerConfig) -> anyhow::Result<()> {
    let state = Arc::new(AppState::new(&cfg)?);
    let app = router(state);
    let listener = tokio::net::TcpListener::bind(cfg.bind).await?;
    tracing::info!(
        bind = %cfg.bind,
        root = ?cfg.root,
        allow = ?cfg.allow,
        cache_mb = cfg.cache_mb,
        max_inflight = cfg.max_inflight,
        "surtgis serve listening"
    );
    eprintln!(
        "SurtGIS Server on http://{}  (demo at /, catalogue at /algorithms, metrics at /metrics)",
        cfg.bind
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

/// Build a Tokio runtime, install a `tracing` subscriber (`RUST_LOG`
/// honoured) and run [`serve`] to completion. For callers without an
/// async runtime of their own, such as the CLI.
pub fn serve_blocking(cfg: ServerConfig) -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(serve(cfg))
}
