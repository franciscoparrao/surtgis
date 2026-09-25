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
//! GET /info?url=…                  source metadata
//! GET /algorithms                  catalogue (name, class, gutter, params)
//! GET /healthz
//! GET /                            MapLibre demo page
//! ```
//!
//! Only local and focal operators tile on the fly; global ones (flow
//! accumulation, watersheds) are materialised offline and served as
//! ordinary sources.

#![deny(missing_docs)]

pub mod algorithms;
pub mod error;
pub mod handlers;
pub mod source;
pub mod tiles;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
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
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: ([127, 0, 0, 1], 8080).into(),
            allow: Vec::new(),
            root: None,
            max_age: 3600,
        }
    }
}

/// Shared state behind every handler.
pub struct AppState {
    /// Source policy.
    pub sources: SourceConfig,
    /// In-memory local sources.
    pub local: LocalCache,
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
            max_age: cfg.max_age,
        })
    }
}

/// The application router over `state`.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(handlers::demo))
        .route("/healthz", get(handlers::healthz))
        .route("/algorithms", get(handlers::algorithms_catalog))
        .route("/info", get(handlers::info))
        .route("/tilejson", get(handlers::tilejson))
        .route("/tiles/{z}/{x}/{y}", get(handlers::tile))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
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
        "surtgis serve listening"
    );
    eprintln!(
        "SurtGIS Server on http://{}  (demo at /, catalogue at /algorithms)",
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
