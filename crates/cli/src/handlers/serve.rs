//! `surtgis serve` — start the dynamic tile server (feature `server`).

use std::path::PathBuf;

use anyhow::{Context, Result};

/// Command-line arguments of `surtgis serve`.
pub struct Args {
    /// Address to listen on (`host:port`).
    pub bind: String,
    /// Directory local sources must live under.
    pub root: Option<PathBuf>,
    /// URL prefixes remote sources may start with.
    pub allow: Vec<String>,
    /// `Cache-Control: max-age` for tiles.
    pub max_age: u32,
    /// L1 tile cache, MiB.
    pub cache_mb: usize,
    /// L2 tile cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Per-tile deadline, ms.
    pub timeout_ms: u64,
    /// Concurrent renders before 503.
    pub max_inflight: usize,
    /// Open COG readers per URL.
    pub pool_per_url: usize,
}

/// Run the server until Ctrl-C.
pub fn handle(args: Args) -> Result<()> {
    let bind = args
        .bind
        .parse()
        .with_context(|| format!("--bind '{}' is not a socket address (host:port)", args.bind))?;
    if args.root.is_none() && args.allow.is_empty() {
        eprintln!(
            "warning: neither --root nor --allow given: every ?url= will be refused. \
             Pass --root <dir> for local GeoTIFFs and/or --allow <url-prefix> for remote COGs."
        );
    }
    surtgis_server::serve_blocking(surtgis_server::ServerConfig {
        bind,
        allow: args.allow,
        root: args.root,
        max_age: args.max_age,
        cache_mb: args.cache_mb,
        cache_dir: args.cache_dir,
        timeout_ms: args.timeout_ms,
        max_inflight: args.max_inflight,
        pool_per_url: args.pool_per_url,
    })
}
