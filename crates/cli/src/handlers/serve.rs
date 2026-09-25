//! `surtgis serve` — start the dynamic tile server (feature `server`).

use std::path::PathBuf;

use anyhow::{Context, Result};

/// Run the server until Ctrl-C.
pub fn handle(bind: String, root: Option<PathBuf>, allow: Vec<String>, max_age: u32) -> Result<()> {
    let bind = bind
        .parse()
        .with_context(|| format!("--bind '{bind}' is not a socket address (host:port)"))?;
    if root.is_none() && allow.is_empty() {
        eprintln!(
            "warning: neither --root nor --allow given: every ?url= will be refused. \
             Pass --root <dir> for local GeoTIFFs and/or --allow <url-prefix> for remote COGs."
        );
    }
    surtgis_server::serve_blocking(surtgis_server::ServerConfig {
        bind,
        allow,
        root,
        max_age,
    })
}
