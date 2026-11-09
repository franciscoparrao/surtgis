//! Pool of open COG readers, keyed by URL.
//!
//! Opening a COG costs a few HTTP range requests for the header and IFD
//! chain; a reader also keeps a tile cache and an HTTP connection alive.
//! Reads need `&mut self`, so each reader sits behind an async mutex and
//! up to `per_url` readers may be open for the same URL to serve
//! concurrent tile requests without serialising them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use surtgis_cloud::{CogReader, CogReaderOptions};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

use crate::error::ServeError;

struct Entry {
    readers: Vec<Arc<AsyncMutex<CogReader>>>,
    last_used: Instant,
}

/// Readers per URL with a soft cap per URL and an idle TTL.
pub struct ReaderPool {
    entries: Mutex<HashMap<String, Entry>>,
    per_url: usize,
    ttl: Duration,
}

impl ReaderPool {
    /// Pool allowing `per_url` readers per URL, dropping URLs idle for `ttl`.
    pub fn new(per_url: usize, ttl: Duration) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            per_url: per_url.max(1),
            ttl,
        }
    }

    /// Lock a reader for `url`: a free one if any, a new one while under
    /// the per-URL cap, else the first existing one (waiting for it).
    pub async fn acquire(&self, url: &str) -> Result<OwnedMutexGuard<CogReader>, ServeError> {
        self.evict_idle();
        let (free, first, can_open) = {
            let mut g = self.entries.lock().unwrap();
            let entry = g.entry(url.to_string()).or_insert_with(|| Entry {
                readers: Vec::new(),
                last_used: Instant::now(),
            });
            entry.last_used = Instant::now();
            let free = entry
                .readers
                .iter()
                .find_map(|r| r.clone().try_lock_owned().ok());
            (
                free,
                entry.readers.first().cloned(),
                entry.readers.len() < self.per_url,
            )
        };
        if let Some(guard) = free {
            return Ok(guard);
        }
        if can_open {
            let reader = CogReader::open(url, CogReaderOptions::default())
                .await
                .map_err(|e| ServeError::Source(format!("failed to open {url}: {e}")))?;
            let arc = Arc::new(AsyncMutex::new(reader));
            let guard = arc.clone().try_lock_owned().expect("fresh mutex");
            let mut g = self.entries.lock().unwrap();
            if let Some(entry) = g.get_mut(url).filter(|e| e.readers.len() < self.per_url) {
                entry.readers.push(arc);
            }
            return Ok(guard);
        }
        match first {
            Some(r) => Ok(r.lock_owned().await),
            None => Err(ServeError::Source("reader pool exhausted".into())),
        }
    }

    fn evict_idle(&self) {
        let mut g = self.entries.lock().unwrap();
        let now = Instant::now();
        g.retain(|_, e| now.duration_since(e.last_used) < self.ttl);
    }

    /// `(urls, readers)` currently held.
    pub fn stats(&self) -> (usize, usize) {
        let g = self.entries.lock().unwrap();
        (g.len(), g.values().map(|e| e.readers.len()).sum())
    }
}
