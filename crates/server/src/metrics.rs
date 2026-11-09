//! Request metrics in the Prometheus text exposition format, without a
//! metrics dependency: a few atomic counters and fixed-bucket histograms.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Latency buckets in seconds (Prometheus cumulative semantics).
const BUCKETS: [f64; 10] = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0];

#[derive(Default)]
struct Histogram {
    counts: [u64; BUCKETS.len()],
    sum: f64,
    total: u64,
}

impl Histogram {
    fn observe(&mut self, secs: f64) {
        for (i, b) in BUCKETS.iter().enumerate() {
            if secs <= *b {
                self.counts[i] += 1;
            }
        }
        self.sum += secs;
        self.total += 1;
    }
}

/// Process-wide counters.
#[derive(Default)]
pub struct Metrics {
    tiles: Mutex<BTreeMap<(String, String), u64>>,
    latency: Mutex<BTreeMap<String, Histogram>>,
    /// Tiles served from the L1 cache.
    pub cache_hits: AtomicU64,
    /// Tiles rendered.
    pub cache_misses: AtomicU64,
    /// Requests rejected because the in-flight limit was reached.
    pub rejected: AtomicU64,
    /// Requests that hit the per-tile timeout.
    pub timeouts: AtomicU64,
}

impl Metrics {
    /// Count a finished tile request for `alg` with an HTTP-ish `status`
    /// label (`ok`, `cached`, `empty`, `error`, `timeout`).
    pub fn tile(&self, alg: &str, status: &str) {
        *self
            .tiles
            .lock()
            .unwrap()
            .entry((alg.to_string(), status.to_string()))
            .or_insert(0) += 1;
    }

    /// Record the render time of a tile for `alg`.
    pub fn observe(&self, alg: &str, secs: f64) {
        self.latency
            .lock()
            .unwrap()
            .entry(alg.to_string())
            .or_default()
            .observe(secs);
    }

    /// Render everything in the Prometheus text format; `extra` are
    /// gauges from other components (name, help, value).
    pub fn render(&self, extra: &[(&str, &str, f64)]) -> String {
        let mut out = String::new();
        out.push_str("# HELP surtgis_tiles_total Tile requests by operator and outcome.\n# TYPE surtgis_tiles_total counter\n");
        for ((alg, status), n) in self.tiles.lock().unwrap().iter() {
            out.push_str(&format!(
                "surtgis_tiles_total{{alg=\"{alg}\",status=\"{status}\"}} {n}\n"
            ));
        }
        out.push_str("# HELP surtgis_tile_seconds Tile render time by operator.\n# TYPE surtgis_tile_seconds histogram\n");
        for (alg, h) in self.latency.lock().unwrap().iter() {
            for (i, b) in BUCKETS.iter().enumerate() {
                out.push_str(&format!(
                    "surtgis_tile_seconds_bucket{{alg=\"{alg}\",le=\"{b}\"}} {}\n",
                    h.counts[i]
                ));
            }
            out.push_str(&format!(
                "surtgis_tile_seconds_bucket{{alg=\"{alg}\",le=\"+Inf\"}} {}\nsurtgis_tile_seconds_sum{{alg=\"{alg}\"}} {}\nsurtgis_tile_seconds_count{{alg=\"{alg}\"}} {}\n",
                h.total, h.sum, h.total
            ));
        }
        for (name, help, value) in [
            (
                "surtgis_tile_cache_hits_total",
                "Tiles served from the L1 cache.",
                self.cache_hits.load(Ordering::Relaxed) as f64,
            ),
            (
                "surtgis_tile_cache_misses_total",
                "Tiles rendered.",
                self.cache_misses.load(Ordering::Relaxed) as f64,
            ),
            (
                "surtgis_rejected_total",
                "Requests rejected by the in-flight limit.",
                self.rejected.load(Ordering::Relaxed) as f64,
            ),
            (
                "surtgis_timeouts_total",
                "Requests that hit the per-tile timeout.",
                self.timeouts.load(Ordering::Relaxed) as f64,
            ),
        ] {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}\n"
            ));
        }
        for (name, help, value) in extra {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}\n"
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposition_has_counters_and_cumulative_buckets() {
        let m = Metrics::default();
        m.tile("hillshade", "ok");
        m.tile("hillshade", "ok");
        m.observe("hillshade", 0.02);
        m.observe("hillshade", 3.0);
        m.cache_hits.fetch_add(1, Ordering::Relaxed);
        let text = m.render(&[("surtgis_pool_readers", "Open readers.", 2.0)]);
        assert!(text.contains("surtgis_tiles_total{alg=\"hillshade\",status=\"ok\"} 2"));
        assert!(text.contains("surtgis_tile_seconds_bucket{alg=\"hillshade\",le=\"0.025\"} 1"));
        assert!(text.contains("surtgis_tile_seconds_bucket{alg=\"hillshade\",le=\"5\"} 2"));
        assert!(text.contains("surtgis_tile_seconds_count{alg=\"hillshade\"} 2"));
        assert!(text.contains("surtgis_tile_cache_hits_total 1"));
        assert!(text.contains("surtgis_pool_readers 2"));
    }
}
