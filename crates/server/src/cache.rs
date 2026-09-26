//! Caches: a byte-bounded LRU of rendered tiles (L1) and an on-disk tile
//! store (L2) laid out as `{layer}/{z}/{x}/{y}.png`, which doubles as a
//! seed for static pyramids.

use std::hash::Hash;
use std::sync::Mutex;

use lru::LruCache;

/// LRU bounded by the summed size of its values, not their count.
pub struct ByteLru<K: Hash + Eq, V> {
    inner: Mutex<Inner<K, V>>,
    budget: usize,
    size_of: fn(&V) -> usize,
}

struct Inner<K: Hash + Eq, V> {
    map: LruCache<K, V>,
    bytes: usize,
    hits: u64,
    misses: u64,
}

impl<K: Hash + Eq + Clone, V: Clone> ByteLru<K, V> {
    /// Cache holding at most `budget` bytes as measured by `size_of`.
    /// A zero budget disables caching (every `get` misses).
    pub fn new(budget: usize, size_of: fn(&V) -> usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                map: LruCache::unbounded(),
                bytes: 0,
                hits: 0,
                misses: 0,
            }),
            budget,
            size_of,
        }
    }

    /// Look up `key`, promoting it to most recently used.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut g = self.inner.lock().unwrap();
        match g.map.get(key).cloned() {
            Some(v) => {
                g.hits += 1;
                Some(v)
            }
            None => {
                g.misses += 1;
                None
            }
        }
    }

    /// Insert `value`, evicting least recently used entries until the
    /// budget holds. A value larger than the whole budget is not kept.
    pub fn put(&self, key: K, value: V) {
        let size = (self.size_of)(&value);
        if self.budget == 0 || size > self.budget {
            return;
        }
        let mut g = self.inner.lock().unwrap();
        if let Some(old) = g.map.push(key, value) {
            g.bytes -= (self.size_of)(&old.1);
        }
        g.bytes += size;
        while g.bytes > self.budget {
            match g.map.pop_lru() {
                Some((_, v)) => g.bytes -= (self.size_of)(&v),
                None => break,
            }
        }
    }

    /// `(entries, bytes, hits, misses)`.
    pub fn stats(&self) -> (usize, usize, u64, u64) {
        let g = self.inner.lock().unwrap();
        (g.map.len(), g.bytes, g.hits, g.misses)
    }

    /// Byte budget.
    pub fn budget(&self) -> usize {
        self.budget
    }
}

/// On-disk tile cache, `{root}/{layer}/{z}/{x}/{y}.png`.
///
/// `layer` is a hash of everything in the request except the tile
/// address (source, operator, params, colormap, rescale), so one folder
/// per rendered layer holds a plain XYZ pyramid that any static tile
/// host can serve. Writes are atomic (temp file + rename), reads are
/// plain file reads; there is no size bound — prune the folder from
/// outside if disk matters.
pub struct DiskCache {
    root: std::path::PathBuf,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
    writes: std::sync::atomic::AtomicU64,
    errors: std::sync::atomic::AtomicU64,
}

impl DiskCache {
    /// Cache rooted at `root` (created if missing).
    pub fn new(root: std::path::PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            hits: Default::default(),
            misses: Default::default(),
            writes: Default::default(),
            errors: Default::default(),
        })
    }

    /// Root directory.
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    /// Path of a tile.
    pub fn path_for(&self, layer: &str, z: u8, x: u32, y: u32) -> std::path::PathBuf {
        self.root
            .join(layer)
            .join(z.to_string())
            .join(x.to_string())
            .join(format!("{y}.png"))
    }

    /// Read a cached tile, if any.
    pub async fn get(&self, layer: &str, z: u8, x: u32, y: u32) -> Option<Vec<u8>> {
        use std::sync::atomic::Ordering::Relaxed;
        match tokio::fs::read(self.path_for(layer, z, x, y)).await {
            Ok(bytes) if !bytes.is_empty() => {
                self.hits.fetch_add(1, Relaxed);
                Some(bytes)
            }
            _ => {
                self.misses.fetch_add(1, Relaxed);
                None
            }
        }
    }

    /// Store a tile atomically (write to a temp file, then rename).
    pub async fn put(&self, layer: &str, z: u8, x: u32, y: u32, bytes: &[u8]) {
        use std::sync::atomic::Ordering::Relaxed;
        let path = self.path_for(layer, z, x, y);
        let result: std::io::Result<()> = async {
            if let Some(dir) = path.parent() {
                tokio::fs::create_dir_all(dir).await?;
            }
            let tmp = path.with_extension(format!("tmp{}", std::process::id()));
            tokio::fs::write(&tmp, bytes).await?;
            tokio::fs::rename(&tmp, &path).await
        }
        .await;
        match result {
            Ok(()) => {
                self.writes.fetch_add(1, Relaxed);
            }
            Err(e) => {
                self.errors.fetch_add(1, Relaxed);
                tracing::warn!(path = %path.display(), error = %e, "disk cache write failed");
            }
        }
    }

    /// `(hits, misses, writes, errors)`.
    pub fn stats(&self) -> (u64, u64, u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (
            self.hits.load(Relaxed),
            self.misses.load(Relaxed),
            self.writes.load(Relaxed),
            self.errors.load(Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_size(v: &Vec<u8>) -> usize {
        v.len()
    }

    #[test]
    fn evicts_least_recently_used_by_bytes() {
        let c: ByteLru<&str, Vec<u8>> = ByteLru::new(10, vec_size);
        c.put("a", vec![0; 4]);
        c.put("b", vec![0; 4]);
        assert!(c.get(&"a").is_some()); // a is now the most recent
        c.put("c", vec![0; 4]); // 12 > 10 → evict b (least recent)
        assert!(c.get(&"b").is_none());
        assert!(c.get(&"a").is_some() && c.get(&"c").is_some());
        let (n, bytes, hits, misses) = c.stats();
        assert_eq!((n, bytes), (2, 8));
        assert_eq!((hits, misses), (3, 1));
    }

    #[test]
    fn oversized_values_and_zero_budget_are_not_kept() {
        let c: ByteLru<&str, Vec<u8>> = ByteLru::new(3, vec_size);
        c.put("big", vec![0; 4]);
        assert!(c.get(&"big").is_none());
        let z: ByteLru<&str, Vec<u8>> = ByteLru::new(0, vec_size);
        z.put("x", vec![]);
        assert!(z.get(&"x").is_none());
    }

    #[test]
    fn replacing_a_key_accounts_bytes_once() {
        let c: ByteLru<&str, Vec<u8>> = ByteLru::new(10, vec_size);
        c.put("a", vec![0; 4]);
        c.put("a", vec![0; 6]);
        assert_eq!(c.stats().1, 6);
    }

    #[tokio::test]
    async fn disk_cache_roundtrip_layout_and_atomicity() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::new(dir.path().join("tiles")).unwrap();
        assert!(cache.get("abc", 3, 4, 5).await.is_none());
        cache.put("abc", 3, 4, 5, b"png-bytes").await;
        assert_eq!(
            cache.get("abc", 3, 4, 5).await.as_deref(),
            Some(&b"png-bytes"[..])
        );
        assert!(dir.path().join("tiles/abc/3/4/5.png").is_file());
        // No temp file left behind; other layers untouched.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path().join("tiles/abc/3/4"))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(leftovers, vec!["5.png".to_string()]);
        assert!(cache.get("other", 3, 4, 5).await.is_none());
        assert_eq!(cache.stats(), (1, 2, 1, 0));
    }
}
