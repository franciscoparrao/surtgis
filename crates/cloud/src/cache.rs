//! LRU tile cache for COG tiles.

use lru::LruCache;
use std::num::NonZeroUsize;

/// Key for cached tiles: (IFD index, tile index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileKey {
    pub ifd_idx: usize,
    pub tile_idx: usize,
}

/// LRU cache storing decompressed tile data.
pub struct TileCache {
    inner: LruCache<TileKey, Vec<u8>>,
}

impl TileCache {
    /// Create a new cache with the given capacity (number of tiles).
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            inner: LruCache::new(cap),
        }
    }

    /// Get a cached tile's raw (decompressed) bytes, if present.
    pub fn get(&mut self, key: &TileKey) -> Option<&Vec<u8>> {
        self.inner.get(key)
    }

    /// Insert a tile into the cache.
    pub fn insert(&mut self, key: TileKey, data: Vec<u8>) {
        self.inner.put(key, data);
    }

    /// Number of tiles currently cached.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear all cached tiles.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_get() {
        let mut cache = TileCache::new(2);
        let key = TileKey {
            ifd_idx: 0,
            tile_idx: 5,
        };
        cache.insert(key, vec![1, 2, 3]);
        assert_eq!(cache.get(&key), Some(&vec![1, 2, 3]));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = TileCache::new(2);
        let k1 = TileKey { ifd_idx: 0, tile_idx: 0 };
        let k2 = TileKey { ifd_idx: 0, tile_idx: 1 };
        let k3 = TileKey { ifd_idx: 0, tile_idx: 2 };

        cache.insert(k1, vec![1]);
        cache.insert(k2, vec![2]);
        cache.insert(k3, vec![3]); // evicts k1

        assert!(cache.get(&k1).is_none());
        assert!(cache.get(&k2).is_some());
        assert!(cache.get(&k3).is_some());
    }
}
