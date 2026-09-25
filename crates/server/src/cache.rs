//! Byte-bounded LRU caches: rendered tiles (L1) and in-memory local sources.

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
}
