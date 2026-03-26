/// Compatibility layer for rayon/sequential execution.
///
/// When the `parallel` feature is enabled, this re-exports rayon's parallel iterators.
/// When disabled (e.g., for WASM builds), it provides sequential fallbacks that
/// implement the same API surface used by our algorithms.
#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

#[cfg(not(feature = "parallel"))]
mod sequential {
    /// Sequential stand-in for `rayon::prelude::IntoParallelIterator`.
    ///
    /// Calls `into_iter()` instead of `into_par_iter()`, so the rest of the
    /// iterator chain (`.flat_map()`, `.collect()`, etc.) resolves to the
    /// standard `Iterator` methods.
    pub trait IntoParallelIterator {
        type Iter;
        type Item;
        fn into_par_iter(self) -> Self::Iter;
    }

    impl<I: IntoIterator> IntoParallelIterator for I {
        type Iter = I::IntoIter;
        type Item = I::Item;
        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }

    /// Sequential stand-in for `rayon::prelude::IntoParallelRefMutIterator`.
    ///
    /// Calls `iter_mut()` instead of `par_iter_mut()`.
    pub trait IntoParallelRefMutIterator<'data> {
        type Iter: Iterator;
        type Item: Send + 'data;
        fn par_iter_mut(&'data mut self) -> Self::Iter;
    }

    impl<'data, T: Send + 'data> IntoParallelRefMutIterator<'data> for Vec<T> {
        type Iter = std::slice::IterMut<'data, T>;
        type Item = T;
        fn par_iter_mut(&'data mut self) -> Self::Iter {
            self.iter_mut()
        }
    }

    /// Sequential stand-in for rayon's ParallelSliceMut
    pub trait ParallelSliceMut<T: Send> {
        fn par_chunks_mut(&mut self, chunk_size: usize) -> std::slice::ChunksMut<T>;
        fn par_rchunks_mut(&mut self, chunk_size: usize) -> std::slice::RChunksMut<T>;
    }

    impl<T: Send> ParallelSliceMut<T> for [T] {
        fn par_chunks_mut(&mut self, chunk_size: usize) -> std::slice::ChunksMut<T> {
            self.chunks_mut(chunk_size)
        }
        fn par_rchunks_mut(&mut self, chunk_size: usize) -> std::slice::RChunksMut<T> {
            self.rchunks_mut(chunk_size)
        }
    }
}

#[cfg(not(feature = "parallel"))]
pub use sequential::*;
