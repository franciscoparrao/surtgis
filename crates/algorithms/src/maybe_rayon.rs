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
}

#[cfg(not(feature = "parallel"))]
pub use sequential::*;
