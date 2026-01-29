//! Parallel processing strategies

use rayon::prelude::*;

/// Processing mode for algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingMode {
    /// Single-threaded processing
    Sequential,
    /// Parallel processing using all available cores
    Parallel,
    /// Parallel with specified number of threads
    ParallelWith(usize),
}

impl Default for ProcessingMode {
    fn default() -> Self {
        ProcessingMode::Parallel
    }
}

/// Strategy for parallel execution
pub trait ParallelStrategy {
    /// Execute a function over indices in parallel
    fn par_for_each<F>(&self, range: std::ops::Range<usize>, f: F)
    where
        F: Fn(usize) + Sync + Send;

    /// Map a function over indices and collect results
    fn par_map<T, F>(&self, range: std::ops::Range<usize>, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync + Send;
}

impl ParallelStrategy for ProcessingMode {
    fn par_for_each<F>(&self, range: std::ops::Range<usize>, f: F)
    where
        F: Fn(usize) + Sync + Send,
    {
        match self {
            ProcessingMode::Sequential => {
                for i in range {
                    f(i);
                }
            }
            ProcessingMode::Parallel => {
                range.into_par_iter().for_each(f);
            }
            ProcessingMode::ParallelWith(threads) => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(*threads)
                    .build()
                    .expect("Failed to build thread pool");
                pool.install(|| {
                    range.into_par_iter().for_each(f);
                });
            }
        }
    }

    fn par_map<T, F>(&self, range: std::ops::Range<usize>, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync + Send,
    {
        match self {
            ProcessingMode::Sequential => range.map(f).collect(),
            ProcessingMode::Parallel => range.into_par_iter().map(f).collect(),
            ProcessingMode::ParallelWith(threads) => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(*threads)
                    .build()
                    .expect("Failed to build thread pool");
                pool.install(|| range.into_par_iter().map(f).collect())
            }
        }
    }
}

/// Get the number of available CPU cores
pub fn num_cpus() -> usize {
    rayon::current_num_threads()
}

/// Configure the global thread pool
pub fn set_num_threads(threads: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok(); // Ignore if already initialized
}
