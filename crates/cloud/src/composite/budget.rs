//! A byte budget that bounds concurrent allocations *by construction*.
//!
//! The composite's peak RAM used to be governed by a calibrated model
//! (`plan_strips` with empirical inflation constants from the v0.6.22→v0.7.1
//! RAM saga): estimate how big each working buffer will be, solve for a strip
//! height that fits a budget, and hope the estimate holds. When the estimate
//! is wrong the process either overshoots (OOM) or undershoots (needless
//! extra strips).
//!
//! [`MemoryBudget`] inverts that: a caller *acquires* a permit for the bytes
//! it is about to allocate, and the acquisition waits until that much budget
//! is free. Total outstanding permits can never exceed the budget, so peak
//! RAM is bounded no matter how wrong any size estimate is — and concurrency
//! self-throttles (a burst of downloads simply waits for earlier buffers to
//! drop instead of all allocating at once). [`TrackedRaster`] ties a permit
//! to a raster's lifetime, so the budget is returned exactly when the raster
//! is freed.
//!
//! Permits are counted in whole MiB (tokio's semaphore takes `u32` permits;
//! byte granularity would overflow and waste no memory to track). A request
//! is rounded up to the next MiB and clamped to the whole budget, so a single
//! over-large allocation serializes against the full budget rather than
//! deadlocking.

use std::ops::Deref;
use std::sync::Arc;

use surtgis_core::Raster;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// One mebibyte.
const MIB: usize = 1024 * 1024;

/// A byte budget backed by a counting semaphore (permits in MiB units).
///
/// Cheap to [`clone`](Self::clone): all clones share one budget.
#[derive(Clone)]
pub struct MemoryBudget {
    sem: Arc<Semaphore>,
    mib_total: usize,
}

impl MemoryBudget {
    /// Create a budget of `total_bytes`, rounded down to whole MiB (minimum
    /// 1 MiB, capped at the semaphore's maximum).
    pub fn new(total_bytes: usize) -> Self {
        let mib_total = (total_bytes / MIB).clamp(1, Semaphore::MAX_PERMITS);
        Self {
            sem: Arc::new(Semaphore::new(mib_total)),
            mib_total,
        }
    }

    /// Total budget in bytes (whole MiB).
    pub fn total_bytes(&self) -> usize {
        self.mib_total * MIB
    }

    /// Bytes currently available (not held by any live permit).
    pub fn available_bytes(&self) -> usize {
        self.sem.available_permits() * MIB
    }

    /// Round a byte count up to the MiB permits it needs, clamped to the
    /// whole budget.
    fn mib_for(&self, bytes: usize) -> usize {
        bytes.div_ceil(MIB).clamp(1, self.mib_total)
    }

    /// Acquire budget for `bytes`, waiting until enough is free. The returned
    /// [`BudgetPermit`] returns the budget when dropped.
    ///
    /// The request is clamped to the whole budget, so an allocation larger
    /// than the budget takes all of it (serializing against every other
    /// holder) instead of waiting forever.
    pub async fn acquire(&self, bytes: usize) -> BudgetPermit {
        let mib = self.mib_for(bytes);
        let permit = self
            .sem
            .clone()
            .acquire_many_owned(mib as u32)
            .await
            .expect("MemoryBudget semaphore is never closed");
        BudgetPermit { _permit: permit }
    }

    /// Try to acquire budget for `bytes` without waiting. Returns `None` if
    /// not enough is currently free.
    pub fn try_acquire(&self, bytes: usize) -> Option<BudgetPermit> {
        let mib = self.mib_for(bytes);
        self.sem
            .clone()
            .try_acquire_many_owned(mib as u32)
            .ok()
            .map(|permit| BudgetPermit { _permit: permit })
    }

    /// Acquire budget and attach it to `raster`, so the budget is held for
    /// exactly as long as the raster is alive.
    pub async fn track(&self, raster: Raster<f64>) -> TrackedRaster {
        let permit = self.acquire(raster_bytes(&raster)).await;
        TrackedRaster { raster, permit }
    }
}

/// Held budget; returns its bytes to the [`MemoryBudget`] when dropped.
pub struct BudgetPermit {
    _permit: OwnedSemaphorePermit,
}

/// A [`Raster<f64>`] whose bytes are charged to a [`MemoryBudget`] for its
/// lifetime. Dereferences to the raster; dropping it (or [`into_inner`]) frees
/// the budget.
///
/// [`into_inner`]: TrackedRaster::into_inner
pub struct TrackedRaster {
    raster: Raster<f64>,
    permit: BudgetPermit,
}

impl TrackedRaster {
    /// Wrap a raster with an already-acquired permit.
    pub fn new(raster: Raster<f64>, permit: BudgetPermit) -> Self {
        Self { raster, permit }
    }

    /// Borrow the underlying raster.
    pub fn raster(&self) -> &Raster<f64> {
        &self.raster
    }

    /// Take the raster out, releasing the budget permit. The bytes are no
    /// longer tracked after this — use when handing ownership to code that
    /// will free them promptly and shouldn't hold the budget meanwhile.
    pub fn into_inner(self) -> Raster<f64> {
        drop(self.permit);
        self.raster
    }
}

impl Deref for TrackedRaster {
    type Target = Raster<f64>;
    fn deref(&self) -> &Raster<f64> {
        &self.raster
    }
}

/// Decoded size of an `f64` raster in bytes (`rows × cols × 8`).
pub fn raster_bytes(raster: &Raster<f64>) -> usize {
    let (rows, cols) = raster.shape();
    rows * cols * std::mem::size_of::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rounds_down_and_floors_at_one_mib() {
        assert_eq!(MemoryBudget::new(0).total_bytes(), MIB);
        assert_eq!(MemoryBudget::new(MIB / 2).total_bytes(), MIB);
        assert_eq!(MemoryBudget::new(4 * MIB + 7).total_bytes(), 4 * MIB);
    }

    #[tokio::test]
    async fn acquire_and_drop_round_trips_the_budget() {
        let b = MemoryBudget::new(8 * MIB);
        assert_eq!(b.available_bytes(), 8 * MIB);
        let p = b.acquire(3 * MIB).await;
        assert_eq!(b.available_bytes(), 5 * MIB);
        let p2 = b.acquire(5 * MIB).await;
        assert_eq!(b.available_bytes(), 0);
        drop(p);
        assert_eq!(b.available_bytes(), 3 * MIB);
        drop(p2);
        assert_eq!(b.available_bytes(), 8 * MIB);
    }

    #[test]
    fn try_acquire_fails_when_insufficient() {
        let b = MemoryBudget::new(4 * MIB);
        let p = b.try_acquire(3 * MIB).expect("fits");
        assert_eq!(b.available_bytes(), MIB);
        assert!(b.try_acquire(2 * MIB).is_none(), "only 1 MiB left");
        drop(p);
        assert!(b.try_acquire(2 * MIB).is_some());
    }

    #[tokio::test]
    async fn acquire_waits_until_budget_frees() {
        let b = MemoryBudget::new(4 * MIB);
        let hold = b.acquire(4 * MIB).await; // whole budget
        assert_eq!(b.available_bytes(), 0);

        let b2 = b.clone();
        let waiter = tokio::spawn(async move {
            // Blocks until `hold` drops.
            let _p = b2.acquire(2 * MIB).await;
            true
        });

        // Give the waiter a chance to park; it must not have completed.
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());

        drop(hold);
        assert!(waiter.await.unwrap());
    }

    #[tokio::test]
    async fn oversized_request_clamps_to_whole_budget() {
        let b = MemoryBudget::new(4 * MIB);
        // Ask for more than the whole budget: clamps to 4 MiB, still succeeds.
        let p = b.acquire(100 * MIB).await;
        assert_eq!(b.available_bytes(), 0);
        drop(p);
        assert_eq!(b.available_bytes(), 4 * MIB);
    }

    #[tokio::test]
    async fn tracked_raster_frees_budget_on_drop_and_into_inner() {
        let b = MemoryBudget::new(16 * MIB);
        let r = Raster::<f64>::filled(512, 512, 0.0); // 512*512*8 = 2 MiB
        assert_eq!(raster_bytes(&r), 2 * MIB);

        let tracked = b.track(r).await;
        assert_eq!(b.available_bytes(), 14 * MIB);
        assert_eq!(tracked.shape(), (512, 512)); // Deref to Raster

        let raster = tracked.into_inner();
        assert_eq!(
            b.available_bytes(),
            16 * MIB,
            "into_inner releases the permit"
        );
        drop(raster);

        let tracked2 = b.track(Raster::<f64>::filled(512, 512, 1.0)).await;
        assert_eq!(b.available_bytes(), 14 * MIB);
        drop(tracked2);
        assert_eq!(b.available_bytes(), 16 * MIB, "drop releases the permit");
    }
}
