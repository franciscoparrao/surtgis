//! Building blocks of the STAC multiband composite (**experimental**,
//! feature `unstable`).
//!
//! Extracted from the CLI composite pipeline (audit R2 recommendation R8,
//! step 1 of 3): the pure planning, per-tile and per-pixel logic lives here
//! with its own test suite, while orchestration (STAC search, downloads,
//! progress reporting, GeoTIFF output) remains in the caller. Later R8
//! steps add a `CompositeEngine`/`StripSink` orchestrator and a
//! serializable `CompositePlan` (checkpoint/resume manifest) on top of
//! these primitives.
//!
//! Being `unstable`-gated, this module is **exempt from the 1.x stability
//! guarantee** — its API may change in minor releases.

mod plan;
mod reduce;
mod tiles;

#[cfg(feature = "native")]
mod engine;
#[cfg(feature = "native")]
mod resolver;
#[cfg(feature = "native")]
mod spec;

pub use plan::{
    ALLOC_OVERHEAD_FRAC, BudgetBreakdown, BudgetProfile, MASK_INFLATION_CALIB, SearchEstimate,
    StripBounds, StripPlan, StripPlanInput, budget_profile, estimate_search_limit, plan_strips,
    select_dates_by_coverage, strip_bounds,
};
pub use reduce::{
    GAP_FILL_PASSES, composite_scene_strips, coverage_order, fill_from_scenes_by_coverage,
    fill_gaps_neighbor_mean, median_composite,
};
pub use tiles::{
    TileOutcome, classify_benign_tile_error, cog_cache_key, mosaic_tile_rasters,
    overview_for_target_resolution, reproject_bbox_between_crs, retry_jitter_ms, unify_tile_crs,
};

#[cfg(feature = "native")]
pub use engine::{
    AssetResolver, CompositeEngine, CompositeProgress, CompositeReport, MaskApplier, NoProgress,
    StripSink,
};
#[cfg(feature = "native")]
pub use resolver::DefaultAssetResolver;
#[cfg(feature = "native")]
pub use spec::{CompositeSpec, OutputGrid};
