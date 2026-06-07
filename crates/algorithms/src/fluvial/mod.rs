//! Fluvial-tectonic morphometry.
//!
//! Network-level metrics that complement the cell-level hydrology already
//! shipped in [`crate::hydrology`]. The five v1 deliverables (channel
//! steepness `ksn`, chi `χ`, knickpoint detection, concavity `θ`, divide
//! migration) target the geomorphology-from-topography literature initiated
//! by Wobus et al. 2006 and Perron & Royden 2013 — the lingua franca for
//! detecting tectonic signals in river networks.
//!
//! ## Why a separate submodule
//!
//! `hydrology/` works at the **cell** level (flow direction, accumulation,
//! filling). Fluvial morphometry works at the **network** level (upstream
//! traversal along channels, segment-wise smoothing, basin-wise regression).
//! The primitives differ enough that mixing them would clutter both.
//!
//! ## Sprint 1 status
//!
//! Only [`stream_traversal::StreamGraph`] is implemented. The five public
//! algorithms (chi, ksn, knickpoint, concavity, divide_migration) are
//! introduced in subsequent sprints (see `docs/SPEC_morfometria_fluvial_tectonica.md`
//! §9).

pub mod channel_steepness;
pub mod chi;
pub mod concavity;
pub mod divide_migration;
pub mod export;
pub mod knickpoint;
pub mod long_profile;
pub mod stream_traversal;
pub mod swath_profile;

pub use channel_steepness::{KsnError, KsnParams, KsnResult, KsnSegment, channel_steepness};
pub use chi::{ChiError, ChiParams, chi_transform};
pub use concavity::{ConcavityError, ConcavityParams, ConcavityResult, concavity_index};
pub use divide_migration::{
    DivideMigrationError, DivideMigrationParams, DivideSegment, divide_migration,
};
pub use export::{
    ExportError, write_chi_csv, write_ksn_segments_csv, write_ksn_segments_json,
    write_long_profiles_csv, write_long_profiles_json, write_swath_csv, write_swath_json,
};
pub use knickpoint::{
    Knickpoint, KnickpointError, KnickpointParams, KnickpointPolarity, knickpoint_detection,
};
pub use long_profile::{
    LongProfile, LongProfileError, LongProfileNode, LongProfileParams, long_profile,
};
pub use stream_traversal::{StreamGraph, StreamGraphError, build_stream_graph};
pub use swath_profile::{SwathError, SwathParams, SwathProfile, SwathStats, swath_profile};
