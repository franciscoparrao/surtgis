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
pub mod stream_traversal;

pub use channel_steepness::{channel_steepness, KsnError, KsnParams, KsnResult, KsnSegment};
pub use chi::{chi_transform, ChiError, ChiParams};
pub use stream_traversal::{build_stream_graph, StreamGraph, StreamGraphError};
