//! `tract`-backed [`TileModel`] — pure-Rust ONNX runtime (no C++, no
//! CUDA, no `onnxruntime`), preserving SurtGIS's single-binary
//! property. See `SPEC_SURTGIS_ONNX_INFERENCE.md` at the workspace
//! root for the full design contract.
//!
//! Feature-gated (`onnx`, off by default): the default build never
//! links `tract-onnx`.
//!
//! **Scaffold status:** [`OnnxModel::load`] and [`OnnxModel::infer`]
//! are stubs. The orchestration this backend plugs into
//! ([`super::run_tiled`] — tiling, halo extraction, seam-free
//! stitching) is fully implemented and tested without any ML runtime;
//! wiring up a real `tract` graph here is deferred work.

use super::{TileInput, TileModel, TileOutput};
use std::path::Path;
use surtgis_core::{Error, Result};

/// A [`TileModel`] backed by a loaded ONNX graph.
///
/// `required_halo` is declared by the caller rather than inferred from
/// the graph (SPEC §7, open question: reliable receptive-field
/// introspection from an arbitrary ONNX graph isn't solved here) — get
/// it wrong and [`super::run_tiled`] will still catch a shape mismatch
/// via [`Error::Algorithm`], just late and per-tile rather than at load
/// time.
///
/// `#[allow(dead_code)]`: every field is read by the [`TileModel`] impl
/// below, but [`OnnxModel::load`] never actually constructs a `Self` —
/// it always returns `Err` until the real `tract` graph loading lands.
/// Rust's dead-code analysis is syntactic (does a `Self { .. }`
/// constructor exist anywhere in the crate?), not reachability-aware,
/// so it can't see that this is deliberate; the explicit `allow`
/// avoids a lint noise regression once a real constructor is added.
#[allow(dead_code)]
#[derive(Debug)]
pub struct OnnxModel {
    in_bands: usize,
    out_bands: usize,
    required_halo: usize,
}

impl OnnxModel {
    /// Load a `.onnx` graph from `path` and validate it against
    /// `in_bands`/`out_bands`/`required_halo`.
    ///
    /// **Not yet implemented.** This is the scaffold's one deliberate
    /// stub (SPEC_SURTGIS_ONNX_INFERENCE.md §5: no real graph is
    /// trained or executed by this scaffold). Returns a controlled
    /// error rather than panicking, so a CLI/library caller gets a
    /// clear message instead of a crash — see the SPEC before wiring
    /// up `tract_onnx::onnx().model_for_path(path)...into_optimized()`.
    pub fn load(
        _path: &Path,
        _in_bands: usize,
        _out_bands: usize,
        _required_halo: usize,
    ) -> Result<Self> {
        Err(Error::Other(
            "OnnxModel::load is not yet implemented (scaffold only). \
             Tiling/halo/stitching orchestration is implemented and \
             tested in `surtgis_algorithms::inference::run_tiled`; \
             loading and running a real tract-onnx graph is specified \
             in SPEC_SURTGIS_ONNX_INFERENCE.md but not wired up yet."
                .to_string(),
        ))
    }
}

impl TileModel for OnnxModel {
    fn in_bands(&self) -> usize {
        self.in_bands
    }

    fn out_bands(&self) -> usize {
        self.out_bands
    }

    fn required_halo(&self) -> usize {
        self.required_halo
    }

    fn infer(&self, _tile: &TileInput) -> Result<TileOutput> {
        // Unreachable today: `OnnxModel::load` always errors before a
        // `Self` exists to call this on. Left as `todo!()` (rather than
        // an `Err`, like `load`) because this is the exact call site
        // SPEC_SURTGIS_ONNX_INFERENCE.md §2.1 describes implementing —
        // running the loaded tract graph on `tile.bands` (already
        // halo-padded per `required_halo`) and returning the
        // core-sized output.
        todo!(
            "run the loaded tract graph on `tile.bands` and return a \
             TileOutput sized [out_bands, core_rows, core_cols] — see \
             SPEC_SURTGIS_ONNX_INFERENCE.md §2.1-2.2"
        )
    }
}
