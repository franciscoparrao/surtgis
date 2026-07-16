//! Streaming `CubeSource` over a dated STAC scene series (**stub**).
//!
//! Design: `SPEC_SURTGIS_TEMPORAL_STREAMING.md` §2.3. [`StacClient::search_all`]
//! already resolves a bbox+collection query to a `Vec<StacItem>` with
//! `datetime` preserved per scene — the eye axis this source needs. What's
//! missing is the orchestration: resolving that scene list once, then
//! decoding only the row range a given [`CubeSource::chunk`] call asks for,
//! within a `MemoryBudget`-style RAM cap, applying the per-scene cloud mask
//! as `NaN` (matching the convention `surtgis_algorithms::temporal` already
//! uses everywhere else). Not implemented yet — this stub exists so
//! `surtgis_algorithms::temporal::reduce_temporal` has a second, real
//! `CubeSource` type to type-check against besides the in-memory `Cube`.
//!
//! [`StacClient::search_all`]: crate::stac_client::StacClient::search_all

use surtgis_core::{CubeChunk, CubeSource, GeoTransform, Result};

/// Streaming [`CubeSource`] over a STAC collection (bbox + date range + one
/// band/index per scene).
///
/// **Stub**: [`StacCubeSource::new`] and [`CubeSource::chunk`] are
/// unimplemented (`todo!()`). See the module docs for the design.
pub struct StacCubeSource {
    /// STAC collection ID (e.g. `"sentinel-2-l2a"`).
    pub collection: String,
    /// Target bbox `[minx, miny, maxx, maxy]`, in the target CRS.
    pub bbox: [f64; 4],
    /// Asset or derived index to read per scene (e.g. `"ndvi"`, or a raw
    /// band name resolved via the collection's asset table).
    pub band: String,
    shape: (usize, usize),
    transform: GeoTransform,
    times: Vec<i64>,
}

impl StacCubeSource {
    /// Resolve the dated scene list for `collection`/`bbox`/`band` (via
    /// [`StacClient::search_all`](crate::stac_client::StacClient::search_all))
    /// and fix the target grid. Does not download or decode any pixel data —
    /// that happens lazily, per row range, in [`CubeSource::chunk`].
    pub fn new(_collection: String, _bbox: [f64; 4], _band: String) -> Result<Self> {
        todo!(
            "StacCubeSource::new: resolve dated scenes via StacClient::search_all, \
             align to the target grid, defer decode to chunk()"
        )
    }
}

impl CubeSource for StacCubeSource {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn transform(&self) -> GeoTransform {
        self.transform
    }

    fn times(&self) -> &[i64] {
        &self.times
    }

    fn chunk(&self, _row0: usize, _rows: usize) -> Result<CubeChunk<'_, f64>> {
        todo!(
            "StacCubeSource::chunk: decode [row0, row0+rows) for every resolved scene \
             within a MemoryBudget, applying the cloud mask as NaN"
        )
    }
}
