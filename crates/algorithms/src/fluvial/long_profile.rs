//! Stream long-profile extraction — outlet → source main stem.
//!
//! For each outlet in a stream network, walks upstream along the
//! largest-area path at every confluence and emits the per-node
//! profile (cumulative distance, elevation, drainage area, and
//! optionally χ and ksn pulled from caller-supplied rasters). This is
//! the underlying data product for χ-elevation plots, longitudinal
//! profile figures, knickpoint annotations, and the comparison panels
//! standard in tectonic-geomorphology papers (Wobus et al. 2006;
//! Perron & Royden 2013; Mudd et al. 2018).
//!
//! ## Why main-stem only
//!
//! TopoToolbox and LSDTopoTools both treat "the profile" of a basin
//! as the largest-area path from outlet to source. Tributary profiles
//! are recovered separately by walking from each headwater. We follow
//! that convention; users wanting per-tributary profiles can iterate
//! confluences in [`super::stream_traversal::StreamGraph::upstream_links`]
//! and call this function on each headwater set.
//!
//! ## Decoupled from χ / ksn computation
//!
//! χ and ksn are passed in as **optional rasters** — the caller
//! computes them with [`super::chi_transform`] or
//! [`super::channel_steepness`] first, then hands the rasters to this
//! function. Keeping the responsibilities separate avoids hard-wiring
//! a particular θ_ref or A₀ choice inside the profile extractor and
//! lets callers mix-and-match (e.g. ksn from a different smoothing
//! window than χ uses).

use surtgis_core::{Raster, Result};

use super::stream_traversal::{StreamGraph, StreamGraphError, build_stream_graph};

/// Parameters for [`long_profile`].
#[derive(Debug, Clone)]
pub struct LongProfileParams {
    /// Pixel size in metres. Caller-supplied because the function
    /// does not introspect the raster transform — matches the rest
    /// of the fluvial module's convention.
    pub cell_size_m: f64,
}

impl Default for LongProfileParams {
    fn default() -> Self {
        Self { cell_size_m: 30.0 }
    }
}

/// One node along a stream long profile.
#[derive(Debug, Clone, Copy)]
pub struct LongProfileNode {
    /// Cell-centre coordinates in the source raster's CRS.
    pub coord: (f64, f64),
    /// Cumulative along-channel distance from the outlet, in metres.
    pub distance_from_outlet_m: f64,
    /// DEM elevation at the cell.
    pub elevation: f64,
    /// Drainage area in m² (flow_acc × cell_size²).
    pub area_m2: f64,
    /// χ value at the cell. `None` when no χ raster was supplied;
    /// `Some(NaN)` when the χ raster is supplied but masked at this
    /// cell (off-network or below threshold).
    pub chi: Option<f64>,
    /// ksn value at the cell. `None` when no ksn raster was supplied;
    /// `Some(NaN)` when masked.
    pub ksn: Option<f64>,
}

/// One outlet's long profile.
#[derive(Debug, Clone)]
pub struct LongProfile {
    /// Outlet cell-centre coordinates.
    pub outlet_coord: (f64, f64),
    /// Nodes ordered outlet → source along the main-stem path.
    pub nodes: Vec<LongProfileNode>,
}

/// Errors specific to long-profile extraction.
#[derive(Debug, thiserror::Error)]
pub enum LongProfileError {
    #[error(transparent)]
    Graph(#[from] StreamGraphError),
    #[error("input raster shapes disagree: {0:?} vs {1:?}")]
    ShapeMismatch((usize, usize), (usize, usize)),
    #[error("LongProfileParams.cell_size_m must be > 0 (got {0})")]
    NonPositiveCellSize(f64),
}

/// Extract one long profile per outlet, walking upstream along the
/// largest-area path at every confluence.
///
/// Returns one [`LongProfile`] per outlet found in `stream`. The
/// nodes are ordered outlet → source so that
/// `nodes[0].distance_from_outlet_m == 0` and the distance increases
/// monotonically along the path.
///
/// ## Inputs
///
/// - `stream` — binary stream-network raster (1 = channel).
/// - `flow_dir` — D8 flow direction; same encoding as
///   [`crate::hydrology::flow_direction`].
/// - `flow_acc` — flow accumulation in **cells** (multiplied by
///   `cell_size_m²` internally to obtain m²).
/// - `dem` — elevation raster.
/// - `chi_raster` / `ksn_raster` — optional per-cell χ and ksn
///   layers. When `Some`, the value at each profile node is copied
///   into the corresponding `LongProfileNode.chi` / `.ksn` field.
///
/// All rasters must share the stream raster's shape; a mismatch
/// returns [`LongProfileError::ShapeMismatch`].
#[allow(clippy::too_many_arguments)]
pub fn long_profile(
    stream: &Raster<u8>,
    flow_dir: &Raster<u8>,
    flow_acc: &Raster<f64>,
    dem: &Raster<f64>,
    chi_raster: Option<&Raster<f64>>,
    ksn_raster: Option<&Raster<f64>>,
    params: LongProfileParams,
) -> Result<Vec<LongProfile>> {
    if params.cell_size_m <= 0.0 {
        return Err(surtgis_core::Error::Other(
            LongProfileError::NonPositiveCellSize(params.cell_size_m).to_string(),
        ));
    }
    let s_shape = stream.shape();
    for (other, label) in [
        (flow_dir.shape(), "flow_dir"),
        (flow_acc.shape(), "flow_acc"),
        (dem.shape(), "dem"),
    ] {
        if other != s_shape {
            return Err(surtgis_core::Error::Other(
                LongProfileError::ShapeMismatch(s_shape, other).to_string(),
            ));
        }
        let _ = label;
    }
    if let Some(r) = chi_raster
        && r.shape() != s_shape
    {
        return Err(surtgis_core::Error::Other(
            LongProfileError::ShapeMismatch(s_shape, r.shape()).to_string(),
        ));
    }
    if let Some(r) = ksn_raster
        && r.shape() != s_shape
    {
        return Err(surtgis_core::Error::Other(
            LongProfileError::ShapeMismatch(s_shape, r.shape()).to_string(),
        ));
    }

    let graph = build_stream_graph(stream, flow_dir)
        .map_err(|e| surtgis_core::Error::Other(LongProfileError::from(e).to_string()))?;

    let cell_area = params.cell_size_m * params.cell_size_m;
    let diag_factor = std::f64::consts::SQRT_2;

    let gt = stream.transform();
    let to_cell_centre = |r: usize, c: usize| -> (f64, f64) {
        let (x0, y0) = stream.pixel_to_geo(c, r);
        (x0 + 0.5 * gt.pixel_width, y0 + 0.5 * gt.pixel_height)
    };

    // Per-node access helpers.
    let value_at = |raster: &Raster<f64>, i: usize| -> f64 {
        let (r, c) = graph.stream_cells[i];
        raster.get(r, c).unwrap_or(f64::NAN)
    };

    let mut profiles = Vec::new();
    for outlet in graph.outlets() {
        let outlet_coord = {
            let (r, c) = graph.stream_cells[outlet];
            to_cell_centre(r, c)
        };

        let mut nodes: Vec<LongProfileNode> = Vec::new();
        let mut cur = outlet;
        let mut dist = 0.0;
        loop {
            let (r, c) = graph.stream_cells[cur];
            let coord = to_cell_centre(r, c);
            let elevation = value_at(dem, cur);
            let area_m2 = value_at(flow_acc, cur) * cell_area;
            let chi = chi_raster.map(|x| value_at(x, cur));
            let ksn = ksn_raster.map(|x| value_at(x, cur));
            nodes.push(LongProfileNode {
                coord,
                distance_from_outlet_m: dist,
                elevation,
                area_m2,
                chi,
                ksn,
            });

            // Pick the largest-area upstream neighbour to follow the
            // main stem; stop at headwaters.
            let ups = &graph.upstream_links[cur];
            if ups.is_empty() {
                break;
            }
            let mut best = ups[0];
            let mut best_area = value_at(flow_acc, best);
            for &u in &ups[1..] {
                let a = value_at(flow_acc, u);
                if a > best_area {
                    best_area = a;
                    best = u;
                }
            }

            // Step length from cur → best in raster (row, col) space.
            // |dr| + |dc| == 1 → cardinal; == 2 → diagonal.
            let (rb, cb) = graph.stream_cells[best];
            let dr = (rb as isize - r as isize).abs();
            let dc = (cb as isize - c as isize).abs();
            let step = if dr + dc == 1 {
                params.cell_size_m
            } else {
                params.cell_size_m * diag_factor
            };
            dist += step;
            cur = best;
        }

        profiles.push(LongProfile {
            outlet_coord,
            nodes,
        });
    }

    Ok(profiles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::raster::Raster;

    fn raster_u8(data: Vec<Vec<u8>>) -> Raster<u8> {
        let rows = data.len();
        let cols = data[0].len();
        Raster::from_array(
            Array2::from_shape_vec((rows, cols), data.into_iter().flatten().collect()).unwrap(),
        )
    }
    fn raster_f64(data: Vec<Vec<f64>>) -> Raster<f64> {
        let rows = data.len();
        let cols = data[0].len();
        Raster::from_array(
            Array2::from_shape_vec((rows, cols), data.into_iter().flatten().collect()).unwrap(),
        )
    }

    /// 6-cell straight east-flowing channel: outlet at column 0,
    /// source at column 5. Distance must increase by `cell_size_m`
    /// at every step; elevation must match the DEM.
    #[test]
    fn straight_channel_distance_is_cell_size_per_step() {
        let stream = raster_u8(vec![vec![1, 1, 1, 1, 1, 1]]);
        // Encoding 5 = W (each cell points west, i.e. column 0 = outlet).
        let flow_dir = raster_u8(vec![vec![0, 5, 5, 5, 5, 5]]);
        let flow_acc = raster_f64(vec![vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]);
        let dem = raster_f64(vec![vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]);
        let params = LongProfileParams { cell_size_m: 30.0 };

        let profiles =
            long_profile(&stream, &flow_dir, &flow_acc, &dem, None, None, params).unwrap();
        assert_eq!(profiles.len(), 1);
        let p = &profiles[0];
        assert_eq!(p.nodes.len(), 6);
        for (i, n) in p.nodes.iter().enumerate() {
            assert_eq!(n.distance_from_outlet_m, i as f64 * 30.0);
            assert_eq!(n.elevation, i as f64);
            assert!(n.chi.is_none());
            assert!(n.ksn.is_none());
        }
        // Outlet drainage area uses cell_size² to convert from cells.
        assert_eq!(p.nodes[0].area_m2, 6.0 * 30.0 * 30.0);
    }

    /// At a confluence the walker takes the larger-area upstream
    /// link. Y-shaped network: 2 headwaters with different areas
    /// merging into a single trunk.
    #[test]
    fn main_stem_picks_largest_area_at_confluence() {
        // Layout (3×5), '.' = no stream:
        //   [B  B  .  .  .]   row 0   (small tributary B)
        //   [.  C  T  T  T]   row 1   (confluence + trunk to outlet)
        //   [A  A  A  .  .]   row 2   (big tributary A)
        //
        // Flow:
        //   A: (2,2) → (2,1) → (2,0) → (1,1)   area 1000 each
        //   B: (0,1) → (0,0) → (1,1)           area 10 each
        //   confluence (1,1) is downstream end; trunk: (1,1)→(1,2)→(1,3)→(1,4) outlet
        let stream = raster_u8(vec![
            vec![1, 1, 0, 0, 0],
            vec![0, 1, 1, 1, 1],
            vec![1, 1, 1, 0, 0],
        ]);
        // Flow direction codes (E=1, NE=2, N=3, NW=4, W=5, SW=6, S=7, SE=8):
        //   row 0: (0,0)→W=5 doesn't reach a downstream cell; let's make
        //   (0,1)→W=5 and (0,0)→SE=8 instead so B exits via the confluence.
        //   row 1: (1,1)→E=1, (1,2)→E=1, (1,3)→E=1, (1,4)→0 outlet
        //   row 2: (2,0)→NE=2 (to (1,1)), (2,1)→W=5, (2,2)→W=5
        let flow_dir = raster_u8(vec![
            vec![8, 5, 0, 0, 0],
            vec![0, 1, 1, 1, 0],
            vec![2, 5, 5, 0, 0],
        ]);
        let flow_acc = raster_f64(vec![
            vec![10.0, 20.0, 0.0, 0.0, 0.0],
            vec![0.0, 2030.0, 2040.0, 2050.0, 2060.0],
            vec![3000.0, 2000.0, 1000.0, 0.0, 0.0],
        ]);
        let dem = raster_f64(vec![
            vec![100.0, 90.0, 0.0, 0.0, 0.0],
            vec![0.0, 50.0, 40.0, 30.0, 20.0],
            vec![80.0, 70.0, 60.0, 0.0, 0.0],
        ]);
        let params = LongProfileParams { cell_size_m: 30.0 };

        let profiles =
            long_profile(&stream, &flow_dir, &flow_acc, &dem, None, None, params).unwrap();
        assert_eq!(profiles.len(), 1);
        let p = &profiles[0];

        // First nodes from outlet (1,4) inward to confluence (1,1):
        // (1,4) → (1,3) → (1,2) → (1,1).
        assert_eq!(p.nodes[0].coord, p.outlet_coord);
        // At (1,1), the two upstream tributaries are (2,0) (area 3000)
        // and (0,0) (area 10). The walker must pick (2,0). The next
        // node after (1,1) must therefore be from row 2.
        let confluence_idx = p
            .nodes
            .iter()
            .position(|n| n.elevation == 50.0)
            .expect("confluence node present");
        let next = p.nodes[confluence_idx + 1];
        // Walking from confluence (1,1) into tributary A enters at
        // (2,0) which has elevation 80 (the downstream end of A);
        // tributary B's downstream end at (0,0) has elevation 100.
        // 80 means the walker correctly chose A (3000 area) over B
        // (10 area).
        assert_eq!(
            next.elevation, 80.0,
            "expected to enter tributary A at (2,0) elevation 80, saw {}",
            next.elevation
        );
        // Tributary A drainage area at (2,2) is 1000; B at (0,1) is 20.
        // Last node area should reflect A.
        let last = p.nodes.last().unwrap();
        assert!(
            last.area_m2 >= 1000.0 * 30.0 * 30.0,
            "last node area {} too small to be tributary A's headwater",
            last.area_m2
        );
    }

    /// chi and ksn rasters propagate to the profile when supplied.
    #[test]
    fn optional_chi_ksn_are_threaded_through() {
        let stream = raster_u8(vec![vec![1, 1, 1]]);
        let flow_dir = raster_u8(vec![vec![0, 5, 5]]);
        let flow_acc = raster_f64(vec![vec![3.0, 2.0, 1.0]]);
        let dem = raster_f64(vec![vec![0.0, 1.0, 2.0]]);
        let chi = raster_f64(vec![vec![0.0, 0.5, 1.0]]);
        let ksn = raster_f64(vec![vec![f64::NAN, 20.0, 25.0]]);
        let params = LongProfileParams { cell_size_m: 1.0 };

        let profiles = long_profile(
            &stream,
            &flow_dir,
            &flow_acc,
            &dem,
            Some(&chi),
            Some(&ksn),
            params,
        )
        .unwrap();
        let p = &profiles[0];
        assert_eq!(p.nodes[0].chi, Some(0.0));
        assert_eq!(p.nodes[1].chi, Some(0.5));
        // NaN doesn't equal itself, so use `is_nan` for the outlet's ksn.
        assert!(p.nodes[0].ksn.unwrap().is_nan());
        assert_eq!(p.nodes[1].ksn, Some(20.0));
    }
}
