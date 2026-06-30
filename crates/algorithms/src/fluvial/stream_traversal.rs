//! Network traversal primitive shared by all fluvial-tectonic algorithms.
//!
//! Builds a graph view of a stream raster: every stream cell is indexed,
//! and for each node we know its downstream link (if any) and its upstream
//! neighbours. This is the structure on top of which chi-transform,
//! channel-steepness, knickpoint detection and divide-migration are
//! implemented in subsequent sprints.
//!
//! ## D8 convention
//!
//! Matches `crate::hydrology::flow_direction`:
//!
//! ```text
//!   4  3  2
//!   5  0  1
//!   6  7  8
//! ```
//!
//! Code `0` means "pit or flat — no defined downstream direction". A stream
//! cell with `flow_dir == 0` is treated as a **terminal** node (its
//! `downstream_link` is `None`).
//!
//! ## What counts as an outlet
//!
//! A stream cell is an outlet if any of:
//!
//! - Its flow_dir points outside the raster (boundary outlet).
//! - Its flow_dir points to a non-stream cell (the channel ends in
//!   non-channelised terrain — could be the coast, a lake, or below the
//!   stream-network threshold).
//! - Its flow_dir is `0` (pit/flat).
//! - Its flow_dir is `NaN`/NoData (treated as a cut, same as `0`).
//!
//! Outlets are flagged so downstream consumers (e.g. chi-transform) can
//! iterate per-subnetwork without re-walking the raster.

use surtgis_core::Raster;

/// D8 neighbour offsets keyed by the 1-8 flow-direction code (index 0 is
/// the "pit/flat" sentinel and is never indexed into this table). Mirrors
/// the table in [`crate::hydrology::flow_direction`].
///
/// Encoding (matches hydrology/flow_direction.rs):
///   1=E, 2=NE, 3=N, 4=NW, 5=W, 6=SW, 7=S, 8=SE.
const D8_OFFSETS: [(isize, isize); 9] = [
    (0, 0),   // 0: pit/flat — sentinel, never used
    (0, 1),   // 1: E
    (-1, 1),  // 2: NE
    (-1, 0),  // 3: N
    (-1, -1), // 4: NW
    (0, -1),  // 5: W
    (1, -1),  // 6: SW
    (1, 0),   // 7: S
    (1, 1),   // 8: SE
];

/// Errors that can be raised by [`build_stream_graph`].
#[derive(Debug, thiserror::Error)]
pub enum StreamGraphError {
    /// The stream and flow-direction rasters had incompatible shapes.
    #[error("stream and flow_dir rasters disagree on shape: stream is {0:?}, flow_dir is {1:?}")]
    ShapeMismatch((usize, usize), (usize, usize)),
}

/// Graph view of a stream-network raster.
///
/// Every stream cell appears exactly once in `stream_cells`. Indices into
/// `stream_cells` are the canonical node identifiers; the other two vectors
/// are aligned to it.
///
/// `downstream_link[i]` is:
///
/// - `Some(j)` when node `i` flows into node `j` (another stream cell).
/// - `None` when node `i` is an outlet (boundary, lake/coast, pit/flat,
///   or NoData downstream).
///
/// `upstream_links[i]` lists every node whose `downstream_link == Some(i)`.
/// For a single-channel cell this has length 1; at a confluence it has
/// length 2 (or more in pathological cases).
///
/// `is_outlet[i]` is `true` iff `downstream_link[i].is_none()`; provided
/// for convenience so consumers can iterate outlets without scanning.
#[derive(Debug, Clone)]
pub struct StreamGraph {
    /// `(row, col)` of each stream cell; its index is the node identifier.
    pub stream_cells: Vec<(usize, usize)>,
    /// Downstream node each node flows into, or `None` for outlets.
    pub downstream_link: Vec<Option<usize>>,
    /// Upstream nodes feeding into each node (length ≥ 2 at confluences).
    pub upstream_links: Vec<Vec<usize>>,
    /// Whether each node is an outlet (has no downstream link).
    pub is_outlet: Vec<bool>,
}

impl StreamGraph {
    /// Number of stream nodes.
    pub fn len(&self) -> usize {
        self.stream_cells.len()
    }

    /// Whether the graph has no stream cells.
    pub fn is_empty(&self) -> bool {
        self.stream_cells.is_empty()
    }

    /// Indices of every outlet node.
    pub fn outlets(&self) -> impl Iterator<Item = usize> + '_ {
        self.is_outlet
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
    }
}

/// Build a [`StreamGraph`] from a stream-network raster and a D8
/// flow-direction raster.
///
/// `stream` is the binary stream raster (1 = channel, 0 = non-channel) as
/// produced by [`crate::hydrology::stream_network`]. `flow_dir` is the
/// D8-encoded direction raster from [`crate::hydrology::flow_direction`].
///
/// The two rasters must share dimensions; mismatch returns
/// [`StreamGraphError::ShapeMismatch`].
///
/// ## Complexity
///
/// O(N) where N = number of cells in the raster. Two passes: one to index
/// stream cells, one to resolve downstream links and populate
/// upstream-links bins.
pub fn build_stream_graph(
    stream: &Raster<u8>,
    flow_dir: &Raster<u8>,
) -> Result<StreamGraph, StreamGraphError> {
    let stream_shape = stream.shape();
    let fd_shape = flow_dir.shape();
    if stream_shape != fd_shape {
        return Err(StreamGraphError::ShapeMismatch(stream_shape, fd_shape));
    }
    let (rows, cols) = stream_shape;

    // Pass 1: index every stream cell. We use a flat lookup table sized
    // rows*cols holding Option<usize>; this trades O(N) extra memory for
    // O(1) (r,c) -> node-index resolution in pass 2 (vs O(log N) for a
    // HashMap). At the raster sizes SurtGIS targets the flat table is the
    // right call.
    let mut node_index: Vec<Option<u32>> = vec![None; rows * cols];
    let mut stream_cells: Vec<(usize, usize)> = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            // .get returns Result; outside-bounds is not reachable here, so
            // unwrap is fine. Treat any read failure as non-stream.
            if stream.get(r, c).map(|v| v == 1).unwrap_or(false) {
                node_index[r * cols + c] = Some(stream_cells.len() as u32);
                stream_cells.push((r, c));
            }
        }
    }

    // Pass 2: resolve downstream links. For each stream node, follow its
    // flow_dir code by one step; if that step lands on another stream cell
    // we record the link. Boundary, non-stream, pit/flat, and NoData all
    // produce a `None` downstream link (i.e. mark the node as an outlet).
    let n = stream_cells.len();
    let mut downstream_link: Vec<Option<usize>> = vec![None; n];
    let mut is_outlet: Vec<bool> = vec![true; n];

    for (idx, &(r, c)) in stream_cells.iter().enumerate() {
        let fd_val = match flow_dir.get(r, c) {
            Ok(v) => v,
            Err(_) => continue, // unreachable in practice; defensive
        };
        if fd_val == 0 || fd_val > 8 {
            // pit/flat or out-of-range (NoData-as-0xff etc.) — outlet.
            continue;
        }
        let (dr, dc) = D8_OFFSETS[fd_val as usize];
        let nr = r as isize + dr;
        let nc = c as isize + dc;
        if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
            // Boundary outlet.
            continue;
        }
        let (nr, nc) = (nr as usize, nc as usize);
        match node_index[nr * cols + nc] {
            Some(downstream_idx) => {
                downstream_link[idx] = Some(downstream_idx as usize);
                is_outlet[idx] = false;
            }
            None => {
                // Channel ends in non-channelised terrain (coast, lake,
                // sub-threshold). Outlet.
            }
        }
    }

    // Pass 3: invert downstream_link into upstream_links. Pre-allocate
    // capacity 1 per bin — confluences are sparse, single-channel is the
    // common case.
    let mut upstream_links: Vec<Vec<usize>> = (0..n).map(|_| Vec::with_capacity(1)).collect();
    for (i, &down) in downstream_link.iter().enumerate() {
        if let Some(j) = down {
            upstream_links[j].push(i);
        }
    }

    Ok(StreamGraph {
        stream_cells,
        downstream_link,
        upstream_links,
        is_outlet,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::raster::Raster;

    fn raster_u8(data: Vec<Vec<u8>>) -> Raster<u8> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<u8> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        Raster::from_array(arr)
    }

    #[test]
    fn linear_channel_indexes_correctly() {
        // Stream: a single horizontal line, 5 cells wide.
        //   . . . . .
        //   1 1 1 1 1   ← channel
        //   . . . . .
        //
        // Flow_dir: every channel cell flows east (code 1). The rightmost
        // cell points beyond the raster → boundary outlet.
        let stream = raster_u8(vec![
            vec![0, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 1],
            vec![0, 0, 0, 0, 0],
        ]);
        let flow_dir = raster_u8(vec![
            vec![0, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 1],
            vec![0, 0, 0, 0, 0],
        ]);

        let g = build_stream_graph(&stream, &flow_dir).unwrap();
        assert_eq!(g.len(), 5);

        // Cells are indexed in row-major order: (1,0), (1,1), ..., (1,4).
        for i in 0..5 {
            assert_eq!(g.stream_cells[i], (1, i));
        }

        // Each cell except the rightmost links to its eastern neighbour.
        assert_eq!(g.downstream_link[0], Some(1));
        assert_eq!(g.downstream_link[1], Some(2));
        assert_eq!(g.downstream_link[2], Some(3));
        assert_eq!(g.downstream_link[3], Some(4));
        // Rightmost cell flows out of the raster → outlet.
        assert_eq!(g.downstream_link[4], None);
        assert!(g.is_outlet[4]);
        assert!(!g.is_outlet[0]);

        // Upstream links are the inverse.
        assert_eq!(g.upstream_links[0], Vec::<usize>::new());
        assert_eq!(g.upstream_links[1], vec![0]);
        assert_eq!(g.upstream_links[4], vec![3]);

        let outlets: Vec<usize> = g.outlets().collect();
        assert_eq!(outlets, vec![4]);
    }

    #[test]
    fn confluence_records_multiple_upstream() {
        // Two tributaries (rows 0 and 2) meet at (1,2), then exit east.
        //   1 1 1 . .       flow_dir row 0: 1 1 7 . .   (E, E, S into row 1 col 2)
        //   . . 1 1 1       flow_dir row 1: . . 1 1 1   (E along main stem)
        //   1 1 1 . .       flow_dir row 2: 1 1 3 . .   (E, E, N into row 1 col 2)
        let stream = raster_u8(vec![
            vec![1, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 1],
            vec![1, 1, 1, 0, 0],
        ]);
        let flow_dir = raster_u8(vec![
            vec![1, 1, 7, 0, 0],
            vec![0, 0, 1, 1, 1],
            vec![1, 1, 3, 0, 0],
        ]);

        let g = build_stream_graph(&stream, &flow_dir).unwrap();
        assert_eq!(g.len(), 9);

        // Find the confluence node (1, 2).
        let confluence = g
            .stream_cells
            .iter()
            .position(|&xy| xy == (1, 2))
            .expect("confluence cell present");

        // It must have exactly two upstream tributaries: (0,2) and (2,2).
        let upstream_coords: Vec<(usize, usize)> = g.upstream_links[confluence]
            .iter()
            .map(|&i| g.stream_cells[i])
            .collect();
        assert_eq!(upstream_coords.len(), 2);
        assert!(upstream_coords.contains(&(0, 2)));
        assert!(upstream_coords.contains(&(2, 2)));

        // Boundary outlet is (1, 4).
        let outlet = g.stream_cells.iter().position(|&xy| xy == (1, 4)).unwrap();
        assert_eq!(g.downstream_link[outlet], None);
        assert!(g.is_outlet[outlet]);
    }

    #[test]
    fn pit_cell_is_outlet() {
        // Three-cell channel where the downstream-most cell has flow_dir=0
        // (pit / no defined outflow). That cell is an outlet even though
        // it's interior to the raster.
        let stream = raster_u8(vec![vec![1, 1, 1, 0, 0]]);
        let flow_dir = raster_u8(vec![vec![1, 1, 0, 0, 0]]);

        let g = build_stream_graph(&stream, &flow_dir).unwrap();
        assert_eq!(g.len(), 3);
        assert_eq!(g.downstream_link[2], None);
        assert!(g.is_outlet[2]);
    }

    #[test]
    fn channel_ending_in_non_stream_terrain_is_outlet() {
        // Single-row stream of length 2, flowing east, but the cell east of
        // the rightmost stream cell is non-stream (lake / coast / below
        // threshold). Right-cell must register as outlet.
        let stream = raster_u8(vec![vec![1, 1, 0, 0, 0]]);
        let flow_dir = raster_u8(vec![vec![1, 1, 0, 0, 0]]);

        let g = build_stream_graph(&stream, &flow_dir).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g.downstream_link[0], Some(1));
        assert_eq!(g.downstream_link[1], None);
        assert!(g.is_outlet[1]);
    }

    #[test]
    fn shape_mismatch_errors() {
        let stream = raster_u8(vec![vec![1, 1]]);
        let flow_dir = raster_u8(vec![vec![1, 1, 1]]);
        let err = build_stream_graph(&stream, &flow_dir).unwrap_err();
        assert!(matches!(err, StreamGraphError::ShapeMismatch(_, _)));
    }

    #[test]
    fn out_of_range_flow_dir_value_treated_as_outlet() {
        // Defensive: if a flow_dir cell holds something outside 0..=8
        // (e.g. a NoData sentinel encoded as 255), the cell should be
        // treated as an outlet rather than panicking or indexing past
        // D8_OFFSETS.
        let stream = raster_u8(vec![vec![1, 1, 0]]);
        let flow_dir = raster_u8(vec![vec![1, 255, 0]]);

        let g = build_stream_graph(&stream, &flow_dir).unwrap();
        assert_eq!(g.len(), 2);
        // (0,1) has flow_dir=255 → outlet
        let oob = g.stream_cells.iter().position(|&xy| xy == (0, 1)).unwrap();
        assert!(g.is_outlet[oob]);
        assert_eq!(g.downstream_link[oob], None);
    }
}
