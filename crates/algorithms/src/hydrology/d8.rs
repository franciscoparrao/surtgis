//! Canonical D8 direction encoding, offset tables and helpers.
//!
//! This module is the **single source of truth** for the D8 flow-direction
//! contract shared by the hydrology, terrain and fluvial modules. The
//! direction codes written to (and read from) flow-direction rasters are:
//!
//! ```text
//!   4  3  2
//!   5  0  1
//!   6  7  8
//! ```
//!
//! - `0` = pit or flat cell (no outflow)
//! - `1`–`8` = counter-clockwise from East: E, NE, N, NW, W, SW, S, SE
//!
//! Offsets are expressed as `(row_offset, col_offset)` in raster coordinates
//! (row increases downwards, i.e. towards the South).
//!
//! **This encoding is a public contract**: it is what [`flow_direction`]
//! writes into output rasters, so files already produced by SurtGIS depend
//! on it. It must never change.
//!
//! Two table layouts are provided:
//!
//! - *Direction-indexed* tables ([`D8_OFFSETS`], [`D8_OFFSETS_9`],
//!   [`D8_ROW_OFF`]/[`D8_COL_OFF`]) whose index maps to the direction code
//!   above. Use these whenever a D8 code is being encoded or decoded.
//! - *Scan-order* tables ([`SCAN_OFFSETS`], [`SCAN_DISTANCE`]) that visit the
//!   3×3 neighborhood in row-major order. These carry **no** direction
//!   semantics and exist for neighborhood-iteration algorithms (depression
//!   filling, breaching, …) whose tie-breaking behaviour depends on the
//!   historical iteration order.
//!
//! [`flow_direction`]: crate::hydrology::flow_direction

use std::f64::consts::SQRT_2;

/// D8 neighbor offsets `(row_offset, col_offset)`, indexed by
/// `direction code - 1` (i.e. index 0 = direction 1 = East).
///
/// Order: E, NE, N, NW, W, SW, S, SE (counter-clockwise from East).
pub const D8_OFFSETS: [(isize, isize); 8] = [
    (0, 1),   // 1: E
    (-1, 1),  // 2: NE
    (-1, 0),  // 3: N
    (-1, -1), // 4: NW
    (0, -1),  // 5: W
    (1, -1),  // 6: SW
    (1, 0),   // 7: S
    (1, 1),   // 8: SE
];

/// Unit distance to each D8 neighbor, indexed like [`D8_OFFSETS`]
/// (`direction code - 1`): `1.0` for cardinal moves, `√2` for diagonals.
///
/// Multiply by the cell size to obtain a grid distance, or use
/// [`distance`] for a code-based lookup.
pub const D8_DISTANCE: [f64; 8] = [1.0, SQRT_2, 1.0, SQRT_2, 1.0, SQRT_2, 1.0, SQRT_2];

/// D8 neighbor offsets indexed **directly by direction code** (0–8).
///
/// Index 0 is the `(0, 0)` sentinel for pit/flat cells (no outflow) and
/// must never be used to move; indices 1–8 match the canonical encoding.
/// Convenient when a raster of direction codes is followed without
/// subtracting 1 (e.g. stream traversal, watershed path tracing).
pub const D8_OFFSETS_9: [(isize, isize); 9] = [
    (0, 0),   // 0: pit/flat — sentinel, never a real move
    (0, 1),   // 1: E
    (-1, 1),  // 2: NE
    (-1, 0),  // 3: N
    (-1, -1), // 4: NW
    (0, -1),  // 5: W
    (1, -1),  // 6: SW
    (1, 0),   // 7: S
    (1, 1),   // 8: SE
];

/// Row offsets of [`D8_OFFSETS_9`] as a structure-of-arrays table,
/// indexed directly by direction code (0 = sentinel).
pub const D8_ROW_OFF: [isize; 9] = [0, 0, -1, -1, -1, 0, 1, 1, 1];

/// Column offsets of [`D8_OFFSETS_9`] as a structure-of-arrays table,
/// indexed directly by direction code (0 = sentinel).
pub const D8_COL_OFF: [isize; 9] = [0, 1, 1, 0, -1, -1, -1, 0, 1];

/// 3×3 neighborhood offsets in **row-major scan order**:
/// NW, N, NE, W, E, SW, S, SE.
///
/// ⚠️ The index of this table does **not** correspond to a D8 direction
/// code — never use it to encode/decode directions. It exists for
/// neighborhood-iteration algorithms (Planchon–Darboux fill,
/// Priority-Flood, breaching, depression labelling, downslope index)
/// whose tie-breaking results depend on this historical visit order.
pub const SCAN_OFFSETS: [(isize, isize); 8] = [
    (-1, -1), // NW
    (-1, 0),  // N
    (-1, 1),  // NE
    (0, -1),  // W
    (0, 1),   // E
    (1, -1),  // SW
    (1, 0),   // S
    (1, 1),   // SE
];

/// Unit distances matching [`SCAN_OFFSETS`] entry-by-entry
/// (`√2` for diagonals, `1.0` for cardinal moves).
pub const SCAN_DISTANCE: [f64; 8] = [SQRT_2, 1.0, SQRT_2, 1.0, 1.0, SQRT_2, 1.0, SQRT_2];

/// Encode a neighbor offset as a D8 direction code.
///
/// Returns `Some(1..=8)` for the eight valid unit offsets, `None` for
/// `(0, 0)` or any offset outside the 3×3 neighborhood.
///
/// # Examples
/// ```
/// use surtgis_algorithms::hydrology::d8;
/// assert_eq!(d8::encode(0, 1), Some(1)); // East
/// assert_eq!(d8::encode(1, 0), Some(7)); // South
/// assert_eq!(d8::encode(0, 0), None); // no movement
/// ```
pub fn encode(row_offset: isize, col_offset: isize) -> Option<u8> {
    D8_OFFSETS
        .iter()
        .position(|&(dr, dc)| dr == row_offset && dc == col_offset)
        .map(|idx| (idx + 1) as u8)
}

/// Decode a D8 direction code into its `(row_offset, col_offset)`.
///
/// Returns `None` for code `0` (pit/flat, no outflow) and for any code
/// greater than `8`.
///
/// # Examples
/// ```
/// use surtgis_algorithms::hydrology::d8;
/// assert_eq!(d8::decode(1), Some((0, 1))); // East
/// assert_eq!(d8::decode(0), None); // pit
/// ```
pub fn decode(dir: u8) -> Option<(isize, isize)> {
    match dir {
        1..=8 => Some(D8_OFFSETS[(dir - 1) as usize]),
        _ => None,
    }
}

/// Grid distance travelled when flowing in direction `dir` on a grid with
/// the given cell size (`cell_size` for cardinal moves, `cell_size · √2`
/// for diagonal moves).
///
/// Returns `None` for code `0` or any invalid code.
pub fn distance(dir: u8, cell_size: f64) -> Option<f64> {
    match dir {
        1..=8 => Some(D8_DISTANCE[(dir - 1) as usize] * cell_size),
        _ => None,
    }
}

/// Opposite D8 direction (rotate by 180°): E↔W, NE↔SW, N↔S, NW↔SE.
///
/// Useful to test whether a neighbor drains *into* the current cell: the
/// neighbor at direction `d` flows towards us iff its code is
/// `opposite(d)`. Returns `0` for code `0` or any invalid code.
pub fn opposite(dir: u8) -> u8 {
    match dir {
        1..=8 => ((dir - 1 + 4) % 8) + 1,
        _ => 0,
    }
}

/// Coordinates of the downstream (receiving) cell when flowing from
/// `(row, col)` in direction `dir`, on a `rows × cols` grid.
///
/// Returns `None` if `dir` is `0`/invalid or if the move would leave the
/// grid bounds.
///
/// # Examples
/// ```
/// use surtgis_algorithms::hydrology::d8;
/// assert_eq!(d8::downstream(1, 1, 1, 3, 3), Some((1, 2))); // East
/// assert_eq!(d8::downstream(0, 0, 3, 3, 3), None); // North off-grid
/// ```
pub fn downstream(
    row: usize,
    col: usize,
    dir: u8,
    rows: usize,
    cols: usize,
) -> Option<(usize, usize)> {
    let (dr, dc) = decode(dir)?;
    let nr = row as isize + dr;
    let nc = col as isize + dc;
    if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
        return None;
    }
    Some((nr as usize, nc as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_consistent() {
        // D8_OFFSETS_9 == sentinel + D8_OFFSETS
        assert_eq!(D8_OFFSETS_9[0], (0, 0));
        for d in 1..=8usize {
            assert_eq!(D8_OFFSETS_9[d], D8_OFFSETS[d - 1]);
            assert_eq!(D8_ROW_OFF[d], D8_OFFSETS[d - 1].0);
            assert_eq!(D8_COL_OFF[d], D8_OFFSETS[d - 1].1);
        }
        // Distances match offsets (diagonal iff both components non-zero)
        for (i, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
            let expected = if dr != 0 && dc != 0 { SQRT_2 } else { 1.0 };
            assert_eq!(D8_DISTANCE[i], expected);
        }
        for (i, &(dr, dc)) in SCAN_OFFSETS.iter().enumerate() {
            let expected = if dr != 0 && dc != 0 { SQRT_2 } else { 1.0 };
            assert_eq!(SCAN_DISTANCE[i], expected);
        }
        // Scan table covers the same 8 neighbors
        for &off in &SCAN_OFFSETS {
            assert!(D8_OFFSETS.contains(&off));
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        for d in 1..=8u8 {
            let (dr, dc) = decode(d).unwrap();
            assert_eq!(encode(dr, dc), Some(d));
        }
        assert_eq!(decode(0), None);
        assert_eq!(decode(9), None);
        assert_eq!(encode(0, 0), None);
        assert_eq!(encode(2, 0), None);
    }

    #[test]
    fn canonical_codes() {
        // The public raster contract: 1=E, 3=N, 5=W, 7=S
        assert_eq!(decode(1), Some((0, 1)));
        assert_eq!(decode(3), Some((-1, 0)));
        assert_eq!(decode(5), Some((0, -1)));
        assert_eq!(decode(7), Some((1, 0)));
        assert_eq!(decode(2), Some((-1, 1))); // NE
        assert_eq!(decode(4), Some((-1, -1))); // NW
        assert_eq!(decode(6), Some((1, -1))); // SW
        assert_eq!(decode(8), Some((1, 1))); // SE
    }

    #[test]
    fn opposite_directions() {
        assert_eq!(opposite(1), 5); // E → W
        assert_eq!(opposite(2), 6); // NE → SW
        assert_eq!(opposite(3), 7); // N → S
        assert_eq!(opposite(4), 8); // NW → SE
        assert_eq!(opposite(5), 1);
        assert_eq!(opposite(6), 2);
        assert_eq!(opposite(7), 3);
        assert_eq!(opposite(8), 4);
        assert_eq!(opposite(0), 0);
        assert_eq!(opposite(9), 0);
        // opposite is an involution and negates the offset
        for d in 1..=8u8 {
            assert_eq!(opposite(opposite(d)), d);
            let (dr, dc) = decode(d).unwrap();
            assert_eq!(decode(opposite(d)), Some((-dr, -dc)));
        }
    }

    #[test]
    fn distance_lookup() {
        assert_eq!(distance(1, 10.0), Some(10.0));
        assert_eq!(distance(2, 10.0), Some(10.0 * SQRT_2));
        assert_eq!(distance(0, 10.0), None);
        assert_eq!(distance(255, 10.0), None);
    }

    #[test]
    fn downstream_bounds() {
        // Interior cell: all 8 directions valid
        for d in 1..=8u8 {
            assert!(downstream(1, 1, d, 3, 3).is_some());
        }
        // Corner (0,0): N/NE/NW/W/SW off-grid
        assert_eq!(downstream(0, 0, 3, 3, 3), None); // N
        assert_eq!(downstream(0, 0, 5, 3, 3), None); // W
        assert_eq!(downstream(0, 0, 1, 3, 3), Some((0, 1))); // E
        assert_eq!(downstream(0, 0, 8, 3, 3), Some((1, 1))); // SE
        // Pit code never moves
        assert_eq!(downstream(1, 1, 0, 3, 3), None);
    }
}
