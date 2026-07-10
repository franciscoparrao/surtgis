//! Drainage assignment over flat surfaces (Garbrecht & Martz 1997).
//!
//! Hydrologically conditioned DEMs (sink-filled with epsilon 0) and
//! integer-quantized DEMs contain *flats*: connected regions of exactly equal
//! elevation where the D8 rule finds no downslope neighbor and leaves the
//! direction code at `0`. Every cell of a filled depression, lake surface or
//! coastal plain then reads as a pit, and downstream algorithms (flow
//! accumulation, watershed, stream network, HAND) silently lose the drainage
//! through those regions.
//!
//! This module implements the double-gradient method of Garbrecht & Martz
//! (1997) using the O(n) formulation of Barnes, Lehman & Mulla (2014):
//! drainage over a flat is directed simultaneously *towards lower terrain*
//! and *away from higher terrain*. Two breadth-first passes build an integer
//! potential surface (`FlatMask`) over each drainable flat — one growing away
//! from the flat's higher rim, one (double-weighted, which guarantees
//! monotone descent to the outlets) growing inward from the cells where the
//! flat meets draining terrain — and the D8 rule applied to that surface
//! yields the directions.
//!
//! A flat can drain two ways: through a same-elevation neighbor that already
//! has a flow direction (an interior outlet), or by touching the raster edge
//! or a nodata region (its water leaves the DEM — a coastal plain against a
//! masked sea, a filled valley floor at the DEM border). Boundary-touching
//! cells keep code `0` — SurtGIS directions never point off-grid or into
//! nodata, so they act as terminal sinks — while the rest of the flat is
//! routed toward them. Only flats with neither kind of outlet (e.g. an
//! unfilled closed depression floor) are left untouched, their cells keeping
//! direction `0` exactly like single-cell pits.
//!
//! References:
//! - Garbrecht, J., Martz, L.W. (1997). The assignment of drainage direction
//!   over flat surfaces in raster digital elevation models. *Journal of
//!   Hydrology* 193, 204–213.
//! - Barnes, R., Lehman, C., Mulla, D. (2014). An efficient assignment of
//!   drainage direction over flat surfaces in raster digital elevation
//!   models. *Computers & Geosciences* 62, 128–135.

use std::collections::VecDeque;

use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use super::d8::{D8_DISTANCE, D8_OFFSETS};

/// Summary of one [`resolve_flats`] pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct FlatResolutionStats {
    /// Number of drainable flats found (connected equal-elevation regions
    /// with at least one outlet onto draining terrain, the raster edge or
    /// nodata).
    pub flats: usize,
    /// Cells that had direction `0` and were assigned a direction.
    pub resolved_cells: usize,
    /// Cells that still have direction `0` afterwards: true pits, the cells
    /// of fully enclosed flats, and boundary-touching flat cells (which act
    /// as the flat's terminal sinks).
    pub unresolved_cells: usize,
}

/// Assign D8 drainage directions across flat surfaces in-place
/// (Garbrecht & Martz 1997, via Barnes et al. 2014).
///
/// `directions` must be the D8 raster produced from `dem` (canonical
/// encoding of the [`d8`](super::d8) module: `0` = no outflow, `1`–`8`
/// counter-clockwise from East). Cells with code `0` that belong to a
/// drainable flat — a connected region of equal elevation with at least one
/// cell adjacent to a same-elevation draining neighbor, to the raster edge
/// or to nodata — receive a direction; flat cells touching the boundary
/// stay at `0` and act as the flat's terminal sinks. True pits and fully
/// enclosed flats are left at `0`. Directions of already-draining cells are
/// never modified, and no assigned direction ever points into nodata or off
/// the grid.
///
/// [`flow_direction`](crate::hydrology::flow_direction) applies this
/// automatically; call it directly only when composing with a custom D8
/// raster.
///
/// # Errors
/// Returns [`Error::ShapeMismatch`] if the two rasters disagree in shape.
pub fn resolve_flats(
    dem: &Raster<f64>,
    directions: &mut Raster<u8>,
) -> Result<FlatResolutionStats> {
    let (rows, cols) = dem.shape();
    if directions.shape() != (rows, cols) {
        return Err(Error::ShapeMismatch {
            expected: (rows, cols),
            got: directions.shape(),
            context: "resolve_flats: dem vs directions".to_string(),
        });
    }
    let n = rows * cols;
    let idx = |r: usize, c: usize| r * cols + c;

    // Snapshot of the incoming directions: edge detection and outlet tests
    // must see the pre-resolution state while we write the new codes.
    let dir: Vec<u8> = directions.data().iter().copied().collect();

    let elev_at = |i: usize| unsafe { dem.get_unchecked(i / cols, i % cols) };
    let in_bounds = |r: isize, c: isize| r >= 0 && c >= 0 && r < rows as isize && c < cols as isize;

    // Pass 1 — classify the undrained cells into flat edges (Barnes alg. 3).
    // Low edge: an undrained cell with a same-elevation neighbor that drains
    // (the flat's outlet rim), or a flat cell touching the raster edge or
    // nodata (the flat's water leaves the DEM there — think a coastal plain
    // against a masked sea). Boundary-touching cells keep code `0` (SurtGIS
    // directions never point off-grid or into nodata: they become terminal
    // sinks, exactly like a sloped cell whose only descent is off the DEM)
    // but they seed the towards-lower gradient so the flat interior drains
    // toward them. High edge: an undrained cell next to higher terrain (the
    // flat's upper rim). Low takes precedence.
    let mut low_edges: VecDeque<usize> = VecDeque::new();
    let mut high_edges: Vec<usize> = Vec::new();
    let mut undrained = 0usize;
    for r in 0..rows {
        for c in 0..cols {
            let i = idx(r, c);
            if dir[i] != 0 {
                continue;
            }
            let e = elev_at(i);
            if dem.is_nodata(e) {
                continue;
            }
            undrained += 1;
            let mut is_low = false;
            let mut is_high = false;
            let mut touches_boundary = false;
            let mut has_equal = false;
            for &(dr, dc) in &D8_OFFSETS {
                let (nr, nc) = (r as isize + dr, c as isize + dc);
                if !in_bounds(nr, nc) {
                    touches_boundary = true;
                    continue;
                }
                let ni = idx(nr as usize, nc as usize);
                let ne = elev_at(ni);
                if dem.is_nodata(ne) {
                    touches_boundary = true;
                    continue;
                }
                if dir[ni] != 0 && ne == e {
                    is_low = true;
                    break;
                }
                if ne > e {
                    is_high = true;
                } else if ne == e {
                    has_equal = true;
                }
            }
            if is_low || (touches_boundary && has_equal) {
                low_edges.push_back(i);
            } else if is_high {
                high_edges.push(i);
            }
        }
    }
    if low_edges.is_empty() {
        return Ok(FlatResolutionStats {
            flats: 0,
            resolved_cells: 0,
            unresolved_cells: undrained,
        });
    }

    // Pass 2 — label each drainable flat: BFS from the low edges across
    // undrained cells of exactly equal elevation (Barnes alg. 4).
    let mut label = vec![0u32; n];
    let mut n_labels = 0u32;
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &seed in &low_edges {
        if label[seed] != 0 {
            continue;
        }
        n_labels += 1;
        let e = elev_at(seed);
        label[seed] = n_labels;
        queue.push_back(seed);
        while let Some(i) = queue.pop_front() {
            let (r, c) = (i / cols, i % cols);
            for &(dr, dc) in &D8_OFFSETS {
                let (nr, nc) = (r as isize + dr, c as isize + dc);
                if !in_bounds(nr, nc) {
                    continue;
                }
                let ni = idx(nr as usize, nc as usize);
                if label[ni] == 0 && dir[ni] == 0 && elev_at(ni) == e {
                    label[ni] = n_labels;
                    queue.push_back(ni);
                }
            }
        }
    }

    // Pass 3 — gradient away from higher terrain (Barnes alg. 5): level-order
    // BFS from the labeled high-edge cells. `mask[i]` = BFS level (1-based);
    // `flat_height[l]` = the deepest level reached inside flat `l`.
    const MARKER: usize = usize::MAX;
    let mut mask = vec![0i32; n];
    let mut flat_height = vec![0i32; n_labels as usize + 1];
    let mut level: i32 = 1;
    let mut away: VecDeque<usize> = high_edges.into_iter().filter(|&i| label[i] != 0).collect();
    if !away.is_empty() {
        away.push_back(MARKER);
        while let Some(i) = away.pop_front() {
            if i == MARKER {
                if away.is_empty() {
                    break;
                }
                level += 1;
                away.push_back(MARKER);
                continue;
            }
            if mask[i] != 0 {
                continue;
            }
            mask[i] = level;
            flat_height[label[i] as usize] = level;
            let (r, c) = (i / cols, i % cols);
            for &(dr, dc) in &D8_OFFSETS {
                let (nr, nc) = (r as isize + dr, c as isize + dc);
                if !in_bounds(nr, nc) {
                    continue;
                }
                let ni = idx(nr as usize, nc as usize);
                if label[ni] == label[i] && dir[ni] == 0 && mask[ni] == 0 {
                    away.push_back(ni);
                }
            }
        }
    }

    // Pass 4 — gradient towards lower terrain, combined with pass 3
    // (Barnes alg. 6). The away-values are negated, then a level-order BFS
    // from the low edges folds them into the final potential:
    //   mask = (flat_height - away_level) + 2 · towards_level
    // The doubled towards-term dominates, guaranteeing every flat cell a
    // strictly descending path to an outlet; the away-term bends that flow
    // around the flat's higher rim (the Garbrecht–Martz signature).
    for i in 0..n {
        if label[i] != 0 {
            mask[i] = -mask[i];
        }
    }
    let mut towards = low_edges;
    level = 1;
    towards.push_back(MARKER);
    while let Some(i) = towards.pop_front() {
        if i == MARKER {
            if towards.is_empty() {
                break;
            }
            level += 1;
            towards.push_back(MARKER);
            continue;
        }
        if mask[i] > 0 {
            continue;
        }
        mask[i] = if mask[i] < 0 {
            flat_height[label[i] as usize] + mask[i] + 2 * level
        } else {
            2 * level
        };
        let (r, c) = (i / cols, i % cols);
        for &(dr, dc) in &D8_OFFSETS {
            let (nr, nc) = (r as isize + dr, c as isize + dc);
            if !in_bounds(nr, nc) {
                continue;
            }
            let ni = idx(nr as usize, nc as usize);
            if label[ni] == label[i] && mask[ni] <= 0 {
                towards.push_back(ni);
            }
        }
    }

    // Pass 5 — D8 over the potential surface. Same-elevation draining
    // neighbors (the outlets) act as potential 0, so low-edge cells exit the
    // flat; everyone else follows the steepest in-flat descent.
    let out = directions
        .data_mut()
        .as_slice_mut()
        .expect("direction raster is contiguous");
    let mut resolved = 0usize;
    for i in 0..n {
        if label[i] == 0 {
            continue;
        }
        let (r, c) = (i / cols, i % cols);
        let e = elev_at(i);
        let mut best_dir = 0u8;
        let mut best_grad = 0.0f64;
        for (k, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
            let (nr, nc) = (r as isize + dr, c as isize + dc);
            if !in_bounds(nr, nc) {
                continue;
            }
            let ni = idx(nr as usize, nc as usize);
            let ne = elev_at(ni);
            if dem.is_nodata(ne) {
                continue;
            }
            let neighbor_potential = if label[ni] == label[i] {
                mask[ni]
            } else if dir[ni] != 0 && ne == e {
                0 // outlet: same elevation, already drains
            } else {
                continue;
            };
            let grad = (mask[i] - neighbor_potential) as f64 / D8_DISTANCE[k];
            if grad > best_grad {
                best_grad = grad;
                best_dir = (k + 1) as u8;
            }
        }
        if best_dir != 0 {
            out[i] = best_dir;
            resolved += 1;
        }
    }

    Ok(FlatResolutionStats {
        flats: n_labels as usize,
        resolved_cells: resolved,
        unresolved_cells: undrained - resolved,
    })
}

#[cfg(test)]
mod tests {
    use super::super::d8;
    use super::super::flow_direction::flow_direction;
    use super::super::priority_flood::priority_flood_flat;
    use super::*;
    use surtgis_core::GeoTransform;

    fn raster(rows: usize, cols: usize, fill: f64) -> Raster<f64> {
        let mut dem = Raster::new(rows, cols);
        dem.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for r in 0..rows {
            for c in 0..cols {
                dem.set(r, c, fill).unwrap();
            }
        }
        dem
    }

    /// Follow D8 codes downstream until a cell with no outflow; panics on
    /// missing directions inside the path or on cycles.
    fn trace_to_sink(fdir: &Raster<u8>, mut r: usize, mut c: usize) -> (usize, usize) {
        let (rows, cols) = fdir.shape();
        for _ in 0..rows * cols + 1 {
            let d = fdir.get(r, c).unwrap();
            if d == 0 {
                return (r, c);
            }
            let (nr, nc) = d8::downstream(r, c, d, rows, cols).expect("direction points off-grid");
            (r, c) = (nr, nc);
        }
        panic!("cycle detected while tracing from a flat cell");
    }

    #[test]
    fn walled_flat_drains_to_single_outlet() {
        // 7×7: high rim at z=20 around a flat floor at z=10, with one notch
        // at (3,6)=5 next to the floor. Every cell must resolve and every
        // flow path must end at the notch.
        let mut dem = raster(7, 7, 20.0);
        for r in 1..6 {
            for c in 1..6 {
                dem.set(r, c, 10.0).unwrap();
            }
        }
        dem.set(3, 6, 5.0).unwrap();

        let fdir = flow_direction(&dem).unwrap();
        for r in 0..7 {
            for c in 0..7 {
                if (r, c) == (3, 6) {
                    assert_eq!(fdir.get(r, c).unwrap(), 0, "outlet cell is the only sink");
                    continue;
                }
                assert_ne!(fdir.get(r, c).unwrap(), 0, "cell ({r},{c}) left unresolved");
                assert_eq!(trace_to_sink(&fdir, r, c), (3, 6), "path from ({r},{c})");
            }
        }
    }

    #[test]
    fn away_from_higher_gradient_bends_flow() {
        // Walled flat basin (rows 1–5 × cols 1–8 at z=10, rim at z=20) with
        // one outlet notch at (6,8)=5. Garbrecht–Martz: the cells along the
        // northern wall must first move away from it (southward component,
        // SW/S/SE) — never run east/west hugging the wall, which is what a
        // towards-lower-only gradient would produce on the western half.
        let mut dem = raster(7, 10, 20.0);
        for r in 1..6 {
            for c in 1..9 {
                dem.set(r, c, 10.0).unwrap();
            }
        }
        dem.set(6, 8, 5.0).unwrap();

        let fdir = flow_direction(&dem).unwrap();
        for c in 1..9 {
            let d = fdir.get(1, c).unwrap();
            assert!(
                (6..=8).contains(&d),
                "cell (1,{c}) should leave the wall southward, got dir {d}"
            );
        }
        // And every basin cell still reaches the outlet.
        for r in 1..6 {
            for c in 1..9 {
                assert_eq!(trace_to_sink(&fdir, r, c), (6, 8), "path from ({r},{c})");
            }
        }
    }

    #[test]
    fn enclosed_flat_stays_unresolved() {
        // Crater: a flat floor fully enclosed by a higher rim has no outlet
        // of any kind (no draining same-elevation neighbor, no boundary
        // contact): nothing must resolve.
        let mut dem = raster(7, 7, 20.0);
        for r in 1..6 {
            for c in 1..6 {
                dem.set(r, c, 10.0).unwrap();
            }
        }
        let mut fdir = flow_direction_raw(&dem);
        let stats = resolve_flats(&dem, &mut fdir).unwrap();
        assert_eq!(stats.flats, 0);
        assert_eq!(stats.resolved_cells, 0);
        assert_eq!(stats.unresolved_cells, 25);
        for r in 1..6 {
            for c in 1..6 {
                assert_eq!(fdir.get(r, c).unwrap(), 0, "crater cell ({r},{c})");
            }
        }
    }

    #[test]
    fn open_flat_drains_toward_border() {
        // A fully flat raster drains off every edge: the border ring stays 0
        // (terminal sinks — directions never point off-grid) and the whole
        // interior is routed toward it.
        let dem = raster(5, 5, 10.0);
        let mut fdir = flow_direction_raw(&dem);
        let stats = resolve_flats(&dem, &mut fdir).unwrap();
        assert_eq!(stats.flats, 1);
        assert_eq!(stats.resolved_cells, 9);
        assert_eq!(stats.unresolved_cells, 16);
        for r in 1..4 {
            for c in 1..4 {
                assert_ne!(fdir.get(r, c).unwrap(), 0, "interior cell ({r},{c})");
                let (sr, sc) = trace_to_sink(&fdir, r, c);
                assert!(
                    sr == 0 || sr == 4 || sc == 0 || sc == 4,
                    "path from ({r},{c}) must end on the border, got ({sr},{sc})"
                );
            }
        }
        for c in 0..5 {
            assert_eq!(fdir.get(0, c).unwrap(), 0, "border sink (0,{c})");
            assert_eq!(fdir.get(4, c).unwrap(), 0, "border sink (4,{c})");
        }
    }

    #[test]
    fn filled_depression_drains_through_spill() {
        // Depression in a bowl → flat-fill (epsilon 0) creates an exactly
        // flat floor. After resolution every filled cell must drain out.
        let mut dem = raster(9, 9, 0.0);
        for r in 0..9 {
            for c in 0..9 {
                // Outer slope draining east, with a 3×3 depression inside
                let z = (9 - c) as f64;
                dem.set(r, c, z).unwrap();
            }
        }
        for r in 3..6 {
            for c in 3..6 {
                dem.set(r, c, 1.0).unwrap(); // pit floor below its rim
            }
        }
        let filled = priority_flood_flat(&dem).unwrap();
        let fdir = flow_direction(&filled).unwrap();
        for r in 3..6 {
            for c in 3..6 {
                assert_ne!(
                    fdir.get(r, c).unwrap(),
                    0,
                    "filled cell ({r},{c}) unresolved"
                );
                let (sr, sc) = trace_to_sink(&fdir, r, c);
                // Every path must leave the depression and reach the east
                // edge of the raster (the regional outlet direction).
                assert_eq!(sc, 8, "path from ({r},{c}) ended at ({sr},{sc})");
            }
        }
    }

    #[test]
    fn nodata_is_a_sink_but_never_a_target() {
        // Open flat with a nodata hole: nodata is drainable terrain (a
        // masked sea or lake) — the flat routes toward it — but no direction
        // may ever point into it. Hole-adjacent and border cells stay 0 as
        // terminal sinks.
        let mut dem = raster(7, 7, 10.0);
        dem.set_nodata(Some(-9999.0));
        dem.set(3, 3, -9999.0).unwrap();

        let fdir = flow_direction(&dem).unwrap();
        let (rows, cols) = fdir.shape();
        for r in 0..7 {
            for c in 0..7 {
                if (r, c) == (3, 3) {
                    continue;
                }
                let d = fdir.get(r, c).unwrap();
                let on_border = r == 0 || r == 6 || c == 0 || c == 6;
                let next_to_hole = r.abs_diff(3) <= 1 && c.abs_diff(3) <= 1;
                if on_border || next_to_hole {
                    assert_eq!(d, 0, "boundary sink ({r},{c}) must stay 0");
                    continue;
                }
                assert_ne!(d, 0, "interior cell ({r},{c}) unresolved");
                let (nr, nc) = d8::downstream(r, c, d, rows, cols).unwrap();
                assert_ne!((nr, nc), (3, 3), "cell ({r},{c}) drains into nodata");
            }
        }
    }

    #[test]
    fn pit_is_not_a_flat() {
        // A single-cell pit has no same-elevation neighbors: it must remain
        // 0 and count as unresolved.
        let mut dem = raster(5, 5, 10.0);
        dem.set(2, 2, 1.0).unwrap();
        let mut fdir = flow_direction_raw(&dem);
        let before = fdir.get(2, 2).unwrap();
        assert_eq!(before, 0);
        let stats = resolve_flats(&dem, &mut fdir).unwrap();
        assert_eq!(fdir.get(2, 2).unwrap(), 0);
        assert_eq!(stats.unresolved_cells, 1);
    }

    #[test]
    fn shape_mismatch_is_an_error() {
        let dem = raster(5, 5, 10.0);
        let mut wrong: Raster<u8> = Raster::new(4, 5);
        assert!(resolve_flats(&dem, &mut wrong).is_err());
    }

    /// Initial D8 without flat resolution, for tests that exercise
    /// `resolve_flats` in isolation.
    fn flow_direction_raw(dem: &Raster<f64>) -> Raster<u8> {
        let (rows, cols) = dem.shape();
        let mut out: Raster<u8> = dem.with_same_meta::<u8>(rows, cols);
        out.set_nodata(Some(0));
        for r in 0..rows {
            for c in 0..cols {
                let e = dem.get(r, c).unwrap();
                if dem.is_nodata(e) {
                    continue;
                }
                let mut best = 0u8;
                let mut max_drop = 0.0;
                for (k, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                    let (nr, nc) = (r as isize + dr, c as isize + dc);
                    if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                        continue;
                    }
                    let ne = dem.get(nr as usize, nc as usize).unwrap();
                    if dem.is_nodata(ne) {
                        continue;
                    }
                    let drop = (e - ne) / D8_DISTANCE[k];
                    if drop > max_drop {
                        max_drop = drop;
                        best = (k + 1) as u8;
                    }
                }
                out.set(r, c, best).unwrap();
            }
        }
        out
    }
}
