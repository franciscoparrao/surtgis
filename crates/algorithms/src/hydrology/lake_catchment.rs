//! Lake catchments: the area draining into a lake, with the lake itself as
//! the outlet.
//!
//! A standard `fill → flow direction → watershed` chain cannot be pointed at
//! a lake: filling erases the lake (it *is* a depression), and a single pour
//! pixel misses everything that drains into the rest of the shoreline. Here
//! the lake is given as a mask, **every lake cell is an outlet**, and the
//! depressions elsewhere are filled with priority-flood, so the catchment is
//! the union of everything whose D8 path ends anywhere in the lake. Lakes
//! upstream of it drain through and belong to its catchment.
//!
//! [`water_body_at`] builds that mask from a water raster (e.g. ESA
//! WorldCover class 80) and a point near the lake, with a morphological
//! opening that keeps a lake from being merged with the river it drains to.

use std::collections::VecDeque;

use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use super::{PriorityFloodParams, WatershedParams, flow_direction, priority_flood, watershed};

/// Parameters for [`lake_catchment`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LakeCatchmentParams {
    /// Priority-flood gradient imposed on filled flats, in elevation
    /// units per cell (so filled areas still drain).
    pub epsilon: f64,
    /// `None` (default): the **connected** catchment — every depression is
    /// filled and spills onward, so a closed basin whose overflow reaches
    /// the lake belongs to it (the HydroSHEDS/HydroLAKES convention).
    /// `Some(d)`: the **local** catchment — depressions at least `d` deep
    /// (elevation units) are kept as sinks of their own, so what drains
    /// into them is excluded; shallower ones (DEM noise) are filled. In
    /// permafrost lake districts the two can differ by orders of magnitude.
    pub sink_depth: Option<f64>,
}

impl Default for LakeCatchmentParams {
    fn default() -> Self {
        Self {
            epsilon: 1e-4,
            sink_depth: None,
        }
    }
}

/// Codes of [`LakeCatchment::mask`].
pub mod mask_code {
    /// Outside the catchment.
    pub const OUTSIDE: u8 = 0;
    /// Land that drains into the lake.
    pub const CATCHMENT: u8 = 1;
    /// The lake.
    pub const LAKE: u8 = 2;
    /// No DEM data.
    pub const NODATA: u8 = 255;
}

/// Result of [`lake_catchment`]. Areas and lengths are in map units of
/// the DEM (square metres and metres on a projected grid).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LakeCatchment {
    /// [`mask_code`] per cell, on the DEM grid.
    pub mask: Raster<u8>,
    /// Cells of the lake.
    pub lake_cells: usize,
    /// Cells of the catchment, lake included.
    pub catchment_cells: usize,
    /// Lake area.
    pub lake_area: f64,
    /// Catchment area, lake included.
    pub catchment_area: f64,
    /// Lake perimeter: cell edges between the lake and anything else.
    pub lake_perimeter: f64,
    /// Median DEM elevation over the lake (lake surfaces are noisy in
    /// global DEMs, so the median, not the first cell).
    pub lake_elevation: f64,
    /// Mean DEM elevation over the catchment's land cells.
    pub catchment_mean_elevation: f64,
    /// The catchment reaches the grid edge or a nodata cell: the window
    /// was too small and the areas are lower bounds.
    pub touches_edge: bool,
    /// The lake itself reaches the grid edge or nodata.
    pub lake_touches_edge: bool,
    /// Depressions kept as sinks of their own (`sink_depth` set).
    pub sinks: usize,
}

/// Delineate the catchment of the lake marked by the non-zero cells of
/// `lake` (same grid as `dem`).
///
/// # Errors
/// Shape mismatch, an empty lake mask, or a lake made only of nodata.
pub fn lake_catchment(
    dem: &Raster<f64>,
    lake: &Raster<u8>,
    params: LakeCatchmentParams,
) -> Result<LakeCatchment> {
    let (rows, cols) = dem.shape();
    if lake.shape() != (rows, cols) {
        return Err(Error::Other(format!(
            "lake mask is {:?}, DEM is {:?}; they must share one grid",
            lake.shape(),
            (rows, cols)
        )));
    }
    let z = |r: usize, c: usize| dem.data()[(r, c)];
    let valid = |r: usize, c: usize| {
        let v = z(r, c);
        v.is_finite() && !dem.nodata().is_some_and(|nd| v == nd)
    };
    let is_lake = |r: usize, c: usize| lake.data()[(r, c)] != 0;

    let mut outlets = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if is_lake(r, c) {
                outlets.push((r, c));
            }
        }
    }
    if outlets.is_empty() {
        return Err(Error::Other("lake mask has no lake cell".into()));
    }
    let mut lake_z: Vec<f64> = outlets
        .iter()
        .filter(|&&(r, c)| valid(r, c))
        .map(|&(r, c)| z(r, c))
        .collect();
    if lake_z.is_empty() {
        return Err(Error::Other("the lake lies entirely on DEM nodata".into()));
    }
    lake_z.sort_by(|a, b| a.total_cmp(b));
    let lake_elevation = lake_z[lake_z.len() / 2];

    // Condition the DEM with the lake as an outlet: lake cells are nodata
    // for the fill, which priority-flood treats as a drain (like the sea),
    // so a lake sitting in a closed basin is not filled up to the basin's
    // spill level (that would route the basin away from the lake). Then
    // the lake is put below everything so D8 drains into it.
    let mut cond = dem.clone();
    for &(r, c) in &outlets {
        cond.data_mut()[(r, c)] = f64::NAN;
    }
    let fill = |c: &Raster<f64>| {
        priority_flood(
            c,
            PriorityFloodParams {
                epsilon: params.epsilon,
            },
        )
    };
    let mut filled = fill(&cond)?;
    let mut sinks: Vec<(usize, usize)> = Vec::new();
    if let Some(min_depth) = params.sink_depth {
        // Depressions = connected cells the fill raised; the deepest cell of
        // each one at least `min_depth` deep becomes an outlet of its own,
        // and the DEM is filled again with those outlets.
        let raised = |r: usize, c: usize| {
            let (f, d) = (filled.data()[(r, c)], dem.data()[(r, c)]);
            f.is_finite() && d.is_finite() && f - d > params.epsilon * 0.5 + 1e-9
        };
        let mut seen = vec![false; rows * cols];
        for r0 in 0..rows {
            for c0 in 0..cols {
                if seen[r0 * cols + c0] || !raised(r0, c0) {
                    continue;
                }
                let mut q = VecDeque::from([(r0, c0)]);
                seen[r0 * cols + c0] = true;
                let mut deepest = ((r0, c0), 0.0f64);
                while let Some((r, c)) = q.pop_front() {
                    let depth = filled.data()[(r, c)] - dem.data()[(r, c)];
                    if depth > deepest.1 {
                        deepest = ((r, c), depth);
                    }
                    for dr in -1isize..=1 {
                        for dc in -1isize..=1 {
                            let (nr, nc) = (r as isize + dr, c as isize + dc);
                            if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                                continue;
                            }
                            let (nr, nc) = (nr as usize, nc as usize);
                            if !seen[nr * cols + nc] && raised(nr, nc) {
                                seen[nr * cols + nc] = true;
                                q.push_back((nr, nc));
                            }
                        }
                    }
                }
                if deepest.1 >= min_depth {
                    sinks.push(deepest.0);
                }
            }
        }
        if !sinks.is_empty() {
            for &(r, c) in &sinks {
                cond.data_mut()[(r, c)] = f64::NAN;
            }
            filled = fill(&cond)?;
        }
    }
    let floor = filled
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    let sink = if floor.is_finite() { floor - 1.0 } else { 0.0 };
    for &(r, c) in outlets.iter().chain(sinks.iter()) {
        filled.data_mut()[(r, c)] = sink;
    }
    let fdir = flow_direction(&filled)?;
    let n_outlets = outlets.len();
    let basins = watershed(
        &fdir,
        WatershedParams {
            pour_points: outlets,
        },
    )?;

    // Edge = grid border or a nodata neighbour (reprojected windows have
    // nodata margins).
    let at_edge = |r: usize, c: usize| -> bool {
        if r == 0 || c == 0 || r + 1 == rows || c + 1 == cols {
            return true;
        }
        !(valid(r - 1, c) && valid(r + 1, c) && valid(r, c - 1) && valid(r, c + 1))
    };

    let mut mask = dem.with_same_meta::<u8>(rows, cols);
    mask.set_nodata(Some(mask_code::NODATA));
    let (mut catchment_cells, mut touches_edge, mut lake_touches_edge) = (0usize, false, false);
    let (mut zsum, mut zn, mut edges) = (0.0f64, 0usize, 0usize);
    for r in 0..rows {
        for c in 0..cols {
            let code = if is_lake(r, c) {
                if at_edge(r, c) {
                    lake_touches_edge = true;
                }
                for (dr, dc) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let (nr, nc) = (r as isize + dr, c as isize + dc);
                    if nr < 0
                        || nc < 0
                        || nr >= rows as isize
                        || nc >= cols as isize
                        || !is_lake(nr as usize, nc as usize)
                    {
                        edges += 1;
                    }
                }
                mask_code::LAKE
            } else if basins.data()[(r, c)] > 0 && valid(r, c) {
                zsum += z(r, c);
                zn += 1;
                mask_code::CATCHMENT
            } else if valid(r, c) {
                mask_code::OUTSIDE
            } else {
                mask_code::NODATA
            };
            if code == mask_code::LAKE || code == mask_code::CATCHMENT {
                catchment_cells += 1;
                if code == mask_code::CATCHMENT && at_edge(r, c) {
                    touches_edge = true;
                }
            }
            mask.data_mut()[(r, c)] = code;
        }
    }
    let t = dem.transform();
    let cell_area = (t.pixel_width * t.pixel_height).abs();
    Ok(LakeCatchment {
        mask,
        lake_cells: n_outlets,
        catchment_cells,
        lake_area: n_outlets as f64 * cell_area,
        catchment_area: catchment_cells as f64 * cell_area,
        lake_perimeter: edges as f64 * t.pixel_width.abs(),
        lake_elevation,
        catchment_mean_elevation: if zn > 0 { zsum / zn as f64 } else { f64::NAN },
        touches_edge,
        lake_touches_edge,
        sinks: sinks.len(),
    })
}

/// Parameters for [`water_body_at`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WaterBodyParams {
    /// Search radius, in cells, for the nearest water cell to the point.
    pub search_radius: usize,
    /// Radius, in cells, of the square opening applied before taking the
    /// connected component: water necks up to about `2 * opening_radius`
    /// cells wide (a river leaving the lake) are cut. 0 disables it.
    pub opening_radius: usize,
}

impl Default for WaterBodyParams {
    fn default() -> Self {
        Self {
            search_radius: 10,
            opening_radius: 2,
        }
    }
}

/// A water body extracted by [`water_body_at`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WaterBody {
    /// 1 on the water body, 0 elsewhere.
    pub mask: Raster<u8>,
    /// Cells of the water body.
    pub cells: usize,
    /// Cells of the raw (unopened) connected component it belongs to.
    /// Much larger than `cells` means the opening cut it from a river or
    /// from another part of the same lake — worth a look.
    pub raw_component_cells: usize,
}

/// The water body nearest to `(row, col)` in `water` (non-zero = water):
/// the 8-connected component of the opened water mask closest to the
/// point (within `search_radius`), grown back into the raw water by the
/// opening radius so its shoreline is not eroded. `None` when no water
/// cell lies within the search radius.
pub fn water_body_at(
    water: &Raster<u8>,
    row: usize,
    col: usize,
    params: WaterBodyParams,
) -> Option<WaterBody> {
    let (rows, cols) = water.shape();
    let raw: Vec<bool> = water.data().iter().map(|&v| v != 0).collect();
    let r_open = params.opening_radius;
    let opened = if r_open > 0 {
        let eroded = square_filter(&raw, rows, cols, r_open, true);
        let o = square_filter(&eroded, rows, cols, r_open, false);
        if o.iter().any(|&b| b) { o } else { raw.clone() }
    } else {
        raw.clone()
    };

    let seed = nearest_true(&opened, rows, cols, row, col, params.search_radius)
        .or_else(|| nearest_true(&raw, rows, cols, row, col, params.search_radius))?;
    let use_opened = opened[seed.0 * cols + seed.1];
    let base = if use_opened { &opened } else { &raw };
    let mut body = flood(base, rows, cols, seed);
    if use_opened && r_open > 0 {
        // Grow back into the raw water by the opening radius.
        let grown = square_filter(&body, rows, cols, r_open, false);
        for i in 0..rows * cols {
            body[i] = body[i] || (grown[i] && raw[i]);
        }
    }
    let raw_component = flood(&raw, rows, cols, seed);

    let mut mask = water.with_same_meta::<u8>(rows, cols);
    mask.set_nodata(None);
    let mut cells = 0;
    for (i, v) in mask.data_mut().iter_mut().enumerate() {
        *v = u8::from(body[i]);
        cells += usize::from(body[i]);
    }
    Some(WaterBody {
        mask,
        cells,
        raw_component_cells: raw_component.iter().filter(|&&b| b).count(),
    })
}

/// Square erosion (`all`) or dilation (any) of radius `r`, separable, O(n).
fn square_filter(src: &[bool], rows: usize, cols: usize, r: usize, erode: bool) -> Vec<bool> {
    let pass = |get: &dyn Fn(usize, usize) -> bool, n_lines: usize, len: usize| -> Vec<bool> {
        let mut out = vec![false; n_lines * len];
        for line in 0..n_lines {
            // prefix count of `true` along the line
            let mut pre = vec![0usize; len + 1];
            for i in 0..len {
                pre[i + 1] = pre[i] + usize::from(get(line, i));
            }
            for i in 0..len {
                let lo = i.saturating_sub(r);
                let hi = (i + r + 1).min(len);
                let ones = pre[hi] - pre[lo];
                // Outside the grid counts as not-water for erosion.
                let full = 2 * r + 1;
                out[line * len + i] = if erode { ones == full } else { ones > 0 };
            }
        }
        out
    };
    let rowpass = pass(&|r_, c| src[r_ * cols + c], rows, cols);
    let colpass = pass(&|c, r_| rowpass[r_ * cols + c], cols, rows);
    // colpass is laid out [col][row]; transpose back
    let mut out = vec![false; rows * cols];
    for c in 0..cols {
        for r_ in 0..rows {
            out[r_ * cols + c] = colpass[c * rows + r_];
        }
    }
    out
}

fn nearest_true(
    m: &[bool],
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
    radius: usize,
) -> Option<(usize, usize)> {
    let mut best: Option<((usize, usize), usize)> = None;
    let r0 = row.saturating_sub(radius);
    let r1 = (row + radius).min(rows.saturating_sub(1));
    let c0 = col.saturating_sub(radius);
    let c1 = (col + radius).min(cols.saturating_sub(1));
    for r in r0..=r1 {
        for c in c0..=c1 {
            if m[r * cols + c] {
                let d = r.abs_diff(row).pow(2) + c.abs_diff(col).pow(2);
                if best.is_none_or(|(_, bd)| d < bd) {
                    best = Some(((r, c), d));
                }
            }
        }
    }
    best.map(|(p, _)| p)
}

fn flood(m: &[bool], rows: usize, cols: usize, seed: (usize, usize)) -> Vec<bool> {
    let mut seen = vec![false; rows * cols];
    if !m[seed.0 * cols + seed.1] {
        return seen;
    }
    let mut q = VecDeque::from([seed]);
    seen[seed.0 * cols + seed.1] = true;
    while let Some((r, c)) = q.pop_front() {
        for dr in -1isize..=1 {
            for dc in -1isize..=1 {
                let (nr, nc) = (r as isize + dr, c as isize + dc);
                if (dr, dc) == (0, 0)
                    || nr < 0
                    || nc < 0
                    || nr >= rows as isize
                    || nc >= cols as isize
                {
                    continue;
                }
                let i = nr as usize * cols + nc as usize;
                if m[i] && !seen[i] {
                    seen[i] = true;
                    q.push_back((nr as usize, nc as usize));
                }
            }
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    /// Two valleys separated by a ridge at column 20; a lake in the left
    /// valley; a second, upstream lake that drains into it.
    fn terrain() -> (Raster<f64>, Raster<u8>) {
        let (rows, cols) = (40, 40);
        let mut dem: Raster<f64> = Raster::new(rows, cols);
        dem.set_transform(GeoTransform::new(0.0, 1200.0, 30.0, -30.0));
        for r in 0..rows {
            for c in 0..cols {
                // Left valley axis at c = 6, right valley axis at c = 33,
                // divide at c = 20; both valleys fall to the south; small
                // noise so D8 is not degenerate.
                let cross = if c <= 20 {
                    (c as f64 - 6.0).abs()
                } else {
                    (c as f64 - 33.0).abs()
                } * 1.5;
                let z = cross - r as f64 * 0.5 + ((r * 7 + c * 3) % 5) as f64 * 0.01;
                dem.set(r, c, 100.0 + z).unwrap();
            }
        }
        // lake: a pit in the left valley
        let mut lake: Raster<u8> = Raster::new(rows, cols);
        lake.set_transform(*dem.transform());
        for r in 28..32 {
            for c in 4..9 {
                dem.set(r, c, 70.0).unwrap();
                lake.set(r, c, 1).unwrap();
            }
        }
        (dem, lake)
    }

    #[test]
    fn lake_is_the_outlet_and_the_ridge_bounds_the_catchment() {
        let (dem, lake) = terrain();
        let lc = lake_catchment(&dem, &lake, LakeCatchmentParams::default()).unwrap();
        assert_eq!(lc.lake_cells, 20);
        assert!((lc.lake_area - 20.0 * 900.0).abs() < 1e-6);
        assert!((lc.lake_elevation - 70.0).abs() < 1e-9);
        // nothing east of the ridge drains into the lake
        for r in 0..40 {
            for c in 22..40 {
                assert_ne!(
                    lc.mask.get(r, c).unwrap(),
                    mask_code::CATCHMENT,
                    "({r},{c})"
                );
            }
        }
        // the cells straight upslope of the lake drain into it
        assert_eq!(lc.mask.get(10, 6).unwrap(), mask_code::CATCHMENT);
        assert!(lc.catchment_cells > lc.lake_cells);
        // perimeter of a 4 x 5 block: 18 cell edges
        assert!((lc.lake_perimeter - 18.0 * 30.0).abs() < 1e-9);
    }

    #[test]
    fn filling_does_not_erase_the_lake() {
        // The lake is a closed pit: a plain fill would flatten it away.
        let (dem, lake) = terrain();
        let lc = lake_catchment(&dem, &lake, LakeCatchmentParams::default()).unwrap();
        for r in 28..32 {
            for c in 4..9 {
                assert_eq!(lc.mask.get(r, c).unwrap(), mask_code::LAKE);
            }
        }
    }

    #[test]
    fn edge_and_nodata_are_reported() {
        let (mut dem, lake) = terrain();
        let lc = lake_catchment(&dem, &lake, LakeCatchmentParams::default()).unwrap();
        assert!(lc.touches_edge, "the valley reaches the north edge");
        // a nodata collar next to the lake counts as edge for the lake
        dem.set(27, 6, f64::NAN).unwrap();
        let lc = lake_catchment(&dem, &lake, LakeCatchmentParams::default()).unwrap();
        assert!(lc.lake_touches_edge);
    }

    #[test]
    fn errors_on_empty_lake_and_shape_mismatch() {
        let (dem, _) = terrain();
        let empty: Raster<u8> = Raster::new(40, 40);
        assert!(lake_catchment(&dem, &empty, LakeCatchmentParams::default()).is_err());
        let wrong: Raster<u8> = Raster::new(39, 40);
        assert!(lake_catchment(&dem, &wrong, LakeCatchmentParams::default()).is_err());
    }

    #[test]
    fn water_body_cuts_the_river_but_keeps_the_shore() {
        // 12 x 12 lake with a 2-cell-wide river leaving east.
        let (rows, cols) = (30, 40);
        let mut w: Raster<u8> = Raster::new(rows, cols);
        for r in 9..21 {
            for c in 4..16 {
                w.set(r, c, 1).unwrap();
            }
        }
        for r in 14..16 {
            for c in 16..40 {
                w.set(r, c, 1).unwrap();
            }
        }
        let wb = water_body_at(&w, 15, 10, WaterBodyParams::default()).unwrap();
        // the full lake, the river's first cells (regrown) but not the river
        for r in 9..21 {
            for c in 4..16 {
                assert_eq!(wb.mask.get(r, c).unwrap(), 1, "lake cell ({r},{c})");
            }
        }
        assert_eq!(wb.mask.get(14, 30).unwrap(), 0, "river cut");
        assert!(wb.cells < 12 * 12 + 2 * 4);
        assert_eq!(wb.raw_component_cells, 144 + 48);
        // no opening: the river comes along
        let p = WaterBodyParams {
            opening_radius: 0,
            ..Default::default()
        };
        let all = water_body_at(&w, 15, 10, p).unwrap();
        assert_eq!(all.cells, 144 + 48);
        // nothing near the point
        let far = WaterBodyParams {
            search_radius: 2,
            ..Default::default()
        };
        assert!(water_body_at(&w, 2, 38, far).is_none());
    }

    #[test]
    fn square_filter_matches_brute_force() {
        let (rows, cols) = (17, 23);
        let src: Vec<bool> = (0..rows * cols).map(|i| (i * 37 % 11) < 7).collect();
        for &erode in &[true, false] {
            let fast = square_filter(&src, rows, cols, 2, erode);
            for r in 0..rows as isize {
                for c in 0..cols as isize {
                    let mut all = true;
                    let mut any = false;
                    for dr in -2..=2isize {
                        for dc in -2..=2isize {
                            let (nr, nc) = (r + dr, c + dc);
                            let v = nr >= 0
                                && nc >= 0
                                && nr < rows as isize
                                && nc < cols as isize
                                && src[nr as usize * cols + nc as usize];
                            all &= v;
                            any |= v;
                        }
                    }
                    let want = if erode { all } else { any };
                    assert_eq!(
                        fast[r as usize * cols + c as usize],
                        want,
                        "({r},{c}) erode={erode}"
                    );
                }
            }
        }
    }
    #[test]
    fn lake_in_a_closed_basin_drains_its_whole_basin() {
        // A crater: rim at 15 cells from the centre, lake pit at the centre.
        // Filling the basin to its spill level must not steal it from the lake.
        let n = 41usize;
        let mut dem: Raster<f64> = Raster::new(n, n);
        dem.set_transform(GeoTransform::new(0.0, n as f64 * 30.0, 30.0, -30.0));
        let mut lake: Raster<u8> = Raster::new(n, n);
        lake.set_transform(*dem.transform());
        for r in 0..n {
            for c in 0..n {
                let d = ((r as f64 - 20.0).powi(2) + (c as f64 - 20.0).powi(2)).sqrt();
                dem.set(
                    r,
                    c,
                    100.0 - (d - 15.0).abs() + ((r * 7 + c * 3) % 5) as f64 * 1e-3,
                )
                .unwrap();
                if d <= 2.0 {
                    lake.set(r, c, 1).unwrap();
                }
            }
        }
        let lc = lake_catchment(&dem, &lake, LakeCatchmentParams::default()).unwrap();
        // Everything inside the rim (d < 14) drains into the lake.
        for r in 0..n {
            for c in 0..n {
                let d = ((r as f64 - 20.0).powi(2) + (c as f64 - 20.0).powi(2)).sqrt();
                let code = lc.mask.get(r, c).unwrap();
                if d < 14.0 {
                    assert!(
                        code == mask_code::CATCHMENT || code == mask_code::LAKE,
                        "({r},{c}) d={d:.1}"
                    );
                } else if d > 16.5 {
                    assert_eq!(code, mask_code::OUTSIDE, "({r},{c}) d={d:.1}");
                }
            }
        }
        assert!(!lc.touches_edge);
    }
    #[test]
    fn local_catchment_excludes_deep_side_depressions() {
        // Tilted plane draining south into the lake, with a 3 m deep pit
        // upslope: connected, the pit spills into the lake; local
        // (sink_depth 1 m), its basin is its own.
        let n = 41usize;
        let mut dem: Raster<f64> = Raster::new(n, n);
        dem.set_transform(GeoTransform::new(0.0, n as f64 * 30.0, 30.0, -30.0));
        let mut lake: Raster<u8> = Raster::new(n, n);
        lake.set_transform(*dem.transform());
        for r in 0..n {
            for c in 0..n {
                let cross = (c as f64 - 20.0).abs() * 0.3;
                dem.set(r, c, 100.0 - r as f64 * 0.5 + cross).unwrap();
            }
        }
        for r in 34..38 {
            for c in 18..23 {
                lake.set(r, c, 1).unwrap();
                dem.set(r, c, 80.0).unwrap();
            }
        }
        // pit at (10, 20): 3 m below its surroundings, 5 x 5 cells
        for r in 8..13 {
            for c in 18..23 {
                let z = dem.get(r, c).unwrap();
                dem.set(r, c, z - 3.0).unwrap();
            }
        }
        let connected = lake_catchment(&dem, &lake, LakeCatchmentParams::default()).unwrap();
        let p = LakeCatchmentParams {
            sink_depth: Some(1.0),
            ..Default::default()
        };
        let local = lake_catchment(&dem, &lake, p).unwrap();
        assert_eq!(connected.sinks, 0);
        assert!(local.sinks >= 1);
        assert_ne!(
            connected.mask.get(10, 20).unwrap(),
            mask_code::OUTSIDE,
            "pit spills into the lake"
        );
        assert_eq!(
            local.mask.get(10, 20).unwrap(),
            mask_code::OUTSIDE,
            "pit is its own sink"
        );
        assert!(local.catchment_cells < connected.catchment_cells);
        // cells between the pit and the lake still drain into the lake
        assert_eq!(local.mask.get(25, 20).unwrap(), mask_code::CATCHMENT);
    }
}
