//! LAHARZ semi-empirical lahar / debris-flow inundation model
//! (Iverson, Schilling & Vallance 1998; Schilling 1998).
//!
//! LAHARZ delineates a flow-confined inundation hazard zone from two
//! volume-scaled areas calibrated from documented events:
//!
//! ```text
//! A = c_A · V^(2/3)     cross-sectional inundation area (valley-perpendicular)
//! B = c_B · V^(2/3)     planimetric (map-view) inundation area
//! ```
//!
//! Starting from one or more user-supplied sources on the drainage, the model
//! marches downstream along the D8 flow path. At each step it raises an
//! inundation stage on the valley-perpendicular transect until the wetted
//! cross-section reaches `A`, marking the flooded cells and their depth. It
//! accumulates the planimetric area of newly flooded cells and stops once that
//! area reaches `B`. The result is a downstream-tapering inundation footprint;
//! with several sources the footprint is their union.
//!
//! Coefficient pairs `(c_A, c_B)` are flow-type specific (Griswold & Iverson
//! 2008): see [`LaharzFlowType`].
//!
//! **Seeding matters.** Seed *proximal channel cells*, not the summit: a summit
//! cell's D8 steepest descent often runs down the wrong side of a multi-drainage
//! edifice (the steepest line from the peak need not coincide with the channels
//! that actually carry the lahars). Pick source cells on the proximal reaches of
//! the target drainages — for example channel heads inside the energy cone you
//! already compute ([`crate::hydrology::energy_cone`]) — and pass them all.
//!
//! Scope / known limitation: single-path-per-source, D8-confined, with a stepped
//! valley-perpendicular transect. It captures the LAHARZ volume scaling and
//! cross-section filling but does not split flow at distributary junctions or
//! reproduce the original ArcInfo proximal-hazard-zone construction. Note also
//! that in steep, confined channels the cross-section fills mostly *vertically*,
//! so small-to-moderate volumes can produce a long, narrow (near-thalweg)
//! ribbon; the runout of the lowest volumes is the least trustworthy output and
//! is a target for coefficient/cross-section calibration against documented
//! events.
//!
//! # References
//! - Iverson, R.M., Schilling, S.P., Vallance, J.W. (1998). Objective
//!   delineation of lahar-inundation hazard zones. *GSA Bulletin* 110, 972–984.
//! - Griswold, J.P. & Iverson, R.M. (2008). Mobility statistics and automated
//!   hazard mapping for debris flows and rock avalanches. *USGS SIR 2007-5276*.

use std::collections::HashSet;

use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// D8 neighbour offsets for codes 1..=8 (matches `flow_direction`).
const D8_OFFSETS: [(isize, isize); 8] = [
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// Flow type, selecting the `(c_A, c_B)` mobility coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaharzFlowType {
    /// Lahars: `A = 0.05 V^2/3`, `B = 200 V^2/3` (Iverson et al. 1998).
    Lahar,
    /// Debris flows: `A = 0.1 V^2/3`, `B = 20 V^2/3` (Griswold & Iverson 2008).
    DebrisFlow,
    /// Rock avalanches: `A = 0.2 V^2/3`, `B = 20 V^2/3` (Griswold & Iverson 2008).
    RockAvalanche,
}

impl LaharzFlowType {
    /// Return the `(c_A, c_B)` coefficient pair.
    pub fn coefficients(self) -> (f64, f64) {
        match self {
            LaharzFlowType::Lahar => (0.05, 200.0),
            LaharzFlowType::DebrisFlow => (0.1, 20.0),
            LaharzFlowType::RockAvalanche => (0.2, 20.0),
        }
    }

    /// Default lateral-spread aspect ratio for the preset. The lahar value is
    /// calibrated against documented Nevados de Chillán events (Ñuble fixture);
    /// the others default to `0` (canonical) pending their own calibration.
    pub fn default_spread_aspect(self) -> f64 {
        match self {
            // Calibrated on the Diguillín (Nevados de Chillán) fixture: with the
            // valley-confined cross-section this brings 1e6 m³ runout from ~65 km
            // to ~6 km (the documented 2020/2021 proximal lahars deposited within
            // 1–5 km) and 1e7 to ~12 km (observed ~15). The largest volumes
            // (≥1e8) remain ~30 % long where the valley confines the spread.
            LaharzFlowType::Lahar => 1000.0,
            LaharzFlowType::DebrisFlow => 0.0,
            LaharzFlowType::RockAvalanche => 0.0,
        }
    }
}

/// Parameters for [`laharz`].
#[derive(Debug, Clone)]
pub struct LaharzParams {
    /// Source cells `(row, col)` on the drainage. **Seed proximal channel cells,
    /// not the summit**: a summit cell's D8 steepest descent often runs down the
    /// wrong drainage for a multi-drainage edifice. Several sources route as
    /// independent flows and the footprint is their union.
    pub sources: Vec<(usize, usize)>,
    /// Flow volume in m³ (applied to each source).
    pub volume_m3: f64,
    /// Cross-sectional area coefficient `c_A` (overrides the flow-type default
    /// when set via [`LaharzParams::from_flow_type`] you get the preset).
    pub coeff_cross: f64,
    /// Planimetric area coefficient `c_B`.
    pub coeff_plan: f64,
    /// Minimum cross-section **width-to-mean-depth aspect ratio** (lateral
    /// spread). `0` reproduces the canonical fill-to-area-A behaviour, which in
    /// confined channels produces a deep, near-thalweg thread and
    /// unphysically long small-volume runout. With `aspect > 0`, when the
    /// natural cross-section is narrower than `w = sqrt(aspect · A)`, the flow
    /// is spread laterally across the valley floor (mantling up to the local
    /// ridge crests) to that width, shortening the runout. Calibrated against
    /// documented Nevados de Chillán lahars (see the flow-type presets).
    pub spread_aspect: f64,
    /// Safety cap on the number of downstream path cells visited per source.
    pub max_path_cells: usize,
}

impl LaharzParams {
    /// Build parameters from a flow-type preset, with one or more sources.
    pub fn from_flow_type(
        sources: Vec<(usize, usize)>,
        volume_m3: f64,
        flow_type: LaharzFlowType,
    ) -> Self {
        let (c_a, c_b) = flow_type.coefficients();
        Self {
            sources,
            volume_m3,
            coeff_cross: c_a,
            coeff_plan: c_b,
            spread_aspect: flow_type.default_spread_aspect(),
            max_path_cells: 100_000,
        }
    }
}

/// Compute a LAHARZ inundation-depth raster from a source down a D8 flow path.
///
/// Returns flow depth (stage − ground) in metres where inundated, `0` elsewhere,
/// and `NaN` on nodata cells.
///
/// # Errors
/// Errors on a non-positive volume/coefficients/cell size, a source outside the
/// grid or on nodata.
pub fn laharz(
    dem: &Raster<f64>,
    flow_dir: &Raster<u8>,
    params: LaharzParams,
) -> Result<Raster<f64>> {
    if params.volume_m3 <= 0.0 {
        return Err(Error::Other("volume_m3 must be positive".into()));
    }
    if params.coeff_cross <= 0.0 || params.coeff_plan <= 0.0 {
        return Err(Error::Other("coefficients must be positive".into()));
    }
    let cs = dem.cell_size();
    if cs <= 0.0 {
        return Err(Error::Other("cell size must be positive".into()));
    }

    let (rows, cols) = dem.shape();
    let (fr, fc) = dem.shape();
    if flow_dir.shape() != (fr, fc) {
        return Err(Error::SizeMismatch {
            er: rows,
            ec: cols,
            ar: flow_dir.shape().0,
            ac: flow_dir.shape().1,
        });
    }
    let nodata = dem.nodata();
    let is_nd = |v: f64| v.is_nan() || nodata.map(|nd| v == nd).unwrap_or(false);
    if params.sources.is_empty() {
        return Err(Error::Other("at least one source is required".into()));
    }
    for &(sr, sc) in &params.sources {
        if sr >= rows || sc >= cols {
            return Err(Error::Other(format!(
                "source ({sr}, {sc}) is outside the {rows}x{cols} grid"
            )));
        }
        if is_nd(unsafe { dem.get_unchecked(sr, sc) }) {
            return Err(Error::Other(format!(
                "source ({sr}, {sc}) falls on a nodata cell"
            )));
        }
    }

    let two_thirds = params.volume_m3.powf(2.0 / 3.0);
    let a_cross = params.coeff_cross * two_thirds;
    let b_plan = params.coeff_plan * two_thirds;
    let cell_area = cs * cs;

    // Inundation depth output (shared across sources; combined by max depth).
    let mut depth = dem.with_same_meta::<f64>(rows, cols);
    depth.set_nodata(Some(f64::NAN));
    for r in 0..rows {
        for c in 0..cols {
            let z = unsafe { dem.get_unchecked(r, c) };
            depth.set(r, c, if is_nd(z) { f64::NAN } else { 0.0 }).ok();
        }
    }

    let in_bounds = |r: isize, c: isize| r >= 0 && c >= 0 && r < rows as isize && c < cols as isize;

    // Route each source independently — each gets its own planimetric budget B
    // (an independent flow of the given volume) — and union the depths.
    for &(sr, sc) in &params.sources {
        let mut flooded: HashSet<(usize, usize)> = HashSet::new();
        let mut plan_area = 0.0_f64;
        let mut on_path: HashSet<(usize, usize)> = HashSet::new();
        let (mut r, mut c) = (sr, sc);

        for _ in 0..params.max_path_cells {
            if !on_path.insert((r, c)) {
                break; // D8 loop guard
            }
            let z_p = unsafe { dem.get_unchecked(r, c) };
            if is_nd(z_p) {
                break;
            }
            let code = unsafe { flow_dir.get_unchecked(r, c) };

            // Perpendicular step from the downstream direction (rotate ±90°).
            // For a pit/flat (code 0) use a default E-W transect so the terminal
            // cell still receives some inundation.
            let (dr, dc) = if (1..=8).contains(&code) {
                D8_OFFSETS[(code - 1) as usize]
            } else {
                (0, 0)
            };
            let (pr, pc) = if dr == 0 && dc == 0 {
                (0, 1)
            } else {
                (-dc, dr)
            };
            let perp_len = ((pr * pr + pc * pc) as f64).sqrt() * cs;

            // Solve the stage h so the wetted cross-section equals A.
            let h = solve_stage(dem, r, c, z_p, pr, pc, perp_len, a_cross, &is_nd, in_bounds);
            let nat_w = transect_width(dem, r, c, z_p, h, pr, pc, perp_len, &is_nd, in_bounds);

            // Lateral spread: if the natural cross-section is narrower than the
            // target aspect width, mantle the valley floor up to that width
            // (stopping at ridge crests); otherwise fill naturally to area A.
            let target_w = if params.spread_aspect > 0.0 {
                (params.spread_aspect * a_cross).sqrt()
            } else {
                0.0
            };
            if target_w > nat_w {
                spread_transect(
                    dem,
                    &mut depth,
                    &mut flooded,
                    &mut plan_area,
                    cell_area,
                    r,
                    c,
                    z_p,
                    a_cross,
                    target_w,
                    perp_len,
                    pr,
                    pc,
                    &is_nd,
                    in_bounds,
                );
            } else {
                mark_transect(
                    dem,
                    &mut depth,
                    &mut flooded,
                    &mut plan_area,
                    cell_area,
                    r,
                    c,
                    z_p,
                    h,
                    pr,
                    pc,
                    &is_nd,
                    in_bounds,
                );
            }

            if plan_area >= b_plan {
                break;
            }

            // Step downstream.
            if !(1..=8).contains(&code) {
                break; // pit / flat: stop
            }
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if !in_bounds(nr, nc) {
                break;
            }
            r = nr as usize;
            c = nc as usize;
        }
    }

    Ok(depth)
}

/// Wetted cross-sectional area of the transect through `(r, c)` at stage
/// `z_p + h`, summing the centre cell and both perpendicular arms until a wall
/// (cell at or above stage) or the grid edge.
#[allow(clippy::too_many_arguments)]
fn cross_area(
    dem: &Raster<f64>,
    r: usize,
    c: usize,
    z_p: f64,
    h: f64,
    pr: isize,
    pc: isize,
    perp_len: f64,
    is_nd: &impl Fn(f64) -> bool,
    in_bounds: impl Fn(isize, isize) -> bool + Copy,
) -> f64 {
    let stage = z_p + h;
    let mut area = h * perp_len; // centre cell
    for sign in [1isize, -1] {
        let mut k = 1isize;
        loop {
            let rr = r as isize + sign * k * pr;
            let cc = c as isize + sign * k * pc;
            if !in_bounds(rr, cc) {
                break;
            }
            let z = unsafe { dem.get_unchecked(rr as usize, cc as usize) };
            // Confine the cross-section to the valley: a wall (>= stage), a
            // nodata cell, or a cell *below* the thalweg (the perpendicular has
            // left the valley onto an open downslope) ends the arm.
            if is_nd(z) || z >= stage || z < z_p {
                break;
            }
            area += (stage - z) * perp_len;
            k += 1;
        }
    }
    area
}

/// Binary-search the stage `h` whose [`cross_area`] equals `target` (A).
#[allow(clippy::too_many_arguments)]
fn solve_stage(
    dem: &Raster<f64>,
    r: usize,
    c: usize,
    z_p: f64,
    pr: isize,
    pc: isize,
    perp_len: f64,
    target: f64,
    is_nd: &impl Fn(f64) -> bool,
    in_bounds: impl Fn(isize, isize) -> bool + Copy,
) -> f64 {
    // Bracket: grow the upper bound until the area exceeds the target.
    let mut hi = perp_len.max(1.0);
    let mut guard = 0;
    while cross_area(dem, r, c, z_p, hi, pr, pc, perp_len, is_nd, in_bounds) < target {
        hi *= 2.0;
        guard += 1;
        if guard > 200 {
            break; // unbounded (e.g. flat plain): cap the stage
        }
    }
    let mut lo = 0.0;
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        let a = cross_area(dem, r, c, z_p, mid, pr, pc, perp_len, is_nd, in_bounds);
        if a < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Flood the transect at stage `z_p + h`: set depths and add newly flooded
/// cells' map area to `plan_area`.
#[allow(clippy::too_many_arguments)]
fn mark_transect(
    dem: &Raster<f64>,
    depth: &mut Raster<f64>,
    flooded: &mut HashSet<(usize, usize)>,
    plan_area: &mut f64,
    cell_area: f64,
    r: usize,
    c: usize,
    z_p: f64,
    h: f64,
    pr: isize,
    pc: isize,
    is_nd: &impl Fn(f64) -> bool,
    in_bounds: impl Fn(isize, isize) -> bool + Copy,
) {
    let stage = z_p + h;
    let mut flood_cell = |rr: usize, cc: usize, z: f64| {
        let d = stage - z;
        if d <= 0.0 {
            return;
        }
        let prev = depth.get(rr, cc).unwrap_or(0.0);
        if d > prev {
            depth.set(rr, cc, d).ok();
        }
        if flooded.insert((rr, cc)) {
            *plan_area += cell_area;
        }
    };

    // Centre.
    flood_cell(r, c, z_p);
    // Arms.
    for sign in [1isize, -1] {
        let mut k = 1isize;
        loop {
            let rr = r as isize + sign * k * pr;
            let cc = c as isize + sign * k * pc;
            if !in_bounds(rr, cc) {
                break;
            }
            let z = unsafe { dem.get_unchecked(rr as usize, cc as usize) };
            // Confine the cross-section to the valley: a wall (>= stage), a
            // nodata cell, or a cell *below* the thalweg (the perpendicular has
            // left the valley onto an open downslope) ends the arm.
            if is_nd(z) || z >= stage || z < z_p {
                break;
            }
            flood_cell(rr as usize, cc as usize, z);
            k += 1;
        }
    }
}

/// Flooded width (metres) of the transect at stage `z_p + h`: the centre cell
/// plus both perpendicular arms out to the first wall (cell at or above stage).
#[allow(clippy::too_many_arguments)]
fn transect_width(
    dem: &Raster<f64>,
    r: usize,
    c: usize,
    z_p: f64,
    h: f64,
    pr: isize,
    pc: isize,
    perp_len: f64,
    is_nd: &impl Fn(f64) -> bool,
    in_bounds: impl Fn(isize, isize) -> bool + Copy,
) -> f64 {
    let stage = z_p + h;
    let mut cells = 1usize; // centre
    for sign in [1isize, -1] {
        let mut k = 1isize;
        loop {
            let rr = r as isize + sign * k * pr;
            let cc = c as isize + sign * k * pc;
            if !in_bounds(rr, cc) {
                break;
            }
            let z = unsafe { dem.get_unchecked(rr as usize, cc as usize) };
            // Confine the cross-section to the valley: a wall (>= stage), a
            // nodata cell, or a cell *below* the thalweg (the perpendicular has
            // left the valley onto an open downslope) ends the arm.
            if is_nd(z) || z >= stage || z < z_p {
                break;
            }
            cells += 1;
            k += 1;
        }
    }
    cells as f64 * perp_len
}

/// Lateral-spread fill: mantle the valley floor with a uniform sheet of depth
/// `d = A / target_w` out to a half-width of `target_w / 2` on each arm, but
/// stopping at the first ridge crest (where the transect starts to descend into
/// the neighbouring catchment) so the flow does not leak across drainage
/// divides. Accumulates newly flooded cells into `plan_area`.
#[allow(clippy::too_many_arguments)]
fn spread_transect(
    dem: &Raster<f64>,
    depth: &mut Raster<f64>,
    flooded: &mut HashSet<(usize, usize)>,
    plan_area: &mut f64,
    cell_area: f64,
    r: usize,
    c: usize,
    z_p: f64,
    a_cross: f64,
    target_w: f64,
    perp_len: f64,
    pr: isize,
    pc: isize,
    is_nd: &impl Fn(f64) -> bool,
    in_bounds: impl Fn(isize, isize) -> bool + Copy,
) {
    let d = a_cross / target_w; // uniform sheet thickness
    let half_cells = (target_w / (2.0 * perp_len)).round().max(1.0) as isize;

    let mut flood_cell = |rr: usize, cc: usize| {
        let prev = depth.get(rr, cc).unwrap_or(0.0);
        if d > prev {
            depth.set(rr, cc, d).ok();
        }
        if flooded.insert((rr, cc)) {
            *plan_area += cell_area;
        }
    };

    flood_cell(r, c);
    for sign in [1isize, -1] {
        let mut prev_z = z_p;
        let mut rising = false;
        for k in 1..=half_cells {
            let rr = r as isize + sign * k * pr;
            let cc = c as isize + sign * k * pc;
            if !in_bounds(rr, cc) {
                break;
            }
            let z = unsafe { dem.get_unchecked(rr as usize, cc as usize) };
            if is_nd(z) || z < z_p {
                break; // nodata, or left the valley onto an open downslope
            }
            // Ridge crest: once we have climbed the valley wall, a descent means
            // we are crossing into the next catchment — stop the arm.
            if z > prev_z {
                rising = true;
            } else if rising && z < prev_z {
                break;
            }
            flood_cell(rr as usize, cc as usize);
            prev_z = z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    fn raster_f64(data: Vec<f64>, rows: usize, cols: usize, cs: f64) -> Raster<f64> {
        let arr = Array2::from_shape_vec((rows, cols), data).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, rows as f64 * cs, cs, -cs));
        r
    }

    fn raster_u8(data: Vec<u8>, rows: usize, cols: usize, cs: f64) -> Raster<u8> {
        let arr = Array2::from_shape_vec((rows, cols), data).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, rows as f64 * cs, cs, -cs));
        r
    }

    // Prismatic V-valley draining due south (code 7), thalweg at column j0.
    // z(i,j) = Z0 - g*i + s*|j - j0|.
    fn v_valley(
        rows: usize,
        cols: usize,
        j0: usize,
        z0: f64,
        g: f64,
        s: f64,
    ) -> (Raster<f64>, Raster<u8>) {
        let mut z = vec![0.0; rows * cols];
        let mut fd = vec![7u8; rows * cols]; // all flow south
        for i in 0..rows {
            for j in 0..cols {
                let dj = (j as isize - j0 as isize).unsigned_abs();
                z[i * cols + j] = z0 - g * i as f64 + s * dj as f64;
            }
        }
        // bottom row: pit (no outflow)
        for j in 0..cols {
            fd[(rows - 1) * cols + j] = 0;
        }
        (
            raster_f64(z, rows, cols, 1.0),
            raster_u8(fd, rows, cols, 1.0),
        )
    }

    #[test]
    fn test_v_valley_analytic() {
        // s = 1 m/col, cs = 1. With A = 9, stage h = 3 (cross-section
        // 1*[3 + 2*((3-1)+(3-2))] = 1*[3 + 2*3] = 9), flooding columns
        // j0, j0±1, j0±2 -> width 5 m. With B = 20 the run stops after
        // 4 rows (4 * 5 cells * 1 m² = 20).
        let (dem, fd) = v_valley(12, 11, 5, 1000.0, 2.0, 1.0);
        // Choose V and coefficients so A = 9, B = 20.
        // A = c_A V^(2/3); pick V=1000 (V^(2/3)=100) -> c_A=0.09, c_B=0.20.
        let out = laharz(
            &dem,
            &fd,
            LaharzParams {
                sources: vec![(0, 5)],
                volume_m3: 1000.0,
                coeff_cross: 0.09,
                coeff_plan: 0.20,
                spread_aspect: 0.0,
                max_path_cells: 1000,
            },
        )
        .unwrap();

        // Source row flooded across 5 cells (cols 3..=7), depth tapers.
        assert!(out.get(0, 5).unwrap() > 2.9, "thalweg depth ~3");
        assert!(out.get(0, 3).unwrap() > 0.9 && out.get(0, 7).unwrap() > 0.9);
        assert!(out.get(0, 2).unwrap() < 1e-6, "col 2 should be dry");
        assert!(out.get(0, 8).unwrap() < 1e-6, "col 8 should be dry");

        // Planimetric area B=20 => ~4 rows inundated; row 4 onward dry.
        let flooded_rows = (0..12)
            .filter(|&i| (0..11).any(|j| out.get(i, j).unwrap() > 1e-6))
            .count();
        assert_eq!(
            flooded_rows, 4,
            "expected 4 flooded rows, got {flooded_rows}"
        );
    }

    #[test]
    fn test_larger_volume_floods_more() {
        // A long, moderately wide valley so neither run saturates the width:
        // a larger volume gives a larger cross-section A and thus a wider
        // (and not shorter) inundation footprint.
        let rows = 100;
        let cols = 31;
        let (dem, fd) = v_valley(rows, cols, 15, 1000.0, 2.0, 1.0);
        let small = laharz(
            &dem,
            &fd,
            LaharzParams::from_flow_type(vec![(0, 15)], 50.0, LaharzFlowType::Lahar),
        )
        .unwrap();
        let big = laharz(
            &dem,
            &fd,
            LaharzParams::from_flow_type(vec![(0, 15)], 500.0, LaharzFlowType::Lahar),
        )
        .unwrap();
        let area = |r: &Raster<f64>| {
            (0..rows)
                .flat_map(|i| (0..cols).map(move |j| (i, j)))
                .filter(|&(i, j)| r.get(i, j).unwrap() > 1e-6)
                .count()
        };
        assert!(
            area(&big) > area(&small),
            "larger V must flood more: big={} small={}",
            area(&big),
            area(&small)
        );
    }

    #[test]
    fn test_flow_type_coefficients() {
        assert_eq!(LaharzFlowType::Lahar.coefficients(), (0.05, 200.0));
        assert_eq!(LaharzFlowType::DebrisFlow.coefficients(), (0.1, 20.0));
        assert_eq!(LaharzFlowType::RockAvalanche.coefficients(), (0.2, 20.0));
    }

    #[test]
    fn test_validation_errors() {
        let (dem, fd) = v_valley(5, 5, 2, 100.0, 1.0, 1.0);
        let p = |src: (usize, usize), v, ca, cb| LaharzParams {
            sources: vec![src],
            volume_m3: v,
            coeff_cross: ca,
            coeff_plan: cb,
            spread_aspect: 0.0,
            max_path_cells: 100,
        };
        assert!(laharz(&dem, &fd, p((0, 2), 0.0, 0.05, 200.0)).is_err()); // V=0
        assert!(laharz(&dem, &fd, p((9, 9), 1000.0, 0.05, 200.0)).is_err()); // OOB
        assert!(laharz(&dem, &fd, p((0, 2), 1000.0, 0.0, 200.0)).is_err()); // c_A=0
        // empty sources is an error
        assert!(
            laharz(
                &dem,
                &fd,
                LaharzParams {
                    sources: vec![],
                    volume_m3: 1000.0,
                    coeff_cross: 0.05,
                    coeff_plan: 200.0,
                    spread_aspect: 0.0,
                    max_path_cells: 100,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn test_spread_shortens_runout_in_confined_valley() {
        // Steep, narrow V-gorge: the canonical fill makes a deep, near-thalweg
        // thread that runs far; lateral spread widens it and so shortens the
        // runout (fewer downstream rows reached for the same B). The valley is
        // long and B/A small, so B (not the valley end) is the binding limit.
        let rows = 400;
        let cols = 21;
        let (dem, fd) = v_valley(rows, cols, 10, 5000.0, 2.0, 5.0);
        let flooded_rows = |aspect: f64| {
            let p = LaharzParams {
                sources: vec![(0, 10)],
                volume_m3: 100.0,
                coeff_cross: 2.0,
                coeff_plan: 20.0,
                spread_aspect: aspect,
                max_path_cells: 1000,
            };
            let out = laharz(&dem, &fd, p).unwrap();
            (0..rows)
                .filter(|&i| (0..cols).any(|j| out.get(i, j).unwrap() > 1e-6))
                .count()
        };
        let canonical = flooded_rows(0.0);
        let spread = flooded_rows(600.0);
        assert!(
            spread < canonical,
            "lateral spread should shorten runout: spread={spread} canonical={canonical}"
        );
    }

    #[test]
    fn test_multi_source_union() {
        // Two parallel thalwegs (cols 3 and 11) draining south; seeding both
        // floods both channels — strictly more than seeding one.
        let rows = 30;
        let cols = 15;
        let mut z = vec![0.0; rows * cols];
        let mut fd = vec![7u8; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let d3 = (j as isize - 3).unsigned_abs();
                let d11 = (j as isize - 11).unsigned_abs();
                z[i * cols + j] = 1000.0 - 2.0 * i as f64 + d3.min(d11) as f64;
            }
        }
        for j in 0..cols {
            fd[(rows - 1) * cols + j] = 0;
        }
        let dem = raster_f64(z, rows, cols, 1.0);
        let fdir = raster_u8(fd, rows, cols, 1.0);
        let area = |r: &Raster<f64>| {
            (0..rows)
                .flat_map(|i| (0..cols).map(move |j| (i, j)))
                .filter(|&(i, j)| r.get(i, j).unwrap() > 1e-6)
                .count()
        };
        let one = laharz(
            &dem,
            &fdir,
            LaharzParams::from_flow_type(vec![(0, 3)], 2000.0, LaharzFlowType::Lahar),
        )
        .unwrap();
        let both = laharz(
            &dem,
            &fdir,
            LaharzParams::from_flow_type(vec![(0, 3), (0, 11)], 2000.0, LaharzFlowType::Lahar),
        )
        .unwrap();
        assert!(
            area(&both) > area(&one),
            "two sources must flood more than one: {} !> {}",
            area(&both),
            area(&one)
        );
    }
}
