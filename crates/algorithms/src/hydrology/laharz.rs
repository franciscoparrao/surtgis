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
//! Starting from a user-supplied source on the drainage, the model marches
//! downstream along the D8 flow path. At each step it raises an inundation
//! stage on the valley-perpendicular transect until the wetted cross-section
//! reaches `A`, marking the flooded cells and their depth. It accumulates the
//! planimetric area of newly flooded cells and stops once that area reaches
//! `B`. The result is a downstream-tapering inundation footprint.
//!
//! Coefficient pairs `(c_A, c_B)` are flow-type specific (Griswold & Iverson
//! 2008): see [`LaharzFlowType`]. The source is typically placed at the outlet
//! of a proximal hazard zone (e.g. from an energy cone — see
//! [`crate::hydrology::energy_cone`]).
//!
//! Scope: this is a single-path, D8-confined implementation with a stepped
//! valley-perpendicular transect. It captures the LAHARZ volume scaling and
//! cross-section filling; it does not split flow at distributary junctions or
//! reproduce the original ArcInfo proximal-hazard-zone construction.
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
}

/// Parameters for [`laharz`].
#[derive(Debug, Clone)]
pub struct LaharzParams {
    /// Source cell `(row, col)` on the drainage.
    pub source: (usize, usize),
    /// Flow volume in m³.
    pub volume_m3: f64,
    /// Cross-sectional area coefficient `c_A` (overrides the flow-type default
    /// when set via [`LaharzParams::from_flow_type`] you get the preset).
    pub coeff_cross: f64,
    /// Planimetric area coefficient `c_B`.
    pub coeff_plan: f64,
    /// Safety cap on the number of downstream path cells visited.
    pub max_path_cells: usize,
}

impl LaharzParams {
    /// Build parameters from a flow-type preset.
    pub fn from_flow_type(
        source: (usize, usize),
        volume_m3: f64,
        flow_type: LaharzFlowType,
    ) -> Self {
        let (c_a, c_b) = flow_type.coefficients();
        Self {
            source,
            volume_m3,
            coeff_cross: c_a,
            coeff_plan: c_b,
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
    let (sr, sc) = params.source;
    if sr >= rows || sc >= cols {
        return Err(Error::Other(format!(
            "source ({sr}, {sc}) is outside the {rows}x{cols} grid"
        )));
    }
    let nodata = dem.nodata();
    let is_nd = |v: f64| v.is_nan() || nodata.map(|nd| v == nd).unwrap_or(false);
    if is_nd(unsafe { dem.get_unchecked(sr, sc) }) {
        return Err(Error::Other("source falls on a nodata cell".into()));
    }

    let two_thirds = params.volume_m3.powf(2.0 / 3.0);
    let a_cross = params.coeff_cross * two_thirds;
    let b_plan = params.coeff_plan * two_thirds;
    let cell_area = cs * cs;

    // Inundation depth output and the set of already-flooded cells (for the
    // planimetric area B, counted once per cell).
    let mut depth = dem.with_same_meta::<f64>(rows, cols);
    depth.set_nodata(Some(f64::NAN));
    for r in 0..rows {
        for c in 0..cols {
            let z = unsafe { dem.get_unchecked(r, c) };
            depth.set(r, c, if is_nd(z) { f64::NAN } else { 0.0 }).ok();
        }
    }

    let mut flooded: HashSet<(usize, usize)> = HashSet::new();
    let mut plan_area = 0.0_f64;

    let in_bounds = |r: isize, c: isize| r >= 0 && c >= 0 && r < rows as isize && c < cols as isize;

    let (mut r, mut c) = (sr, sc);
    let mut on_path: HashSet<(usize, usize)> = HashSet::new();

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

        // Flood the transect at this stage; accumulate new planimetric area.
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
            if is_nd(z) || z >= stage {
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
            if is_nd(z) || z >= stage {
                break;
            }
            flood_cell(rr as usize, cc as usize, z);
            k += 1;
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
                source: (0, 5),
                volume_m3: 1000.0,
                coeff_cross: 0.09,
                coeff_plan: 0.20,
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
            LaharzParams::from_flow_type((0, 15), 50.0, LaharzFlowType::Lahar),
        )
        .unwrap();
        let big = laharz(
            &dem,
            &fd,
            LaharzParams::from_flow_type((0, 15), 500.0, LaharzFlowType::Lahar),
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
        let p = |src, v, ca, cb| LaharzParams {
            source: src,
            volume_m3: v,
            coeff_cross: ca,
            coeff_plan: cb,
            max_path_cells: 100,
        };
        assert!(laharz(&dem, &fd, p((0, 2), 0.0, 0.05, 200.0)).is_err()); // V=0
        assert!(laharz(&dem, &fd, p((9, 9), 1000.0, 0.05, 200.0)).is_err()); // OOB
        assert!(laharz(&dem, &fd, p((0, 2), 1000.0, 0.0, 200.0)).is_err()); // c_A=0
    }
}
