//! D-infinity (D∞) flow direction algorithm
//!
//! Computes continuous flow direction angles based on triangular facets
//! fitted to the 3×3 neighborhood. Flow direction is the steepest
//! downslope angle, which can point in any direction (0–2π).
//!
//! For flow accumulation, the flow from each cell is partitioned
//! between the two D8 neighbors that bracket the D∞ angle, with
//! proportions based on angular proximity.
//!
//! Reference:
//! Tarboton, D.G. (1997). A new method for the determination of flow
//! directions and upslope areas in grid digital elevation models.
//! *Water Resources Research*, 33(2), 309–319.

use ndarray::Array2;
use surtgis_core::Result;
use surtgis_core::raster::Raster;

/// D8 neighbor offsets (E, NE, N, NW, W, SW, S, SE)
const D8_OFFSETS: [(isize, isize); 8] = [
    (0, 1),   // 0: E
    (-1, 1),  // 1: NE
    (-1, 0),  // 2: N
    (-1, -1), // 3: NW
    (0, -1),  // 4: W
    (1, -1),  // 5: SW
    (1, 0),   // 6: S
    (1, 1),   // 7: SE
];

/// Distance factors for each D8 direction
const D8_DIST: [f64; 8] = [
    1.0,
    std::f64::consts::SQRT_2,
    1.0,
    std::f64::consts::SQRT_2,
    1.0,
    std::f64::consts::SQRT_2,
    1.0,
    std::f64::consts::SQRT_2,
];

/// Tarboton (1997) facet decomposition — matches TauDEM exactly.
///
/// Each facet is defined by a CARDINAL neighbor (e1 direction) and an
/// adjacent DIAGONAL neighbor (e2 direction). The base_angle is the
/// azimuth of the cardinal direction; `sign` controls whether θ is
/// added (+1) or subtracted (-1) to reach the diagonal.
///
/// Facet tuple: (cardinal_idx, diagonal_idx, base_angle, sign)
const TARBOTON_FACETS: [(usize, usize, f64, f64); 8] = [
    // K=1 in TauDEM: E→NE,   angle = 0     + θ
    (0, 1, 0.0, 1.0),
    // K=2 in TauDEM: N→NE,   angle = π/2   - θ
    (2, 1, std::f64::consts::FRAC_PI_2, -1.0),
    // K=3 in TauDEM: N→NW,   angle = π/2   + θ
    (2, 3, std::f64::consts::FRAC_PI_2, 1.0),
    // K=4 in TauDEM: W→NW,   angle = π     - θ
    (4, 3, std::f64::consts::PI, -1.0),
    // K=5 in TauDEM: W→SW,   angle = π     + θ
    (4, 5, std::f64::consts::PI, 1.0),
    // K=6 in TauDEM: S→SW,   angle = 3π/2  - θ
    (6, 5, 3.0 * std::f64::consts::FRAC_PI_2, -1.0),
    // K=7 in TauDEM: S→SE,   angle = 3π/2  + θ
    (6, 7, 3.0 * std::f64::consts::FRAC_PI_2, 1.0),
    // K=8 in TauDEM: E→SE,   angle = 2π    - θ
    (0, 7, 2.0 * std::f64::consts::PI, -1.0),
];

/// Result of D-infinity computation
pub struct DinfResult {
    /// Flow direction angles in radians (0 = East, counterclockwise).
    /// NaN for nodata/pit cells.
    pub angles: Raster<f64>,
    /// Flow accumulation (contributing area in cell counts)
    pub accumulation: Raster<f64>,
}

/// Compute D-infinity flow direction angles.
///
/// For each cell, fits 8 triangular facets to the 3×3 neighborhood
/// and selects the steepest downslope facet. The flow angle is
/// continuous (not restricted to 8 directions).
///
/// # Arguments
/// * `dem` - Input DEM (should be hydrologically conditioned)
///
/// # Returns
/// `Raster<f64>` with flow direction angles in radians.
/// -1.0 indicates a pit/flat cell (no downslope).
pub fn flow_direction_dinf(dem: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();
    let cs = dem.cell_size();
    let diag = cs * std::f64::consts::SQRT_2;

    let mut angles = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    for row in 0..rows {
        for col in 0..cols {
            let z0 = unsafe { dem.get_unchecked(row, col) };
            if z0.is_nan() || nodata.is_some_and(|nd| (z0 - nd).abs() < f64::EPSILON) {
                continue;
            }

            // Get all 8 neighbors
            let mut zn = [f64::NAN; 8];
            let mut all_valid = true;

            for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                let nr = row as isize + dr;
                let nc = col as isize + dc;

                if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                    all_valid = false;
                    continue;
                }

                let nval = unsafe { dem.get_unchecked(nr as usize, nc as usize) };
                if nval.is_nan() || nodata.is_some_and(|nd| (nval - nd).abs() < f64::EPSILON) {
                    all_valid = false;
                    continue;
                }
                zn[idx] = nval;
            }

            // Find steepest facet using Tarboton (1997) decomposition.
            // Each facet uses a CARDINAL neighbor for e1 and an adjacent
            // DIAGONAL neighbor for e2, with d1 = d2 = cell_size.
            let mut best_slope = 0.0_f64;
            let mut best_angle = -1.0_f64;

            for &(card_idx, diag_idx, base_angle, sign) in &TARBOTON_FACETS {
                if zn[card_idx].is_nan() || zn[diag_idx].is_nan() {
                    continue;
                }

                // e1: slope from center toward cardinal neighbor (d1 = cs)
                // e2: cross-slope from cardinal toward diagonal (d2 = cs)
                let e1 = (z0 - zn[card_idx]) / cs;
                let e2 = (zn[card_idx] - zn[diag_idx]) / cs;

                let theta: f64;
                let slope: f64;

                if e1 == 0.0 && e2 == 0.0 {
                    continue; // flat facet
                }

                // Use atan2 (matching TauDEM's VSLOPE) to correctly handle
                // all sign combinations of e1 and e2.
                let raw = e2.atan2(e1);
                let ad = std::f64::consts::FRAC_PI_4; // atan2(cs, cs) for square grid

                if raw < 0.0 {
                    // Steepest direction is along the cardinal edge
                    theta = 0.0;
                    slope = e1;
                } else if raw > ad {
                    // Steepest direction is along the diagonal edge
                    theta = ad;
                    slope = (z0 - zn[diag_idx]) / diag;
                } else {
                    theta = raw;
                    slope = (e1 * e1 + e2 * e2).sqrt();
                }

                if slope > best_slope {
                    best_slope = slope;
                    let mut angle = base_angle + sign * theta;
                    // Normalize to [0, 2π)
                    let two_pi = 2.0 * std::f64::consts::PI;
                    if angle >= two_pi {
                        angle -= two_pi;
                    }
                    if angle < 0.0 {
                        angle += two_pi;
                    }
                    best_angle = angle;
                }
            }

            // Fallback for edge/nodata cells: D8-style single-direction
            if best_angle < 0.0 && !all_valid {
                let cardinal_angles = [
                    0.0,
                    std::f64::consts::FRAC_PI_4,
                    std::f64::consts::FRAC_PI_2,
                    3.0 * std::f64::consts::FRAC_PI_4,
                    std::f64::consts::PI,
                    5.0 * std::f64::consts::FRAC_PI_4,
                    3.0 * std::f64::consts::FRAC_PI_2,
                    7.0 * std::f64::consts::FRAC_PI_4,
                ];
                for (idx, _) in D8_OFFSETS.iter().enumerate() {
                    if zn[idx].is_nan() {
                        continue;
                    }
                    let dist = D8_DIST[idx] * cs;
                    let slope = (z0 - zn[idx]) / dist;
                    if slope > best_slope {
                        best_slope = slope;
                        best_angle = cardinal_angles[idx];
                    }
                }
            }

            angles[(row, col)] = best_angle;
        }
    }

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = angles;

    Ok(output)
}

/// The two D8 receivers of a D∞ angle with their flow fractions.
///
/// Flow is split between the two D8 neighbors bracketing the angle,
/// proportional to the angular offset within the π/4 sector
/// (Tarboton 1997, eq. 6). On exact sector boundaries all flow goes
/// to a single neighbor (the second fraction is 0).
fn dinf_receivers(angle: f64) -> [(usize, f64); 2] {
    let pi4 = std::f64::consts::FRAC_PI_4;
    let sector = ((angle / pi4) % 8.0).floor() as usize;
    let lower_idx = sector.min(7);
    let upper_idx = (lower_idx + 1) % 8;
    let alpha = (angle - (lower_idx as f64) * pi4).clamp(0.0, pi4);
    let frac_upper = alpha / pi4;
    [(lower_idx, 1.0 - frac_upper), (upper_idx, frac_upper)]
}

/// Calculate D-infinity flow accumulation from a D∞ angle raster.
///
/// Each cell receives the (fractional) count of all upstream cells
/// that flow into it. Headwater cells have accumulation = 0, matching
/// the D8 [`flow_accumulation`](super::flow_accumulation) convention.
///
/// Flow from each cell is split between the two D8 neighbors
/// bracketing its angle, proportional to the angular offset.
/// Processing order is a dependency-driven topological sort, so only
/// the angle raster is needed (no DEM). Cells with negative or NaN
/// angle (pits / nodata) accumulate but do not forward flow.
///
/// # Arguments
/// * `angles` - D∞ flow angles in radians (output from
///   [`flow_direction_dinf`]; TauDEM `ang` rasters use the same
///   convention)
///
/// # Returns
/// `Raster<f64>` with fractional accumulation values; NaN where the
/// input angle is NaN.
pub fn flow_accumulation_dinf(angles: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = angles.shape();

    let receivers_of = |row: usize, col: usize| -> Option<[(usize, f64); 2]> {
        let a = unsafe { angles.get_unchecked(row, col) };
        if a.is_nan() || a < 0.0 {
            None
        } else {
            Some(dinf_receivers(a))
        }
    };

    // Step 1: in-degree = number of incoming flow edges per cell
    let mut in_degree = Array2::<u32>::zeros((rows, cols));
    for row in 0..rows {
        for col in 0..cols {
            let Some(recv) = receivers_of(row, col) else {
                continue;
            };
            for (idx, frac) in recv {
                if frac <= 0.0 {
                    continue;
                }
                let (dr, dc) = D8_OFFSETS[idx];
                let nr = row as isize + dr;
                let nc = col as isize + dc;
                if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                    in_degree[(nr as usize, nc as usize)] += 1;
                }
            }
        }
    }

    // Step 2-3: topological propagation from headwaters
    let mut accumulation = Array2::<f64>::zeros((rows, cols));
    let mut queue: Vec<(usize, usize)> = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            if in_degree[(row, col)] == 0 {
                queue.push((row, col));
            }
        }
    }

    let mut processed = 0usize;
    while let Some((row, col)) = queue.pop() {
        processed += 1;
        let Some(recv) = receivers_of(row, col) else {
            continue;
        };
        let outflow = accumulation[(row, col)] + 1.0;
        for (idx, frac) in recv {
            if frac <= 0.0 {
                continue;
            }
            let (dr, dc) = D8_OFFSETS[idx];
            let nr = row as isize + dr;
            let nc = col as isize + dc;
            if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                continue;
            }
            let (nr, nc) = (nr as usize, nc as usize);
            accumulation[(nr, nc)] += outflow * frac;
            in_degree[(nr, nc)] -= 1;
            if in_degree[(nr, nc)] == 0 {
                queue.push((nr, nc));
            }
        }
    }

    // Tarboton-conformant angles cannot form cycles (receivers are
    // strictly downslope), but arbitrary input rasters can. Forward
    // each remaining cell once, row-major, so the result is finite
    // everywhere instead of silently dropping basins.
    if processed < rows * cols {
        for row in 0..rows {
            for col in 0..cols {
                if in_degree[(row, col)] == 0 {
                    continue;
                }
                let Some(recv) = receivers_of(row, col) else {
                    continue;
                };
                let outflow = accumulation[(row, col)] + 1.0;
                for (idx, frac) in recv {
                    if frac <= 0.0 {
                        continue;
                    }
                    let (dr, dc) = D8_OFFSETS[idx];
                    let nr = row as isize + dr;
                    let nc = col as isize + dc;
                    if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                        accumulation[(nr as usize, nc as usize)] += outflow * frac;
                    }
                }
            }
        }
    }

    // NaN out nodata cells
    for row in 0..rows {
        for col in 0..cols {
            let a = unsafe { angles.get_unchecked(row, col) };
            if a.is_nan() {
                accumulation[(row, col)] = f64::NAN;
            }
        }
    }

    let mut acc_raster = angles.with_same_meta::<f64>(rows, cols);
    acc_raster.set_nodata(Some(f64::NAN));
    *acc_raster.data_mut() = accumulation;
    Ok(acc_raster)
}

/// Compute D-infinity flow direction and accumulation.
///
/// Convenience wrapper: [`flow_direction_dinf`] followed by
/// [`flow_accumulation_dinf`].
///
/// # Arguments
/// * `dem` - Input DEM (should be hydrologically conditioned)
///
/// # Returns
/// `DinfResult` with both flow direction angles and accumulation
pub fn flow_dinf(dem: &Raster<f64>) -> Result<DinfResult> {
    let angles = flow_direction_dinf(dem)?;
    let accumulation = flow_accumulation_dinf(&angles)?;
    Ok(DinfResult {
        angles,
        accumulation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_dinf_east_slope() {
        // DEM sloping east: flow should be ~0 radians (east)
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - col) as f64 * 10.0).unwrap();
            }
        }

        let angles = flow_direction_dinf(&dem).unwrap();
        let center = angles.get(2, 2).unwrap();

        // Should be approximately 0 radians (east direction)
        assert!(
            center.abs() < 0.5 || (center - 2.0 * std::f64::consts::PI).abs() < 0.5,
            "Flow should point east (~0 rad), got {}",
            center
        );
    }

    #[test]
    fn test_dinf_south_slope() {
        // DEM sloping south: flow should be ~3π/2 radians
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 10.0).unwrap();
            }
        }

        let angles = flow_direction_dinf(&dem).unwrap();
        let center = angles.get(2, 2).unwrap();

        // South direction = 3π/2 ≈ 4.712
        let target = 3.0 * std::f64::consts::FRAC_PI_2;
        assert!(
            (center - target).abs() < 0.5,
            "Flow should point south (~{:.3} rad), got {}",
            target,
            center
        );
    }

    #[test]
    fn test_dinf_pit_returns_negative() {
        // Central pit: flow direction should be -1 (no outflow)
        let mut dem = Raster::new(3, 3);
        dem.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        for row in 0..3 {
            for col in 0..3 {
                dem.set(row, col, 5.0).unwrap();
            }
        }
        dem.set(1, 1, 1.0).unwrap(); // pit

        let angles = flow_direction_dinf(&dem).unwrap();
        let center = angles.get(1, 1).unwrap();
        assert!(
            center < 0.0,
            "Pit should have negative flow angle, got {}",
            center
        );
    }

    #[test]
    fn test_dinf_accumulation_convergent() {
        // Pit in center: all flow converges
        let mut dem = Raster::new(3, 3);
        dem.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        for row in 0..3 {
            for col in 0..3 {
                dem.set(row, col, 5.0).unwrap();
            }
        }
        dem.set(1, 1, 1.0).unwrap();

        let result = flow_dinf(&dem).unwrap();
        let center_acc = result.accumulation.get(1, 1).unwrap();

        assert!(
            center_acc > 5.0,
            "Center pit should accumulate most neighbors, got {}",
            center_acc
        );
    }

    #[test]
    fn test_dinf_receivers_split() {
        // Angle exactly between E (0) and NE (π/4) → 50/50 split
        let recv = dinf_receivers(std::f64::consts::FRAC_PI_4 / 2.0);
        assert_eq!(recv[0].0, 0);
        assert_eq!(recv[1].0, 1);
        assert!((recv[0].1 - 0.5).abs() < 1e-12);
        assert!((recv[1].1 - 0.5).abs() < 1e-12);

        // Exact cardinal angle → all flow to that neighbor
        let recv = dinf_receivers(std::f64::consts::FRAC_PI_2);
        assert_eq!(recv[0].0, 2);
        assert!((recv[0].1 - 1.0).abs() < 1e-12);
        assert!(recv[1].1.abs() < 1e-12);

        // Last sector wraps to E
        let recv = dinf_receivers(7.5 * std::f64::consts::FRAC_PI_4);
        assert_eq!(recv[0].0, 7);
        assert_eq!(recv[1].0, 0);
        assert!((recv[0].1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_dinf_accumulation_east_plane() {
        // Plane sloping east: interior accumulation grows along each row
        let mut dem = Raster::new(7, 7);
        dem.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));
        for row in 0..7 {
            for col in 0..7 {
                dem.set(row, col, (7 - col) as f64 * 10.0).unwrap();
            }
        }

        let result = flow_dinf(&dem).unwrap();
        let acc = &result.accumulation;
        // Interior row: monotone increase eastward, headwater west edge ≈ 0
        assert!(acc.get(3, 0).unwrap() < 0.5);
        for col in 1..6 {
            assert!(
                acc.get(3, col).unwrap() >= acc.get(3, col - 1).unwrap(),
                "accumulation should not decrease eastward at col {}",
                col
            );
        }
        assert!(acc.get(3, 5).unwrap() > 3.0);
    }

    #[test]
    fn test_dinf_accumulation_mass_conservation() {
        // Inward-draining bowl: every valid cell's flow eventually
        // reaches the central pit, so pit accumulation = N - 1.
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                let dr = row as f64 - 2.0;
                let dc = col as f64 - 2.0;
                dem.set(row, col, (dr * dr + dc * dc).sqrt() * 10.0)
                    .unwrap();
            }
        }

        let result = flow_dinf(&dem).unwrap();
        let center = result.accumulation.get(2, 2).unwrap();
        assert!(
            (center - 24.0).abs() < 1e-9,
            "bowl pit should accumulate all 24 upstream cells, got {}",
            center
        );
    }

    #[test]
    fn test_dinf_accumulation_from_angles_matches_wrapper() {
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                let base = (row + col) as f64;
                let var = ((row * 7 + col * 13) % 100) as f64 / 10.0;
                dem.set(row, col, base + var).unwrap();
            }
        }

        let angles = flow_direction_dinf(&dem).unwrap();
        let standalone = flow_accumulation_dinf(&angles).unwrap();
        let combined = flow_dinf(&dem).unwrap();

        for row in 0..10 {
            for col in 0..10 {
                let a = standalone.get(row, col).unwrap();
                let b = combined.accumulation.get(row, col).unwrap();
                assert!(
                    (a - b).abs() < 1e-12 || (a.is_nan() && b.is_nan()),
                    "mismatch at ({},{}): {} vs {}",
                    row,
                    col,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_dinf_accumulation_nan_propagation() {
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - col) as f64 * 10.0).unwrap();
            }
        }
        dem.set(2, 2, f64::NAN).unwrap();

        let result = flow_dinf(&dem).unwrap();
        assert!(result.accumulation.get(2, 2).unwrap().is_nan());
        assert!(result.accumulation.nodata().is_some_and(|nd| nd.is_nan()));
    }

    #[test]
    fn test_dinf_angle_range() {
        // All valid angles should be in [0, 2π) or -1 for pits
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                let base = (row + col) as f64;
                let var = ((row * 7 + col * 13) % 100) as f64 / 10.0;
                dem.set(row, col, base + var).unwrap();
            }
        }

        let angles = flow_direction_dinf(&dem).unwrap();
        let (rows, cols) = angles.shape();
        let two_pi = 2.0 * std::f64::consts::PI;

        for row in 0..rows {
            for col in 0..cols {
                let a = angles.get(row, col).unwrap();
                if a.is_nan() {
                    continue;
                }
                assert!(
                    a < 0.0 || (a >= 0.0 && a <= two_pi + 0.01),
                    "Angle at ({},{}) should be in [-1, 2π], got {}",
                    row,
                    col,
                    a
                );
            }
        }
    }
}
