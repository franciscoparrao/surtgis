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
use surtgis_core::raster::Raster;
use surtgis_core::Result;

/// D8 neighbor offsets for 8 facets (E, NE, N, NW, W, SW, S, SE)
/// Each facet is defined by two adjacent cardinal/diagonal neighbors.
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
    1.0, std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
    1.0, std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
];

/// The 8 triangular facets around a cell.
/// Each facet is defined by the center cell and two adjacent neighbors.
/// facet_neighbors[i] = (neighbor_a_index, neighbor_b_index) into D8_OFFSETS.
/// The angle of facet i starts at the azimuth of neighbor a.
const FACET_NEIGHBORS: [(usize, usize); 8] = [
    (0, 1), // E-NE facet (azimuth 0 to π/4)
    (1, 2), // NE-N facet (π/4 to π/2)
    (2, 3), // N-NW facet (π/2 to 3π/4)
    (3, 4), // NW-W facet (3π/4 to π)
    (4, 5), // W-SW facet (π to 5π/4)
    (5, 6), // SW-S facet (5π/4 to 3π/2)
    (6, 7), // S-SE facet (3π/2 to 7π/4)
    (7, 0), // SE-E facet (7π/4 to 2π)
];

/// Base angle for each facet (azimuth in radians, clockwise from East=0)
const FACET_BASE_ANGLE: [f64; 8] = [
    0.0,                                // E
    std::f64::consts::FRAC_PI_4,        // NE (π/4)
    std::f64::consts::FRAC_PI_2,        // N (π/2)
    3.0 * std::f64::consts::FRAC_PI_4,  // NW (3π/4)
    std::f64::consts::PI,               // W (π)
    5.0 * std::f64::consts::FRAC_PI_4,  // SW (5π/4)
    3.0 * std::f64::consts::FRAC_PI_2,  // S (3π/2)
    7.0 * std::f64::consts::FRAC_PI_4,  // SE (7π/4)
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
/// Raster<f64> with flow direction angles in radians.
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
            if z0.is_nan() || nodata.map_or(false, |nd| (z0 - nd).abs() < f64::EPSILON) {
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
                if nval.is_nan() || nodata.map_or(false, |nd| (nval - nd).abs() < f64::EPSILON) {
                    all_valid = false;
                    continue;
                }
                zn[idx] = nval;
            }

            // Find steepest facet
            let mut best_slope = 0.0_f64;
            let mut best_angle = -1.0_f64;

            for facet in 0..8 {
                let (a_idx, b_idx) = FACET_NEIGHBORS[facet];

                if zn[a_idx].is_nan() || zn[b_idx].is_nan() {
                    // Fall back to single-direction slope for available neighbors
                    continue;
                }

                // Tarboton's facet decomposition:
                // e1 = (z0 - z_a) / d1  (slope along the cardinal edge)
                // e2 = (z_a - z_b) / d2  (slope along the perpendicular)
                // where d1 and d2 are distances.

                // For facets where neighbor a is cardinal, d1=cs, d2=cs
                // For facets where neighbor a is diagonal, d1=diag, d2=cs
                let d1 = D8_DIST[a_idx] * cs;
                let d2 = cs; // perpendicular distance within the facet

                let e1 = (z0 - zn[a_idx]) / d1;
                let e2 = (zn[a_idx] - zn[b_idx]) / d2;

                // Steepest direction within this facet
                let mut theta: f64; // angle within facet [0, π/4]
                let slope: f64;

                if e1 == 0.0 && e2 == 0.0 {
                    continue; // flat facet
                }

                if e2 == 0.0 {
                    theta = 0.0;
                    slope = e1;
                } else {
                    theta = (e2 / e1).atan();
                    if theta < 0.0 {
                        theta = 0.0;
                        slope = e1;
                    } else if theta > std::f64::consts::FRAC_PI_4 {
                        theta = std::f64::consts::FRAC_PI_4;
                        // Slope along the diagonal to neighbor b
                        slope = (z0 - zn[b_idx]) / diag;
                    } else {
                        slope = (e1 * e1 + e2 * e2).sqrt();
                    }
                }

                if slope > best_slope {
                    best_slope = slope;
                    best_angle = FACET_BASE_ANGLE[facet] + theta;
                }
            }

            // If no steepest facet found but we have downslope neighbors,
            // fall back to D8-style single-direction
            if best_angle < 0.0 && !all_valid {
                for (idx, _) in D8_OFFSETS.iter().enumerate() {
                    if zn[idx].is_nan() {
                        continue;
                    }
                    let dist = D8_DIST[idx] * cs;
                    let slope = (z0 - zn[idx]) / dist;
                    if slope > best_slope {
                        best_slope = slope;
                        best_angle = FACET_BASE_ANGLE[idx];
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

/// Compute D-infinity flow direction and accumulation.
///
/// Flow from each cell is split between the two D8 neighbors
/// bracketing the D∞ angle, proportional to the angular offset.
///
/// # Arguments
/// * `dem` - Input DEM (should be hydrologically conditioned)
///
/// # Returns
/// `DinfResult` with both flow direction angles and accumulation
pub fn flow_dinf(dem: &Raster<f64>) -> Result<DinfResult> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();

    // Step 1: Compute D∞ angles
    let angle_raster = flow_direction_dinf(dem)?;

    // Step 2: Compute accumulation using topological sort
    // Sort cells by elevation (highest first)
    let mut cells: Vec<(usize, usize, f64)> = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let z = unsafe { dem.get_unchecked(row, col) };
            let is_nd = z.is_nan()
                || nodata.map_or(false, |nd| (z - nd).abs() < f64::EPSILON);
            if !is_nd {
                cells.push((row, col, z));
            }
        }
    }
    cells.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut accumulation = Array2::<f64>::ones((rows, cols));
    // Mark nodata cells
    for row in 0..rows {
        for col in 0..cols {
            let z = unsafe { dem.get_unchecked(row, col) };
            if z.is_nan() || nodata.map_or(false, |nd| (z - nd).abs() < f64::EPSILON) {
                accumulation[(row, col)] = 0.0;
            }
        }
    }

    let pi4 = std::f64::consts::FRAC_PI_4;
    let two_pi = 2.0 * std::f64::consts::PI;

    for &(row, col, _) in &cells {
        let angle = angle_raster.get(row, col).unwrap();
        if angle < 0.0 || angle.is_nan() {
            continue; // pit cell
        }

        let current_acc = accumulation[(row, col)];

        // Determine which two D8 neighbors bracket this angle
        // facet index = floor(angle / (π/4))
        let facet = ((angle / pi4) % 8.0).floor() as usize;
        let facet = facet.min(7); // safety clamp

        let (a_idx, b_idx) = FACET_NEIGHBORS[facet];

        // Angular offset within the facet
        let base = FACET_BASE_ANGLE[facet];
        let mut alpha = angle - base;
        if alpha < 0.0 { alpha += two_pi; }
        if alpha > pi4 { alpha = pi4; }

        // Proportion to neighbor b (at base + π/4)
        let frac_b = alpha / pi4;
        let frac_a = 1.0 - frac_b;

        // Distribute to neighbor a
        let (dr_a, dc_a) = D8_OFFSETS[a_idx];
        let nr_a = row as isize + dr_a;
        let nc_a = col as isize + dc_a;
        if nr_a >= 0 && nc_a >= 0 && (nr_a as usize) < rows && (nc_a as usize) < cols {
            accumulation[(nr_a as usize, nc_a as usize)] += current_acc * frac_a;
        }

        // Distribute to neighbor b
        let (dr_b, dc_b) = D8_OFFSETS[b_idx];
        let nr_b = row as isize + dr_b;
        let nc_b = col as isize + dc_b;
        if nr_b >= 0 && nc_b >= 0 && (nr_b as usize) < rows && (nc_b as usize) < cols {
            accumulation[(nr_b as usize, nc_b as usize)] += current_acc * frac_b;
        }
    }

    // Subtract self-contribution
    for row in 0..rows {
        for col in 0..cols {
            let z = unsafe { dem.get_unchecked(row, col) };
            let is_nd = z.is_nan()
                || nodata.map_or(false, |nd| (z - nd).abs() < f64::EPSILON);
            if !is_nd {
                accumulation[(row, col)] -= 1.0;
                if accumulation[(row, col)] < 0.0 {
                    accumulation[(row, col)] = 0.0;
                }
            }
        }
    }

    let mut acc_raster = dem.with_same_meta::<f64>(rows, cols);
    acc_raster.set_nodata(Some(f64::NAN));
    *acc_raster.data_mut() = accumulation;

    Ok(DinfResult {
        angles: angle_raster,
        accumulation: acc_raster,
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
            target, center
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
                if a.is_nan() { continue; }
                assert!(
                    a < 0.0 || (a >= 0.0 && a <= two_pi + 0.01),
                    "Angle at ({},{}) should be in [-1, 2π], got {}",
                    row, col, a
                );
            }
        }
    }
}
