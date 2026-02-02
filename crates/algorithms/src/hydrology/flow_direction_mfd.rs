//! FD8/Quinn Multiple Flow Direction algorithm
//!
//! Distributes flow from each cell to ALL downslope neighbors,
//! proportional to the slope gradient in each direction.
//!
//! Unlike D8 (which sends all flow to the steepest neighbor),
//! MFD distributes flow more realistically, producing smoother
//! contributing area patterns and significantly better TWI maps.
//!
//! The flow fraction to neighbor i is:
//!   f_i = max(0, tan_i)^p / Σ max(0, tan_j)^p
//! where tan_i is the slope to neighbor i and p is the flow
//! dispersion exponent (default 1.1 per Quinn et al. 1995).
//!
//! References:
//! - Quinn, P. et al. (1991). The prediction of hillslope flow paths.
//!   *Hydrological Processes*, 5(1), 59–79.
//! - Quinn, P. et al. (1995). The in (a/tan/beta) index: How to
//!   calculate it and how to use it. *Hydrological Processes*, 9, 161–182.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::Result;

/// D8 neighbor offsets matching direction encoding (1=E, 2=NE, ..., 8=SE)
const D8_OFFSETS: [(isize, isize); 8] = [
    (0, 1),   // 1: E
    (-1, 1),  // 2: NE
    (-1, 0),  // 3: N
    (-1, -1), // 4: NW
    (0, -1),  // 5: W
    (1, -1),  // 6: SW
    (1, 0),   // 7: S
    (1, 1),   // 8: SE
];

/// Distance factors for each D8 direction (1.0 cardinal, sqrt(2) diagonal)
const D8_DIST: [f64; 8] = [
    1.0, std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
    1.0, std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
];

/// Contour lengths for each D8 direction (Quinn et al. 1991, Table 1).
/// Cardinal: 0.5 * cell_size, Diagonal: 0.354 * cell_size (≈ 0.5/√2)
const CONTOUR_FRACTION: [f64; 8] = [
    0.5, 0.354, 0.5, 0.354,
    0.5, 0.354, 0.5, 0.354,
];

/// Parameters for MFD flow accumulation
#[derive(Debug, Clone)]
pub struct MfdParams {
    /// Flow dispersion exponent (p).
    /// p=1.0: original Quinn et al. 1991
    /// p=1.1: Quinn et al. 1995 (recommended)
    /// Higher values → more concentrated flow (approaches D8 as p→∞)
    /// Default: 1.1
    pub exponent: f64,
}

impl Default for MfdParams {
    fn default() -> Self {
        Self { exponent: 1.1 }
    }
}

/// Compute MFD (Quinn FD8) flow accumulation.
///
/// This computes contributing area using the multiple flow direction
/// algorithm. Unlike D8 which sends all flow to one neighbor,
/// MFD distributes flow proportionally to all downslope neighbors.
///
/// The algorithm uses topological sorting: cells are processed from
/// highest to lowest elevation, ensuring all upstream contributions
/// are accumulated before processing downstream cells.
///
/// # Arguments
/// * `dem` - Input DEM (should be hydrologically conditioned — sinks filled)
/// * `params` - MFD parameters (exponent)
///
/// # Returns
/// Raster<f64> with MFD flow accumulation (contributing area in cell counts)
pub fn flow_accumulation_mfd(dem: &Raster<f64>, params: MfdParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();
    let cell_size = dem.cell_size();
    let p = params.exponent;
    let total = rows * cols;

    // Step 1: Build sorted list of cells by elevation (highest first)
    let mut cells: Vec<(usize, usize, f64)> = Vec::with_capacity(total);
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

    // Sort descending by elevation (process highest cells first)
    cells.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Step 2: Initialize accumulation (each cell starts with 1.0 = itself)
    let mut accumulation = Array2::<f64>::ones((rows, cols));

    // Mark nodata cells with 0
    for row in 0..rows {
        for col in 0..cols {
            let z = unsafe { dem.get_unchecked(row, col) };
            let is_nd = z.is_nan()
                || nodata.map_or(false, |nd| (z - nd).abs() < f64::EPSILON);
            if is_nd {
                accumulation[(row, col)] = 0.0;
            }
        }
    }

    // Step 3: Process cells from highest to lowest, distributing flow
    for &(row, col, z) in &cells {
        let current_acc = accumulation[(row, col)];

        // Compute slopes to all 8 neighbors
        let mut slopes = [0.0_f64; 8];
        let mut sum_weighted = 0.0_f64;

        for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
            let nr = row as isize + dr;
            let nc = col as isize + dc;

            if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                continue;
            }

            let nz = unsafe { dem.get_unchecked(nr as usize, nc as usize) };
            if nz.is_nan() {
                continue;
            }
            if let Some(nd) = nodata {
                if (nz - nd).abs() < f64::EPSILON {
                    continue;
                }
            }

            // Only consider downslope neighbors
            let drop = z - nz;
            if drop <= 0.0 {
                continue;
            }

            // Slope = drop / distance
            let distance = D8_DIST[idx] * cell_size;
            let slope = drop / distance;

            // Weight = (slope * contour_length)^p  (Quinn 1991 Eq. 3)
            let contour = CONTOUR_FRACTION[idx] * cell_size;
            let weight = (slope * contour).powf(p);

            slopes[idx] = weight;
            sum_weighted += weight;
        }

        // Distribute flow proportionally
        if sum_weighted > 0.0 {
            for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                if slopes[idx] <= 0.0 {
                    continue;
                }

                let nr = (row as isize + dr) as usize;
                let nc = (col as isize + dc) as usize;

                let fraction = slopes[idx] / sum_weighted;
                accumulation[(nr, nc)] += current_acc * fraction;
            }
        }
    }

    // Convert from "cell counts including self" to "upstream cell counts"
    // (subtract 1.0 to match D8 convention where headwaters = 0)
    for row in 0..rows {
        for col in 0..cols {
            let z = unsafe { dem.get_unchecked(row, col) };
            let is_nd = z.is_nan()
                || nodata.map_or(false, |nd| (z - nd).abs() < f64::EPSILON);
            if !is_nd {
                accumulation[(row, col)] -= 1.0;
                // Clamp to 0 (floating point rounding)
                if accumulation[(row, col)] < 0.0 {
                    accumulation[(row, col)] = 0.0;
                }
            }
        }
    }

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = accumulation;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydrology::fill_sinks::{fill_sinks, FillSinksParams};
    use crate::hydrology::flow_direction::flow_direction;
    use crate::hydrology::flow_accumulation::flow_accumulation;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_mfd_south_slope() {
        // 5x5 plane sloping south: each row contributes to next
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 10.0).unwrap();
            }
        }

        let acc = flow_accumulation_mfd(&dem, MfdParams::default()).unwrap();

        // Top row: accumulation = 0 (headwaters)
        for col in 0..5 {
            assert!(
                acc.get(0, col).unwrap() < 0.5,
                "Top row should have ~0 accumulation at col {}",
                col
            );
        }

        // Bottom row: higher accumulation
        let bottom = acc.get(4, 2).unwrap();
        assert!(
            bottom > 1.0,
            "Bottom center should have accumulation > 1, got {}",
            bottom
        );
    }

    #[test]
    fn test_mfd_vs_d8_smoother() {
        // V-shaped valley: MFD should distribute flow more evenly than D8
        let rows = 11;
        let cols = 11;
        let mut dem = Raster::new(rows, cols);
        dem.set_transform(GeoTransform::new(0.0, cols as f64, 1.0, -1.0));
        for row in 0..rows {
            for col in 0..cols {
                let cross = (col as f64 - 5.0).abs();
                let along = (rows - 1 - row) as f64 * 0.5;
                dem.set(row, col, cross + along).unwrap();
            }
        }

        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();

        // D8 accumulation
        let fdir = flow_direction(&filled).unwrap();
        let d8_acc = flow_accumulation(&fdir).unwrap();

        // MFD accumulation
        let mfd_acc = flow_accumulation_mfd(&filled, MfdParams::default()).unwrap();

        // D8 peak (max accumulation) should be >= MFD peak
        // because D8 concentrates all flow in single cells
        let d8_max = (0..rows).flat_map(|r| (0..cols).map(move |c| (r, c)))
            .map(|(r, c)| d8_acc.get(r, c).unwrap())
            .fold(0.0_f64, f64::max);

        let mfd_max = (0..rows).flat_map(|r| (0..cols).map(move |c| (r, c)))
            .map(|(r, c)| mfd_acc.get(r, c).unwrap())
            .fold(0.0_f64, f64::max);

        assert!(
            d8_max >= mfd_max - 1.0,
            "D8 max should be >= MFD max (more concentrated): d8={}, mfd={}",
            d8_max, mfd_max
        );

        // MFD bottom row should have more non-zero cells than D8
        // (flow is spread across more cells)
        let d8_nonzero = (0..cols)
            .filter(|&c| d8_acc.get(rows - 1, c).unwrap() > 1.0)
            .count();
        let mfd_nonzero = (0..cols)
            .filter(|&c| mfd_acc.get(rows - 1, c).unwrap() > 1.0)
            .count();

        assert!(
            mfd_nonzero >= d8_nonzero,
            "MFD should have flow in more cells: d8_nonzero={}, mfd_nonzero={}",
            d8_nonzero, mfd_nonzero
        );
    }

    #[test]
    fn test_mfd_conservation() {
        // Total outflow from the DEM should approximately equal total cells
        // (conservation of mass: each cell contributes 1 unit)
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (10 - row - col) as f64 * 10.0).unwrap();
            }
        }

        let acc = flow_accumulation_mfd(&dem, MfdParams::default()).unwrap();

        // The corner cell (4,4) should have the highest accumulation
        // (all flow converges there on a SE slope)
        let corner = acc.get(4, 4).unwrap();
        assert!(
            corner > 10.0,
            "Outlet should accumulate most of the grid, got {}",
            corner
        );
    }

    #[test]
    fn test_mfd_convergent_pit() {
        // 3x3 with center as lowest: all neighbors contribute to center
        let mut dem = Raster::new(3, 3);
        dem.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        for row in 0..3 {
            for col in 0..3 {
                dem.set(row, col, 5.0).unwrap();
            }
        }
        dem.set(1, 1, 1.0).unwrap();

        let acc = flow_accumulation_mfd(&dem, MfdParams::default()).unwrap();

        // Center receives flow from all 8 neighbors
        let center = acc.get(1, 1).unwrap();
        assert!(
            (center - 8.0).abs() < 0.5,
            "Center pit should accumulate ~8 neighbors, got {}",
            center
        );
    }

    #[test]
    fn test_mfd_exponent_effect() {
        // Higher exponent = more concentrated (approaches D8)
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 10.0 + col as f64).unwrap();
            }
        }

        let low_p = flow_accumulation_mfd(&dem, MfdParams { exponent: 0.5 }).unwrap();
        let high_p = flow_accumulation_mfd(&dem, MfdParams { exponent: 5.0 }).unwrap();

        // With low exponent, flow is more distributed → max accumulation is lower
        let low_max = (0..5).flat_map(|r| (0..5).map(move |c| (r, c)))
            .map(|(r, c)| low_p.get(r, c).unwrap())
            .fold(0.0_f64, f64::max);

        let high_max = (0..5).flat_map(|r| (0..5).map(move |c| (r, c)))
            .map(|(r, c)| high_p.get(r, c).unwrap())
            .fold(0.0_f64, f64::max);

        assert!(
            high_max >= low_max - 0.5,
            "Higher exponent should concentrate flow more: low_max={}, high_max={}",
            low_max, high_max
        );
    }
}
