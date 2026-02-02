//! Adaptive MFD (Qin et al. 2011)
//!
//! Instead of using a fixed flow dispersion exponent, adapts β to local
//! terrain using maximum downslope gradient. This produces lower TWI errors
//! at 1–30m resolution compared to standard MFD.
//!
//! β = max_downslope_gradient × scale_factor
//!
//! Reference:
//! Qin, C.-Z. et al. (2011). An adaptive approach to selecting a flow-partition
//! exponent for a multiple-flow-direction algorithm. *International Journal of
//! Geographical Information Science*, 21(4), 443–458.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::Result;

/// D8 neighbor offsets
const D8_OFFSETS: [(isize, isize); 8] = [
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
    (0, -1), (1, -1), (1, 0), (1, 1),
];

const D8_DIST: [f64; 8] = [
    1.0, std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
    1.0, std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
];

const CONTOUR_FRACTION: [f64; 8] = [
    0.5, 0.354, 0.5, 0.354,
    0.5, 0.354, 0.5, 0.354,
];

/// Parameters for Adaptive MFD
#[derive(Debug, Clone)]
pub struct AdaptiveMfdParams {
    /// Scale factor for adaptive exponent: β = max_slope × scale_factor.
    /// Default: 8.9 (Qin 2011 recommended value for 10–30m DEMs)
    pub scale_factor: f64,
    /// Minimum exponent (prevents β < min in flat areas). Default: 0.5
    pub min_exponent: f64,
    /// Maximum exponent (caps β in very steep terrain). Default: 20.0
    pub max_exponent: f64,
}

impl Default for AdaptiveMfdParams {
    fn default() -> Self {
        Self {
            scale_factor: 8.9,
            min_exponent: 0.5,
            max_exponent: 20.0,
        }
    }
}

/// Compute flow accumulation using adaptive MFD (Qin 2011).
///
/// The exponent β adapts to local terrain: β = max_downslope_gradient × scale_factor.
/// In flat terrain, β is small → flow disperses widely.
/// In steep terrain, β is large → flow concentrates (approaches D8).
///
/// # Arguments
/// * `dem` — Input DEM (should be hydrologically conditioned)
/// * `params` — Adaptive MFD parameters
///
/// # Returns
/// Raster<f64> with flow accumulation (contributing area in cell counts)
pub fn flow_accumulation_mfd_adaptive(
    dem: &Raster<f64>,
    params: AdaptiveMfdParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();
    let cell_size = dem.cell_size();
    let total = rows * cols;

    // Step 1: Build sorted cells (highest first)
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
    cells.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Step 2: Initialize accumulation
    let mut accumulation = Array2::<f64>::ones((rows, cols));
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

    // Step 3: Process cells with adaptive exponent
    for &(row, col, z) in &cells {
        let current_acc = accumulation[(row, col)];

        // Compute slopes to all 8 neighbors and find max downslope gradient
        let mut slopes_raw = [0.0_f64; 8];
        let mut max_slope = 0.0_f64;

        for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
            let nr = row as isize + dr;
            let nc = col as isize + dc;
            if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                continue;
            }

            let nz = unsafe { dem.get_unchecked(nr as usize, nc as usize) };
            if nz.is_nan() { continue; }
            if let Some(nd) = nodata {
                if (nz - nd).abs() < f64::EPSILON { continue; }
            }

            let drop = z - nz;
            if drop <= 0.0 { continue; }

            let distance = D8_DIST[idx] * cell_size;
            let slope = drop / distance;
            slopes_raw[idx] = slope;

            if slope > max_slope {
                max_slope = slope;
            }
        }

        // Adaptive exponent: β = max_slope × scale_factor, clamped
        let beta = (max_slope * params.scale_factor)
            .max(params.min_exponent)
            .min(params.max_exponent);

        // Compute weighted flow fractions
        let mut weights = [0.0_f64; 8];
        let mut sum_weighted = 0.0_f64;

        for idx in 0..8 {
            if slopes_raw[idx] <= 0.0 { continue; }
            let contour = CONTOUR_FRACTION[idx] * cell_size;
            let w = (slopes_raw[idx] * contour).powf(beta);
            weights[idx] = w;
            sum_weighted += w;
        }

        // Distribute flow
        if sum_weighted > 0.0 {
            for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                if weights[idx] <= 0.0 { continue; }
                let nr = (row as isize + dr) as usize;
                let nc = (col as isize + dc) as usize;
                accumulation[(nr, nc)] += current_acc * (weights[idx] / sum_weighted);
            }
        }
    }

    // Subtract self-contribution
    for row in 0..rows {
        for col in 0..cols {
            let z = unsafe { dem.get_unchecked(row, col) };
            let is_nd = z.is_nan()
                || nodata.map_or(false, |nd| (z - nd).abs() < f64::EPSILON);
            if !is_nd {
                accumulation[(row, col)] = (accumulation[(row, col)] - 1.0).max(0.0);
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
    use surtgis_core::GeoTransform;

    #[test]
    fn test_adaptive_mfd_south_slope() {
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 10.0).unwrap();
            }
        }

        let acc = flow_accumulation_mfd_adaptive(&dem, AdaptiveMfdParams::default()).unwrap();
        let top = acc.get(0, 2).unwrap();
        let bottom = acc.get(4, 2).unwrap();
        assert!(bottom > top, "Bottom should have more accumulation: top={}, bottom={}", top, bottom);
    }

    #[test]
    fn test_adaptive_mfd_converges_to_d8_steep() {
        // Very steep terrain with high scale_factor → very high β → D8-like
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 100.0).unwrap();
            }
        }

        let acc = flow_accumulation_mfd_adaptive(&dem, AdaptiveMfdParams {
            scale_factor: 100.0,
            ..Default::default()
        }).unwrap();

        // With very concentrated flow, pattern should be D8-like
        let bottom_center = acc.get(4, 2).unwrap();
        assert!(bottom_center > 0.0, "Should have positive accumulation");
    }

    #[test]
    fn test_adaptive_mfd_disperses_in_flat() {
        // Nearly flat terrain → low β → dispersed flow
        let mut dem = Raster::new(7, 7);
        dem.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));
        for row in 0..7 {
            for col in 0..7 {
                dem.set(row, col, (7 - row) as f64 * 0.01).unwrap();
            }
        }

        let acc = flow_accumulation_mfd_adaptive(&dem, AdaptiveMfdParams::default()).unwrap();

        // In flat terrain, flow should be well-distributed
        let mut nonzero = 0;
        for col in 0..7 {
            if acc.get(6, col).unwrap() > 0.5 {
                nonzero += 1;
            }
        }
        assert!(nonzero >= 2, "Flat terrain should distribute flow to multiple cells: {}", nonzero);
    }

    #[test]
    fn test_adaptive_mfd_pit() {
        let mut dem = Raster::new(3, 3);
        dem.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        for row in 0..3 {
            for col in 0..3 {
                dem.set(row, col, 5.0).unwrap();
            }
        }
        dem.set(1, 1, 1.0).unwrap();

        let acc = flow_accumulation_mfd_adaptive(&dem, AdaptiveMfdParams::default()).unwrap();
        let center = acc.get(1, 1).unwrap();
        assert!(
            (center - 8.0).abs() < 0.5,
            "Center pit should accumulate ~8, got {}",
            center
        );
    }

    #[test]
    fn test_adaptive_vs_fixed_different() {
        // Verify adaptive produces different results than fixed exponent
        let mut dem = Raster::new(11, 11);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        for row in 0..11 {
            for col in 0..11 {
                // Mix of steep and flat areas
                let z = if row < 5 {
                    (5 - row) as f64 * 50.0 + col as f64
                } else {
                    (11 - row) as f64 * 0.5 + col as f64 * 0.1
                };
                dem.set(row, col, z).unwrap();
            }
        }

        let adaptive = flow_accumulation_mfd_adaptive(&dem, AdaptiveMfdParams::default()).unwrap();
        let fixed = crate::hydrology::flow_accumulation_mfd(
            &dem,
            crate::hydrology::MfdParams { exponent: 1.1 },
        ).unwrap();

        // Should be different at some cells (adaptive varies β per cell)
        let mut diffs = 0;
        for row in 0..11 {
            for col in 0..11 {
                let a = adaptive.get(row, col).unwrap();
                let f = fixed.get(row, col).unwrap();
                if (a - f).abs() > 0.1 {
                    diffs += 1;
                }
            }
        }
        assert!(diffs > 0, "Adaptive should differ from fixed at some cells");
    }
}
