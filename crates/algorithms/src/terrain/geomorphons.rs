//! Geomorphons — pattern recognition approach to landform classification
//!
//! Jasiewicz & Stepinski (2013): Classifies terrain into 10 landform elements
//! by comparing each cell's elevation to surrounding cells along 8 directions.
//!
//! Each direction produces a ternary value (+1, 0, -1) creating a pattern
//! ("geomorphon") that maps to a landform class.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Geomorphon landform classes
pub mod class {
    pub const FLAT: u8 = 1;
    pub const PEAK: u8 = 2;
    pub const RIDGE: u8 = 3;
    pub const SHOULDER: u8 = 4;
    pub const SPUR: u8 = 5;
    pub const SLOPE: u8 = 6;
    pub const HOLLOW: u8 = 7;
    pub const FOOTSLOPE: u8 = 8;
    pub const VALLEY: u8 = 9;
    pub const PIT: u8 = 10;
}

/// Parameters for geomorphon computation
#[derive(Debug, Clone)]
pub struct GeomorphonParams {
    /// Search radius in cells (default 10)
    pub radius: usize,
    /// Flatness threshold in degrees (default 1.0)
    pub flatness_threshold: f64,
}

impl Default for GeomorphonParams {
    fn default() -> Self {
        Self {
            radius: 10,
            flatness_threshold: 1.0,
        }
    }
}

/// 8 direction vectors (N, NE, E, SE, S, SW, W, NW)
const DIRECTIONS: [(isize, isize); 8] = [
    (-1, 0),  // N
    (-1, 1),  // NE
    (0, 1),   // E
    (1, 1),   // SE
    (1, 0),   // S
    (1, -1),  // SW
    (0, -1),  // W
    (-1, -1), // NW
];

/// Compute geomorphons landform classification
///
/// For each cell, looks along 8 directions up to `radius` distance.
/// In each direction, computes the zenith and nadir angles to determine
/// if the terrain is higher (+), lower (-), or at the same level (0).
/// The resulting 8-element ternary pattern classifies the landform.
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - Search radius and flatness threshold
///
/// # Returns
/// Raster<u8> with landform class codes (1-10)
pub fn geomorphons(dem: &Raster<f64>, params: GeomorphonParams) -> Result<Raster<u8>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Radius must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();
    let threshold_rad = params.flatness_threshold.to_radians();

    let output_data: Vec<u8> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![0u8; cols];

            for col in 0..cols {
                let z0 = unsafe { dem.get_unchecked(row, col) };
                if z0.is_nan() {
                    continue;
                }

                let mut pattern = [0i8; 8];

                for (dir_idx, &(dr, dc)) in DIRECTIONS.iter().enumerate() {
                    // Find the maximum angle looking up (zenith) and down (nadir)
                    let mut max_up_angle = f64::NEG_INFINITY;
                    let mut max_down_angle = f64::NEG_INFINITY;

                    for step in 1..=params.radius {
                        let nr = row as isize + dr * step as isize;
                        let nc = col as isize + dc * step as isize;

                        if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                            break;
                        }

                        let z = unsafe { dem.get_unchecked(nr as usize, nc as usize) };
                        if z.is_nan() {
                            break;
                        }

                        let dist = step as f64 * cell_size
                            * if dr != 0 && dc != 0 { std::f64::consts::SQRT_2 } else { 1.0 };
                        let dz = z - z0;
                        let angle = (dz / dist).atan();

                        if angle > 0.0 && angle > max_up_angle {
                            max_up_angle = angle;
                        }
                        if angle < 0.0 && (-angle) > max_down_angle {
                            max_down_angle = -angle;
                        }
                    }

                    // Determine ternary code
                    // +1 = terrain is higher (looking up dominates)
                    // -1 = terrain is lower (looking down dominates)
                    //  0 = flat (both angles below threshold)
                    if max_up_angle > threshold_rad && max_up_angle >= max_down_angle {
                        pattern[dir_idx] = 1;  // Higher
                    } else if max_down_angle > threshold_rad && max_down_angle > max_up_angle {
                        pattern[dir_idx] = -1; // Lower
                    } else {
                        pattern[dir_idx] = 0;  // Flat
                    }
                }

                row_data[col] = classify_pattern(&pattern);
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<u8>(rows, cols);
    output.set_nodata(Some(0));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

/// Classify a ternary pattern into one of 10 landform classes
///
/// Based on counting the number of + and - values:
/// - All +  = pit (surrounded by higher terrain)
/// - All -  = peak (surrounded by lower terrain)
/// - Mixed  = various landform types based on connectivity
fn classify_pattern(pattern: &[i8; 8]) -> u8 {
    let n_plus = pattern.iter().filter(|&&v| v > 0).count();
    let n_minus = pattern.iter().filter(|&&v| v < 0).count();
    let n_zero = pattern.iter().filter(|&&v| v == 0).count();

    // Count connected + and - segments
    let plus_segments = count_segments(pattern, 1);
    let minus_segments = count_segments(pattern, -1);

    match (n_plus, n_minus, n_zero) {
        (8, 0, 0) => class::PIT,       // All higher → pit
        (0, 8, 0) => class::PEAK,      // All lower → peak
        (0, 0, 8) => class::FLAT,      // All flat → flat

        // Ridge-like: mostly lower, no higher
        (0, n, _) if n >= 6 => class::RIDGE,
        // Valley-like: mostly higher, no lower
        (n, 0, _) if n >= 6 => class::VALLEY,

        // Shoulder: mostly lower with some flat
        (0, n, _) if n >= 4 => class::SHOULDER,
        // Footslope: mostly higher with some flat
        (n, 0, _) if n >= 4 => class::FOOTSLOPE,

        // Spur: mostly lower, few higher (promontory)
        (p, m, _) if m > p && m >= 5 && plus_segments <= 1 => class::SPUR,
        // Hollow: mostly higher, few lower (concavity)
        (p, m, _) if p > m && p >= 5 && minus_segments <= 1 => class::HOLLOW,

        // Ridge with some variation
        (_, m, _) if m >= 5 && minus_segments == 1 => class::RIDGE,
        // Valley with some variation
        (p, _, _) if p >= 5 && plus_segments == 1 => class::VALLEY,

        // Spur: lower on two sides, higher on one
        (p, m, _) if m > p && m >= 3 => class::SPUR,
        // Hollow: higher on two sides, lower on one
        (p, m, _) if p > m && p >= 3 => class::HOLLOW,

        // General slope: asymmetric pattern
        _ => class::SLOPE,
    }
}

/// Count the number of connected segments of a given value in circular array
fn count_segments(pattern: &[i8; 8], target: i8) -> usize {
    let mut count = 0;
    let mut in_segment = false;

    // Check circular continuity: start from a non-target position
    let mut start = 0;
    for i in 0..8 {
        if pattern[i] != target {
            start = i;
            break;
        }
    }

    for i in 0..8 {
        let idx = (start + i) % 8;
        if pattern[idx] == target {
            if !in_segment {
                count += 1;
                in_segment = true;
            }
        } else {
            in_segment = false;
        }
    }

    // Edge case: all values match
    if pattern.iter().all(|&v| v == target) {
        return 1;
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_dem(size: usize) -> Raster<f64> {
        let mut dem = Raster::new(size, size);
        dem.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        dem
    }

    #[test]
    fn test_geomorphons_peak() {
        // High center, low surroundings → peak
        let mut dem = make_dem(21);
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, 100.0 - (dx * dx + dy * dy).sqrt() * 5.0).unwrap();
            }
        }

        let result = geomorphons(&dem, GeomorphonParams { radius: 8, flatness_threshold: 1.0 }).unwrap();
        let center = result.get(10, 10).unwrap();
        assert_eq!(center, class::PEAK, "Center of hill should be peak, got {}", center);
    }

    #[test]
    fn test_geomorphons_pit() {
        // Low center, high surroundings → pit
        let mut dem = make_dem(21);
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, (dx * dx + dy * dy).sqrt() * 5.0).unwrap();
            }
        }

        let result = geomorphons(&dem, GeomorphonParams { radius: 8, flatness_threshold: 1.0 }).unwrap();
        let center = result.get(10, 10).unwrap();
        assert_eq!(center, class::PIT, "Center of depression should be pit, got {}", center);
    }

    #[test]
    fn test_geomorphons_flat() {
        let dem = Raster::filled(21, 21, 100.0_f64);
        let result = geomorphons(&dem, GeomorphonParams { radius: 5, flatness_threshold: 1.0 }).unwrap();
        let center = result.get(10, 10).unwrap();
        assert_eq!(center, class::FLAT, "Flat surface should be FLAT, got {}", center);
    }

    #[test]
    fn test_geomorphons_radius_zero() {
        let dem = Raster::filled(5, 5, 100.0_f64);
        assert!(geomorphons(&dem, GeomorphonParams { radius: 0, flatness_threshold: 1.0 }).is_err());
    }

    #[test]
    fn test_classify_all_plus() {
        let pattern = [1, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(classify_pattern(&pattern), class::PIT);
    }

    #[test]
    fn test_classify_all_minus() {
        let pattern = [-1, -1, -1, -1, -1, -1, -1, -1];
        assert_eq!(classify_pattern(&pattern), class::PEAK);
    }

    #[test]
    fn test_classify_all_zero() {
        let pattern = [0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(classify_pattern(&pattern), class::FLAT);
    }

    #[test]
    fn test_count_segments() {
        assert_eq!(count_segments(&[1, 1, -1, -1, 1, 1, -1, -1], 1), 2);
        assert_eq!(count_segments(&[1, 1, 1, 1, 1, 1, 1, 1], 1), 1);
        assert_eq!(count_segments(&[-1, -1, -1, -1, -1, -1, -1, -1], 1), 0);
    }
}
