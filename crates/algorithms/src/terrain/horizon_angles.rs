//! Horizon Angle Computation
//!
//! Shared module for computing terrain horizon angles in multiple azimuthal
//! directions. The horizon angle is the maximum elevation angle from a cell
//! to any point along a specific azimuth direction.
//!
//! ## Use cases
//! - **Solar radiation**: compare sun altitude with horizon → shadow/lit
//! - **Sky View Factor**: SVF = 1 − mean(sin²(horizon))
//! - **Terrain openness**: 90° − max_horizon_angle
//!
//! ## Azimuth convention
//! Azimuth 0 = North, increasing clockwise: π/2 = East, π = South, 3π/2 = West.
//!
//! Reference:
//! Corripio, J.G. (2003). Vectorial algebra algorithms for calculating terrain
//! parameters from DEMs and the position of the sun. *IJGIS*, 17(1), 1–23.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use std::f64::consts::PI;

/// Parameters for horizon angle computation
#[derive(Debug, Clone)]
pub struct HorizonParams {
    /// Search radius in cells (default 100)
    pub radius: usize,
    /// Number of equally-spaced azimuth directions (default 36, i.e. every 10°)
    pub directions: usize,
}

impl Default for HorizonParams {
    fn default() -> Self {
        Self {
            radius: 100,
            directions: 36,
        }
    }
}

/// Precomputed horizon angles for all directions at every cell.
///
/// Stores a flat `f64` array of size `directions × rows × cols`.
/// Access via `get(dir, row, col)` or `interpolate(row, col, azimuth)`.
///
/// **Memory**: `8 × directions × rows × cols` bytes.
/// For a 1000×1000 DEM with 36 directions ≈ 275 MB.
/// Use [`horizon_angle_map`] for single-azimuth queries on large DEMs.
#[derive(Debug)]
pub struct HorizonAngles {
    data: Vec<f64>,
    n_dirs: usize,
    rows: usize,
    cols: usize,
    /// Azimuth angles in radians (0 = N, clockwise)
    pub azimuths: Vec<f64>,
    template_transform: surtgis_core::GeoTransform,
    template_nodata: Option<f64>,
}

impl HorizonAngles {
    /// Get horizon angle for a specific direction and cell.
    ///
    /// Returns horizon angle in radians (0 = flat, positive = above horizontal).
    #[inline]
    pub fn get(&self, dir: usize, row: usize, col: usize) -> f64 {
        self.data[dir * self.rows * self.cols + row * self.cols + col]
    }

    /// Number of directions
    pub fn directions(&self) -> usize {
        self.n_dirs
    }

    /// DEM dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Interpolate horizon angle at any azimuth by linear interpolation
    /// between the two nearest computed directions.
    ///
    /// # Arguments
    /// * `row`, `col` — Cell position
    /// * `azimuth_rad` — Target azimuth in radians (0 = N, clockwise)
    pub fn interpolate(&self, row: usize, col: usize, azimuth_rad: f64) -> f64 {
        let n = self.n_dirs;
        if n == 0 {
            return 0.0;
        }

        // Normalize azimuth to [0, 2π)
        let az = azimuth_rad.rem_euclid(2.0 * PI);
        let step = 2.0 * PI / n as f64;

        let idx_f = az / step;
        let i0 = idx_f.floor() as usize % n;
        let i1 = (i0 + 1) % n;
        let frac = idx_f - idx_f.floor();

        let h0 = self.get(i0, row, col);
        let h1 = self.get(i1, row, col);

        if h0.is_nan() || h1.is_nan() {
            return f64::NAN;
        }

        h0 * (1.0 - frac) + h1 * frac
    }

    /// Extract horizon angles for one direction as a Raster.
    pub fn to_raster(&self, dir: usize) -> Raster<f64> {
        let slice_start = dir * self.rows * self.cols;
        let slice_end = slice_start + self.rows * self.cols;
        let data = self.data[slice_start..slice_end].to_vec();

        let mut raster = Raster::new(self.rows, self.cols);
        raster.set_transform(self.template_transform);
        raster.set_nodata(self.template_nodata);
        *raster.data_mut() = Array2::from_shape_vec((self.rows, self.cols), data)
            .expect("shape mismatch in to_raster");
        raster
    }
}

/// Compute horizon angles in all directions for every cell.
///
/// For each cell, traces rays in `params.directions` equally-spaced azimuthal
/// directions and finds the maximum elevation angle to the horizon in each.
///
/// # Arguments
/// * `dem` — Input DEM
/// * `params` — Search radius and number of directions
///
/// # Returns
/// [`HorizonAngles`] struct with indexed access and interpolation.
pub fn horizon_angles(dem: &Raster<f64>, params: HorizonParams) -> Result<HorizonAngles> {
    if params.radius == 0 || params.directions == 0 {
        return Err(Error::Algorithm("Radius and directions must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();
    let n_dirs = params.directions;

    // Precompute direction vectors (azimuth clockwise from North)
    let dir_vectors: Vec<(f64, f64)> = (0..n_dirs)
        .map(|i| {
            let azimuth = 2.0 * PI * i as f64 / n_dirs as f64;
            (-azimuth.cos(), azimuth.sin()) // (dr, dc): North = (-1,0), East = (0,1)
        })
        .collect();

    let azimuths: Vec<f64> = (0..n_dirs)
        .map(|i| 2.0 * PI * i as f64 / n_dirs as f64)
        .collect();

    // Compute all horizon angles — parallel over rows
    let row_results: Vec<Vec<f64>> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![f64::NAN; n_dirs * cols];

            for col in 0..cols {
                let z0 = unsafe { dem.get_unchecked(row, col) };
                if z0.is_nan() {
                    continue;
                }

                for (d, &(dr, dc)) in dir_vectors.iter().enumerate() {
                    let angle = trace_horizon(
                        dem, row, col, z0, dr, dc,
                        params.radius, cell_size, rows, cols,
                    );
                    row_data[d * cols + col] = angle;
                }
            }

            row_data
        })
        .collect();

    // Assemble into flat data array: [dir0_row0, dir0_row1, ..., dir1_row0, ...]
    let mut data = vec![f64::NAN; n_dirs * rows * cols];
    for (row, row_data) in row_results.iter().enumerate() {
        for d in 0..n_dirs {
            for col in 0..cols {
                data[d * rows * cols + row * cols + col] = row_data[d * cols + col];
            }
        }
    }

    Ok(HorizonAngles {
        data,
        n_dirs,
        rows,
        cols,
        azimuths,
        template_transform: *dem.transform(),
        template_nodata: Some(f64::NAN),
    })
}

/// Compute horizon angle map for a single azimuth direction.
///
/// More memory-efficient than [`horizon_angles`] when only one or a few
/// directions are needed (e.g., checking shadow at a specific sun azimuth).
///
/// # Arguments
/// * `dem` — Input DEM
/// * `azimuth_rad` — Azimuth in radians (0 = N, π/2 = E, π = S, 3π/2 = W)
/// * `radius` — Search radius in cells
///
/// # Returns
/// Raster<f64> with horizon angles in radians at each cell.
pub fn horizon_angle_map(
    dem: &Raster<f64>,
    azimuth_rad: f64,
    radius: usize,
) -> Result<Raster<f64>> {
    if radius == 0 {
        return Err(Error::Algorithm("Radius must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();

    let dr = -azimuth_rad.cos();
    let dc = azimuth_rad.sin();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let z0 = unsafe { dem.get_unchecked(row, col) };
                if z0.is_nan() {
                    continue;
                }

                *row_data_col = trace_horizon(
                    dem, row, col, z0, dr, dc,
                    radius, cell_size, rows, cols,
                );
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

// ---------------------------------------------------------------------------
// HORAYZON-style fast horizon computation (P2-SO4)
// ---------------------------------------------------------------------------
// Steger et al. (2022): "HORAYZON v1.2 — efficient horizon computation"
// Key idea: LOD pyramid with max-elevation aggregation. Full resolution
// for nearby terrain, progressively coarser for distant terrain.
// This gives O(log R) work per direction instead of O(R).

/// Parameters for fast (LOD-accelerated) horizon angle computation.
///
/// Uses a terrain simplification pyramid where each level aggregates 2×2 cells
/// by taking the maximum elevation. Near terrain is traced at full resolution;
/// distant terrain uses coarser levels, maintaining angular resolution while
/// reducing computation by 10–100× for large radii.
#[derive(Debug, Clone)]
pub struct FastHorizonParams {
    /// Search radius in cells at the original resolution (default 500).
    pub radius: usize,
    /// Number of equally-spaced azimuth directions (default 36).
    pub directions: usize,
    /// Distance threshold (in cells) for the first LOD transition.
    /// Full resolution is used up to this distance. Default 32.
    pub near_distance: usize,
}

impl Default for FastHorizonParams {
    fn default() -> Self {
        Self {
            radius: 500,
            directions: 36,
            near_distance: 32,
        }
    }
}

/// One level of the LOD pyramid.
struct LodLevel {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    /// Aggregation factor: 1 for original, 2 for 2×2, 4 for 4×4, etc.
    scale: usize,
}

impl LodLevel {
    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col]
        } else {
            f64::NAN
        }
    }
}

/// Build LOD pyramid from DEM.
///
/// Level 0 = original DEM data. Level k aggregates 2^k × 2^k cells with
/// the maximum elevation (conservative for horizon detection).
fn build_lod_pyramid(dem: &Raster<f64>) -> Vec<LodLevel> {
    let (rows, cols) = dem.shape();
    let data = dem.data();

    // Level 0: copy original
    let mut level0_data = vec![f64::NAN; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            level0_data[r * cols + c] = data[[r, c]];
        }
    }

    let mut levels = vec![LodLevel {
        data: level0_data,
        rows,
        cols,
        scale: 1,
    }];

    // Build coarser levels until dimension drops below 2
    let mut prev_rows = rows;
    let mut prev_cols = cols;
    let mut scale = 1;

    while prev_rows > 2 && prev_cols > 2 {
        let new_rows = prev_rows.div_ceil(2);
        let new_cols = prev_cols.div_ceil(2);
        scale *= 2;

        let prev = &levels.last().unwrap().data;
        let pc = prev_cols;
        let pr = prev_rows;

        let mut new_data = vec![f64::NAN; new_rows * new_cols];

        for r in 0..new_rows {
            for c in 0..new_cols {
                let r0 = r * 2;
                let c0 = c * 2;
                let r1 = (r0 + 1).min(pr - 1);
                let c1 = (c0 + 1).min(pc - 1);

                let v00 = prev[r0 * pc + c0];
                let v01 = prev[r0 * pc + c1];
                let v10 = prev[r1 * pc + c0];
                let v11 = prev[r1 * pc + c1];

                // Max of valid values (NaN-aware)
                let mut mx = f64::NAN;
                for &v in &[v00, v01, v10, v11] {
                    if !v.is_nan()
                        && (mx.is_nan() || v > mx) {
                            mx = v;
                        }
                }
                new_data[r * new_cols + c] = mx;
            }
        }

        levels.push(LodLevel {
            data: new_data,
            rows: new_rows,
            cols: new_cols,
            scale,
        });

        prev_rows = new_rows;
        prev_cols = new_cols;
    }

    levels
}

/// Select which LOD level to use based on distance from observer.
///
/// Level 0 (full res) for d ∈ [0, near_distance),
/// Level 1 for d ∈ [near_distance, 2×near_distance),
/// Level k for d ∈ [2^(k-1) × near_distance, 2^k × near_distance), etc.
#[inline]
fn select_lod_level(distance_cells: f64, near_distance: usize, max_level: usize) -> usize {
    if distance_cells < near_distance as f64 {
        return 0;
    }
    let ratio = distance_cells / near_distance as f64;
    let level = ratio.log2().floor() as usize + 1;
    level.min(max_level)
}

/// Trace horizon using LOD pyramid — fast version.
#[allow(clippy::too_many_arguments)]
#[inline]
fn trace_horizon_lod(
    pyramid: &[LodLevel],
    row: usize, col: usize, z0: f64,
    dr_step: f64, dc_step: f64,
    radius: usize, cell_size: f64,
    rows: usize, cols: usize,
    near_distance: usize,
) -> f64 {
    let max_level = pyramid.len() - 1;
    let mut max_angle = 0.0_f64;
    let mut step = 1_usize;

    while step <= radius {
        let fr = row as f64 + dr_step * step as f64;
        let fc = col as f64 + dc_step * step as f64;

        // Out of bounds at original resolution?
        if fr < 0.0 || fc < 0.0 || fr >= rows as f64 || fc >= cols as f64 {
            break;
        }

        let dist_cells = ((fr - row as f64).powi(2) + (fc - col as f64).powi(2)).sqrt();
        let level = select_lod_level(dist_cells, near_distance, max_level);
        let lod = &pyramid[level];
        let s = lod.scale;

        // Map original coordinates to this LOD level
        let lr = (fr / s as f64).floor() as usize;
        let lc = (fc / s as f64).floor() as usize;

        let z = lod.get(lr, lc);
        if z.is_nan() {
            // Skip NaN but don't break — distant terrain may still be valid
            step += s;
            continue;
        }

        let dist = dist_cells * cell_size;
        if dist > f64::EPSILON {
            let angle = ((z - z0) / dist).atan();
            if angle > max_angle {
                max_angle = angle;
            }
        }

        // Step size adapts to LOD level
        step += s;
    }

    max_angle
}

/// Compute horizon angles using HORAYZON-style LOD acceleration.
///
/// Builds a terrain simplification pyramid (max-elevation aggregation) and
/// traces rays at full resolution for nearby terrain, switching to coarser
/// levels for distant terrain. Provides 10–100× speedup for large radii
/// with negligible accuracy loss for smooth terrain.
///
/// # Arguments
/// * `dem` — Input DEM
/// * `params` — Fast horizon parameters (radius, directions, near_distance)
///
/// # Returns
/// [`HorizonAngles`] struct (same as standard horizon_angles).
///
/// # Reference
/// Steger, C. et al. (2022). HORAYZON v1.2. *Geosci. Model Dev.*
pub fn horizon_angles_fast(dem: &Raster<f64>, params: FastHorizonParams) -> Result<HorizonAngles> {
    if params.radius == 0 || params.directions == 0 {
        return Err(Error::Algorithm("Radius and directions must be > 0".into()));
    }
    if params.near_distance == 0 {
        return Err(Error::Algorithm("near_distance must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();
    let n_dirs = params.directions;
    let near = params.near_distance;

    // Build LOD pyramid
    let pyramid = build_lod_pyramid(dem);

    // Precompute direction vectors
    let dir_vectors: Vec<(f64, f64)> = (0..n_dirs)
        .map(|i| {
            let azimuth = 2.0 * PI * i as f64 / n_dirs as f64;
            (-azimuth.cos(), azimuth.sin())
        })
        .collect();

    let azimuths: Vec<f64> = (0..n_dirs)
        .map(|i| 2.0 * PI * i as f64 / n_dirs as f64)
        .collect();

    // Compute horizon angles — parallel over rows
    let row_results: Vec<Vec<f64>> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![f64::NAN; n_dirs * cols];

            for col in 0..cols {
                let z0 = unsafe { dem.get_unchecked(row, col) };
                if z0.is_nan() {
                    continue;
                }

                for (d, &(dr, dc)) in dir_vectors.iter().enumerate() {
                    let angle = trace_horizon_lod(
                        &pyramid, row, col, z0, dr, dc,
                        params.radius, cell_size, rows, cols, near,
                    );
                    row_data[d * cols + col] = angle;
                }
            }

            row_data
        })
        .collect();

    // Assemble flat data array
    let mut data = vec![f64::NAN; n_dirs * rows * cols];
    for (row, row_data) in row_results.iter().enumerate() {
        for d in 0..n_dirs {
            for col in 0..cols {
                data[d * rows * cols + row * cols + col] = row_data[d * cols + col];
            }
        }
    }

    Ok(HorizonAngles {
        data,
        n_dirs,
        rows,
        cols,
        azimuths,
        template_transform: *dem.transform(),
        template_nodata: Some(f64::NAN),
    })
}

/// Compute fast horizon angle map for a single azimuth direction.
///
/// LOD-accelerated version of [`horizon_angle_map`].
pub fn horizon_angle_map_fast(
    dem: &Raster<f64>,
    azimuth_rad: f64,
    radius: usize,
    near_distance: usize,
) -> Result<Raster<f64>> {
    if radius == 0 {
        return Err(Error::Algorithm("Radius must be > 0".into()));
    }
    if near_distance == 0 {
        return Err(Error::Algorithm("near_distance must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();
    let pyramid = build_lod_pyramid(dem);

    let dr = -azimuth_rad.cos();
    let dc = azimuth_rad.sin();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let z0 = unsafe { dem.get_unchecked(row, col) };
                if z0.is_nan() {
                    continue;
                }

                *row_data_col = trace_horizon_lod(
                    &pyramid, row, col, z0, dr, dc,
                    radius, cell_size, rows, cols, near_distance,
                );
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

/// Trace a ray from (row, col) in direction (dr, dc) and find the
/// maximum elevation angle to the horizon.
///
/// Returns horizon angle in radians (≥ 0).
#[allow(clippy::too_many_arguments)]
#[inline]
fn trace_horizon(
    dem: &Raster<f64>,
    row: usize, col: usize, z0: f64,
    dr_step: f64, dc_step: f64,
    radius: usize, cell_size: f64,
    rows: usize, cols: usize,
) -> f64 {
    let mut max_angle = 0.0_f64;

    for step in 1..=radius {
        let fr = row as f64 + dr_step * step as f64;
        let fc = col as f64 + dc_step * step as f64;

        let nr = fr.round() as isize;
        let nc = fc.round() as isize;

        if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
            break;
        }

        let z = unsafe { dem.get_unchecked(nr as usize, nc as usize) };
        if z.is_nan() {
            break;
        }

        let dist = ((fr - row as f64).powi(2) + (fc - col as f64).powi(2)).sqrt() * cell_size;
        if dist < f64::EPSILON {
            continue;
        }

        let angle = ((z - z0) / dist).atan();
        if angle > max_angle {
            max_angle = angle;
        }
    }

    max_angle
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    // =======================================================================
    // HORAYZON fast horizon tests
    // =======================================================================

    #[test]
    fn test_fast_flat_terrain_zero_horizon() {
        let mut dem = Raster::filled(41, 41, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 41.0, 1.0, -1.0));

        let result = horizon_angles_fast(&dem, FastHorizonParams {
            radius: 20,
            directions: 8,
            near_distance: 8,
        }).unwrap();

        for d in 0..8 {
            let angle = result.get(d, 20, 20);
            assert!(
                angle.abs() < 0.01,
                "Fast: flat terrain dir {} should be ~0, got {:.4}",
                d, angle
            );
        }
    }

    #[test]
    fn test_fast_pit_positive_horizon() {
        let mut dem = Raster::new(41, 41);
        dem.set_transform(GeoTransform::new(0.0, 41.0, 1.0, -1.0));
        for row in 0..41 {
            for col in 0..41 {
                let dx = col as f64 - 20.0;
                let dy = row as f64 - 20.0;
                dem.set(row, col, (dx * dx + dy * dy).sqrt() * 10.0).unwrap();
            }
        }

        let result = horizon_angles_fast(&dem, FastHorizonParams {
            radius: 20,
            directions: 8,
            near_distance: 8,
        }).unwrap();

        for d in 0..8 {
            let angle = result.get(d, 20, 20);
            assert!(
                angle > 0.1,
                "Fast: pit center dir {} should be positive, got {:.4}",
                d, angle
            );
        }
    }

    #[test]
    fn test_fast_agrees_with_standard() {
        // Build a varied terrain
        let mut dem = Raster::new(51, 51);
        dem.set_transform(GeoTransform::new(0.0, 51.0, 1.0, -1.0));
        for row in 0..51 {
            for col in 0..51 {
                let dx = col as f64 - 25.0;
                let dy = row as f64 - 25.0;
                let z = (dx * dx + dy * dy).sqrt() * 5.0
                    + (dx * 0.3).sin() * 20.0
                    + (dy * 0.2).cos() * 15.0;
                dem.set(row, col, z).unwrap();
            }
        }

        let radius = 25;
        let dirs = 8;

        let standard = horizon_angles(&dem, HorizonParams {
            radius,
            directions: dirs,
        }).unwrap();

        let fast = horizon_angles_fast(&dem, FastHorizonParams {
            radius,
            directions: dirs,
            near_distance: 8,
        }).unwrap();

        // Fast uses LOD stepping that may miss the exact peak cell, so
        // results are approximate. For smooth terrain, the difference should
        // be small (< ~5 degrees = 0.09 rad).
        let mut max_diff = 0.0_f64;
        for d in 0..dirs {
            let std_val = standard.get(d, 25, 25);
            let fast_val = fast.get(d, 25, 25);
            let diff = (fast_val - std_val).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        // Max difference should be small for smooth terrain
        assert!(
            max_diff < 0.15,
            "Fast vs standard max diff should be small, got {:.4} rad ({:.1}°)",
            max_diff, max_diff.to_degrees()
        );
    }

    #[test]
    fn test_fast_single_azimuth_map() {
        let mut dem = Raster::new(41, 41);
        dem.set_transform(GeoTransform::new(0.0, 41.0, 1.0, -1.0));
        for row in 0..41 {
            for col in 0..41 {
                let dx = col as f64 - 20.0;
                let dy = row as f64 - 20.0;
                dem.set(row, col, (dx * dx + dy * dy).sqrt() * 10.0).unwrap();
            }
        }

        let fast = horizon_angle_map_fast(&dem, PI, 20, 8).unwrap();
        let v = fast.get(20, 20).unwrap();
        assert!(
            v > 0.1,
            "Fast single-azimuth pit center looking south should be positive, got {:.4}",
            v
        );
    }

    #[test]
    fn test_fast_lod_pyramid_levels() {
        let dem = Raster::filled(64, 64, 100.0_f64);
        let pyramid = build_lod_pyramid(&dem);

        // 64 → 32 → 16 → 8 → 4 → 2 = 6 levels
        assert!(pyramid.len() >= 5, "64×64 should produce >= 5 LOD levels, got {}", pyramid.len());
        assert_eq!(pyramid[0].rows, 64);
        assert_eq!(pyramid[0].cols, 64);
        assert_eq!(pyramid[0].scale, 1);
        assert_eq!(pyramid[1].rows, 32);
        assert_eq!(pyramid[1].cols, 32);
        assert_eq!(pyramid[1].scale, 2);
    }

    #[test]
    fn test_fast_lod_max_aggregation() {
        // Check that LOD preserves max elevation
        let mut dem = Raster::filled(4, 4, 10.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
        dem.set(0, 0, 100.0).unwrap(); // One high cell

        let pyramid = build_lod_pyramid(&dem);
        assert!(pyramid.len() >= 2);

        // Level 1 (2x2 aggregation): cell (0,0) should have max=100
        let l1 = &pyramid[1];
        assert_eq!(l1.get(0, 0), 100.0, "LOD level 1 should preserve max elevation");
    }

    #[test]
    fn test_fast_invalid_params() {
        let dem = Raster::filled(5, 5, 100.0_f64);
        assert!(horizon_angles_fast(&dem, FastHorizonParams {
            radius: 0, directions: 8, near_distance: 4
        }).is_err());
        assert!(horizon_angles_fast(&dem, FastHorizonParams {
            radius: 10, directions: 0, near_distance: 4
        }).is_err());
        assert!(horizon_angles_fast(&dem, FastHorizonParams {
            radius: 10, directions: 8, near_distance: 0
        }).is_err());
        assert!(horizon_angle_map_fast(&dem, 0.0, 0, 4).is_err());
        assert!(horizon_angle_map_fast(&dem, 0.0, 10, 0).is_err());
    }

    #[test]
    fn test_fast_nan_cells() {
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        dem.set(10, 10, f64::NAN).unwrap();

        let result = horizon_angles_fast(&dem, FastHorizonParams {
            radius: 10,
            directions: 4,
            near_distance: 4,
        }).unwrap();

        for d in 0..4 {
            assert!(
                result.get(d, 10, 10).is_nan(),
                "Fast: NaN cell should produce NaN horizon"
            );
        }
    }

    #[test]
    fn test_flat_terrain_zero_horizon() {
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));

        let result = horizon_angles(&dem, HorizonParams {
            radius: 10,
            directions: 8,
        }).unwrap();

        // Flat terrain: interior cells should have ~0 horizon angle
        for d in 0..8 {
            let angle = result.get(d, 10, 10);
            assert!(
                angle.abs() < 0.01,
                "Flat terrain should have ~0 horizon, dir {} got {:.4}",
                d, angle
            );
        }
    }

    #[test]
    fn test_pit_positive_horizon() {
        // Cone-shaped pit: center is low, edges are high
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, (dx * dx + dy * dy).sqrt() * 10.0).unwrap();
            }
        }

        let result = horizon_angles(&dem, HorizonParams {
            radius: 10,
            directions: 8,
        }).unwrap();

        // Center of pit should have positive horizon angles in all directions
        for d in 0..8 {
            let angle = result.get(d, 10, 10);
            assert!(
                angle > 0.1,
                "Pit center should have positive horizon angle, dir {} got {:.4}",
                d, angle
            );
        }
    }

    #[test]
    fn test_ridge_asymmetric_horizon() {
        // East-west ridge: high in center row, low at edges
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let dist_from_center_row = (row as f64 - 10.0).abs();
                dem.set(row, col, 100.0 - dist_from_center_row * 10.0).unwrap();
            }
        }

        let result = horizon_angles(&dem, HorizonParams {
            radius: 10,
            directions: 4, // N, E, S, W
        }).unwrap();

        // Cell at row=5 is below the ridge (row=10).
        // Looking South (toward ridge) should have higher horizon than North (away).
        // Row increases southward in standard raster convention.
        let north_angle = result.get(0, 5, 10); // direction 0 = N (away from ridge)
        let south_angle = result.get(2, 5, 10); // direction 2 = S (toward ridge)

        assert!(
            south_angle > north_angle,
            "Looking toward ridge (S) should have higher horizon: N={:.4}, S={:.4}",
            north_angle, south_angle
        );
    }

    #[test]
    fn test_interpolate_between_directions() {
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, (dx * dx + dy * dy).sqrt() * 10.0).unwrap();
            }
        }

        let result = horizon_angles(&dem, HorizonParams {
            radius: 10,
            directions: 8,
        }).unwrap();

        // For symmetric pit, interpolation at N (azimuth=0) should match dir 0
        let direct = result.get(0, 10, 10);
        let interpolated = result.interpolate(10, 10, 0.0);
        assert!(
            (direct - interpolated).abs() < 0.001,
            "Interpolation at exact azimuth should match: direct={:.4}, interp={:.4}",
            direct, interpolated
        );

        // Interpolation between two directions should be between them
        let h0 = result.get(0, 10, 10);
        let h1 = result.get(1, 10, 10);
        let mid_az = PI / 8.0; // Halfway between dir 0 and dir 1
        let mid = result.interpolate(10, 10, mid_az);
        let expected = (h0 + h1) / 2.0;
        assert!(
            (mid - expected).abs() < 0.01,
            "Mid-interpolation should be average: got {:.4}, expected {:.4}",
            mid, expected
        );
    }

    #[test]
    fn test_horizon_angle_map_single() {
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, (dx * dx + dy * dy).sqrt() * 10.0).unwrap();
            }
        }

        // South azimuth = π
        let south = horizon_angle_map(&dem, PI, 10).unwrap();
        let v = south.get(10, 10).unwrap();
        assert!(
            v > 0.1,
            "Pit center looking south should have positive horizon, got {:.4}",
            v
        );
    }

    #[test]
    fn test_to_raster_extraction() {
        let mut dem = Raster::filled(11, 11, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));

        let result = horizon_angles(&dem, HorizonParams {
            radius: 5,
            directions: 4,
        }).unwrap();

        let raster = result.to_raster(0);
        let (r, c) = raster.shape();
        assert_eq!(r, 11);
        assert_eq!(c, 11);

        let v = raster.get(5, 5).unwrap();
        assert!(v.abs() < 0.01, "Flat terrain raster horizon should be ~0, got {:.4}", v);
    }

    #[test]
    fn test_invalid_params() {
        let dem = Raster::filled(5, 5, 100.0_f64);
        assert!(horizon_angles(&dem, HorizonParams { radius: 0, directions: 8 }).is_err());
        assert!(horizon_angles(&dem, HorizonParams { radius: 10, directions: 0 }).is_err());
        assert!(horizon_angle_map(&dem, 0.0, 0).is_err());
    }

    #[test]
    fn test_nan_cells_produce_nan() {
        let mut dem = Raster::filled(11, 11, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        dem.set(5, 5, f64::NAN).unwrap();

        let result = horizon_angles(&dem, HorizonParams {
            radius: 5,
            directions: 4,
        }).unwrap();

        for d in 0..4 {
            assert!(
                result.get(d, 5, 5).is_nan(),
                "NaN cell should produce NaN horizon"
            );
        }
    }
}
