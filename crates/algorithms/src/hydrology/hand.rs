//! HAND — Height Above Nearest Drainage
//!
//! For each cell, traces the D8 flow path downstream until reaching a
//! "stream" cell (flow accumulation ≥ threshold), then computes the
//! elevation difference between the cell and that stream cell.
//!
//! HAND = 0 for stream cells themselves, and increases with distance
//! from the drainage network. It is a fundamental flood-mapping index.
//!
//! Reference:
//! Nobre, A.D. et al. (2011). HAND, a new terrain descriptor using
//! SRTM-DEM. *Mapping Ecology and Conservation*, 275–287.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// D8 neighbor offsets matching flow direction encoding (1=E, 2=NE, ..., 8=SE)
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

/// Parameters for HAND computation
#[derive(Debug, Clone)]
pub struct HandParams {
    /// Flow accumulation threshold to define stream cells.
    /// Cells with accumulation >= this value are considered streams.
    /// Default: 1000.0 (cells)
    pub stream_threshold: f64,
}

impl Default for HandParams {
    fn default() -> Self {
        Self {
            stream_threshold: 1000.0,
        }
    }
}

/// Compute Height Above Nearest Drainage (HAND).
///
/// For each cell, follows the D8 flow direction downstream until a
/// stream cell is reached (accumulation ≥ threshold), then returns
/// the elevation difference.
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `flow_dir` - D8 flow direction raster (from `flow_direction`)
/// * `flow_acc` - Flow accumulation raster (from `flow_accumulation`)
/// * `params` - HAND parameters (stream threshold)
///
/// # Returns
/// Raster<f64> with HAND values (meters). Stream cells have HAND = 0.
/// Cells that cannot reach a stream (e.g., pits) get NaN.
pub fn hand(
    dem: &Raster<f64>,
    flow_dir: &Raster<u8>,
    flow_acc: &Raster<f64>,
    params: HandParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let (fd_rows, fd_cols) = flow_dir.shape();
    let (fa_rows, fa_cols) = flow_acc.shape();

    if rows != fd_rows || cols != fd_cols {
        return Err(Error::SizeMismatch {
            er: rows, ec: cols,
            ar: fd_rows, ac: fd_cols,
        });
    }
    if rows != fa_rows || cols != fa_cols {
        return Err(Error::SizeMismatch {
            er: rows, ec: cols,
            ar: fa_rows, ac: fa_cols,
        });
    }

    let nodata = dem.nodata();
    let threshold = params.stream_threshold;
    let total = rows * cols;

    // Build stream mask: true if accumulation >= threshold
    let mut is_stream = vec![false; total];
    for row in 0..rows {
        for col in 0..cols {
            let acc = unsafe { flow_acc.get_unchecked(row, col) };
            if acc >= threshold {
                is_stream[row * cols + col] = true;
            }
        }
    }

    // For each cell, find its nearest downstream stream cell elevation.
    // We cache results to avoid redundant path tracing.
    // stream_elev[i] = Some(elev) if resolved, None if not yet visited.
    let mut stream_elev: Vec<Option<f64>> = vec![None; total];

    // Pre-fill stream cells
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            if is_stream[idx] {
                stream_elev[idx] = Some(unsafe { dem.get_unchecked(row, col) });
            }
        }
    }

    // Trace each cell downstream to find its stream cell elevation
    for start_row in 0..rows {
        for start_col in 0..cols {
            let start_idx = start_row * cols + start_col;

            if stream_elev[start_idx].is_some() {
                continue; // Already resolved (stream cell or previously traced)
            }

            let z = unsafe { dem.get_unchecked(start_row, start_col) };
            if z.is_nan() || nodata.is_some_and(|nd| (z - nd).abs() < f64::EPSILON) {
                continue; // Nodata
            }

            // Trace the path downstream, collecting cells along the way
            let mut path: Vec<usize> = Vec::new();
            let mut cur_row = start_row;
            let mut cur_col = start_col;
            let mut found_elev: Option<f64> = None;

            // Use visited set to detect cycles (shouldn't happen with proper
            // fill+D8, but guard against infinite loops)
            let max_steps = total; // absolute safety limit
            let mut steps = 0;

            loop {
                let idx = cur_row * cols + cur_col;

                // Check if we've already resolved this cell
                if let Some(elev) = stream_elev[idx] {
                    found_elev = Some(elev);
                    break;
                }

                path.push(idx);
                steps += 1;
                if steps > max_steps {
                    break; // Safety: avoid infinite loops
                }

                // Follow flow direction
                let dir = unsafe { flow_dir.get_unchecked(cur_row, cur_col) };
                if dir == 0 || dir > 8 {
                    break; // Pit or invalid — no downstream path
                }

                let (dr, dc) = D8_OFFSETS[(dir - 1) as usize];
                let nr = cur_row as isize + dr;
                let nc = cur_col as isize + dc;

                if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                    break; // Flows off grid edge
                }

                cur_row = nr as usize;
                cur_col = nc as usize;
            }

            // Cache the result for all cells along the path
            if let Some(elev) = found_elev {
                for &idx in &path {
                    stream_elev[idx] = Some(elev);
                }
            }
            // If found_elev is None, cells remain None (will be NaN in output)
        }
    }

    // Build output: HAND = DEM(cell) - stream_elev(cell)
    let mut output_data = Array2::<f64>::from_elem((rows, cols), f64::NAN);
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            if let Some(se) = stream_elev[idx] {
                let z = unsafe { dem.get_unchecked(row, col) };
                if !z.is_nan() {
                    let h = z - se;
                    // HAND should be >= 0; negative values indicate DEM
                    // inconsistencies (clamp to 0)
                    output_data[(row, col)] = if h < 0.0 { 0.0 } else { h };
                }
            }
        }
    }

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = output_data;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydrology::{flow_accumulation, flow_direction};
    use crate::hydrology::fill_sinks::{fill_sinks, FillSinksParams};
    use surtgis_core::GeoTransform;

    /// V-shaped valley: elevation = |col - center|
    /// All flow converges to the center column and flows south.
    fn v_valley() -> Raster<f64> {
        let rows = 10;
        let cols = 11; // center at col=5
        let mut dem = Raster::new(rows, cols);
        dem.set_transform(GeoTransform::new(0.0, cols as f64, 1.0, -1.0));
        for row in 0..rows {
            for col in 0..cols {
                // V-shape cross-section + south slope
                let cross = (col as f64 - 5.0).abs();
                let along = (rows - 1 - row) as f64 * 0.1; // slight south slope
                dem.set(row, col, cross + along).unwrap();
            }
        }
        dem
    }

    #[test]
    fn test_hand_stream_cells_zero() {
        let dem = v_valley();
        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();
        let fdir = flow_direction(&filled).unwrap();
        let facc = flow_accumulation(&fdir).unwrap();

        // Use a low threshold so center column cells become streams
        let result = hand(&dem, &fdir, &facc, HandParams { stream_threshold: 3.0 }).unwrap();

        // Stream cells (center column, high accumulation) should have HAND ≈ 0
        // Check a cell that is definitely a stream (bottom center has max acc)
        let bottom_center = result.get(9, 5).unwrap();
        assert!(
            bottom_center < 0.5,
            "Stream cell should have HAND ≈ 0, got {}",
            bottom_center
        );
    }

    #[test]
    fn test_hand_increases_from_stream() {
        let dem = v_valley();
        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();
        let fdir = flow_direction(&filled).unwrap();
        let facc = flow_accumulation(&fdir).unwrap();

        let result = hand(&dem, &fdir, &facc, HandParams { stream_threshold: 3.0 }).unwrap();

        // HAND should increase away from the center column
        // Compare a cell 1 step from center vs 3 steps from center (same row)
        let row = 5;
        let h_near = result.get(row, 4).unwrap(); // 1 col from center
        let h_far = result.get(row, 2).unwrap();  // 3 cols from center

        // Both should be non-NaN
        assert!(!h_near.is_nan(), "Near-stream cell should have valid HAND");
        assert!(!h_far.is_nan(), "Far-from-stream cell should have valid HAND");

        // Far cell should have greater or equal HAND
        assert!(
            h_far >= h_near - 0.01,
            "HAND should increase away from stream: near={}, far={}",
            h_near,
            h_far
        );
    }

    #[test]
    fn test_hand_all_non_negative() {
        let dem = v_valley();
        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();
        let fdir = flow_direction(&filled).unwrap();
        let facc = flow_accumulation(&fdir).unwrap();

        let result = hand(&dem, &fdir, &facc, HandParams { stream_threshold: 3.0 }).unwrap();

        let (rows, cols) = result.shape();
        for row in 0..rows {
            for col in 0..cols {
                let val = result.get(row, col).unwrap();
                if !val.is_nan() {
                    assert!(
                        val >= 0.0,
                        "HAND must be >= 0 at ({}, {}), got {}",
                        row, col, val
                    );
                }
            }
        }
    }

    #[test]
    fn test_hand_dimension_mismatch() {
        let dem = Raster::<f64>::new(5, 5);
        let fdir = Raster::<u8>::new(3, 3);
        let facc = Raster::<f64>::new(5, 5);

        let result = hand(&dem, &fdir, &facc, HandParams::default());
        assert!(result.is_err(), "Should error on dimension mismatch");
    }
}
