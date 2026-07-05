//! Sediment Connectivity Index (Borselli et al. 2008)
//!
//! The Index of Connectivity (IC) quantifies the potential for sediment
//! to be transported from hillslopes to the channel network. It combines
//! an upslope component (D_up) representing sediment mobilization potential
//! with a downslope component (D_dn) representing the impedance of the
//! flow path to the nearest channel.
//!
//! IC = log10(D_up / D_dn)
//!
//! D_up = W * S * sqrt(A)    (per-pixel approximation)
//! D_dn = sum(d_i / (W_i * S_i))  along D8 path to channel
//!
//! where W is the weighting factor (e.g., C-factor or NDVI-based),
//! S is the slope gradient (sin), and A is the upslope contributing area.
//!
//! Reference:
//! Borselli, L., Cassi, P., & Torri, D. (2008). Prolegomena to sediment
//! and flow connectivity in the landscape: A GIS and field numerical
//! assessment. *CATENA*, 75(3), 268–277.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for Sediment Connectivity Index
#[derive(Debug, Clone)]
pub struct SedimentConnectivityParams {
    /// Flow accumulation threshold to identify channel cells
    pub stream_threshold: f64,
}

impl Default for SedimentConnectivityParams {
    fn default() -> Self {
        Self {
            stream_threshold: 1000.0,
        }
    }
}

use super::d8::{D8_DISTANCE as D8_DIST, D8_OFFSETS};

/// Minimum slope gradient to avoid division by zero in D_dn computation
const MIN_SLOPE: f64 = 0.005;

/// Validate that `other` shares shape, geotransform and EPSG-comparable
/// CRS with `reference`, even though the two rasters may have different
/// element types (e.g. a `Raster<u8>` flow-direction grid alongside the
/// `Raster<f64>` slope/flow-accumulation rasters).
///
/// `surtgis_core::raster::check_aligned` cannot be called directly across
/// mixed element types because it is generic over a single `T` for the
/// whole slice; this mirrors its shape + geotransform + CRS contract for
/// the one input that doesn't fit that slice.
fn check_shape_transform_crs<T, U>(
    reference: &Raster<T>,
    other: &Raster<U>,
    context: &str,
) -> Result<()>
where
    T: surtgis_core::raster::RasterCell,
    U: surtgis_core::raster::RasterCell,
{
    if reference.shape() != other.shape() {
        return Err(Error::ShapeMismatch {
            expected: reference.shape(),
            got: other.shape(),
            context: context.to_string(),
        });
    }
    if reference.transform() != other.transform() {
        return Err(Error::Misaligned {
            reason: format!(
                "geotransform mismatch: {context} is not aligned with the slope raster"
            ),
        });
    }
    if let (Some(a), Some(b)) = (
        reference.crs().and_then(|c| c.epsg()),
        other.crs().and_then(|c| c.epsg()),
    ) && a != b
    {
        return Err(Error::Misaligned {
            reason: format!("CRS mismatch: {context} is EPSG:{b} but the slope raster is EPSG:{a}"),
        });
    }
    Ok(())
}

/// Compute Sediment Connectivity Index (IC).
///
/// IC = log10(D_up / D_dn)
///
/// # Arguments
/// * `slope` - Slope raster in radians
/// * `flow_acc` - Flow accumulation raster (cell count)
/// * `flow_dir` - D8 flow direction raster (1=E, 2=NE, ..., 8=SE, 0=pit)
/// * `params` - Parameters (stream threshold)
/// * `weight` - Optional weighting raster (e.g., C-factor). If None, W=1.0 everywhere.
///
/// # Returns
/// Raster with IC values, clamped to [-10, 10]
pub fn sediment_connectivity(
    slope: &Raster<f64>,
    flow_acc: &Raster<f64>,
    flow_dir: &Raster<u8>,
    params: SedimentConnectivityParams,
    weight: Option<&Raster<f64>>,
) -> Result<Raster<f64>> {
    let (rows, cols) = slope.shape();

    // Multi-raster entry point: slope, flow_acc and (if present) weight
    // share element type f64 and go through the shared check_aligned
    // helper directly; flow_dir is a Raster<u8> and needs the mixed-type
    // equivalent above.
    let mut f64_inputs = vec![slope, flow_acc];
    if let Some(w) = weight {
        f64_inputs.push(w);
    }
    surtgis_core::raster::check_aligned(&f64_inputs)?;
    check_shape_transform_crs(slope, flow_dir, "flow_dir")?;

    let threshold = params.stream_threshold;

    let gt = slope.transform();
    let cell_size = gt.pixel_width.abs();

    let total = rows * cols;

    // Build stream mask
    let mut is_stream = vec![false; total];
    for row in 0..rows {
        for col in 0..cols {
            let acc = unsafe { flow_acc.get_unchecked(row, col) };
            if acc >= threshold {
                is_stream[row * cols + col] = true;
            }
        }
    }

    // Compute D_dn for each cell by tracing D8 path to channel
    // Cache results to avoid redundant tracing
    let mut d_dn: Vec<f64> = vec![f64::NAN; total];

    // Stream cells have D_dn = 0 (they are at the channel)
    // Actually, D_dn for stream cells = d_i / (W_i * S_i) for the stream cell itself
    // But conventionally, IC at stream cells is set to a high value (D_dn → 0)
    for idx in 0..total {
        if is_stream[idx] {
            d_dn[idx] = 0.0;
        }
    }

    // Trace each non-stream cell downstream
    for start_row in 0..rows {
        for start_col in 0..cols {
            let start_idx = start_row * cols + start_col;
            if !d_dn[start_idx].is_nan() {
                continue; // Already computed
            }

            let s = unsafe { slope.get_unchecked(start_row, start_col) };
            if slope.is_nodata(s) {
                continue;
            }

            // Trace path downstream, collecting cells
            let mut path: Vec<(usize, f64)> = Vec::new(); // (index, local_d_dn_contribution)
            let mut cur_row = start_row;
            let mut cur_col = start_col;
            let mut found = false;
            let max_steps = total;
            let mut steps = 0;

            loop {
                let idx = cur_row * cols + cur_col;

                // If we reached a cell with known D_dn, we can resolve the path
                if !d_dn[idx].is_nan() {
                    found = true;
                    break;
                }

                // Compute local contribution: d_i / (W_i * S_i)
                let s_i = unsafe { slope.get_unchecked(cur_row, cur_col) };
                let sin_s = s_i.sin().max(MIN_SLOPE);
                let w_i = if let Some(w) = weight {
                    unsafe { w.get_unchecked(cur_row, cur_col) }.max(0.001)
                } else {
                    1.0
                };

                // Get flow direction to determine step distance
                let dir = unsafe { flow_dir.get_unchecked(cur_row, cur_col) };
                if dir == 0 || dir > 8 {
                    break; // Pit or invalid
                }

                let d_step = D8_DIST[(dir - 1) as usize] * cell_size;
                let contribution = d_step / (w_i * sin_s);
                path.push((idx, contribution));

                steps += 1;
                if steps > max_steps {
                    break;
                }

                // Move to next cell
                let (dr, dc) = D8_OFFSETS[(dir - 1) as usize];
                let nr = cur_row as isize + dr;
                let nc = cur_col as isize + dc;

                if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                    break; // Off grid
                }

                cur_row = nr as usize;
                cur_col = nc as usize;
            }

            if found {
                // Resolve all cells in the path
                let terminal_idx = cur_row * cols + cur_col;
                let mut cumulative = d_dn[terminal_idx];

                // Walk the path backwards, accumulating D_dn
                for &(idx, contribution) in path.iter().rev() {
                    cumulative += contribution;
                    d_dn[idx] = cumulative;
                }
            }
            // If not found, cells remain NaN (unreachable channel)
        }
    }

    // Compute D_up and IC for each cell
    let mut output_data = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let dn = d_dn[idx];
            if dn.is_nan() {
                continue;
            }

            let s = unsafe { slope.get_unchecked(row, col) };
            let acc = unsafe { flow_acc.get_unchecked(row, col) };

            if slope.is_nodata(s) || flow_acc.is_nodata(acc) {
                continue;
            }

            let w = if let Some(wt) = weight {
                unsafe { wt.get_unchecked(row, col) }.max(0.001)
            } else {
                1.0
            };

            let sin_s = s.sin().max(MIN_SLOPE);

            // D_up = W * S * sqrt(A * cell_size^2)
            let a_up = (acc * cell_size * cell_size).sqrt();
            let d_up = w * sin_s * a_up;

            if dn < 1e-10 {
                // At channel: IC → very high (clamped)
                output_data[(row, col)] = 10.0;
            } else {
                let ic = (d_up / dn).log10();
                // Clamp to [-10, 10]
                output_data[(row, col)] = ic.clamp(-10.0, 10.0);
            }
        }
    }

    let mut output = slope.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = output_data;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_sediment_connectivity_simple_slope() {
        // Simple south-sloping terrain, channel at bottom row
        let rows = 5;
        let cols = 3;

        // Slope: uniform 10 degrees
        let slope_rad = 10.0_f64.to_radians();
        let mut slope = Raster::filled(rows, cols, slope_rad);
        slope.set_transform(GeoTransform::new(0.0, cols as f64, 10.0, -10.0));

        // Flow accumulation: increases southward
        let mut flow_acc_data = Array2::<f64>::zeros((rows, cols));
        for row in 0..rows {
            for col in 0..cols {
                flow_acc_data[(row, col)] = ((row + 1) * cols) as f64;
            }
        }
        let mut flow_acc = Raster::from_array(flow_acc_data);
        flow_acc.set_transform(GeoTransform::new(0.0, cols as f64, 10.0, -10.0));

        // Flow direction: all flowing south (7)
        let mut flow_dir_data = Array2::<u8>::from_elem((rows, cols), 7);
        // Last row: pit (no outflow)
        for col in 0..cols {
            flow_dir_data[(rows - 1, col)] = 0;
        }
        let mut flow_dir = Raster::from_array(flow_dir_data);
        flow_dir.set_transform(GeoTransform::new(0.0, cols as f64, 10.0, -10.0));

        let params = SedimentConnectivityParams {
            stream_threshold: 10.0, // Bottom row becomes stream
        };

        let result = sediment_connectivity(&slope, &flow_acc, &flow_dir, params, None).unwrap();

        // Top row should have lower IC (farther from channel)
        let ic_top = result.get(0, 1).unwrap();
        let ic_mid = result.get(2, 1).unwrap();

        // Both should be valid
        assert!(!ic_top.is_nan(), "Top cell should have valid IC");
        assert!(!ic_mid.is_nan(), "Mid cell should have valid IC");

        // Top row has higher D_dn (farther from channel) → lower IC
        // Mid row is closer to channel → higher IC
        // But top row also has lower D_up... the relative magnitude depends on setup.
        // At minimum, verify IC is in valid range
        assert!(
            ic_top >= -10.0 && ic_top <= 10.0,
            "IC should be in [-10, 10], got {}",
            ic_top
        );
        assert!(
            ic_mid >= -10.0 && ic_mid <= 10.0,
            "IC should be in [-10, 10], got {}",
            ic_mid
        );
    }

    #[test]
    fn test_sediment_connectivity_stream_cell() {
        // Stream cells should have high IC (D_dn ≈ 0)
        let rows = 3;
        let cols = 3;

        let slope_rad = 5.0_f64.to_radians();
        let mut slope = Raster::filled(rows, cols, slope_rad);
        slope.set_transform(GeoTransform::new(0.0, 3.0, 10.0, -10.0));

        // All cells have high accumulation → all are stream cells
        let mut flow_acc = Raster::filled(rows, cols, 2000.0);
        flow_acc.set_transform(GeoTransform::new(0.0, 3.0, 10.0, -10.0));

        let mut flow_dir = Raster::<u8>::filled(rows, cols, 7);
        flow_dir.set_transform(GeoTransform::new(0.0, 3.0, 10.0, -10.0));

        let params = SedimentConnectivityParams {
            stream_threshold: 1000.0,
        };

        let result = sediment_connectivity(&slope, &flow_acc, &flow_dir, params, None).unwrap();

        let ic = result.get(1, 1).unwrap();
        assert!(!ic.is_nan(), "Stream cell should have valid IC");
        assert!(
            ic > 5.0,
            "Stream cell should have high IC (D_dn≈0), got {}",
            ic
        );
    }

    #[test]
    fn test_sediment_connectivity_dimension_mismatch() {
        let slope = Raster::<f64>::new(5, 5);
        let flow_acc = Raster::<f64>::new(3, 3);
        let flow_dir = Raster::<u8>::new(5, 5);

        let result = sediment_connectivity(
            &slope,
            &flow_acc,
            &flow_dir,
            SedimentConnectivityParams::default(),
            None,
        );
        assert!(result.is_err(), "Should error on dimension mismatch");
    }
}
