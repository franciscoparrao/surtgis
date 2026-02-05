//! TFGA — Facet-to-Facet Multiple Flow Direction
//!
//! Li Z. et al. (2024): Divides the central pixel into 8 triangular sub-facets
//! (each shared with an adjacent neighbor pair). For each sub-facet, computes
//! the exact gradient direction using the 3 vertices (center + 2 neighbors),
//! then distributes flow proportionally.
//!
//! Key advantage: ~1 order of magnitude more precise than Quinn MFD, because
//! flow proportions are derived from strict mathematical surface normals rather
//! than simple slope-to-neighbor gradients.
//!
//! Reference:
//! Li, Z. et al. (2024). A facet-to-facet multiple flow direction algorithm.
//! *Computers & Geosciences*.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::Result;

/// Parameters for TFGA flow accumulation
#[derive(Debug, Clone)]
pub struct TfgaParams {
    /// Minimum slope to distribute flow (avoids division by zero). Default: 1e-8
    pub min_slope: f64,
}

impl Default for TfgaParams {
    fn default() -> Self {
        Self { min_slope: 1e-8 }
    }
}

/// 8 neighbors in clockwise order starting from East
const OFFSETS: [(isize, isize); 8] = [
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
    (0, -1), (1, -1), (1, 0), (1, 1),
];

/// Compute TFGA (Facet-to-Facet) flow accumulation.
///
/// For each cell, constructs 8 triangular facets between consecutive neighbor
/// pairs (e.g., E-NE, NE-N, ...). For each facet, computes the planar gradient
/// direction. If the gradient points into the facet's angular range, flow is
/// distributed to the two bounding neighbors proportionally to angle.
///
/// # Arguments
/// * `dem` — Input DEM (hydrologically conditioned)
/// * `params` — TFGA parameters
///
/// # Returns
/// Raster<f64> with flow accumulation (contributing area in cell counts)
pub fn flow_accumulation_tfga(dem: &Raster<f64>, params: TfgaParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();
    let cell_size = dem.cell_size();
    let n = rows * cols;

    // Sort cells highest first
    let mut cells: Vec<(usize, usize, f64)> = Vec::with_capacity(n);
    for row in 0..rows {
        for col in 0..cols {
            let z = unsafe { dem.get_unchecked(row, col) };
            let is_nd = z.is_nan()
                || nodata.is_some_and(|nd| (z - nd).abs() < f64::EPSILON);
            if !is_nd {
                cells.push((row, col, z));
            }
        }
    }
    cells.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut acc = Array2::<f64>::ones((rows, cols));
    // Mark nodata as 0
    for row in 0..rows {
        for col in 0..cols {
            let z = unsafe { dem.get_unchecked(row, col) };
            let is_nd = z.is_nan()
                || nodata.is_some_and(|nd| (z - nd).abs() < f64::EPSILON);
            if is_nd {
                acc[(row, col)] = 0.0;
            }
        }
    }

    // Compute actual angles from offset vectors using atan2(dr, dc)
    let dir_angle: [f64; 8] = std::array::from_fn(|i| {
        let (dr, dc) = OFFSETS[i];
        (dr as f64).atan2(dc as f64).rem_euclid(2.0 * std::f64::consts::PI)
    });

    let two_pi = 2.0 * std::f64::consts::PI;

    for &(row, col, z0) in &cells {
        let current_acc = acc[(row, col)];
        if current_acc <= 0.0 {
            continue;
        }

        // Get neighbor elevations
        let mut nz = [f64::NAN; 8];
        for (i, &(dr, dc)) in OFFSETS.iter().enumerate() {
            let nr = row as isize + dr;
            let nc = col as isize + dc;
            if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                continue;
            }
            let z = unsafe { dem.get_unchecked(nr as usize, nc as usize) };
            if !z.is_nan() {
                if let Some(nd) = nodata
                    && (z - nd).abs() < f64::EPSILON { continue; }
                nz[i] = z;
            }
        }

        // For each facet (pair of consecutive neighbors), compute gradient direction
        let mut flow_to = [0.0_f64; 8]; // total flow fraction to each neighbor
        let mut total_flow = 0.0_f64;

        for i in 0..8 {
            let j = (i + 1) % 8;

            if nz[i].is_nan() || nz[j].is_nan() {
                continue;
            }

            // Facet triangle: center (0,0), neighbor i, neighbor j
            // Compute planar gradient on this triangular facet
            let xi = OFFSETS[i].1 as f64 * cell_size;
            let yi = OFFSETS[i].0 as f64 * cell_size;
            let zi = nz[i] - z0;

            let xj = OFFSETS[j].1 as f64 * cell_size;
            let yj = OFFSETS[j].0 as f64 * cell_size;
            let zj = nz[j] - z0;

            // Gradient of plane through (0,0,0), (xi,yi,zi), (xj,yj,zj):
            // dz/dx and dz/dy from cross product of edge vectors
            let det = xi * yj - xj * yi;
            if det.abs() < 1e-15 {
                continue;
            }

            let dzdx = (zi * yj - zj * yi) / det;
            let dzdy = (xi * zj - xj * zi) / det;

            let slope_mag = (dzdx * dzdx + dzdy * dzdy).sqrt();
            if slope_mag < params.min_slope {
                continue;
            }

            // Gradient direction (downslope = negative gradient)
            // Consistent with atan2(dr, dc): dr-component = -dzdy, dc-component = -dzdx
            let grad_angle = (-dzdy).atan2(-dzdx).rem_euclid(two_pi);

            // Check if gradient falls within this facet's angular range
            let a_i = dir_angle[i];
            let a_j = dir_angle[j];

            // Compute angular midpoint and half-width for the facet
            let (lo, hi) = if a_i <= a_j {
                if a_j - a_i <= std::f64::consts::PI {
                    (a_i, a_j)
                } else {
                    (a_j, a_i + two_pi) // wrap
                }
            } else if a_i - a_j <= std::f64::consts::PI {
                (a_j, a_i)
            } else {
                (a_i, a_j + two_pi) // wrap
            };

            // Normalize gradient angle into the range
            let mut ga = grad_angle;
            if ga < lo { ga += two_pi; }
            if ga > hi + two_pi { ga -= two_pi; }

            if ga >= lo && ga <= hi {
                // Flow is within this facet — distribute to neighbors i and j
                // proportional to angle proximity
                let range = hi - lo;
                let fi = 1.0 - (ga - lo) / range; // closer to i → more flow to i
                let fj = 1.0 - fi;

                let w = slope_mag; // weight by gradient magnitude

                flow_to[i] += fi * w;
                flow_to[j] += fj * w;
                total_flow += w;
            }
        }

        // Fallback: if no facet captured the gradient, use slope-proportional
        // distribution to all downslope neighbors (like Quinn MFD with exp=1)
        if total_flow <= 0.0 {
            for (i, &(dr, dc)) in OFFSETS.iter().enumerate() {
                if nz[i].is_nan() {
                    continue;
                }
                let drop = z0 - nz[i];
                if drop > params.min_slope {
                    let dist = if dr != 0 && dc != 0 {
                        cell_size * std::f64::consts::SQRT_2
                    } else {
                        cell_size
                    };
                    let slope = drop / dist;
                    flow_to[i] = slope;
                    total_flow += slope;
                }
            }
        }

        // Distribute accumulated flow
        if total_flow > 0.0 {
            for (i, &(dr, dc)) in OFFSETS.iter().enumerate() {
                if flow_to[i] <= 0.0 {
                    continue;
                }
                let nr = (row as isize + dr) as usize;
                let nc = (col as isize + dc) as usize;
                acc[(nr, nc)] += current_acc * flow_to[i] / total_flow;
            }
        }
    }

    // Subtract self-contribution
    for row in 0..rows {
        for col in 0..cols {
            let z = unsafe { dem.get_unchecked(row, col) };
            let is_nd = z.is_nan()
                || nodata.is_some_and(|nd| (z - nd).abs() < f64::EPSILON);
            if !is_nd {
                acc[(row, col)] = (acc[(row, col)] - 1.0).max(0.0);
            }
        }
    }

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = acc;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_tfga_south_slope() {
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 10.0).unwrap();
            }
        }

        let acc = flow_accumulation_tfga(&dem, TfgaParams::default()).unwrap();
        let top = acc.get(0, 2).unwrap();
        let bottom = acc.get(4, 2).unwrap();
        assert!(
            bottom > top,
            "Bottom should have more accumulation: top={}, bottom={}",
            top, bottom
        );
    }

    #[test]
    fn test_tfga_pit() {
        let mut dem = Raster::new(3, 3);
        dem.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        for row in 0..3 {
            for col in 0..3 {
                dem.set(row, col, 5.0).unwrap();
            }
        }
        dem.set(1, 1, 1.0).unwrap();

        let acc = flow_accumulation_tfga(&dem, TfgaParams::default()).unwrap();
        let center = acc.get(1, 1).unwrap();
        assert!(
            (center - 8.0).abs() < 1.0,
            "Center pit should accumulate ~8, got {}",
            center
        );
    }

    #[test]
    fn test_tfga_conservation() {
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (10 - row - col) as f64 * 10.0).unwrap();
            }
        }

        let acc = flow_accumulation_tfga(&dem, TfgaParams::default()).unwrap();
        let corner = acc.get(4, 4).unwrap();
        assert!(
            corner > 5.0,
            "SE corner should accumulate significant flow, got {}",
            corner
        );
    }

    #[test]
    fn test_tfga_vs_mfd_different() {
        let mut dem = Raster::new(7, 7);
        dem.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));
        for row in 0..7 {
            for col in 0..7 {
                let z = (7 - row) as f64 * 10.0 + col as f64 * 3.0
                    + ((row as f64 * 0.5).sin()) * 5.0;
                dem.set(row, col, z).unwrap();
            }
        }

        let tfga = flow_accumulation_tfga(&dem, TfgaParams::default()).unwrap();
        let mfd = crate::hydrology::flow_accumulation_mfd(
            &dem,
            crate::hydrology::MfdParams { exponent: 1.1 },
        ).unwrap();

        let mut diffs = 0;
        for row in 0..7 {
            for col in 0..7 {
                let t = tfga.get(row, col).unwrap();
                let m = mfd.get(row, col).unwrap();
                if (t - m).abs() > 0.1 {
                    diffs += 1;
                }
            }
        }
        assert!(diffs > 0, "TFGA should differ from MFD at some cells");
    }
}
