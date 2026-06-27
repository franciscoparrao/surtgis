//! Excess topography (Blöthe, Korup & Schwanghart 2015).
//!
//! Excess topography is the volume/height of terrain that rises above a
//! threshold hillslope angle. It is a first-rate landslide conditioning factor:
//! terrain steeper than the long-term strength-limited angle of a hillslope is
//! "excess" material available for failure.
//!
//! Given a threshold angle `θ`, we reconstruct the maximal surface `S` that
//!
//! - lies at or below the DEM everywhere (`S ≤ z`), and
//! - nowhere exceeds slope `tan θ` (`|S(p) − S(q)| ≤ tan θ · d(p,q)`),
//!
//! and report the excess `z − S ≥ 0`. The maximal slope-limited surface is the
//! lower envelope of cones of slope `tan θ` seated on the DEM:
//! `S(p) = min_q [ z(q) + tan θ · d(p,q) ]`.
//!
//! It is computed with the **fast sweeping method** (Zhao 2005): initialise
//! `S = z` and relax `S(p) ← min(S(p), S(q) + tan θ · d(p,q))` over the eight
//! neighbours, sweeping in the four diagonal orderings, until convergence. This
//! is the `fsm2d` approach used by TopoToolbox's `excesstopography`.
//!
//! # References
//! - Blöthe, J.H., Korup, O., Schwanghart, W. (2015). Large landslides lie low:
//!   Excess topography in the Himalaya-Karakoram ranges. *Geology* 43(6), 523–526.
//! - Zhao, H. (2005). A fast sweeping method for eikonal equations.
//!   *Mathematics of Computation* 74, 603–627.

use std::f64::consts::SQRT_2;

use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for [`excess_topography`].
#[derive(Debug, Clone, Copy)]
pub struct ExcessTopographyParams {
    /// Threshold hillslope angle in degrees (0 < θ < 90).
    pub threshold_degrees: f64,
    /// Maximum number of fast-sweeping rounds (each round is four sweeps).
    pub max_iterations: usize,
    /// Convergence tolerance: stop when the largest cell change in a round
    /// falls at or below this (in elevation units).
    pub tolerance: f64,
}

impl Default for ExcessTopographyParams {
    fn default() -> Self {
        Self {
            threshold_degrees: 30.0,
            max_iterations: 200,
            tolerance: 1e-4,
        }
    }
}

/// Compute excess topography above a threshold hillslope angle.
///
/// Returns a raster of `z − S ≥ 0` (same elevation units as the DEM); cells are
/// `NaN` where the DEM is nodata. See the module docs for the method.
///
/// # Errors
/// Returns an error if the threshold is not in `(0, 90)` or the cell size is
/// non-positive.
pub fn excess_topography(dem: &Raster<f64>, params: ExcessTopographyParams) -> Result<Raster<f64>> {
    if !(params.threshold_degrees > 0.0 && params.threshold_degrees < 90.0) {
        return Err(Error::Other(
            "threshold_degrees must be in the open interval (0, 90)".into(),
        ));
    }
    let cell_size = dem.cell_size();
    if cell_size <= 0.0 {
        return Err(Error::Other("cell size must be positive".into()));
    }

    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();
    let tan_t = params.threshold_degrees.to_radians().tan();
    let card = tan_t * cell_size; // max rise to a cardinal neighbour
    let diag = tan_t * cell_size * SQRT_2; // ... to a diagonal neighbour

    // Working surface S (row-major), initialised to the DEM; nodata -> NaN.
    let mut s: Vec<f64> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let z = unsafe { dem.get_unchecked(r, c) };
            let v = if z.is_nan() || nodata.map(|nd| z == nd).unwrap_or(false) {
                f64::NAN
            } else {
                z
            };
            s.push(v);
        }
    }

    // Eight neighbours: (di, dj, distance-weighted rise).
    let neigh: [(isize, isize, f64); 8] = [
        (-1, 0, card),
        (1, 0, card),
        (0, -1, card),
        (0, 1, card),
        (-1, -1, diag),
        (-1, 1, diag),
        (1, -1, diag),
        (1, 1, diag),
    ];

    // Four diagonal sweep orderings: (reverse_rows, reverse_cols).
    let sweeps = [(false, false), (false, true), (true, false), (true, true)];

    for _round in 0..params.max_iterations {
        let mut max_change = 0.0_f64;
        for &(rev_i, rev_j) in &sweeps {
            for ii in 0..rows {
                let i = if rev_i { rows - 1 - ii } else { ii };
                for jj in 0..cols {
                    let j = if rev_j { cols - 1 - jj } else { jj };
                    let p = i * cols + j;
                    let sp = s[p];
                    if sp.is_nan() {
                        continue;
                    }
                    let mut best = sp;
                    for &(di, dj, d) in &neigh {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        if ni < 0 || nj < 0 || ni >= rows as isize || nj >= cols as isize {
                            continue;
                        }
                        let q = ni as usize * cols + nj as usize;
                        let sq = s[q];
                        if sq.is_nan() {
                            continue;
                        }
                        let cand = sq + d;
                        if cand < best {
                            best = cand;
                        }
                    }
                    if best < sp {
                        let change = sp - best;
                        if change > max_change {
                            max_change = change;
                        }
                        s[p] = best;
                    }
                }
            }
        }
        if max_change <= params.tolerance {
            break;
        }
    }

    // excess = z - S (>= 0); nodata stays NaN.
    let mut out = dem.with_same_meta::<f64>(rows, cols);
    out.set_nodata(Some(f64::NAN));
    for r in 0..rows {
        for c in 0..cols {
            let p = r * cols + c;
            let sv = s[p];
            let val = if sv.is_nan() {
                f64::NAN
            } else {
                let z = unsafe { dem.get_unchecked(r, c) };
                (z - sv).max(0.0)
            };
            out.set(r, c, val).ok();
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    fn dem(data: Vec<f64>, rows: usize, cols: usize, cs: f64) -> Raster<f64> {
        let arr = Array2::from_shape_vec((rows, cols), data).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, rows as f64 * cs, cs, -cs));
        r
    }

    fn params(theta: f64) -> ExcessTopographyParams {
        ExcessTopographyParams {
            threshold_degrees: theta,
            max_iterations: 500,
            tolerance: 1e-9,
        }
    }

    #[test]
    fn test_flat_has_no_excess() {
        let d = dem(vec![100.0; 25], 5, 5, 10.0);
        let ex = excess_topography(&d, params(30.0)).unwrap();
        for v in ex.data().iter() {
            assert!(
                v.abs() < 1e-9,
                "flat terrain must have zero excess, got {v}"
            );
        }
    }

    #[test]
    fn test_ramp_analytic() {
        // DEM(i,j) = 2*j, cs = 1, θ = 45° (tan = 1) -> card = 1.
        // Maximal slope-limited surface S(j) = j (rises at the threshold from
        // the min at j=0). Excess = 2j - j = j.
        let rows = 4;
        let cols = 6;
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = 2.0 * j as f64;
            }
        }
        let d = dem(data, rows, cols, 1.0);
        let ex = excess_topography(&d, params(45.0)).unwrap();
        for i in 0..rows {
            for j in 0..cols {
                let got = ex.get(i, j).unwrap();
                let expect = j as f64;
                assert!(
                    (got - expect).abs() < 1e-6,
                    "excess at (col {j}) = {got}, expected {expect}"
                );
            }
        }
    }

    #[test]
    fn test_threshold_above_terrain_slope_zero_excess() {
        // Same ramp, but a very steep threshold (80°, tan ~ 5.67) exceeds the
        // terrain slope (2 per cell) -> no constraint -> zero excess.
        let rows = 3;
        let cols = 5;
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = 2.0 * j as f64;
            }
        }
        let d = dem(data, rows, cols, 1.0);
        let ex = excess_topography(&d, params(80.0)).unwrap();
        for v in ex.data().iter() {
            assert!(v.abs() < 1e-9, "no excess expected, got {v}");
        }
    }

    #[test]
    fn test_nodata_preserved_and_validation() {
        let mut data = vec![0.0; 9];
        for j in 0..3 {
            for i in 0..3 {
                data[i * 3 + j] = 3.0 * j as f64;
            }
        }
        data[4] = f64::NAN; // centre nodata
        let d = dem(data, 3, 3, 1.0);
        let ex = excess_topography(&d, params(45.0)).unwrap();
        assert!(ex.get(1, 1).unwrap().is_nan());
        // surrounding valid cells still produce finite, non-negative excess
        assert!(ex.get(0, 2).unwrap() >= 0.0);

        assert!(excess_topography(&d, params(0.0)).is_err());
        assert!(excess_topography(&d, params(90.0)).is_err());
    }
}
