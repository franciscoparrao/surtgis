//! Energy-cone (energy-line) mass-flow inundation model (Malin & Sheridan 1982).
//!
//! A first-order screening model for the runout of volcanic mass flows — block-
//! and-ash flows, debris avalanches, and lahars. From a source, an *energy line*
//! descends at a constant angle `φ` (the Heim coefficient `H/L = tan φ`): the
//! kinetic energy available at a point is proportional to the height of this
//! line above the ground. A cell is reached by the flow where the energy line
//! lies at or above the topography:
//!
//! ```text
//! E(p) = H0 − tan(φ) · L(p)        (energy-line elevation at cell p)
//! reached  ⟺  E(p) ≥ z(p)
//! energy height above ground = max(0, E(p) − z(p))
//! ```
//!
//! where `H0 = z(source) + collapse_height` is the apex elevation and `L(p)` is
//! the horizontal distance from the source. With several sources the result is
//! the cell-wise maximum (the union of the cones). Smaller `φ` ⇒ more mobile
//! flow ⇒ longer runout.
//!
//! This is the classic "energy cone": it ignores flow path and barriers (a cell
//! inside the cone is reached even if a ridge intervenes), so it is a
//! conservative envelope, not a routing model. For valley-confined lahar
//! inundation the LAHARZ statistical model (Iverson et al. 1998) is a natural
//! companion.
//!
//! # References
//! - Malin, M.C. & Sheridan, M.F. (1982). Computer-assisted mapping of
//!   pyroclastic surges. *Science* 217, 637–640.
//! - Sheridan, M.F. (1979). Emplacement of pyroclastic flows: a review.
//!   *GSA Special Paper* 180, 125–136.

use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for [`energy_cone`].
#[derive(Debug, Clone)]
pub struct EnergyConeParams {
    /// Source cells `(row, col)` (e.g. vent / collapse points). At least one.
    pub sources: Vec<(usize, usize)>,
    /// Energy-cone angle `φ` in degrees (the `H/L = tan φ` mobility). 0 < φ < 90.
    pub cone_angle_degrees: f64,
    /// Height added to the source elevation to set the apex (collapse height).
    pub collapse_height: f64,
}

/// Compute energy-cone inundation from one or more sources.
///
/// Returns a raster of the energy-line height above the ground,
/// `max(0, E(p) − z(p))`: positive where the flow reaches, `0` where it does
/// not, and `NaN` where the DEM is nodata. Threshold at `> 0` for a reach mask.
///
/// # Errors
/// Returns an error if there are no sources, a source is outside the grid, the
/// cone angle is not in `(0, 90)`, the cell size is non-positive, or a source
/// cell is nodata.
pub fn energy_cone(dem: &Raster<f64>, params: EnergyConeParams) -> Result<Raster<f64>> {
    if params.sources.is_empty() {
        return Err(Error::Other("at least one source is required".into()));
    }
    if !(params.cone_angle_degrees > 0.0 && params.cone_angle_degrees < 90.0) {
        return Err(Error::Other(
            "cone_angle_degrees must be in the open interval (0, 90)".into(),
        ));
    }
    let cell_size = dem.cell_size();
    if cell_size <= 0.0 {
        return Err(Error::Other("cell size must be positive".into()));
    }

    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();
    let is_nd = |v: f64| v.is_nan() || nodata.map(|nd| v == nd).unwrap_or(false);
    let tan_phi = params.cone_angle_degrees.to_radians().tan();

    // Resolve each source's apex elevation H0 = z(source) + collapse_height.
    let mut apexes: Vec<(f64, f64, f64)> = Vec::with_capacity(params.sources.len()); // (row, col, H0)
    for &(sr, sc) in &params.sources {
        if sr >= rows || sc >= cols {
            return Err(Error::Other(format!(
                "source ({sr}, {sc}) is outside the {rows}x{cols} grid"
            )));
        }
        let zs = unsafe { dem.get_unchecked(sr, sc) };
        if is_nd(zs) {
            return Err(Error::Other(format!(
                "source ({sr}, {sc}) falls on a nodata cell"
            )));
        }
        apexes.push((sr as f64, sc as f64, zs + params.collapse_height));
    }

    let mut out = dem.with_same_meta::<f64>(rows, cols);
    out.set_nodata(Some(f64::NAN));

    for r in 0..rows {
        for c in 0..cols {
            let z = unsafe { dem.get_unchecked(r, c) };
            if is_nd(z) {
                out.set(r, c, f64::NAN).ok();
                continue;
            }
            // Best (maximum) energy height above ground over all sources.
            let mut best = 0.0_f64;
            for &(sr, sc, h0) in &apexes {
                let dr = r as f64 - sr;
                let dc = c as f64 - sc;
                let l = (dr * dr + dc * dc).sqrt() * cell_size;
                let e = h0 - tan_phi * l; // energy-line elevation here
                let height = e - z;
                if height > best {
                    best = height;
                }
            }
            out.set(r, c, best).ok();
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

    #[test]
    fn test_flat_disc_analytic() {
        // Flat z=0, apex collapse_height=100, φ=45° (tanφ=1), cs=1.
        // Energy height at horizontal distance L = max(0, 100 − L).
        let n = 11;
        let d = dem(vec![0.0; n * n], n, n, 1.0);
        let out = energy_cone(
            &d,
            EnergyConeParams {
                sources: vec![(5, 5)],
                cone_angle_degrees: 45.0,
                collapse_height: 100.0,
            },
        )
        .unwrap();
        // Source: full 100.
        assert!((out.get(5, 5).unwrap() - 100.0).abs() < 1e-9);
        // 3 cells east (L=3): 97.
        assert!((out.get(5, 8).unwrap() - 97.0).abs() < 1e-9);
        // Diagonal (8,8) from (5,5): L = sqrt(18) ≈ 4.2426 -> ~95.757.
        let exp = 100.0 - (18.0_f64).sqrt();
        assert!((out.get(8, 8).unwrap() - exp).abs() < 1e-9);
    }

    #[test]
    fn test_apex_at_ground_no_inundation() {
        // No collapse height on flat ground: apex == ground, cone descends
        // immediately below the surface -> only the source has zero, rest 0.
        let n = 5;
        let d = dem(vec![10.0; n * n], n, n, 1.0);
        let out = energy_cone(
            &d,
            EnergyConeParams {
                sources: vec![(2, 2)],
                cone_angle_degrees: 30.0,
                collapse_height: 0.0,
            },
        )
        .unwrap();
        for v in out.data().iter() {
            assert!(v.abs() < 1e-9, "expected no inundation, got {v}");
        }
    }

    #[test]
    fn test_two_sources_take_max() {
        let n = 11;
        let d = dem(vec![0.0; n * n], n, n, 1.0);
        let out = energy_cone(
            &d,
            EnergyConeParams {
                sources: vec![(0, 0), (10, 10)],
                cone_angle_degrees: 45.0,
                collapse_height: 50.0,
            },
        )
        .unwrap();
        // Corner (0,0): full 50 from its own source.
        assert!((out.get(0, 0).unwrap() - 50.0).abs() < 1e-9);
        // Opposite corner also full 50 from the second source.
        assert!((out.get(10, 10).unwrap() - 50.0).abs() < 1e-9);
        // Centre (5,5): L = sqrt(50) ≈ 7.07 from either -> 50 - 7.07.
        let exp = 50.0 - (50.0_f64).sqrt();
        assert!((out.get(5, 5).unwrap() - exp).abs() < 1e-6);
    }

    #[test]
    fn test_downhill_runs_further_than_uphill() {
        // Tilted plane sloping down toward +col: z decreases as col increases.
        let rows = 3;
        let cols = 9;
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = 100.0 - 10.0 * j as f64; // drops 10/col
            }
        }
        let d = dem(data, rows, cols, 1.0);
        // Source at column 0 (the high end).
        let out = energy_cone(
            &d,
            EnergyConeParams {
                sources: vec![(1, 0)],
                cone_angle_degrees: 45.0, // tanφ = 1, energy drops 1/cell
                collapse_height: 0.0,
            },
        )
        .unwrap();
        // Downhill (toward +col) ground drops 10/cell but the energy line only
        // drops 1/cell, so inundation height grows with distance downhill.
        let near = out.get(1, 1).unwrap();
        let far = out.get(1, 8).unwrap();
        assert!(
            far > near,
            "downhill inundation should grow: {far} !> {near}"
        );
    }

    #[test]
    fn test_validation_errors() {
        let d = dem(vec![0.0; 9], 3, 3, 1.0);
        // no sources
        assert!(
            energy_cone(
                &d,
                EnergyConeParams {
                    sources: vec![],
                    cone_angle_degrees: 30.0,
                    collapse_height: 10.0
                }
            )
            .is_err()
        );
        // source out of bounds
        assert!(
            energy_cone(
                &d,
                EnergyConeParams {
                    sources: vec![(5, 5)],
                    cone_angle_degrees: 30.0,
                    collapse_height: 10.0
                }
            )
            .is_err()
        );
        // bad angle
        assert!(
            energy_cone(
                &d,
                EnergyConeParams {
                    sources: vec![(1, 1)],
                    cone_angle_degrees: 0.0,
                    collapse_height: 10.0
                }
            )
            .is_err()
        );
    }
}
