//! Ambient occlusion as an intensity layer.
//!
//! Thin wrapper over [`surtgis_algorithms::terrain::sky_view_factor`]. The
//! SVF output is already in `[0, 1]` (1 = open sky, 0 = fully enclosed), so
//! this function just re-shapes it into the relief crate's error / result
//! conventions. We deliberately do **not** invert: a brighter ambient layer
//! means *more* sky exposure, which is what most rayshader recipes assume.

use surtgis_algorithms::terrain::{SvfParams, sky_view_factor};
use surtgis_core::raster::Raster;

use crate::{ReliefError, Result};

/// Compute the ambient-occlusion intensity layer.
///
/// `radius` is the SVF search radius in cells. The number of azimuth
/// directions is fixed at the SVF default (16) — exposing it here would
/// add a knob without a clear use case yet.
///
/// # Errors
///
/// Returns [`ReliefError::Algorithm`] if the underlying SVF call fails
/// (e.g. `radius == 0`).
pub fn ambient_shade(dem: &Raster<f64>, radius: usize) -> Result<Raster<f64>> {
    let params = SvfParams {
        radius,
        ..Default::default()
    };
    sky_view_factor(dem, params).map_err(ReliefError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambient_shade_flat_dem_is_one() {
        let mut dem = Raster::new(10, 10);
        for r in 0..10 {
            for c in 0..10 {
                dem.set(r, c, 100.0).unwrap();
            }
        }
        let out = ambient_shade(&dem, 5).unwrap();
        for v in out.data().iter() {
            if v.is_nan() {
                continue;
            }
            assert!(
                (v - 1.0).abs() < 1e-9,
                "flat DEM should be fully sky-open, got {v}"
            );
        }
    }

    #[test]
    fn ambient_shade_pit_is_below_one() {
        // 1-cell pit at the centre of an otherwise flat plateau. Centre
        // cell should see less sky than the open border cells.
        let mut dem = Raster::new(11, 11);
        for r in 0..11 {
            for c in 0..11 {
                dem.set(r, c, 100.0).unwrap();
            }
        }
        // Tall walls 1-cell wide around row=5, col=5, leaving (5,5) at z=0
        dem.set(5, 5, 0.0).unwrap();
        for &(r, c) in &[(4, 5), (6, 5), (5, 4), (5, 6)] {
            dem.set(r, c, 200.0).unwrap();
        }
        let out = ambient_shade(&dem, 5).unwrap();
        let center = out.get(5, 5).unwrap();
        let corner = out.get(0, 0).unwrap();
        assert!(
            center < corner,
            "pit centre ({center}) should see less sky than open corner ({corner})"
        );
    }

    #[test]
    fn ambient_shade_zero_radius_errors() {
        let mut dem = Raster::new(4, 4);
        for r in 0..4 {
            for c in 0..4 {
                dem.set(r, c, 0.0).unwrap();
            }
        }
        assert!(matches!(
            ambient_shade(&dem, 0),
            Err(ReliefError::Algorithm(_))
        ));
    }
}
