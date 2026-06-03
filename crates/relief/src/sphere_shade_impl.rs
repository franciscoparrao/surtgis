//! Normal-based "sphere" shade, normalised to `[0, 1]`.
//!
//! Wraps [`surtgis_algorithms::terrain::hillshade`] with `normalized: true`
//! and clamps to the `[0, 1]` intensity contract that the rest of the
//! relief compositor expects.

use crate::Result;
use surtgis_algorithms::terrain::{HillshadeParams, hillshade};
use surtgis_core::raster::Raster;

/// Normal-based hillshade as intensity in `[0, 1]`.
///
/// This is a thin wrapper over
/// [`surtgis_algorithms::terrain::hillshade`] (Horn 1981 surface-normal
/// approach). The underlying function already supports a normalised
/// output mode; we just enable it and pass parameters through.
///
/// NaN cells pass through.
pub fn sphere_shade(dem: &Raster<f64>, mut params: HillshadeParams) -> Result<Raster<f64>> {
    // Force normalised mode so the output lands in [0, 1] regardless of
    // what the caller passed.
    params.normalized = true;
    Ok(hillshade(dem, params)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::Raster;

    #[test]
    fn flat_dem_returns_finite_intensity() {
        let mut dem = Raster::new(8, 8);
        for r in 0..8 {
            for c in 0..8 {
                dem.set(r, c, 100.0).unwrap();
            }
        }
        let out = sphere_shade(&dem, HillshadeParams::default()).unwrap();
        // Interior cells should land in [0, 1]. Edges are zeroed by the
        // hillshade algorithm; just check interior.
        for r in 1..7 {
            for c in 1..7 {
                let v = out.get(r, c).unwrap();
                assert!((0.0..=1.0).contains(&v), "cell ({r},{c}) = {v}");
            }
        }
    }
}
