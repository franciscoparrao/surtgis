//! Spectral vegetation and water indices
//!
//! Common remote sensing indices computed from multispectral imagery.
//! All indices operate on single-band rasters (one band per raster).

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Enumeration of supported spectral indices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralIndex {
    /// Normalized Difference Vegetation Index
    NDVI,
    /// Normalized Difference Water Index (McFeeters)
    NDWI,
    /// Modified NDWI (Xu, uses SWIR)
    MNDWI,
    /// Soil Adjusted Vegetation Index
    SAVI,
    /// Enhanced Vegetation Index
    EVI,
    /// Normalized Burn Ratio
    NBR,
    /// Bare Soil Index
    BSI,
    /// Normalized Difference Red Edge Index
    NDRE,
    /// Green Normalized Difference Vegetation Index
    GNDVI,
    /// Normalized Green-Red Difference Index
    NGRDI,
    /// Red Edge Chlorophyll Index
    RECI,
}

// ---------------------------------------------------------------------------
// Generic normalized difference
// ---------------------------------------------------------------------------

/// Compute the normalized difference between two bands:
///
/// `(band_a - band_b) / (band_a + band_b)`
///
/// Result is in the range [-1, 1]. Pixels where both bands are zero
/// or either is nodata are set to NaN.
///
/// # Arguments
/// * `band_a` - Numerator positive band
/// * `band_b` - Numerator negative band
pub fn normalized_difference(band_a: &Raster<f64>, band_b: &Raster<f64>) -> Result<Raster<f64>> {
    check_dimensions(band_a, band_b)?;

    let (rows, cols) = band_a.shape();
    let nodata_a = band_a.nodata();
    let nodata_b = band_b.nodata();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let a = unsafe { band_a.get_unchecked(row, col) };
                let b = unsafe { band_b.get_unchecked(row, col) };

                if is_nodata_f64(a, nodata_a) || is_nodata_f64(b, nodata_b) {
                    continue;
                }

                let sum = a + b;
                if sum.abs() < 1e-10 {
                    continue; // Avoid division by zero
                }

                row_data[col] = (a - b) / sum;
            }
            row_data
        })
        .collect();

    build_output(band_a, rows, cols, data)
}

// ---------------------------------------------------------------------------
// NDVI
// ---------------------------------------------------------------------------

/// Normalized Difference Vegetation Index
///
/// `NDVI = (NIR - Red) / (NIR + Red)`
///
/// Values range from -1 to 1:
/// - Dense vegetation: 0.6 to 0.9
/// - Sparse vegetation: 0.2 to 0.5
/// - Bare soil: 0.1 to 0.2
/// - Water/clouds: -1.0 to 0.0
///
/// # Arguments
/// * `nir` - Near-infrared band
/// * `red` - Red band
pub fn ndvi(nir: &Raster<f64>, red: &Raster<f64>) -> Result<Raster<f64>> {
    normalized_difference(nir, red)
}

// ---------------------------------------------------------------------------
// NDWI
// ---------------------------------------------------------------------------

/// Normalized Difference Water Index (McFeeters, 1996)
///
/// `NDWI = (Green - NIR) / (Green + NIR)`
///
/// Positive values indicate water bodies.
///
/// # Arguments
/// * `green` - Green band
/// * `nir` - Near-infrared band
pub fn ndwi(green: &Raster<f64>, nir: &Raster<f64>) -> Result<Raster<f64>> {
    normalized_difference(green, nir)
}

// ---------------------------------------------------------------------------
// MNDWI
// ---------------------------------------------------------------------------

/// Modified Normalized Difference Water Index (Xu, 2006)
///
/// `MNDWI = (Green - SWIR) / (Green + SWIR)`
///
/// Better discrimination between water and built-up areas than NDWI.
///
/// # Arguments
/// * `green` - Green band
/// * `swir` - Shortwave infrared band
pub fn mndwi(green: &Raster<f64>, swir: &Raster<f64>) -> Result<Raster<f64>> {
    normalized_difference(green, swir)
}

// ---------------------------------------------------------------------------
// NBR
// ---------------------------------------------------------------------------

/// Normalized Burn Ratio
///
/// `NBR = (NIR - SWIR) / (NIR + SWIR)`
///
/// Used for mapping burn severity. Low values indicate burned areas.
///
/// # Arguments
/// * `nir` - Near-infrared band
/// * `swir` - Shortwave infrared band
pub fn nbr(nir: &Raster<f64>, swir: &Raster<f64>) -> Result<Raster<f64>> {
    normalized_difference(nir, swir)
}

// ---------------------------------------------------------------------------
// SAVI
// ---------------------------------------------------------------------------

/// Parameters for SAVI
#[derive(Debug, Clone)]
pub struct SaviParams {
    /// Soil brightness correction factor (0 = high vegetation, 1 = low vegetation)
    /// Default: 0.5
    pub l_factor: f64,
}

impl Default for SaviParams {
    fn default() -> Self {
        Self { l_factor: 0.5 }
    }
}

/// Soil Adjusted Vegetation Index (Huete, 1988)
///
/// `SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)`
///
/// Minimizes soil brightness influences on vegetation indices.
///
/// # Arguments
/// * `nir` - Near-infrared band
/// * `red` - Red band
/// * `params` - SAVI parameters (L factor)
pub fn savi(nir: &Raster<f64>, red: &Raster<f64>, params: SaviParams) -> Result<Raster<f64>> {
    check_dimensions(nir, red)?;

    let (rows, cols) = nir.shape();
    let nodata_nir = nir.nodata();
    let nodata_red = red.nodata();
    let l = params.l_factor;

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let n = unsafe { nir.get_unchecked(row, col) };
                let r = unsafe { red.get_unchecked(row, col) };

                if is_nodata_f64(n, nodata_nir) || is_nodata_f64(r, nodata_red) {
                    continue;
                }

                let denom = n + r + l;
                if denom.abs() < 1e-10 {
                    continue;
                }

                row_data[col] = ((n - r) / denom) * (1.0 + l);
            }
            row_data
        })
        .collect();

    build_output(nir, rows, cols, data)
}

// ---------------------------------------------------------------------------
// EVI
// ---------------------------------------------------------------------------

/// Parameters for EVI
#[derive(Debug, Clone)]
pub struct EviParams {
    /// Gain factor (default: 2.5)
    pub g: f64,
    /// Aerosol coefficient for red band (default: 6.0)
    pub c1: f64,
    /// Aerosol coefficient for blue band (default: 7.5)
    pub c2: f64,
    /// Canopy background adjustment (default: 1.0)
    pub l: f64,
}

impl Default for EviParams {
    fn default() -> Self {
        Self {
            g: 2.5,
            c1: 6.0,
            c2: 7.5,
            l: 1.0,
        }
    }
}

/// Enhanced Vegetation Index (Huete et al., 2002)
///
/// `EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)`
///
/// More sensitive than NDVI in high biomass areas and reduces
/// atmospheric and soil noise.
///
/// # Arguments
/// * `nir` - Near-infrared band
/// * `red` - Red band
/// * `blue` - Blue band
/// * `params` - EVI parameters
pub fn evi(
    nir: &Raster<f64>,
    red: &Raster<f64>,
    blue: &Raster<f64>,
    params: EviParams,
) -> Result<Raster<f64>> {
    check_dimensions(nir, red)?;
    check_dimensions(nir, blue)?;

    let (rows, cols) = nir.shape();
    let nodata_nir = nir.nodata();
    let nodata_red = red.nodata();
    let nodata_blue = blue.nodata();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let n = unsafe { nir.get_unchecked(row, col) };
                let r = unsafe { red.get_unchecked(row, col) };
                let b = unsafe { blue.get_unchecked(row, col) };

                if is_nodata_f64(n, nodata_nir)
                    || is_nodata_f64(r, nodata_red)
                    || is_nodata_f64(b, nodata_blue)
                {
                    continue;
                }

                let denom = n + params.c1 * r - params.c2 * b + params.l;
                if denom.abs() < 1e-10 {
                    continue;
                }

                row_data[col] = params.g * (n - r) / denom;
            }
            row_data
        })
        .collect();

    build_output(nir, rows, cols, data)
}

// ---------------------------------------------------------------------------
// BSI
// ---------------------------------------------------------------------------

/// Bare Soil Index
///
/// `BSI = ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))`
///
/// Highlights bare soil areas. High values indicate bare soil.
///
/// # Arguments
/// * `swir` - Shortwave infrared band
/// * `red` - Red band
/// * `nir` - Near-infrared band
/// * `blue` - Blue band
pub fn bsi(
    swir: &Raster<f64>,
    red: &Raster<f64>,
    nir: &Raster<f64>,
    blue: &Raster<f64>,
) -> Result<Raster<f64>> {
    check_dimensions(swir, red)?;
    check_dimensions(swir, nir)?;
    check_dimensions(swir, blue)?;

    let (rows, cols) = swir.shape();
    let nd_swir = swir.nodata();
    let nd_red = red.nodata();
    let nd_nir = nir.nodata();
    let nd_blue = blue.nodata();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let sw = unsafe { swir.get_unchecked(row, col) };
                let r = unsafe { red.get_unchecked(row, col) };
                let n = unsafe { nir.get_unchecked(row, col) };
                let b = unsafe { blue.get_unchecked(row, col) };

                if is_nodata_f64(sw, nd_swir)
                    || is_nodata_f64(r, nd_red)
                    || is_nodata_f64(n, nd_nir)
                    || is_nodata_f64(b, nd_blue)
                {
                    continue;
                }

                let a_val = sw + r;
                let b_val = n + b;
                let denom = a_val + b_val;

                if denom.abs() < 1e-10 {
                    continue;
                }

                row_data[col] = (a_val - b_val) / denom;
            }
            row_data
        })
        .collect();

    build_output(swir, rows, cols, data)
}

// ---------------------------------------------------------------------------
// NDRE
// ---------------------------------------------------------------------------

/// Normalized Difference Red Edge Index (Gitelson & Merzlyak, 1994)
///
/// `NDRE = (NIR - RedEdge) / (NIR + RedEdge)`
///
/// Sensitive to chlorophyll content in leaves. More effective than NDVI
/// for monitoring vegetation health in mid-to-late growth stages.
///
/// # Arguments
/// * `nir` - Near-infrared band (e.g., Sentinel-2 B8)
/// * `red_edge` - Red edge band (e.g., Sentinel-2 B5 or B6)
pub fn ndre(nir: &Raster<f64>, red_edge: &Raster<f64>) -> Result<Raster<f64>> {
    normalized_difference(nir, red_edge)
}

// ---------------------------------------------------------------------------
// GNDVI
// ---------------------------------------------------------------------------

/// Green Normalized Difference Vegetation Index (Gitelson et al., 1996)
///
/// `GNDVI = (NIR - Green) / (NIR + Green)`
///
/// More sensitive to chlorophyll concentration than NDVI. Useful for
/// assessing vegetation vigor and nitrogen content.
///
/// # Arguments
/// * `nir` - Near-infrared band
/// * `green` - Green band
pub fn gndvi(nir: &Raster<f64>, green: &Raster<f64>) -> Result<Raster<f64>> {
    normalized_difference(nir, green)
}

// ---------------------------------------------------------------------------
// NGRDI
// ---------------------------------------------------------------------------

/// Normalized Green-Red Difference Index (Tucker, 1979)
///
/// `NGRDI = (Green - Red) / (Green + Red)`
///
/// Simple visible-band vegetation index. Can be computed from standard
/// RGB imagery without NIR bands.
///
/// # Arguments
/// * `green` - Green band
/// * `red` - Red band
pub fn ngrdi(green: &Raster<f64>, red: &Raster<f64>) -> Result<Raster<f64>> {
    normalized_difference(green, red)
}

// ---------------------------------------------------------------------------
// RECI
// ---------------------------------------------------------------------------

/// Red Edge Chlorophyll Index (Gitelson et al., 2003)
///
/// `RECI = (NIR / RedEdge) - 1`
///
/// Estimates canopy chlorophyll content. Values typically range from 0 to 10+.
/// Unlike normalized indices, this is a ratio index (not bounded to [-1,1]).
///
/// # Arguments
/// * `nir` - Near-infrared band
/// * `red_edge` - Red edge band
pub fn reci(nir: &Raster<f64>, red_edge: &Raster<f64>) -> Result<Raster<f64>> {
    check_dimensions(nir, red_edge)?;

    let (rows, cols) = nir.shape();
    let nodata_nir = nir.nodata();
    let nodata_re = red_edge.nodata();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let n = unsafe { nir.get_unchecked(row, col) };
                let re = unsafe { red_edge.get_unchecked(row, col) };

                if is_nodata_f64(n, nodata_nir) || is_nodata_f64(re, nodata_re) {
                    continue;
                }

                if re.abs() < 1e-10 {
                    continue; // Avoid division by zero
                }

                row_data[col] = (n / re) - 1.0;
            }
            row_data
        })
        .collect();

    build_output(nir, rows, cols, data)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_nodata_f64(value: f64, nodata: Option<f64>) -> bool {
    if value.is_nan() {
        return true;
    }
    match nodata {
        Some(nd) => (value - nd).abs() < f64::EPSILON,
        None => false,
    }
}

fn check_dimensions(a: &Raster<f64>, b: &Raster<f64>) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(Error::SizeMismatch {
            er: a.rows(),
            ec: a.cols(),
            ar: b.rows(),
            ac: b.cols(),
        });
    }
    Ok(())
}

fn build_output(
    template: &Raster<f64>,
    rows: usize,
    cols: usize,
    data: Vec<f64>,
) -> Result<Raster<f64>> {
    let mut output = template.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_band(rows: usize, cols: usize, value: f64) -> Raster<f64> {
        let mut r = Raster::filled(rows, cols, value);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        r
    }

    fn make_gradient(rows: usize, cols: usize, start: f64, step: f64) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, start + (row * cols + col) as f64 * step)
                    .unwrap();
            }
        }
        r
    }

    #[test]
    fn test_normalized_difference_basic() {
        let a = make_band(5, 5, 0.8);
        let b = make_band(5, 5, 0.2);

        let result = normalized_difference(&a, &b).unwrap();
        let val = result.get(2, 2).unwrap();

        // (0.8 - 0.2) / (0.8 + 0.2) = 0.6
        assert!(
            (val - 0.6).abs() < 1e-10,
            "Expected 0.6, got {}",
            val
        );
    }

    #[test]
    fn test_normalized_difference_range() {
        // Result should always be in [-1, 1]
        let a = make_gradient(10, 10, 0.1, 0.01);
        let b = make_gradient(10, 10, 0.5, -0.005);

        let result = normalized_difference(&a, &b).unwrap();

        for row in 0..10 {
            for col in 0..10 {
                let val = result.get(row, col).unwrap();
                if !val.is_nan() {
                    assert!(
                        val >= -1.0 && val <= 1.0,
                        "ND out of range: {} at ({}, {})",
                        val,
                        row,
                        col
                    );
                }
            }
        }
    }

    #[test]
    fn test_ndvi() {
        let nir = make_band(5, 5, 0.5);
        let red = make_band(5, 5, 0.1);

        let result = ndvi(&nir, &red).unwrap();
        let val = result.get(2, 2).unwrap();

        // (0.5 - 0.1) / (0.5 + 0.1) = 0.4/0.6 ≈ 0.6667
        let expected = (0.5 - 0.1) / (0.5 + 0.1);
        assert!(
            (val - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_ndvi_water() {
        // Water: Red > NIR → negative NDVI
        let nir = make_band(5, 5, 0.05);
        let red = make_band(5, 5, 0.15);

        let result = ndvi(&nir, &red).unwrap();
        let val = result.get(2, 2).unwrap();

        assert!(val < 0.0, "Water should have negative NDVI, got {}", val);
    }

    #[test]
    fn test_ndwi() {
        let green = make_band(5, 5, 0.3);
        let nir = make_band(5, 5, 0.1);

        let result = ndwi(&green, &nir).unwrap();
        let val = result.get(2, 2).unwrap();

        // Positive: water-like
        assert!(val > 0.0, "Expected positive NDWI, got {}", val);
    }

    #[test]
    fn test_savi() {
        let nir = make_band(5, 5, 0.5);
        let red = make_band(5, 5, 0.1);

        let result = savi(&nir, &red, SaviParams::default()).unwrap();
        let val = result.get(2, 2).unwrap();

        // ((0.5 - 0.1) / (0.5 + 0.1 + 0.5)) * 1.5 = (0.4 / 1.1) * 1.5 ≈ 0.5455
        let expected = ((0.5 - 0.1) / (0.5 + 0.1 + 0.5)) * 1.5;
        assert!(
            (val - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_evi() {
        let nir = make_band(5, 5, 0.5);
        let red = make_band(5, 5, 0.1);
        let blue = make_band(5, 5, 0.05);

        let result = evi(&nir, &red, &blue, EviParams::default()).unwrap();
        let val = result.get(2, 2).unwrap();

        let params = EviParams::default();
        let expected =
            params.g * (0.5 - 0.1) / (0.5 + params.c1 * 0.1 - params.c2 * 0.05 + params.l);
        assert!(
            (val - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_bsi() {
        let swir = make_band(5, 5, 0.4);
        let red = make_band(5, 5, 0.3);
        let nir = make_band(5, 5, 0.2);
        let blue = make_band(5, 5, 0.1);

        let result = bsi(&swir, &red, &nir, &blue).unwrap();
        let val = result.get(2, 2).unwrap();

        // ((0.4+0.3) - (0.2+0.1)) / ((0.4+0.3) + (0.2+0.1)) = (0.7-0.3)/1.0 = 0.4
        assert!(
            (val - 0.4).abs() < 1e-10,
            "Expected 0.4, got {}",
            val
        );
    }

    #[test]
    fn test_nodata_handling() {
        let mut nir = make_band(5, 5, 0.5);
        nir.set_nodata(Some(-9999.0));
        nir.set(2, 2, -9999.0).unwrap();

        let red = make_band(5, 5, 0.1);

        let result = ndvi(&nir, &red).unwrap();
        let val = result.get(2, 2).unwrap();

        assert!(val.is_nan(), "Nodata pixel should be NaN, got {}", val);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = make_band(5, 5, 1.0);
        let b = make_band(5, 10, 1.0);

        let result = normalized_difference(&a, &b);
        assert!(result.is_err(), "Should fail on dimension mismatch");
    }

    #[test]
    fn test_ndre() {
        let nir = make_band(5, 5, 0.6);
        let red_edge = make_band(5, 5, 0.3);

        let result = ndre(&nir, &red_edge).unwrap();
        let val = result.get(2, 2).unwrap();

        let expected = (0.6 - 0.3) / (0.6 + 0.3);
        assert!(
            (val - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_gndvi() {
        let nir = make_band(5, 5, 0.5);
        let green = make_band(5, 5, 0.2);

        let result = gndvi(&nir, &green).unwrap();
        let val = result.get(2, 2).unwrap();

        let expected = (0.5 - 0.2) / (0.5 + 0.2);
        assert!(
            (val - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_ngrdi() {
        let green = make_band(5, 5, 0.25);
        let red = make_band(5, 5, 0.15);

        let result = ngrdi(&green, &red).unwrap();
        let val = result.get(2, 2).unwrap();

        let expected = (0.25 - 0.15) / (0.25 + 0.15);
        assert!(
            (val - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_reci() {
        let nir = make_band(5, 5, 0.6);
        let red_edge = make_band(5, 5, 0.2);

        let result = reci(&nir, &red_edge).unwrap();
        let val = result.get(2, 2).unwrap();

        // RECI = (0.6 / 0.2) - 1 = 3.0 - 1.0 = 2.0
        assert!(
            (val - 2.0).abs() < 1e-10,
            "Expected 2.0, got {}",
            val
        );
    }

    #[test]
    fn test_reci_zero_rededge() {
        let nir = make_band(5, 5, 0.5);
        let red_edge = make_band(5, 5, 0.0);

        let result = reci(&nir, &red_edge).unwrap();
        let val = result.get(2, 2).unwrap();

        assert!(val.is_nan(), "Zero red edge should produce NaN, got {}", val);
    }
}
