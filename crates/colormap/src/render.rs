//! Raster-to-RGBA rendering using color schemes.

use crate::scheme::{evaluate, ColorScheme, Rgb};
use surtgis_core::raster::{Raster, RasterElement};

/// Parameters for colormap rendering.
#[derive(Debug, Clone)]
pub struct ColormapParams {
    /// Color scheme to use.
    pub scheme: ColorScheme,
    /// Minimum value for normalization. Values below this are clamped.
    pub min: f64,
    /// Maximum value for normalization. Values above this are clamped.
    pub max: f64,
    /// Color for nodata pixels (RGBA). Default: fully transparent.
    pub nodata_color: [u8; 4],
}

impl ColormapParams {
    /// Create params with the given scheme; min/max must be set separately
    /// or use [`auto_params`] to detect from data.
    pub fn new(scheme: ColorScheme) -> Self {
        Self {
            scheme,
            min: 0.0,
            max: 1.0,
            nodata_color: [0, 0, 0, 0],
        }
    }

    /// Create params with explicit min/max range.
    pub fn with_range(scheme: ColorScheme, min: f64, max: f64) -> Self {
        Self {
            scheme,
            min,
            max,
            nodata_color: [0, 0, 0, 0],
        }
    }
}

/// Auto-detect min/max from a raster, returning `ColormapParams` ready to use.
///
/// Scans all valid (non-nodata) cells to find the data range.
pub fn auto_params<T: RasterElement>(raster: &Raster<T>, scheme: ColorScheme) -> ColormapParams {
    let nodata = raster.nodata();
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for val in raster.data().iter() {
        if val.is_nodata(nodata) {
            continue;
        }
        if let Some(v) = val.to_f64() {
            if v.is_finite() {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
    }

    // Handle edge case: all nodata or constant raster
    if !min.is_finite() || !max.is_finite() {
        min = 0.0;
        max = 1.0;
    } else if (max - min).abs() < f64::EPSILON {
        max = min + 1.0;
    }

    ColormapParams::with_range(scheme, min, max)
}

/// Convert a raster to an RGBA pixel buffer.
///
/// Returns a `Vec<u8>` of length `rows * cols * 4` in row-major order,
/// suitable for uploading as a GPU texture.
///
/// Nodata pixels are rendered with `params.nodata_color` (default: transparent black).
pub fn raster_to_rgba<T: RasterElement>(raster: &Raster<T>, params: &ColormapParams) -> Vec<u8> {
    let rows = raster.rows();
    let cols = raster.cols();
    let nodata = raster.nodata();
    let range = params.max - params.min;
    let inv_range = if range.abs() > f64::EPSILON {
        1.0 / range
    } else {
        1.0
    };

    let mut rgba = vec![0u8; rows * cols * 4];

    for (i, val) in raster.data().iter().enumerate() {
        let offset = i * 4;

        if val.is_nodata(nodata) {
            rgba[offset] = params.nodata_color[0];
            rgba[offset + 1] = params.nodata_color[1];
            rgba[offset + 2] = params.nodata_color[2];
            rgba[offset + 3] = params.nodata_color[3];
            continue;
        }

        match val.to_f64() {
            Some(v) if v.is_finite() => {
                let t = (v - params.min) * inv_range;
                let Rgb { r, g, b } = evaluate(params.scheme, t);
                rgba[offset] = r;
                rgba[offset + 1] = g;
                rgba[offset + 2] = b;
                rgba[offset + 3] = 255;
            }
            _ => {
                // NaN or conversion failure -> nodata color
                rgba[offset] = params.nodata_color[0];
                rgba[offset + 1] = params.nodata_color[1];
                rgba[offset + 2] = params.nodata_color[2];
                rgba[offset + 3] = params.nodata_color[3];
            }
        }
    }

    rgba
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::Raster;

    #[test]
    fn raster_to_rgba_basic() {
        let mut r = Raster::<f64>::new(2, 2);
        r.set(0, 0, 0.0).unwrap();
        r.set(0, 1, 0.5).unwrap();
        r.set(1, 0, 1.0).unwrap();
        r.set(1, 1, f64::NAN).unwrap();
        r.set_nodata(Some(f64::NAN));

        let params = ColormapParams::with_range(ColorScheme::Grayscale, 0.0, 1.0);
        let rgba = raster_to_rgba(&r, &params);

        assert_eq!(rgba.len(), 16); // 4 pixels * 4 bytes

        // pixel (0,0) = 0.0 -> black, opaque
        assert_eq!(rgba[0], 0);
        assert_eq!(rgba[1], 0);
        assert_eq!(rgba[2], 0);
        assert_eq!(rgba[3], 255);

        // pixel (0,1) = 0.5 -> gray, opaque
        assert_eq!(rgba[4], 128);
        assert_eq!(rgba[5], 128);
        assert_eq!(rgba[6], 128);
        assert_eq!(rgba[7], 255);

        // pixel (1,0) = 1.0 -> white, opaque
        assert_eq!(rgba[8], 255);
        assert_eq!(rgba[9], 255);
        assert_eq!(rgba[10], 255);
        assert_eq!(rgba[11], 255);

        // pixel (1,1) = NaN -> transparent
        assert_eq!(rgba[12], 0);
        assert_eq!(rgba[13], 0);
        assert_eq!(rgba[14], 0);
        assert_eq!(rgba[15], 0);
    }

    #[test]
    fn auto_params_range() {
        let mut r = Raster::<f64>::new(1, 3);
        r.set(0, 0, 10.0).unwrap();
        r.set(0, 1, 50.0).unwrap();
        r.set(0, 2, 100.0).unwrap();

        let params = auto_params(&r, ColorScheme::Terrain);
        assert!((params.min - 10.0).abs() < f64::EPSILON);
        assert!((params.max - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn auto_params_all_nodata() {
        let mut r = Raster::<f64>::new(1, 2);
        r.set(0, 0, f64::NAN).unwrap();
        r.set(0, 1, f64::NAN).unwrap();
        r.set_nodata(Some(f64::NAN));

        let params = auto_params(&r, ColorScheme::Terrain);
        assert!((params.min - 0.0).abs() < f64::EPSILON);
        assert!((params.max - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn auto_params_constant_raster() {
        let r = Raster::<f64>::filled(2, 2, 42.0);
        let params = auto_params(&r, ColorScheme::Terrain);
        assert!((params.min - 42.0).abs() < f64::EPSILON);
        assert!((params.max - 43.0).abs() < f64::EPSILON);
    }
}
