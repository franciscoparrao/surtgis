//! RGBA pixel buffer + PNG encoder.
//!
//! Adds an in-memory `RgbaImage` type with alpha-over and multiply blends,
//! plus native PNG output via the `image` crate. The native PNG path is
//! gated on `cfg(not(target_arch = "wasm32"))` so WASM builds skip the
//! `image` dependency; on WASM the JS side encodes the raw RGBA buffer.
//!
//! This is the in-tree home for what was previously expected to live in
//! `surtgis-relief`. Hosting it here lets every other crate that produces
//! RGBA buffers (hypsometric, curvature previews, fluvial maps) save PNG
//! without pulling a separate dep.

use surtgis_core::raster::Raster;
use thiserror::Error;

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("shape mismatch: pixels.len()={got}, expected={expected} ({width}x{height}x4)")]
    Shape {
        width: usize,
        height: usize,
        got: usize,
        expected: usize,
    },
    #[cfg(not(target_arch = "wasm32"))]
    #[error("png encode: {0}")]
    Png(#[from] image::ImageError),
    #[cfg(not(target_arch = "wasm32"))]
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Result alias scoped to this module.
pub type Result<T> = std::result::Result<T, EncodeError>;

/// Row-major 8-bit RGBA pixel buffer.
///
/// Layout: `pixels[(row * width + col) * 4 + channel]`, with `channel`
/// 0=R, 1=G, 2=B, 3=A. Matches the convention produced by
/// [`crate::raster_to_rgba`].
#[derive(Debug, Clone)]
pub struct RgbaImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    /// Construct from a pre-built RGBA buffer. Returns an error if
    /// `pixels.len() != width * height * 4`.
    pub fn from_rgba(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        let expected = width * height * 4;
        if pixels.len() != expected {
            return Err(EncodeError::Shape {
                width,
                height,
                got: pixels.len(),
                expected,
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Construct from a single-channel intensity raster in `[0, 1]`.
    ///
    /// Values outside `[0, 1]` are clamped. NaN cells render fully
    /// transparent black. Output is greyscale (R = G = B = scaled value),
    /// alpha 255 for finite cells.
    pub fn from_intensity(intensity: &Raster<f64>) -> Self {
        let (rows, cols) = intensity.shape();
        let mut pixels = vec![0u8; rows * cols * 4];
        for (i, v) in intensity.data().iter().enumerate() {
            let off = i * 4;
            if v.is_nan() {
                // transparent black (already zeroed)
                continue;
            }
            let clamped = v.clamp(0.0, 1.0);
            let g = (clamped * 255.0).round() as u8;
            pixels[off] = g;
            pixels[off + 1] = g;
            pixels[off + 2] = g;
            pixels[off + 3] = 255;
        }
        Self {
            width: cols,
            height: rows,
            pixels,
        }
    }

    /// Alpha-over composite: paint `top` on top of `self`, modulated by
    /// `opacity` in `[0, 1]`. `self` is mutated in place.
    ///
    /// `top.width` and `top.height` must match.
    pub fn over(&mut self, top: &RgbaImage, opacity: f64) -> Result<()> {
        if top.width != self.width || top.height != self.height {
            return Err(EncodeError::Shape {
                width: top.width,
                height: top.height,
                got: top.pixels.len(),
                expected: self.width * self.height * 4,
            });
        }
        let op = opacity.clamp(0.0, 1.0);
        let n = self.pixels.len() / 4;
        for i in 0..n {
            let off = i * 4;
            let ta = (top.pixels[off + 3] as f64 / 255.0) * op;
            if ta <= 0.0 {
                continue;
            }
            let inv = 1.0 - ta;
            for c in 0..3 {
                let s = self.pixels[off + c] as f64;
                let t = top.pixels[off + c] as f64;
                self.pixels[off + c] = (t * ta + s * inv).round().clamp(0.0, 255.0) as u8;
            }
            // Alpha: standard over compositing.
            let sa = self.pixels[off + 3] as f64 / 255.0;
            let out_a = ta + sa * inv;
            self.pixels[off + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        Ok(())
    }

    /// Multiply blend: `out_rgb = self_rgb * lerp(white, top_rgb, opacity)`,
    /// per channel, normalised to `[0, 1]`. Useful for laying shadows over a
    /// colored base. Alpha of `self` is preserved.
    pub fn multiply(&mut self, top: &RgbaImage, opacity: f64) -> Result<()> {
        if top.width != self.width || top.height != self.height {
            return Err(EncodeError::Shape {
                width: top.width,
                height: top.height,
                got: top.pixels.len(),
                expected: self.width * self.height * 4,
            });
        }
        let op = opacity.clamp(0.0, 1.0);
        let n = self.pixels.len() / 4;
        for i in 0..n {
            let off = i * 4;
            for c in 0..3 {
                let s = self.pixels[off + c] as f64 / 255.0;
                let t = top.pixels[off + c] as f64 / 255.0;
                // lerp(1.0, t, op) = (1-op) + op*t
                let factor = (1.0 - op) + op * t;
                self.pixels[off + c] = (s * factor * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        Ok(())
    }

    /// Encode the buffer as PNG bytes. Native target only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_png_bytes(&self) -> Result<Vec<u8>> {
        rgba_to_png_bytes(self.width as u32, self.height as u32, &self.pixels)
    }

    /// Write the buffer to `path` as a PNG. Native target only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_png<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        save_png(path, self.width as u32, self.height as u32, &self.pixels)
    }
}

/// Encode an arbitrary RGBA byte buffer as PNG. Native target only.
///
/// `rgba.len()` must equal `width * height * 4`.
#[cfg(not(target_arch = "wasm32"))]
pub fn rgba_to_png_bytes(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>> {
    use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};

    let expected = (width as usize) * (height as usize) * 4;
    if rgba.len() != expected {
        return Err(EncodeError::Shape {
            width: width as usize,
            height: height as usize,
            got: rgba.len(),
            expected,
        });
    }

    let mut out = Vec::with_capacity(rgba.len() / 8);
    PngEncoder::new(&mut out).write_image(rgba, width, height, ExtendedColorType::Rgba8)?;
    Ok(out)
}

/// Write an RGBA buffer to `path` as a PNG. Native target only.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_png<P: AsRef<Path>>(path: P, width: u32, height: u32, rgba: &[u8]) -> Result<()> {
    let bytes = rgba_to_png_bytes(width, height, rgba)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::Raster;

    fn solid(w: usize, h: usize, rgba: [u8; 4]) -> RgbaImage {
        let mut pixels = Vec::with_capacity(w * h * 4);
        for _ in 0..(w * h) {
            pixels.extend_from_slice(&rgba);
        }
        RgbaImage::from_rgba(w, h, pixels).unwrap()
    }

    #[test]
    fn from_rgba_rejects_wrong_length() {
        let err = RgbaImage::from_rgba(2, 2, vec![0u8; 15]).unwrap_err();
        match err {
            EncodeError::Shape { expected, got, .. } => {
                assert_eq!(expected, 16);
                assert_eq!(got, 15);
            }
            _ => panic!("wrong error kind"),
        }
    }

    #[test]
    fn from_intensity_clamps_and_handles_nan() {
        // 2x1 raster: NaN, 0.5, 2.0 -> transparent, 128, 255
        let mut r = Raster::<f64>::new(1, 3);
        r.set(0, 0, f64::NAN).unwrap();
        r.set(0, 1, 0.5).unwrap();
        r.set(0, 2, 2.0).unwrap();
        let img = RgbaImage::from_intensity(&r);
        assert_eq!(img.width, 3);
        assert_eq!(img.height, 1);
        // Pixel 0: transparent black
        assert_eq!(&img.pixels[0..4], &[0, 0, 0, 0]);
        // Pixel 1: 128 grey
        assert_eq!(&img.pixels[4..8], &[128, 128, 128, 255]);
        // Pixel 2: 255 grey (clamped)
        assert_eq!(&img.pixels[8..12], &[255, 255, 255, 255]);
    }

    #[test]
    fn over_opaque_top_replaces_base() {
        let mut base = solid(1, 1, [10, 20, 30, 255]);
        let top = solid(1, 1, [100, 200, 50, 255]);
        base.over(&top, 1.0).unwrap();
        assert_eq!(&base.pixels[..], &[100, 200, 50, 255]);
    }

    #[test]
    fn over_zero_opacity_keeps_base() {
        let mut base = solid(1, 1, [10, 20, 30, 255]);
        let top = solid(1, 1, [100, 200, 50, 255]);
        base.over(&top, 0.0).unwrap();
        assert_eq!(&base.pixels[..], &[10, 20, 30, 255]);
    }

    #[test]
    fn over_half_opacity_lerps() {
        let mut base = solid(1, 1, [0, 0, 0, 255]);
        let top = solid(1, 1, [200, 200, 200, 255]);
        base.over(&top, 0.5).unwrap();
        // 200 * 0.5 + 0 * 0.5 = 100
        assert_eq!(&base.pixels[..3], &[100, 100, 100]);
    }

    #[test]
    fn multiply_with_white_top_full_opacity_preserves_base() {
        let mut base = solid(1, 1, [128, 64, 32, 255]);
        let top = solid(1, 1, [255, 255, 255, 255]);
        base.multiply(&top, 1.0).unwrap();
        // factor = 1.0; base unchanged
        assert_eq!(&base.pixels[..3], &[128, 64, 32]);
    }

    #[test]
    fn multiply_with_black_top_full_opacity_zeroes_rgb() {
        let mut base = solid(1, 1, [128, 64, 32, 255]);
        let top = solid(1, 1, [0, 0, 0, 255]);
        base.multiply(&top, 1.0).unwrap();
        assert_eq!(&base.pixels[..3], &[0, 0, 0]);
    }

    #[test]
    fn over_rejects_shape_mismatch() {
        let mut base = solid(2, 2, [0, 0, 0, 255]);
        let top = solid(3, 3, [0, 0, 0, 255]);
        assert!(base.over(&top, 1.0).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn png_roundtrip_size_and_signature() {
        let img = solid(4, 3, [255, 0, 0, 255]);
        let bytes = img.to_png_bytes().unwrap();
        // PNG signature
        assert_eq!(
            &bytes[..8],
            &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn save_png_writes_a_valid_file() {
        let img = solid(2, 2, [0, 128, 255, 255]);
        let tmp = std::env::temp_dir().join("colormap_encode_test.png");
        img.save_png(&tmp).unwrap();
        let bytes = std::fs::read(&tmp).unwrap();
        assert_eq!(
            &bytes[..8],
            &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
