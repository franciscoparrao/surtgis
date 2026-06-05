//! Fluent compositor: stack relief layers and render them to an RGBA image.
//!
//! The accumulator pattern mirrors the rayshader recipe — start with a
//! base colormap of the DEM, then multiply-blend each shading/shadow
//! layer in. Order of `.add_*()` calls is the order of blending; the
//! final `.render()` consumes the builder and returns an [`RgbaImage`].
//!
//! No algorithm logic lives here. Layers are pre-computed by their
//! respective functions ([`crate::ray_shade`], [`crate::sphere_shade`],
//! [`crate::ambient_shade`]) and handed in as `Raster<f64>` intensity
//! masks in `[0, 1]`.

use surtgis_colormap::{
    ColorScheme, ColormapParams, RgbaImage, auto_params, evaluate, raster_to_rgba,
};
use surtgis_core::raster::Raster;

use crate::{ReliefError, Result};

/// Default opacity for a layer when the caller doesn't override it.
const DEFAULT_OPACITY: f64 = 0.5;

/// Order-preserving accumulator for relief layers.
///
/// Hold a base RGBA buffer (from `base_colormap`) and a list of
/// `(layer, opacity, blend)` triples. `render()` folds them in order.
pub struct ReliefBuilder<'a> {
    dem: &'a Raster<f64>,
    base: Option<RgbaImage>,
    layers: Vec<Layer>,
}

#[derive(Debug, Clone, Copy)]
enum Blend {
    Multiply,
    Over,
}

struct Layer {
    image: RgbaImage,
    opacity: f64,
    blend: Blend,
}

impl<'a> ReliefBuilder<'a> {
    /// Begin a new builder bound to `dem`. The DEM is used to size every
    /// layer and (if `base_colormap` is set) to produce the colored base.
    pub fn new(dem: &'a Raster<f64>) -> Self {
        Self {
            dem,
            base: None,
            layers: Vec::new(),
        }
    }

    /// Set the colored base layer: render `dem` to RGBA using `scheme`
    /// with auto-detected min/max. Replaces any previous base.
    pub fn base_colormap(mut self, scheme: ColorScheme) -> Self {
        let params = auto_params(self.dem, scheme);
        let (rows, cols) = self.dem.shape();
        let rgba = raster_to_rgba(self.dem, &params);
        // raster_to_rgba returns Vec<u8> sized rows*cols*4 by construction.
        self.base = Some(
            RgbaImage::from_rgba(cols, rows, rgba)
                .expect("raster_to_rgba sized rows*cols*4 by construction"),
        );
        self
    }

    /// Set the base layer from explicit colormap parameters (skips auto
    /// min/max detection). Useful when you want a fixed elevation scale
    /// across multiple renders.
    pub fn base_with_params(mut self, params: &ColormapParams) -> Self {
        let (rows, cols) = self.dem.shape();
        let rgba = raster_to_rgba(self.dem, params);
        self.base = Some(
            RgbaImage::from_rgba(cols, rows, rgba)
                .expect("raster_to_rgba sized rows*cols*4 by construction"),
        );
        self
    }

    /// Multiply-blend a normal-based hillshade intensity layer
    /// (typically [`crate::sphere_shade`]) onto the stack.
    pub fn add_shade(self, intensity: Raster<f64>, opacity: f64) -> Self {
        self.add_intensity(intensity, opacity, Blend::Multiply)
    }

    /// Multiply-blend a cast-shadow intensity layer (typically
    /// [`crate::ray_shade`]) onto the stack.
    pub fn add_shadow(self, intensity: Raster<f64>, opacity: f64) -> Self {
        self.add_intensity(intensity, opacity, Blend::Multiply)
    }

    /// Multiply-blend an ambient-occlusion intensity layer (typically
    /// [`crate::ambient_shade`]) onto the stack.
    pub fn add_ambient(self, intensity: Raster<f64>, opacity: f64) -> Self {
        self.add_intensity(intensity, opacity, Blend::Multiply)
    }

    /// Alpha-over a pre-built RGBA layer. Use this for water tints or any
    /// composite the caller has already coloured.
    pub fn add_rgba_over(mut self, image: RgbaImage, opacity: f64) -> Self {
        self.layers.push(Layer {
            image,
            opacity,
            blend: Blend::Over,
        });
        self
    }

    /// Alpha-over a water mask (typically from [`crate::detect_water`]),
    /// painting each `1` cell with a fixed colour sampled from `scheme`
    /// and leaving `0` cells transparent.
    ///
    /// The colour is sampled from the scheme at `t = 0.5` so it doesn't
    /// depend on any data range. For shore-to-centre depth gradients
    /// use [`Self::add_water_depth`] instead.
    pub fn add_water(self, mask: Raster<u8>, scheme: ColorScheme) -> Self {
        let (rows, cols) = mask.shape();
        let colour = evaluate(scheme, 0.5);
        let mut pixels = vec![0u8; rows * cols * 4];
        for (i, &m) in mask.data().iter().enumerate() {
            if m == 0 {
                continue;
            }
            let off = i * 4;
            pixels[off] = colour.r;
            pixels[off + 1] = colour.g;
            pixels[off + 2] = colour.b;
            pixels[off + 3] = 255;
        }
        let image = RgbaImage::from_rgba(cols, rows, pixels)
            .expect("water mask sized rows*cols*4 by construction");
        self.add_rgba_over(image, 1.0)
    }

    /// Alpha-over a water mask with a shore-to-centre depth gradient.
    /// Each water cell's colour is sampled from `scheme` at
    /// `t = depth / max_depth`, where depth is the cell's 8-connected
    /// Chebyshev distance to the nearest non-water neighbour (computed
    /// internally via [`crate::water_depth`]). Shore cells → light end
    /// of the scheme; deep centres → dark end.
    ///
    /// Use [`ColorScheme::Water`] (white → cyan → navy) for the
    /// rayshader-style read. If `max_depth` resolves to 0 (no water in
    /// the mask), the layer is fully transparent.
    pub fn add_water_depth(self, mask: Raster<u8>, scheme: ColorScheme) -> Self {
        let (rows, cols) = mask.shape();
        let depth = match crate::water_depth(&mask) {
            Ok(d) => d,
            Err(_) => {
                // Empty mask — emit a fully transparent layer so the
                // builder API stays infallible and the caller can chain.
                return self.add_rgba_over(
                    RgbaImage::from_rgba(cols, rows, vec![0u8; rows * cols * 4])
                        .expect("empty water mask sized rows*cols*4"),
                    1.0,
                );
            }
        };
        let max_depth = depth
            .data()
            .iter()
            .fold(0f32, |acc, &v| if v > acc { v } else { acc });
        let inv_max = if max_depth > 0.0 {
            1.0 / max_depth
        } else {
            0.0
        };

        let mut pixels = vec![0u8; rows * cols * 4];
        for (i, &d) in depth.data().iter().enumerate() {
            if d <= 0.0 {
                continue;
            }
            // Shore (d=1) → t small (light); centre (d=max) → t=1 (dark).
            let t = (d * inv_max).clamp(0.0, 1.0) as f64;
            let colour = evaluate(scheme, t);
            let off = i * 4;
            pixels[off] = colour.r;
            pixels[off + 1] = colour.g;
            pixels[off + 2] = colour.b;
            pixels[off + 3] = 255;
        }
        let image = RgbaImage::from_rgba(cols, rows, pixels)
            .expect("water depth mask sized rows*cols*4 by construction");
        self.add_rgba_over(image, 1.0)
    }

    fn add_intensity(mut self, intensity: Raster<f64>, opacity: f64, blend: Blend) -> Self {
        let layer = RgbaImage::from_intensity(&intensity);
        self.layers.push(Layer {
            image: layer,
            opacity,
            blend,
        });
        self
    }

    /// Collapse the stack into a single [`RgbaImage`]. Fails if any layer's
    /// shape disagrees with the base (or with the DEM, if no base was set).
    pub fn render(self) -> Result<RgbaImage> {
        let (rows, cols) = self.dem.shape();
        let mut out = match self.base {
            Some(base) => base,
            None => {
                // No coloured base — start from neutral white so multiply
                // blends behave intuitively (white * x = x).
                let pixels = vec![255u8; rows * cols * 4];
                RgbaImage::from_rgba(cols, rows, pixels)
                    .expect("white base sized rows*cols*4 by construction")
            }
        };

        for layer in self.layers {
            let opacity = if layer.opacity.is_nan() {
                DEFAULT_OPACITY
            } else {
                layer.opacity
            };
            match layer.blend {
                Blend::Multiply => out.multiply(&layer.image, opacity)?,
                Blend::Over => out.over(&layer.image, opacity)?,
            }
        }

        // Detect a final shape mismatch in case the user passed layers of
        // the wrong size before any blend op caught it (currently `multiply`
        // and `over` already check, so this is belt-and-braces).
        if out.width != cols || out.height != rows {
            return Err(ReliefError::Shape(format!(
                "render produced {}x{}, expected {}x{}",
                out.width, out.height, cols, rows
            )));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RayShadeParams, ray_shade, sphere_shade};
    use surtgis_algorithms::terrain::HillshadeParams;

    fn ramp_dem(rows: usize, cols: usize) -> Raster<f64> {
        let mut d = Raster::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                d.set(r, c, (r + c) as f64).unwrap();
            }
        }
        d
    }

    #[test]
    fn render_with_no_base_or_layers_returns_white() {
        let dem = ramp_dem(4, 4);
        let img = ReliefBuilder::new(&dem).render().unwrap();
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        // Every pixel should be fully white opaque.
        for chunk in img.pixels.chunks_exact(4) {
            assert_eq!(chunk, &[255, 255, 255, 255]);
        }
    }

    #[test]
    fn render_with_base_only_matches_raster_to_rgba() {
        let dem = ramp_dem(4, 4);
        let img = ReliefBuilder::new(&dem)
            .base_colormap(ColorScheme::Terrain)
            .render()
            .unwrap();
        let expected = raster_to_rgba(&dem, &auto_params(&dem, ColorScheme::Terrain));
        assert_eq!(img.pixels, expected);
    }

    #[test]
    fn full_pipeline_renders_without_panic() {
        let dem = ramp_dem(8, 8);
        let sphere = sphere_shade(
            &dem,
            HillshadeParams {
                azimuth: 315.0,
                altitude: 45.0,
                z_factor: 1.0,
                normalized: true,
            },
        )
        .unwrap();
        let shadow = ray_shade(&dem, &RayShadeParams::default()).unwrap();
        let img = ReliefBuilder::new(&dem)
            .base_colormap(ColorScheme::Terrain)
            .add_shade(sphere, 0.5)
            .add_shadow(shadow, 0.5)
            .render()
            .unwrap();
        assert_eq!(img.width, 8);
        assert_eq!(img.height, 8);
        // Should be non-white somewhere — the shading layers must have
        // darkened at least one pixel.
        let any_dark = img
            .pixels
            .chunks_exact(4)
            .any(|c| c[0] < 250 || c[1] < 250 || c[2] < 250);
        assert!(any_dark, "expected at least one shaded pixel");
    }

    #[test]
    fn layer_shape_mismatch_errors() {
        let dem = ramp_dem(4, 4);
        let wrong = Raster::<f64>::new(3, 3);
        let res = ReliefBuilder::new(&dem)
            .base_colormap(ColorScheme::Terrain)
            .add_shade(wrong, 0.5)
            .render();
        assert!(matches!(res, Err(ReliefError::Encode(_))));
    }
}
