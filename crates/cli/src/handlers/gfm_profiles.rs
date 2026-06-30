//! Foundation Model profiles for `extract-patches`.
//!
//! A profile encapsulates the input convention expected by a specific
//! Geospatial Foundation Model (GFM): which bands in what order, the
//! spatial tile size the model was pre-trained at, and per-band
//! normalization statistics applied to inputs.
//!
//! Profiles supported as of v0.9.0:
//!
//!   - `prithvi-v2`  → NASA/IBM Prithvi-EO-2.0-300M
//!                     6 bands (B02, B03, B04, B05, B06, B07), tile 224x224,
//!                     DN units (Sentinel-2 L2A surface reflectance × 10000).
//!                     Normalization stats from the official model config:
//!                     <https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M>
//!
//!   - `clay-v1.5`   → Clay Foundation Model v1.5 (Sentinel-2 path)
//!                     10 bands (B02-B12 less B09/B10), tile 256x256.
//!                     Reflectance [0,1] units. Normalization stats from the
//!                     official Clay metadata.yaml:
//!                     <https://github.com/Clay-foundation/model>
//!
//! Both profiles are intended as sensible defaults for the most common
//! pre-trained variants. Fine-tuned model variants may use different bands
//! or stats; pass `--profile custom` and provide explicit `--norm-mean`
//! and `--norm-std` to override.

use anyhow::{Result, anyhow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GfmProfile {
    PrithviV2,
    ClayV15,
}

#[derive(Clone, Debug)]
pub struct GfmProfileSpec {
    pub name: &'static str,
    /// Hugging Face model identifier or canonical reference
    pub model_target: &'static str,
    /// Band identifiers in the exact order the model expects them
    pub bands_order: Vec<&'static str>,
    /// Spatial tile size (square) the model was pre-trained at
    pub tile_size: usize,
    /// Per-band (mean, std) used for input normalization
    pub band_norm: Vec<(f32, f32)>,
    /// Unit convention expected at the model input (informational, written to meta.json)
    pub expected_unit: &'static str,
    /// Citation URL for the constants above
    pub source_url: &'static str,
}

impl GfmProfile {
    pub fn from_name(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "prithvi-v2" | "prithvi" | "prithvi-eo-2" | "prithvi-eo-2.0" => Ok(Self::PrithviV2),
            "clay-v1.5" | "clay" | "clay-v15" => Ok(Self::ClayV15),
            _ => Err(anyhow!(
                "Unknown GFM profile '{}'. Supported: prithvi-v2, clay-v1.5",
                s
            )),
        }
    }

    pub fn spec(&self) -> GfmProfileSpec {
        match self {
            Self::PrithviV2 => GfmProfileSpec {
                name: "prithvi-v2",
                model_target: "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
                bands_order: vec!["B02", "B03", "B04", "B05", "B06", "B07"],
                tile_size: 224,
                band_norm: vec![
                    (1087.0, 2248.0), // B02 Blue
                    (1342.0, 2179.0), // B03 Green
                    (1433.0, 2178.0), // B04 Red
                    (2734.0, 1850.0), // B05 Red Edge 1
                    (1958.0, 1242.0), // B06 Red Edge 2
                    (1363.0, 1049.0), // B07 Red Edge 3
                ],
                expected_unit: "DN_sr_x10000",
                source_url: "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
            },
            Self::ClayV15 => GfmProfileSpec {
                name: "clay-v1.5",
                model_target: "made-with-clay/Clay/v1.5",
                bands_order: vec![
                    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
                ],
                tile_size: 256,
                band_norm: vec![
                    (0.1182, 0.0461),
                    (0.1262, 0.0468),
                    (0.1389, 0.0596),
                    (0.1497, 0.0635),
                    (0.2010, 0.0780),
                    (0.2353, 0.0950),
                    (0.2455, 0.0987),
                    (0.2547, 0.1023),
                    (0.2099, 0.1153),
                    (0.1421, 0.0890),
                ],
                expected_unit: "reflectance_0_1",
                source_url: "https://github.com/Clay-foundation/model",
            },
        }
    }
}

/// Apply z-score normalization per band, in place. Skips NaN pixels.
///
/// `buf` is laid out [band0_flat..., band1_flat..., ...] of length
/// `n_bands * tile_size * tile_size`.
///
/// Caller is responsible for ensuring `band_norm.len()` equals the number
/// of bands implied by `buf.len() / (tile_size * tile_size)`.
pub fn apply_band_norm(buf: &mut [f32], band_norm: &[(f32, f32)], tile_size: usize) {
    let band_pixels = tile_size * tile_size;
    apply_band_norm_block(buf, band_norm, band_pixels);
}

/// Like `apply_band_norm`, but with the per-band block size (number of
/// scalar values for one band) passed explicitly. Use this for temporal
/// stacks where one band occupies `T × H × W` contiguous f32s rather
/// than `H × W`.
pub fn apply_band_norm_block(buf: &mut [f32], band_norm: &[(f32, f32)], block_size: usize) {
    for (bi, (mean, std)) in band_norm.iter().enumerate() {
        let band_offset = bi * block_size;
        let std_safe = if *std > 1e-10 { *std } else { 1e-10 };
        for px in 0..block_size {
            let v = buf[band_offset + px];
            if v.is_finite() {
                buf[band_offset + px] = (v - mean) / std_safe;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_name_aliases() {
        assert_eq!(
            GfmProfile::from_name("prithvi-v2").unwrap(),
            GfmProfile::PrithviV2
        );
        assert_eq!(
            GfmProfile::from_name("prithvi").unwrap(),
            GfmProfile::PrithviV2
        );
        assert_eq!(
            GfmProfile::from_name("PRITHVI-EO-2.0").unwrap(),
            GfmProfile::PrithviV2
        );
        assert_eq!(
            GfmProfile::from_name("clay-v1.5").unwrap(),
            GfmProfile::ClayV15
        );
        assert_eq!(GfmProfile::from_name("Clay").unwrap(), GfmProfile::ClayV15);
        assert!(GfmProfile::from_name("bogus").is_err());
    }

    #[test]
    fn prithvi_v2_spec_consistent() {
        let s = GfmProfile::PrithviV2.spec();
        assert_eq!(s.bands_order.len(), s.band_norm.len());
        assert_eq!(s.bands_order.len(), 6);
        assert_eq!(s.tile_size, 224);
    }

    #[test]
    fn clay_v15_spec_consistent() {
        let s = GfmProfile::ClayV15.spec();
        assert_eq!(s.bands_order.len(), s.band_norm.len());
        assert_eq!(s.bands_order.len(), 10);
        assert_eq!(s.tile_size, 256);
    }

    #[test]
    fn apply_band_norm_zscore() {
        // Two bands, 2x2 tile = 4 px each. Band 0 mean=2 std=1 → values become x-2
        // Band 1 mean=10 std=2 → values become (x-10)/2
        let mut buf: Vec<f32> = vec![
            2.0, 3.0, 4.0, 5.0, // band 0 raw
            10.0, 12.0, 14.0, 16.0, // band 1 raw
        ];
        let norm = vec![(2.0_f32, 1.0_f32), (10.0_f32, 2.0_f32)];
        apply_band_norm(&mut buf, &norm, 2);
        let expected: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];
        for (got, want) in buf.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {} want {}", got, want);
        }
    }

    #[test]
    fn apply_band_norm_block_temporal() {
        // 1 band, 2 timestamps, 2x2 tile each → block_size = 2 * 2 * 2 = 8 values
        // for that single band. Verifies the temporal path treats T*H*W as one
        // contiguous z-score region rather than per-timestamp.
        let mut buf: Vec<f32> = vec![
            // band 0, t0
            2.0, 3.0, 4.0, 5.0, // band 0, t1
            6.0, 7.0, 8.0, 9.0,
        ];
        let norm = vec![(2.0_f32, 1.0_f32)];
        apply_band_norm_block(&mut buf, &norm, 8);
        let expected: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        for (got, want) in buf.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {} want {}", got, want);
        }
    }

    #[test]
    fn apply_band_norm_preserves_nan() {
        let mut buf: Vec<f32> = vec![2.0, f32::NAN, 4.0, 5.0];
        let norm = vec![(2.0_f32, 1.0_f32)];
        apply_band_norm(&mut buf, &norm, 2);
        assert_eq!(buf[0], 0.0);
        assert!(buf[1].is_nan());
        assert_eq!(buf[2], 2.0);
        assert_eq!(buf[3], 3.0);
    }
}
