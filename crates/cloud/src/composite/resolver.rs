//! A default [`AssetResolver`](super::AssetResolver) over the common
//! Sentinel-2 / Landsat / Sentinel-1 band-naming conventions.

use super::engine::AssetResolver;
use crate::stac_models::StacItem;

/// Common-name → catalog-specific band-code aliases (both directions), so a
/// caller can ask for `"red"` and get `B04` (S2) or `SR_B4` (Landsat).
const ALIASES: &[(&str, &[&str])] = &[
    // Sentinel-2 common name → band code
    ("red", &["B04", "b04", "Red", "SR_B4"]),
    ("green", &["B03", "b03", "Green", "SR_B3"]),
    ("blue", &["B02", "b02", "Blue", "SR_B2"]),
    ("nir", &["B08", "b08", "nir08", "Nir", "SR_B5"]),
    ("nir08", &["B08", "b08", "nir", "SR_B5"]),
    ("nir09", &["B09", "b09"]),
    ("rededge1", &["B05", "b05"]),
    ("rededge2", &["B06", "b06"]),
    ("rededge3", &["B07", "b07"]),
    ("swir16", &["B11", "b11", "swir1", "SWIR1", "SR_B6"]),
    ("swir22", &["B12", "b12", "swir2", "SWIR2", "SR_B7"]),
    ("scl", &["SCL"]),
    ("coastal", &["B01", "b01", "SR_B1"]),
    ("wvp", &["B09", "b09"]),
    // Sentinel-2 band code → common name
    ("B02", &["blue", "Blue"]),
    ("B03", &["green", "Green"]),
    ("B04", &["red", "Red"]),
    ("B08", &["nir", "nir08"]),
    ("B05", &["rededge1"]),
    ("B06", &["rededge2"]),
    ("B07", &["rededge3"]),
    ("B11", &["swir16", "swir1"]),
    ("B12", &["swir22", "swir2"]),
    ("SCL", &["scl"]),
    // Landsat
    ("SR_B1", &["coastal", "B01"]),
    ("SR_B2", &["blue", "Blue", "B02"]),
    ("SR_B3", &["green", "Green", "B03"]),
    ("SR_B4", &["red", "Red", "B04"]),
    ("SR_B5", &["nir", "nir08", "B08"]),
    ("SR_B6", &["swir16", "swir1", "B11"]),
    ("SR_B7", &["swir22", "swir2", "B12"]),
    ("QA_PIXEL", &["qa_pixel", "QA_pixel"]),
    ("qa_pixel", &["QA_PIXEL", "QA_pixel"]),
    // Sentinel-1
    ("vv", &["VV"]),
    ("vh", &["VH"]),
    ("VV", &["vv"]),
    ("VH", &["vh"]),
];

/// Resolves a band/mask key against a STAC item using exact match first, then
/// case-insensitive match, then the [`ALIASES`] table (so `"red"` maps to the
/// catalog's actual band code). Returns the asset href.
///
/// This is the default resolution used by the Python composite binding; the
/// CLI supplies its own equivalent resolver.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultAssetResolver;

impl AssetResolver for DefaultAssetResolver {
    fn resolve(&self, item: &StacItem, key: &str) -> Option<String> {
        // Exact key.
        if let Some(asset) = item.asset(key) {
            return Some(asset.href.clone());
        }
        // Case-insensitive exact match (QA_PIXEL vs qa_pixel).
        let key_lower = key.to_lowercase();
        for (asset_key, asset) in &item.assets {
            if asset_key.to_lowercase() == key_lower {
                return Some(asset.href.clone());
            }
        }
        // Alias table.
        for &(name, alt_keys) in ALIASES {
            if name.to_lowercase() == key_lower {
                for &alt in alt_keys {
                    if let Some(asset) = item.asset(alt) {
                        return Some(asset.href.clone());
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stac_models::{StacAsset, StacItem};
    use std::collections::HashMap;

    fn item_with(assets: &[(&str, &str)]) -> StacItem {
        let mut map = HashMap::new();
        for (k, href) in assets {
            map.insert(
                k.to_string(),
                StacAsset {
                    href: href.to_string(),
                    ..Default::default()
                },
            );
        }
        StacItem {
            assets: map,
            ..Default::default()
        }
    }

    #[test]
    fn exact_key_wins() {
        let item = item_with(&[("B04", "http://x/b04.tif")]);
        assert_eq!(
            DefaultAssetResolver.resolve(&item, "B04").as_deref(),
            Some("http://x/b04.tif")
        );
    }

    #[test]
    fn common_name_resolves_to_band_code() {
        let s2 = item_with(&[("B04", "http://x/b04.tif")]);
        assert_eq!(
            DefaultAssetResolver.resolve(&s2, "red").as_deref(),
            Some("http://x/b04.tif")
        );
        let landsat = item_with(&[("SR_B5", "http://x/sr_b5.tif")]);
        assert_eq!(
            DefaultAssetResolver.resolve(&landsat, "nir").as_deref(),
            Some("http://x/sr_b5.tif")
        );
    }

    #[test]
    fn case_insensitive_match() {
        let item = item_with(&[("qa_pixel", "http://x/qa.tif")]);
        assert_eq!(
            DefaultAssetResolver.resolve(&item, "QA_PIXEL").as_deref(),
            Some("http://x/qa.tif")
        );
    }

    #[test]
    fn missing_asset_is_none() {
        let item = item_with(&[("B02", "http://x/b02.tif")]);
        assert!(DefaultAssetResolver.resolve(&item, "swir22").is_none());
    }
}
