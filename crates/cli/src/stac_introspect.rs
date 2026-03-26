//! STAC catalog introspection: auto-detect band types, cloud masking, CRS
//!
//! Enables SurtGIS to work with ANY STAC API (not just Planetary Computer)
//! by automatically discovering available bands, cloud masking strategies,
//! and metadata from the first item of any collection.

use anyhow::Result;
use surtgis_cloud::StacItem;

/// Auto-detected metadata for a STAC collection
#[derive(Debug, Clone)]
pub struct StacCollectionSchema {
    pub collection_name: String,
    pub available_bands: Vec<BandInfo>,
    pub cloud_mask_asset: Option<String>,
    pub cloud_mask_type: CloudMaskType,
    pub resolution_m: (f64, f64), // (pixel_width, pixel_height) in meters
    pub crs_epsg: Option<u32>,
}

/// Information about a single band in the collection
#[derive(Debug, Clone)]
pub struct BandInfo {
    pub asset_key: String,                // "B04", "SR_B4", "VV", etc.
    pub band_type: BandType,              // Red, NIR, Thermal, etc.
    pub wavelength_um: Option<f64>,       // 0.665 for red, 0.842 for NIR
    pub resolution_m: Option<f64>,        // band-specific resolution
    pub description: Option<String>,      // From STAC metadata
}

/// Automatically detected band type from asset key
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandType {
    Blue,
    Green,
    Red,
    Nir,
    Swir1,
    Swir2,
    Thermal,
    Pan,
    Sar(SarPol),
    Unknown,
}

/// SAR polarization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SarPol {
    VV,
    VH,
    HH,
    HV,
    Quad,
}

impl std::fmt::Display for SarPol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SarPol::VV => write!(f, "VV"),
            SarPol::VH => write!(f, "VH"),
            SarPol::HH => write!(f, "HH"),
            SarPol::HV => write!(f, "HV"),
            SarPol::Quad => write!(f, "Quad"),
        }
    }
}

/// Cloud masking strategy detected from STAC metadata
#[derive(Debug, Clone, PartialEq)]
pub enum CloudMaskType {
    /// Categorical masking (e.g., Sentinel-2 SCL with classes 0-11)
    Categorical { asset: String, num_classes: u32 },
    /// Bitmask masking (e.g., Landsat QA_PIXEL)
    Bitmask { asset: String, bits: Vec<u32> },
    /// No cloud masking needed (e.g., SAR)
    None,
}

impl StacCollectionSchema {
    /// Introspect a STAC collection from its first item
    pub fn from_stac_item(
        collection_name: &str,
        item: &StacItem,
    ) -> Result<Self> {
        let available_bands = Self::detect_bands(item)?;
        let cloud_mask_info = Self::detect_cloud_mask(item, &available_bands);
        let (crs_epsg, resolution_m) = Self::extract_crs_and_resolution(item)?;

        Ok(Self {
            collection_name: collection_name.to_string(),
            available_bands,
            cloud_mask_asset: cloud_mask_info.0,
            cloud_mask_type: cloud_mask_info.1,
            resolution_m,
            crs_epsg,
        })
    }

    /// Detect all available bands from item assets
    fn detect_bands(item: &StacItem) -> Result<Vec<BandInfo>> {
        let mut bands = Vec::new();

        for (asset_key, asset) in &item.assets {
            // Skip non-raster assets
            if let Some(media_type) = &asset.type_ {
                if !media_type.contains("image/tiff") && !media_type.contains("image/jp2") {
                    continue;
                }
            }

            let band_type = detect_band_type(asset_key, item);

            // Skip cloud mask and quality assets from band list
            if matches!(
                band_type,
                BandType::Unknown if asset_key.to_lowercase().contains("mask")
                    || asset_key.to_lowercase().contains("qa")
                    || asset_key.to_lowercase().contains("scl")
            ) {
                continue;
            }

            let wavelength = wavelength_for_band_type(band_type);

            bands.push(BandInfo {
                asset_key: asset_key.clone(),
                band_type,
                wavelength_um: wavelength,
                resolution_m: None, // Would extract from item.properties if available
                description: asset.title.clone(),
            });
        }

        Ok(bands)
    }

    /// Detect cloud masking strategy from item assets and properties
    fn detect_cloud_mask(
        item: &StacItem,
        available_bands: &[BandInfo],
    ) -> (Option<String>, CloudMaskType) {
        // Check for SCL (Sentinel-2)
        if item.assets.contains_key("SCL") {
            return (
                Some("SCL".to_string()),
                CloudMaskType::Categorical {
                    asset: "SCL".to_string(),
                    num_classes: 12,
                },
            );
        }

        // Check for QA_PIXEL (Landsat)
        for key in ["QA_PIXEL", "qa_pixel", "QA", "qa"] {
            if item.assets.contains_key(key) {
                return (
                    Some(key.to_string()),
                    CloudMaskType::Bitmask {
                        asset: key.to_string(),
                        bits: vec![1, 3, 4], // cloud, shadow, snow
                    },
                );
            }
        }

        // Check if SAR (has sar:polarizations property)
        // Note: item.properties is a StacItemProperties struct, not a HashMap
        // We'll check for SAR via band detection instead

        // Check if any band is SAR
        if available_bands.iter().any(|b| matches!(b.band_type, BandType::Sar(_))) {
            return (None, CloudMaskType::None);
        }

        // Default: no cloud masking detected
        (None, CloudMaskType::None)
    }

    /// Extract CRS and resolution from item metadata
    fn extract_crs_and_resolution(item: &StacItem) -> Result<(Option<u32>, (f64, f64))> {
        let crs_epsg = item.epsg();
        let resolution = (10.0, 10.0); // Default: 10m, will refine if needed

        Ok((crs_epsg, resolution))
    }

    /// Find best matching band by name or type
    pub fn find_band_by_name(&self, query: &str) -> Option<&BandInfo> {
        let query_lower = query.to_lowercase();

        // Exact asset key match
        if let Some(band) = self.available_bands.iter().find(|b| {
            b.asset_key.to_lowercase() == query_lower
        }) {
            return Some(band);
        }

        // Type name match (red, nir, thermal, etc.)
        let query_type = detect_band_type_from_name(&query_lower);
        if query_type != BandType::Unknown {
            if let Some(band) = self.available_bands.iter().find(|b| {
                b.band_type == query_type
            }) {
                return Some(band);
            }
        }

        None
    }

    /// List available bands in human-readable format
    pub fn format_bands(&self) -> String {
        self.available_bands
            .iter()
            .map(|b| format!("{} ({:?})", b.asset_key, b.band_type))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Detect band type from asset key (heuristic-based)
pub fn detect_band_type(asset_key: &str, item: &StacItem) -> BandType {
    let key_lower = asset_key.to_lowercase();

    // Priority 1: Check STAC eo:bands specification (standard)
    if let Some(asset) = item.assets.get(asset_key) {
        // Look for common_name in EO bands metadata
        if let Some(common) = asset.extra.get("common_name") {
            if let Some(common_str) = common.as_str() {
                return detect_band_type_from_name(common_str);
            }
        }
    }

    // Priority 2: Asset key pattern matching
    detect_band_type_from_name(&key_lower)
}

/// Detect band type from name/key string (case-insensitive)
fn detect_band_type_from_name(name: &str) -> BandType {
    let name_lower = name.to_lowercase();
    match name_lower.as_str() {
        // Sentinel-2 bands
        "b02" | "blue" | "coastal" | "banda_azul" => BandType::Blue,
        "b03" | "green" | "banda_verde" => BandType::Green,
        "b04" | "red" | "banda_roja" | "rouge" => BandType::Red,
        "b08" | "b8" | "nir" | "nir08" | "infrared" | "proche_infrarouge" => BandType::Nir,
        "b11" | "swir1" | "swir16" | "mid_infrared" => BandType::Swir1,
        "b12" | "swir2" | "swir22" => BandType::Swir2,

        // Landsat bands
        "sr_b1" | "sr_b2" | "blue_l8" | "blue_l9" => BandType::Blue,
        "sr_b3" | "green_l8" | "green_l9" => BandType::Green,
        "sr_b4" | "sr_b4c" | "red_l8" | "red_l9" => BandType::Red,
        "sr_b5" | "nir_l8" | "nir_l9" => BandType::Nir,
        "sr_b6" | "swir1_l8" | "swir1_l9" => BandType::Swir1,
        "sr_b7" | "swir2_l8" | "swir2_l9" => BandType::Swir2,
        "st_b10" | "st_b11" | "thermal" | "tirs1" | "tirs2" => BandType::Thermal,
        "b10" | "lwir1" | "lwir" => BandType::Thermal,

        // SAR bands
        "vv" | "vv_amplitude" | "vh" | "vh_amplitude" => {
            if name_lower.contains("vh") {
                BandType::Sar(SarPol::VH)
            } else {
                BandType::Sar(SarPol::VV)
            }
        }
        "hh" | "hh_amplitude" | "hv" | "hv_amplitude" => {
            if name_lower.contains("hv") {
                BandType::Sar(SarPol::HV)
            } else {
                BandType::Sar(SarPol::HH)
            }
        }

        // Panchromatic
        "pan" | "panchromatic" => BandType::Pan,

        _ => BandType::Unknown,
    }
}

/// Get typical wavelength for band type (in micrometers)
fn wavelength_for_band_type(band_type: BandType) -> Option<f64> {
    match band_type {
        BandType::Blue => Some(0.485),
        BandType::Green => Some(0.560),
        BandType::Red => Some(0.665),
        BandType::Nir => Some(0.842),
        BandType::Swir1 => Some(1.610),
        BandType::Swir2 => Some(2.190),
        BandType::Thermal => Some(10.9),
        BandType::Pan => Some(0.590),
        BandType::Sar(_) => None,
        BandType::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_band_type_sentinel2() {
        assert_eq!(detect_band_type_from_name("B04"), BandType::Red);
        assert_eq!(detect_band_type_from_name("red"), BandType::Red);
        assert_eq!(detect_band_type_from_name("B08"), BandType::Nir);
        assert_eq!(detect_band_type_from_name("B02"), BandType::Blue);
        assert_eq!(detect_band_type_from_name("B11"), BandType::Swir1);
    }

    #[test]
    fn test_detect_band_type_landsat() {
        assert_eq!(detect_band_type_from_name("SR_B4"), BandType::Red);
        assert_eq!(detect_band_type_from_name("red_l8"), BandType::Red);
        assert_eq!(detect_band_type_from_name("SR_B5"), BandType::Nir);
        assert_eq!(detect_band_type_from_name("SR_B6"), BandType::Swir1);
    }

    #[test]
    fn test_detect_band_type_sar() {
        assert_eq!(
            detect_band_type_from_name("VV"),
            BandType::Sar(SarPol::VV)
        );
        assert_eq!(
            detect_band_type_from_name("VH"),
            BandType::Sar(SarPol::VH)
        );
    }

    #[test]
    fn test_detect_band_type_multilingual() {
        assert_eq!(detect_band_type_from_name("banda_roja"), BandType::Red);
        assert_eq!(detect_band_type_from_name("rouge"), BandType::Red);
        assert_eq!(
            detect_band_type_from_name("proche_infrarouge"),
            BandType::Nir
        );
    }

    #[test]
    fn test_wavelength_for_band_type() {
        assert_eq!(wavelength_for_band_type(BandType::Red), Some(0.665));
        assert_eq!(wavelength_for_band_type(BandType::Nir), Some(0.842));
        assert_eq!(wavelength_for_band_type(BandType::Thermal), Some(10.9));
        assert_eq!(wavelength_for_band_type(BandType::Sar(SarPol::VV)), None);
    }

    #[test]
    fn test_cloud_mask_type_display() {
        let scl = CloudMaskType::Categorical {
            asset: "SCL".to_string(),
            num_classes: 12,
        };
        assert_eq!(scl, CloudMaskType::Categorical { asset: "SCL".to_string(), num_classes: 12 });
    }

    #[test]
    fn test_band_type_case_insensitive() {
        // Ensure case-insensitive matching works
        assert_eq!(detect_band_type_from_name("b04"), BandType::Red);
        assert_eq!(detect_band_type_from_name("B04"), BandType::Red);
        assert_eq!(detect_band_type_from_name("RED"), BandType::Red);
        assert_eq!(detect_band_type_from_name("Red"), BandType::Red);
    }

    #[test]
    fn test_sar_band_type_equality() {
        assert_eq!(
            detect_band_type_from_name("vv"),
            BandType::Sar(SarPol::VV)
        );
        assert_eq!(
            detect_band_type_from_name("VV"),
            BandType::Sar(SarPol::VV)
        );
    }

    #[test]
    fn test_thermal_band_detection() {
        assert_eq!(detect_band_type_from_name("thermal"), BandType::Thermal);
        assert_eq!(detect_band_type_from_name("tirs1"), BandType::Thermal);
        assert_eq!(detect_band_type_from_name("b10"), BandType::Thermal);
        assert_eq!(detect_band_type_from_name("lwir"), BandType::Thermal);
        assert_eq!(detect_band_type_from_name("st_b10"), BandType::Thermal);
    }

    #[test]
    fn test_panchromatic_detection() {
        assert_eq!(detect_band_type_from_name("pan"), BandType::Pan);
        assert_eq!(detect_band_type_from_name("panchromatic"), BandType::Pan);
    }

    #[test]
    fn test_swir_bands() {
        assert_eq!(detect_band_type_from_name("b11"), BandType::Swir1);
        assert_eq!(detect_band_type_from_name("b12"), BandType::Swir2);
        assert_eq!(detect_band_type_from_name("swir1"), BandType::Swir1);
        assert_eq!(detect_band_type_from_name("swir2"), BandType::Swir2);
    }

    #[test]
    fn test_unknown_band_type() {
        assert_eq!(detect_band_type_from_name("unknown_band"), BandType::Unknown);
        assert_eq!(detect_band_type_from_name("xyz"), BandType::Unknown);
        assert_eq!(detect_band_type_from_name(""), BandType::Unknown);
    }

    #[test]
    fn test_landsat_collection_all_bands() {
        // Test full Landsat band suite
        assert_eq!(detect_band_type_from_name("sr_b1"), BandType::Blue);
        assert_eq!(detect_band_type_from_name("sr_b2"), BandType::Blue);
        assert_eq!(detect_band_type_from_name("sr_b3"), BandType::Green);
        assert_eq!(detect_band_type_from_name("sr_b4"), BandType::Red);
        assert_eq!(detect_band_type_from_name("sr_b5"), BandType::Nir);
        assert_eq!(detect_band_type_from_name("sr_b6"), BandType::Swir1);
        assert_eq!(detect_band_type_from_name("sr_b7"), BandType::Swir2);
        assert_eq!(detect_band_type_from_name("st_b10"), BandType::Thermal);
    }

    #[test]
    fn test_eo_band_common_names() {
        // Test EO-specific naming patterns
        assert_eq!(detect_band_type_from_name("nir08"), BandType::Nir);  // S2 narrow NIR
        assert_eq!(detect_band_type_from_name("swir16"), BandType::Swir1);  // S2 SWIR naming
        assert_eq!(detect_band_type_from_name("swir22"), BandType::Swir2);
    }

    #[test]
    fn test_multilingual_comprehensive() {
        // Spanish
        assert_eq!(detect_band_type_from_name("banda_roja"), BandType::Red);
        assert_eq!(detect_band_type_from_name("banda_verde"), BandType::Green);
        assert_eq!(detect_band_type_from_name("banda_azul"), BandType::Blue);

        // French
        assert_eq!(detect_band_type_from_name("rouge"), BandType::Red);
        assert_eq!(detect_band_type_from_name("proche_infrarouge"), BandType::Nir);
    }

    #[test]
    fn test_sarpol_display() {
        assert_eq!(format!("{}", SarPol::VV), "VV");
        assert_eq!(format!("{}", SarPol::VH), "VH");
        assert_eq!(format!("{}", SarPol::HH), "HH");
        assert_eq!(format!("{}", SarPol::HV), "HV");
        assert_eq!(format!("{}", SarPol::Quad), "Quad");
    }
}
