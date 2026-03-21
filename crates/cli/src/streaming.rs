//! Cloud-related helper functions: STAC asset resolution, COG reading.

#[cfg(feature = "cloud")]
use anyhow::{Context, Result};
#[cfg(feature = "cloud")]
use tracing::info;

#[cfg(feature = "cloud")]
use surtgis_cloud::blocking::{CogReaderBlocking, StacClientBlocking, read_cog};
#[cfg(feature = "cloud")]
use surtgis_cloud::{BBox, CogReaderOptions, StacItem};

#[cfg(feature = "cloud")]
use crate::helpers::spinner;

#[cfg(feature = "cloud")]
pub fn read_cog_dem(url: &str, bbox: &BBox) -> Result<surtgis_core::Raster<f64>> {
    let pb = spinner("Fetching COG tiles...");
    let opts = CogReaderOptions::default();
    let raster: surtgis_core::Raster<f64> =
        read_cog(url, bbox, opts).context("Failed to read remote COG")?;
    pb.finish_and_clear();
    let (rows, cols) = raster.shape();
    info!(
        "Remote raster: {} x {} ({} cells)",
        cols,
        rows,
        raster.len()
    );
    Ok(raster)
}

/// Sentinel-2 band name aliases: common name -> catalog-specific keys.
/// Tries the exact key first, then aliases.
#[cfg(feature = "cloud")]
pub fn resolve_asset_key<'a>(item: &'a StacItem, key: &'a str) -> Option<(&'a str, &'a surtgis_cloud::stac_models::StacAsset)> {
    // Try exact key first
    if let Some(asset) = item.asset(key) {
        return Some((key, asset));
    }

    // Alias table: common name <-> Sentinel-2 band codes
    let aliases: &[(&str, &[&str])] = &[
        ("red",     &["B04", "b04", "Red"]),
        ("green",   &["B03", "b03", "Green"]),
        ("blue",    &["B02", "b02", "Blue"]),
        ("nir",     &["B08", "b08", "nir08", "Nir"]),
        ("nir08",   &["B08", "b08", "nir"]),
        ("nir09",   &["B09", "b09"]),
        ("rededge1",&["B05", "b05"]),
        ("rededge2",&["B06", "b06"]),
        ("rededge3",&["B07", "b07"]),
        ("swir16",  &["B11", "b11", "swir1", "SWIR1"]),
        ("swir22",  &["B12", "b12", "swir2", "SWIR2"]),
        ("scl",     &["SCL"]),
        ("coastal", &["B01", "b01"]),
        ("wvp",     &["B09", "b09"]),
        // Reverse: band code -> common name
        ("B02",  &["blue", "Blue"]),
        ("B03",  &["green", "Green"]),
        ("B04",  &["red", "Red"]),
        ("B08",  &["nir", "nir08"]),
        ("B05",  &["rededge1"]),
        ("B06",  &["rededge2"]),
        ("B07",  &["rededge3"]),
        ("B11",  &["swir16", "swir1"]),
        ("B12",  &["swir22", "swir2"]),
        ("SCL",  &["scl"]),
    ];

    let key_lower = key.to_lowercase();
    for &(name, alt_keys) in aliases {
        if name.to_lowercase() == key_lower {
            for &alt in alt_keys {
                if let Some(asset) = item.asset(alt) {
                    return Some((alt, asset));
                }
            }
        }
    }
    None
}

/// Fetch a single asset from a STAC item as a raster.
#[cfg(feature = "cloud")]
#[allow(dead_code)]
pub fn fetch_stac_asset(
    item: &StacItem,
    asset_key: &str,
    bbox: &BBox,
    client: &StacClientBlocking,
) -> Result<surtgis_core::Raster<f64>> {
    let (resolved_key, stac_asset) = resolve_asset_key(item, asset_key)
        .ok_or_else(|| {
            let available: Vec<&str> = item.assets.keys().map(|k| k.as_str()).collect();
            anyhow::anyhow!(
                "Item {} missing asset '{}'. Available: {}",
                item.id, asset_key, available.join(", ")
            )
        })?;

    if resolved_key != asset_key {
        info!("Resolved asset '{}' -> '{}'", asset_key, resolved_key);
    }

    let stac_asset = stac_asset.clone();

    let href = client
        .sign_asset_href(&stac_asset.href, item.collection.as_deref().unwrap_or(""))
        .context("Failed to sign asset URL")?;

    let opts = CogReaderOptions::default();
    let mut reader =
        CogReaderBlocking::open(&href, opts).context("Failed to open remote COG")?;

    // Auto-reproject bbox if COG is in a projected CRS
    let read_bb = {
        use surtgis_cloud::reproject;
        let epsg = item
            .epsg()
            .or_else(|| reader.metadata().crs.as_ref().and_then(|c| c.epsg()));
        if let Some(epsg) = epsg {
            if !reproject::is_wgs84(epsg) {
                reproject::reproject_bbox_to_cog(bbox, epsg)
            } else {
                *bbox
            }
        } else {
            *bbox
        }
    };

    let raster: surtgis_core::Raster<f64> = reader
        .read_bbox(&read_bb, None)
        .context("Failed to read bounding box from COG")?;
    Ok(raster)
}
