//! Bridge between STAC search results and the COG reader.
//!
//! Provides convenience functions to read a STAC asset as a [`Raster`] by
//! combining [`StacClient`] (search + sign) with [`CogReader`] (HTTP range
//! fetch + decode).
//!
//! When the COG is stored in a projected CRS (e.g. UTM for Sentinel-2), the
//! WGS84 bbox is automatically reprojected to the COG's native CRS before
//! reading.

use surtgis_core::raster::{Raster, RasterElement};

use crate::cog_reader::{CogReader, CogReaderOptions};
use crate::error::{CloudError, Result};
use crate::reproject;
use crate::stac_client::{StacCatalog, StacClient, StacClientOptions};
use crate::stac_models::{StacItem, StacSearchParams};
use crate::tile_index::BBox;

/// Resolve the effective bbox for reading a COG.
///
/// Tries `proj:epsg` from the STAC item first (avoids an extra HTTP round-trip
/// just to discover the CRS). Falls back to the COG metadata CRS.
fn resolve_read_bbox(bbox: &BBox, item: &StacItem, reader: &CogReader) -> BBox {
    // 1. Try proj:epsg from the STAC item properties
    if let Some(epsg) = item.epsg() {
        if !reproject::is_wgs84(epsg) {
            return reproject::reproject_bbox_to_cog(bbox, epsg);
        }
    }

    // 2. Fallback: CRS from COG metadata
    if let Some(crs) = reader.metadata().crs.as_ref() {
        if let Some(epsg) = crs.epsg() {
            if !reproject::is_wgs84(epsg) {
                return reproject::reproject_bbox_to_cog(bbox, epsg);
            }
        }
    }

    *bbox
}

/// Read a STAC asset as a [`Raster<T>`] via the COG reader.
///
/// 1. Looks up the asset by `asset_key` in the item.
/// 2. Signs the href if needed (Planetary Computer SAS).
/// 3. Opens the COG, auto-reprojects the bbox if needed, and reads.
pub async fn read_stac_asset<T: RasterElement>(
    client: &StacClient,
    item: &StacItem,
    asset_key: &str,
    bbox: &BBox,
    cog_options: CogReaderOptions,
) -> Result<Raster<T>> {
    let asset = item.asset(asset_key).ok_or_else(|| {
        CloudError::Network(format!(
            "asset '{}' not found in item '{}'",
            asset_key, item.id
        ))
    })?;

    let collection = item.collection.as_deref().unwrap_or("");
    let href = client.sign_asset_href(&asset.href, collection).await?;

    let mut reader = CogReader::open(&href, cog_options).await?;
    let read_bbox = resolve_read_bbox(bbox, item, &reader);
    reader.read_bbox(&read_bbox, None).await
}

/// One-shot: search a catalog and read the first matching COG asset.
///
/// Steps:
/// 1. Search the catalog with the given params.
/// 2. Pick the first item.
/// 3. Read the specified asset (or auto-detect a COG) at the given bbox,
///    reprojecting automatically if the COG is in a projected CRS.
pub async fn search_and_read<T: RasterElement>(
    catalog: StacCatalog,
    params: &StacSearchParams,
    asset_key: Option<&str>,
    bbox: &BBox,
) -> Result<Raster<T>> {
    let client = StacClient::new(catalog, StacClientOptions::default())?;
    let results = client.search(params).await?;

    let item = results.features.first().ok_or_else(|| {
        CloudError::Network("STAC search returned no items".to_string())
    })?;

    // Determine which asset to read
    let key = if let Some(k) = asset_key {
        k.to_string()
    } else {
        // Auto-detect first COG asset
        let (k, _) = item.first_cog_asset().ok_or_else(|| {
            CloudError::Network(format!(
                "no COG asset found in item '{}'",
                item.id
            ))
        })?;
        k.clone()
    };

    read_stac_asset(&client, item, &key, bbox, CogReaderOptions::default()).await
}
