//! Bridge between STAC search results and the COG reader.
//!
//! Provides convenience functions to read a STAC asset as a [`Raster`] by
//! combining [`StacClient`] (search + sign) with [`CogReader`] (HTTP range
//! fetch + decode).

use surtgis_core::raster::{Raster, RasterElement};

use crate::cog_reader::{CogReader, CogReaderOptions};
use crate::error::{CloudError, Result};
use crate::stac_client::{StacCatalog, StacClient, StacClientOptions};
use crate::stac_models::{StacItem, StacSearchParams};
use crate::tile_index::BBox;

/// Read a STAC asset as a [`Raster<T>`] via the COG reader.
///
/// 1. Looks up the asset by `asset_key` in the item.
/// 2. Signs the href if needed (Planetary Computer SAS).
/// 3. Opens the COG and reads the requested bbox.
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
    reader.read_bbox(bbox, None).await
}

/// One-shot: search a catalog and read the first matching COG asset.
///
/// Steps:
/// 1. Search the catalog with the given params.
/// 2. Pick the first item.
/// 3. Read the specified asset (or auto-detect a COG) at the given bbox.
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
