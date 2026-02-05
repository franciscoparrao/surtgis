//! WASM bindings for the COG reader.
//!
//! Exposes an async `cog_read_bbox` function that can be called from
//! JavaScript via `wasm-bindgen`.

#[cfg(feature = "wasm")]
mod inner {
    use wasm_bindgen::prelude::*;

    use crate::cog_reader::{CogReader, CogReaderOptions};
    use crate::tile_index::BBox;
    use surtgis_core::io::write_geotiff_to_buffer;

    /// Read a bounding box from a remote COG and return the result as
    /// a GeoTIFF byte array.
    ///
    /// # Arguments
    ///
    /// * `url` — URL of the COG file.
    /// * `min_x`, `min_y`, `max_x`, `max_y` — Geographic bounding box.
    ///
    /// # Returns
    ///
    /// `Uint8Array` containing a GeoTIFF with the requested region.
    #[wasm_bindgen]
    pub async fn cog_read_bbox(
        url: &str,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Result<Vec<u8>, JsValue> {
        let opts = CogReaderOptions::default();
        let mut reader = CogReader::open(url, opts)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let bbox = BBox::new(min_x, min_y, max_x, max_y);
        let raster: surtgis_core::Raster<f64> = reader
            .read_bbox(&bbox, None)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        write_geotiff_to_buffer(&raster, None)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get metadata from a remote COG as a JSON string.
    #[wasm_bindgen]
    pub async fn cog_metadata(url: &str) -> Result<String, JsValue> {
        let opts = CogReaderOptions::default();
        let reader = CogReader::open(url, opts)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let meta = reader.metadata();
        let json = serde_json::json!({
            "url": meta.url,
            "width": meta.width,
            "height": meta.height,
            "tile_width": meta.tile_width,
            "tile_height": meta.tile_height,
            "bits_per_sample": meta.bits_per_sample,
            "sample_format": meta.sample_format,
            "compression": meta.compression,
            "nodata": meta.nodata,
            "num_overviews": meta.num_overviews,
            "geo_transform": {
                "origin_x": meta.geo_transform.origin_x,
                "origin_y": meta.geo_transform.origin_y,
                "pixel_width": meta.geo_transform.pixel_width,
                "pixel_height": meta.geo_transform.pixel_height,
            },
        });

        Ok(json.to_string())
    }

    /// Search a STAC catalog and return results as JSON.
    ///
    /// `catalog` — shorthand (`"pc"`, `"es"`) or full STAC API URL.
    ///
    /// Returns a JSON string with the STAC ItemCollection response.
    #[wasm_bindgen]
    pub async fn stac_search(
        catalog: &str,
        bbox: &str,
        datetime: &str,
        collections: &str,
        limit: u32,
    ) -> Result<String, JsValue> {
        use crate::stac_client::{StacCatalog, StacClient, StacClientOptions};
        use crate::stac_models::StacSearchParams;

        let cat = StacCatalog::from_str_or_url(catalog);
        let client = StacClient::new(cat, StacClientOptions::default())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Parse bbox: "west,south,east,north"
        let mut params = StacSearchParams::new().limit(limit);

        if !bbox.is_empty() {
            let parts: Vec<f64> = bbox
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if parts.len() == 4 {
                params = params.bbox(parts[0], parts[1], parts[2], parts[3]);
            }
        }

        if !datetime.is_empty() {
            params = params.datetime(datetime);
        }

        if !collections.is_empty() {
            let cols: Vec<&str> = collections.split(',').map(|s| s.trim()).collect();
            params = params.collections(&cols);
        }

        let results = client
            .search(&params)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        serde_json::to_string(&results)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
pub use inner::*;
