//! STAC (SpatioTemporal Asset Catalog) data types.
//!
//! Lightweight serde models for STAC Item Search (POST /search) responses,
//! covering the subset needed by SurtGis: bbox, datetime, collections filtering,
//! pagination via `links`, and asset access.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Search request
// ---------------------------------------------------------------------------

/// Body for `POST /search` (STAC API – Item Search).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StacSearchParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Vec<f64>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub datetime: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub collections: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,

    /// Pagination token (next page).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
}

impl StacSearchParams {
    /// Create empty search params.
    pub fn new() -> Self {
        Self {
            bbox: None,
            datetime: None,
            collections: None,
            limit: None,
            token: None,
        }
    }

    /// Set the bounding box `[west, south, east, north]`.
    pub fn bbox(mut self, west: f64, south: f64, east: f64, north: f64) -> Self {
        self.bbox = Some(vec![west, south, east, north]);
        self
    }

    /// Set datetime or datetime range (e.g. `"2024-06-01/2024-06-30"`).
    pub fn datetime(mut self, dt: &str) -> Self {
        self.datetime = Some(dt.to_string());
        self
    }

    /// Set collection filter.
    pub fn collections(mut self, cols: &[&str]) -> Self {
        self.collections = Some(cols.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Set maximum items per page.
    pub fn limit(mut self, n: u32) -> Self {
        self.limit = Some(n);
        self
    }

    /// Set pagination token.
    pub fn token(mut self, tok: &str) -> Self {
        self.token = Some(tok.to_string());
        self
    }
}

impl Default for StacSearchParams {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// A STAC Item Collection (GeoJSON FeatureCollection).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StacItemCollection {
    #[serde(rename = "type")]
    pub type_: String,

    pub features: Vec<StacItem>,

    #[serde(default)]
    pub links: Vec<StacLink>,

    /// Some catalogs return `numberMatched` or `context.matched`.
    #[serde(rename = "numberMatched", skip_serializing_if = "Option::is_none")]
    pub number_matched: Option<u64>,

    /// Some catalogs return `numberReturned` or `context.returned`.
    #[serde(rename = "numberReturned", skip_serializing_if = "Option::is_none")]
    pub number_returned: Option<u64>,

    /// Earth Search / some catalogs use `context` instead of numberMatched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<serde_json::Value>,
}

impl StacItemCollection {
    /// Find the `"next"` pagination link, if any.
    pub fn next_link(&self) -> Option<&StacLink> {
        self.links.iter().find(|l| l.rel == "next")
    }

    /// Whether there is a next page.
    pub fn has_next(&self) -> bool {
        self.next_link().is_some()
    }

    /// Total number of items in this page.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// A single STAC Item (GeoJSON Feature).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StacItem {
    #[serde(rename = "type")]
    pub type_: String,

    /// Unique item identifier.
    pub id: String,

    /// Geometry as raw JSON (we don't need to parse it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometry: Option<serde_json::Value>,

    /// Bounding box `[west, south, east, north]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Vec<f64>>,

    pub properties: StacItemProperties,

    pub assets: HashMap<String, StacAsset>,

    /// Collection this item belongs to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,

    #[serde(default)]
    pub links: Vec<StacLink>,
}

impl StacItem {
    /// Get an asset by key.
    pub fn asset(&self, key: &str) -> Option<&StacAsset> {
        self.assets.get(key)
    }

    /// Get the EPSG code from the `proj:epsg` property, if available.
    ///
    /// Most STAC items from Sentinel-2 / Landsat include this via the
    /// [projection extension](https://github.com/stac-extensions/projection).
    pub fn epsg(&self) -> Option<u32> {
        self.properties
            .extra
            .get("proj:epsg")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
    }

    /// Find the first asset that looks like a COG (role contains "data" and
    /// media type is GeoTIFF or the href ends in `.tif`/`.tiff`).
    pub fn first_cog_asset(&self) -> Option<(&String, &StacAsset)> {
        self.assets.iter().find(|(_, a)| {
            let is_data_role = a
                .roles
                .as_ref()
                .map(|r| r.iter().any(|role| role == "data"))
                .unwrap_or(false);
            let is_geotiff = a
                .type_
                .as_ref()
                .map(|t| {
                    t.contains("geotiff")
                        || t.contains("geo+tiff")
                        || t.contains("cloud-optimized")
                })
                .unwrap_or(false);
            let href_tiff = a.href.ends_with(".tif") || a.href.ends_with(".tiff");
            is_data_role && (is_geotiff || href_tiff)
        })
    }
}

/// STAC Item properties.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StacItemProperties {
    /// ISO 8601 datetime.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub datetime: Option<String>,

    /// Cloud cover percentage (EO extension).
    #[serde(rename = "eo:cloud_cover", skip_serializing_if = "Option::is_none")]
    pub eo_cloud_cover: Option<f64>,

    /// Platform name (e.g., "sentinel-2a").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platform: Option<String>,

    /// Constellation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constellation: Option<String>,

    /// GSD (ground sample distance).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gsd: Option<f64>,

    /// All other properties we don't model explicitly.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// A single STAC Asset (file reference).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StacAsset {
    /// URL to the asset file.
    pub href: String,

    /// Media type (e.g., `"image/tiff; application=geotiff; profile=cloud-optimized"`).
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,

    /// Human-readable title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Roles: `["data"]`, `["thumbnail"]`, `["overview"]`, etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roles: Option<Vec<String>>,

    /// All other asset fields.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// A STAC Link (used for pagination and related resources).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StacLink {
    /// Relationship: `"self"`, `"root"`, `"next"`, `"prev"`, etc.
    pub rel: String,

    /// Target URL.
    pub href: String,

    /// HTTP method for the link (default GET, but `"next"` often uses POST).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,

    /// Request body for POST-based pagination.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<serde_json::Value>,

    /// Merge mode: if true, merge body with previous request body.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merge: Option<bool>,

    /// Media type of the linked resource.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": "S2A_MSIL2A_20240615T105621_R094_T30TVK_20240615T164132",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-3.95, 40.22], [-2.84, 40.22], [-2.84, 41.21], [-3.95, 41.21], [-3.95, 40.22]]]
      },
      "bbox": [-3.95, 40.22, -2.84, 41.21],
      "properties": {
        "datetime": "2024-06-15T10:56:21Z",
        "eo:cloud_cover": 5.2,
        "platform": "sentinel-2a",
        "gsd": 10.0,
        "proj:epsg": 32630
      },
      "assets": {
        "red": {
          "href": "https://example.com/B04.tif",
          "type": "image/tiff; application=geotiff; profile=cloud-optimized",
          "title": "Band 4 - Red",
          "roles": ["data"]
        },
        "nir": {
          "href": "https://example.com/B08.tif",
          "type": "image/tiff; application=geotiff; profile=cloud-optimized",
          "title": "Band 8 - NIR",
          "roles": ["data"]
        },
        "thumbnail": {
          "href": "https://example.com/thumb.png",
          "type": "image/png",
          "title": "Thumbnail",
          "roles": ["thumbnail"]
        }
      },
      "collection": "sentinel-2-l2a",
      "links": []
    }
  ],
  "links": [
    {
      "rel": "next",
      "href": "https://earth-search.aws.element84.com/v1/search",
      "method": "POST",
      "body": {"token": "abc123"},
      "merge": true
    },
    {
      "rel": "self",
      "href": "https://earth-search.aws.element84.com/v1/search"
    }
  ],
  "numberMatched": 42,
  "numberReturned": 1
}"#;

    #[test]
    fn parse_item_collection() {
        let col: StacItemCollection = serde_json::from_str(FIXTURE).unwrap();
        assert_eq!(col.type_, "FeatureCollection");
        assert_eq!(col.features.len(), 1);
        assert_eq!(col.number_matched, Some(42));
        assert_eq!(col.number_returned, Some(1));
    }

    #[test]
    fn parse_item() {
        let col: StacItemCollection = serde_json::from_str(FIXTURE).unwrap();
        let item = &col.features[0];
        assert_eq!(item.id, "S2A_MSIL2A_20240615T105621_R094_T30TVK_20240615T164132");
        assert_eq!(item.collection.as_deref(), Some("sentinel-2-l2a"));
        assert!(item.geometry.is_some());
        assert_eq!(item.bbox.as_ref().unwrap().len(), 4);
    }

    #[test]
    fn parse_properties() {
        let col: StacItemCollection = serde_json::from_str(FIXTURE).unwrap();
        let props = &col.features[0].properties;
        assert_eq!(props.datetime.as_deref(), Some("2024-06-15T10:56:21Z"));
        assert!((props.eo_cloud_cover.unwrap() - 5.2).abs() < f64::EPSILON);
        assert_eq!(props.platform.as_deref(), Some("sentinel-2a"));
        assert!((props.gsd.unwrap() - 10.0).abs() < f64::EPSILON);
        // Extra fields should be captured by flatten
        assert!(props.extra.contains_key("proj:epsg"));
    }

    #[test]
    fn asset_lookup() {
        let col: StacItemCollection = serde_json::from_str(FIXTURE).unwrap();
        let item = &col.features[0];

        assert!(item.asset("red").is_some());
        assert!(item.asset("nir").is_some());
        assert!(item.asset("nonexistent").is_none());

        let red = item.asset("red").unwrap();
        assert_eq!(red.href, "https://example.com/B04.tif");
        assert!(red.type_.as_ref().unwrap().contains("geotiff"));
        assert_eq!(red.roles.as_ref().unwrap(), &["data"]);
    }

    #[test]
    fn epsg_from_proj_extension() {
        let col: StacItemCollection = serde_json::from_str(FIXTURE).unwrap();
        let item = &col.features[0];
        assert_eq!(item.epsg(), Some(32630));
    }

    #[test]
    fn first_cog_asset_finds_data_geotiff() {
        let col: StacItemCollection = serde_json::from_str(FIXTURE).unwrap();
        let item = &col.features[0];
        let (key, asset) = item.first_cog_asset().unwrap();
        // Should find one of the data assets (red or nir), both are COGs
        assert!(key == "red" || key == "nir");
        assert!(asset.href.ends_with(".tif"));
    }

    #[test]
    fn pagination_links() {
        let col: StacItemCollection = serde_json::from_str(FIXTURE).unwrap();
        assert!(col.has_next());

        let next = col.next_link().unwrap();
        assert_eq!(next.rel, "next");
        assert_eq!(next.method.as_deref(), Some("POST"));
        assert!(next.body.is_some());
        assert_eq!(next.merge, Some(true));
    }

    #[test]
    fn builder_serializes_correctly() {
        let params = StacSearchParams::new()
            .bbox(-3.75, 40.38, -3.65, 40.45)
            .datetime("2024-06-01/2024-06-30")
            .collections(&["sentinel-2-l2a"])
            .limit(5);

        let json = serde_json::to_value(&params).unwrap();
        assert_eq!(json["bbox"], serde_json::json!([-3.75, 40.38, -3.65, 40.45]));
        assert_eq!(json["datetime"], "2024-06-01/2024-06-30");
        assert_eq!(json["collections"], serde_json::json!(["sentinel-2-l2a"]));
        assert_eq!(json["limit"], 5);
        assert!(json.get("token").is_none());
    }

    #[test]
    fn empty_params_has_no_fields() {
        let params = StacSearchParams::new();
        let json = serde_json::to_value(&params).unwrap();
        let obj = json.as_object().unwrap();
        assert!(obj.is_empty());
    }

    #[test]
    fn no_next_link_when_absent() {
        let col = StacItemCollection {
            type_: "FeatureCollection".to_string(),
            features: vec![],
            links: vec![StacLink {
                rel: "self".to_string(),
                href: "https://example.com".to_string(),
                method: None,
                body: None,
                merge: None,
                type_: None,
            }],
            number_matched: None,
            number_returned: None,
            context: None,
        };
        assert!(!col.has_next());
        assert!(col.next_link().is_none());
    }
}
