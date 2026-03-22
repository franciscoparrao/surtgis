//! Vector data structures and operations.
//!
//! Supports reading features from multiple formats:
//! - **GeoJSON** (always available)
//! - **Shapefile** (requires `shapefile` feature)
//! - **GeoPackage** (requires `geopackage` feature)
//!
//! Use [`read_vector`] for automatic format detection by file extension,
//! or call the format-specific readers directly.

pub mod geojson_reader;
pub mod rasterize;

#[cfg(feature = "shapefile")]
pub mod shapefile_reader;
#[cfg(feature = "geopackage")]
pub mod gpkg_reader;

use geo_types::Geometry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use geojson_reader::{parse_geojson, read_geojson};
pub use rasterize::{clip_raster, clip_raster_by_polygon, rasterize_polygons};

#[cfg(feature = "shapefile")]
pub use shapefile_reader::read_shapefile;
#[cfg(feature = "geopackage")]
pub use gpkg_reader::{read_gpkg, list_gpkg_layers};

/// Attribute value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

/// A geographic feature with geometry and attributes
#[derive(Debug, Clone)]
pub struct Feature {
    /// Feature geometry
    pub geometry: Option<Geometry<f64>>,
    /// Feature attributes
    pub properties: HashMap<String, AttributeValue>,
    /// Optional feature ID
    pub id: Option<String>,
}

impl Feature {
    /// Create a new feature with geometry
    pub fn new(geometry: Geometry<f64>) -> Self {
        Self {
            geometry: Some(geometry),
            properties: HashMap::new(),
            id: None,
        }
    }

    /// Create a feature with no geometry
    pub fn empty() -> Self {
        Self {
            geometry: None,
            properties: HashMap::new(),
            id: None,
        }
    }

    /// Set an attribute
    pub fn set_property(&mut self, key: impl Into<String>, value: AttributeValue) {
        self.properties.insert(key.into(), value);
    }

    /// Get an attribute
    pub fn get_property(&self, key: &str) -> Option<&AttributeValue> {
        self.properties.get(key)
    }
}

/// Collection of features
#[derive(Debug, Clone, Default)]
pub struct FeatureCollection {
    pub features: Vec<Feature>,
}

impl FeatureCollection {
    pub fn new() -> Self {
        Self { features: Vec::new() }
    }

    pub fn push(&mut self, feature: Feature) {
        self.features.push(feature);
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Feature> {
        self.features.iter()
    }
}

impl IntoIterator for FeatureCollection {
    type Item = Feature;
    type IntoIter = std::vec::IntoIter<Feature>;

    fn into_iter(self) -> Self::IntoIter {
        self.features.into_iter()
    }
}

/// Read vector features from any supported format, auto-detected by file extension.
///
/// Supported extensions:
/// - `.geojson`, `.json` -> GeoJSON
/// - `.shp` -> Shapefile (requires `shapefile` feature)
/// - `.gpkg` -> GeoPackage (requires `geopackage` feature)
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use surtgis_core::vector::read_vector;
///
/// let fc = read_vector(Path::new("basins.shp")).unwrap();
/// println!("Read {} features", fc.len());
/// ```
pub fn read_vector(path: &std::path::Path) -> crate::error::Result<FeatureCollection> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "geojson" | "json" => geojson_reader::read_geojson(path),
        #[cfg(feature = "shapefile")]
        "shp" => shapefile_reader::read_shapefile(path),
        #[cfg(feature = "geopackage")]
        "gpkg" => gpkg_reader::read_gpkg(path, None),
        _ => {
            let mut supported = vec![".geojson", ".json"];
            #[cfg(feature = "shapefile")]
            supported.push(".shp");
            #[cfg(feature = "geopackage")]
            supported.push(".gpkg");
            Err(crate::error::Error::Other(format!(
                "Unsupported vector format: '.{}'. Supported: {}",
                ext,
                supported.join(", ")
            )))
        }
    }
}
