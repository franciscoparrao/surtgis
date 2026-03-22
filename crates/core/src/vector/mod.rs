//! Vector data structures and operations.

pub mod geojson_reader;
pub mod rasterize;

use geo_types::Geometry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use geojson_reader::{parse_geojson, read_geojson};
pub use rasterize::{clip_raster, clip_raster_by_polygon, rasterize_polygons};

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
