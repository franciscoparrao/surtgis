//! GeoJSON reader: parse GeoJSON files/strings into FeatureCollection.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};
use geo_types::{Coord, Geometry, LineString, MultiPolygon, Point, Polygon};

use super::{AttributeValue, Feature, FeatureCollection};

/// Read a GeoJSON file into a FeatureCollection.
pub fn read_geojson(path: &Path) -> Result<FeatureCollection> {
    let content = std::fs::read_to_string(path)?;
    parse_geojson(&content)
}

/// Parse a GeoJSON string into a FeatureCollection.
pub fn parse_geojson(json: &str) -> Result<FeatureCollection> {
    let root: Value =
        serde_json::from_str(json).map_err(|e| Error::Other(format!("Invalid JSON: {e}")))?;

    let root_type = root
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    match root_type {
        "FeatureCollection" => parse_feature_collection(&root),
        "Feature" => {
            let feature = parse_feature(&root)?;
            let mut fc = FeatureCollection::new();
            fc.push(feature);
            Ok(fc)
        }
        // Bare geometry (wrap in a feature)
        "Point" | "Polygon" | "MultiPolygon" | "LineString" | "MultiLineString"
        | "MultiPoint" | "GeometryCollection" => {
            let geom = parse_geometry(&root)?;
            let mut fc = FeatureCollection::new();
            fc.push(Feature::new(geom));
            Ok(fc)
        }
        _ => Err(Error::Other(format!(
            "Unsupported GeoJSON type: '{root_type}'"
        ))),
    }
}

fn parse_feature_collection(value: &Value) -> Result<FeatureCollection> {
    let features_array = value
        .get("features")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::Other("Missing 'features' array in FeatureCollection".into()))?;

    let mut fc = FeatureCollection::new();
    for feat_val in features_array {
        fc.push(parse_feature(feat_val)?);
    }
    Ok(fc)
}

fn parse_feature(value: &Value) -> Result<Feature> {
    let geometry = match value.get("geometry") {
        Some(Value::Null) | None => None,
        Some(geom_val) => Some(parse_geometry(geom_val)?),
    };

    let properties = match value.get("properties") {
        Some(props_val) if props_val.is_object() => parse_properties(props_val),
        _ => HashMap::new(),
    };

    let id = value.get("id").and_then(|v| match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    });

    Ok(Feature {
        geometry,
        properties,
        id,
    })
}

fn parse_geometry(value: &Value) -> Result<Geometry<f64>> {
    let geom_type = value
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::Other("Geometry missing 'type' field".into()))?;

    match geom_type {
        "Point" => {
            let coords = value
                .get("coordinates")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Other("Point missing 'coordinates'".into()))?;
            let (x, y) = parse_coord_pair(coords)?;
            Ok(Geometry::Point(Point::new(x, y)))
        }
        "Polygon" => {
            let rings = value
                .get("coordinates")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Other("Polygon missing 'coordinates'".into()))?;
            let polygon = parse_polygon_rings(rings)?;
            Ok(Geometry::Polygon(polygon))
        }
        "MultiPolygon" => {
            let polys_coords = value
                .get("coordinates")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Other("MultiPolygon missing 'coordinates'".into()))?;
            let mut polygons = Vec::with_capacity(polys_coords.len());
            for poly_rings in polys_coords {
                let rings = poly_rings
                    .as_array()
                    .ok_or_else(|| Error::Other("Invalid MultiPolygon ring array".into()))?;
                polygons.push(parse_polygon_rings(rings)?);
            }
            Ok(Geometry::MultiPolygon(MultiPolygon::new(polygons)))
        }
        _ => Err(Error::Other(format!(
            "Unsupported geometry type: '{geom_type}'"
        ))),
    }
}

fn parse_polygon_rings(rings: &[Value]) -> Result<Polygon<f64>> {
    if rings.is_empty() {
        return Err(Error::Other("Polygon has no rings".into()));
    }

    let exterior = parse_coord_ring(
        rings[0]
            .as_array()
            .ok_or_else(|| Error::Other("Invalid exterior ring".into()))?,
    )?;

    let mut interiors = Vec::new();
    for ring_val in &rings[1..] {
        let ring = ring_val
            .as_array()
            .ok_or_else(|| Error::Other("Invalid interior ring".into()))?;
        interiors.push(parse_coord_ring(ring)?);
    }

    Ok(Polygon::new(exterior, interiors))
}

fn parse_coord_ring(coords: &[Value]) -> Result<LineString<f64>> {
    let mut points = Vec::with_capacity(coords.len());
    for coord_val in coords {
        let arr = coord_val
            .as_array()
            .ok_or_else(|| Error::Other("Coordinate is not an array".into()))?;
        let (x, y) = parse_coord_pair(arr)?;
        points.push(Coord { x, y });
    }
    Ok(LineString::new(points))
}

fn parse_coord_pair(arr: &[Value]) -> Result<(f64, f64)> {
    if arr.len() < 2 {
        return Err(Error::Other(
            "Coordinate array must have at least 2 elements".into(),
        ));
    }
    let x = arr[0]
        .as_f64()
        .ok_or_else(|| Error::Other("Coordinate X is not a number".into()))?;
    let y = arr[1]
        .as_f64()
        .ok_or_else(|| Error::Other("Coordinate Y is not a number".into()))?;
    Ok((x, y))
}

fn parse_properties(value: &Value) -> HashMap<String, AttributeValue> {
    let mut map = HashMap::new();
    if let Some(obj) = value.as_object() {
        for (key, val) in obj {
            let attr = match val {
                Value::Null => AttributeValue::Null,
                Value::Bool(b) => AttributeValue::Bool(*b),
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        AttributeValue::Int(i)
                    } else if let Some(f) = n.as_f64() {
                        AttributeValue::Float(f)
                    } else {
                        AttributeValue::String(n.to_string())
                    }
                }
                Value::String(s) => AttributeValue::String(s.clone()),
                // Nested objects/arrays stored as JSON string
                other => AttributeValue::String(other.to_string()),
            };
            map.insert(key.clone(), attr);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_GEOJSON: &str = r#"{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "basin1",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
                },
                "properties": {
                    "name": "Basin 1",
                    "area_km2": 42.5,
                    "id": 1
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]]
                },
                "properties": {
                    "name": "Basin 2",
                    "area_km2": 18.3,
                    "id": 2
                }
            }
        ]
    }"#;

    #[test]
    fn test_parse_feature_collection() {
        let fc = parse_geojson(SAMPLE_GEOJSON).unwrap();
        assert_eq!(fc.len(), 2);

        let f0 = &fc.features[0];
        assert_eq!(f0.id.as_deref(), Some("basin1"));
        assert!(f0.geometry.is_some());

        match f0.get_property("name") {
            Some(AttributeValue::String(s)) => assert_eq!(s, "Basin 1"),
            other => panic!("Expected String, got {:?}", other),
        }
        match f0.get_property("area_km2") {
            Some(AttributeValue::Float(v)) => assert!((v - 42.5).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
        match f0.get_property("id") {
            Some(AttributeValue::Int(v)) => assert_eq!(*v, 1),
            other => panic!("Expected Int, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_single_feature() {
        let json = r#"{
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [100.0, 0.5]
            },
            "properties": {"name": "test"}
        }"#;
        let fc = parse_geojson(json).unwrap();
        assert_eq!(fc.len(), 1);
        match &fc.features[0].geometry {
            Some(Geometry::Point(p)) => {
                assert!((p.x() - 100.0).abs() < 1e-10);
                assert!((p.y() - 0.5).abs() < 1e-10);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_multipolygon() {
        let json = r#"{
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[[0,0],[1,0],[1,1],[0,1],[0,0]]],
                    [[[2,2],[3,2],[3,3],[2,3],[2,2]]]
                ]
            },
            "properties": {}
        }"#;
        let fc = parse_geojson(json).unwrap();
        assert_eq!(fc.len(), 1);
        match &fc.features[0].geometry {
            Some(Geometry::MultiPolygon(mp)) => assert_eq!(mp.0.len(), 2),
            other => panic!("Expected MultiPolygon, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_polygon_with_hole() {
        let json = r#"{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0,0],[10,0],[10,10],[0,10],[0,0]],
                    [[2,2],[8,2],[8,8],[2,8],[2,2]]
                ]
            },
            "properties": {}
        }"#;
        let fc = parse_geojson(json).unwrap();
        match &fc.features[0].geometry {
            Some(Geometry::Polygon(p)) => {
                assert_eq!(p.exterior().0.len(), 5);
                assert_eq!(p.interiors().len(), 1);
                assert_eq!(p.interiors()[0].0.len(), 5);
            }
            other => panic!("Expected Polygon, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_json() {
        assert!(parse_geojson("not json").is_err());
    }

    #[test]
    fn test_bare_geometry() {
        let json = r#"{
            "type": "Polygon",
            "coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]
        }"#;
        let fc = parse_geojson(json).unwrap();
        assert_eq!(fc.len(), 1);
        assert!(fc.features[0].geometry.is_some());
    }
}
