//! Shapefile (.shp) reader
//!
//! Reads ESRI Shapefiles into [`FeatureCollection`] using the `shapefile` crate.
//! Automatically reads the associated `.dbf` (attributes) and `.shx` (index) files.

use std::collections::HashMap;
use std::path::Path;

use crate::crs::CRS;
use crate::error::{Error, Result};
use geo_types::Geometry;

use super::{AttributeValue, Feature, FeatureCollection};

/// Read a Shapefile (.shp) into a [`FeatureCollection`].
///
/// Automatically reads the associated `.dbf` (attributes) and `.shx` (index) files
/// that must be present alongside the `.shp` file.
///
/// # Supported geometry types
///
/// All standard shapefile geometry types are supported via conversion to `geo_types`:
/// - Point / PointM / PointZ
/// - Polyline / PolylineM / PolylineZ
/// - Polygon / PolygonM / PolygonZ
/// - Multipoint / MultipointM / MultipointZ
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use surtgis_core::vector::read_shapefile;
///
/// let fc = read_shapefile(Path::new("basins.shp")).unwrap();
/// println!("Read {} features", fc.len());
/// ```
pub fn read_shapefile(path: &Path) -> Result<FeatureCollection> {
    let mut reader = shapefile::Reader::from_path(path)
        .map_err(|e| Error::Other(format!("Cannot open shapefile '{}': {}", path.display(), e)))?;

    let mut features = FeatureCollection::with_crs(read_prj_sidecar(path));

    for result in reader.iter_shapes_and_records() {
        let (shape, record) =
            result.map_err(|e| Error::Other(format!("Error reading shapefile record: {}", e)))?;

        // Convert shapefile::Shape -> geo_types::Geometry via TryFrom (geo-types feature)
        let geometry: Option<Geometry<f64>> = match shape {
            shapefile::Shape::NullShape => None,
            other => other.try_into().ok(),
        };

        // Convert dbase::Record -> HashMap<String, AttributeValue>
        let properties = convert_record(record);

        let mut feature = match geometry {
            Some(geom) => Feature::new(geom),
            None => Feature::empty(),
        };
        feature.properties = properties;
        features.push(feature);
    }

    Ok(features)
}

/// Read the `.prj` sidecar file next to a shapefile, if present.
///
/// The `.prj` file holds the CRS as ESRI WKT. There is no standard,
/// dependency-free way to resolve arbitrary ESRI WKT to an EPSG code, so it
/// is kept as opaque WKT via [`CRS::from_wkt`] — still enough for
/// [`super::rasterize::rasterize_polygons`] to detect an outright CRS
/// mismatch when the reference raster's CRS is *also* WKT-only and the
/// strings disagree, though the common useful case is EPSG vs EPSG.
/// Returns `None` (unknown), not a guess, when the sidecar is missing,
/// unreadable, or empty.
fn read_prj_sidecar(shp_path: &Path) -> Option<CRS> {
    let prj_path = shp_path.with_extension("prj");
    let content = std::fs::read_to_string(prj_path).ok()?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(CRS::from_wkt(trimmed.to_string()))
    }
}

/// Convert a dbase `Record` into our `AttributeValue` map.
fn convert_record(record: shapefile::dbase::Record) -> HashMap<String, AttributeValue> {
    let mut properties = HashMap::new();

    for (name, value) in record.into_iter() {
        let attr = match value {
            shapefile::dbase::FieldValue::Character(Some(s)) => AttributeValue::String(s),
            shapefile::dbase::FieldValue::Numeric(Some(f)) => AttributeValue::Float(f),
            shapefile::dbase::FieldValue::Float(Some(f)) => AttributeValue::Float(f as f64),
            shapefile::dbase::FieldValue::Integer(i) => AttributeValue::Int(i as i64),
            shapefile::dbase::FieldValue::Double(f) => AttributeValue::Float(f),
            shapefile::dbase::FieldValue::Currency(f) => AttributeValue::Float(f),
            shapefile::dbase::FieldValue::Logical(Some(b)) => AttributeValue::Bool(b),
            shapefile::dbase::FieldValue::Memo(s) => AttributeValue::String(s),
            shapefile::dbase::FieldValue::Date(Some(d)) => {
                AttributeValue::String(format!("{:04}-{:02}-{:02}", d.year(), d.month(), d.day()))
            }
            shapefile::dbase::FieldValue::DateTime(dt) => {
                AttributeValue::String(format!("{:?}", dt))
            }
            // All None/null variants
            _ => AttributeValue::Null,
        };
        properties.insert(name, attr);
    }

    properties
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_nonexistent_shapefile() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nonexistent.shp");
        let result = read_shapefile(&missing);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_shapefile_roundtrip() {
        use shapefile::dbase::{FieldName, FieldValue, TableWriterBuilder};
        use shapefile::{Point, PolygonRing, Writer};

        let dir = std::env::temp_dir().join("surtgis_shp_test");
        std::fs::create_dir_all(&dir).unwrap();
        let shp_path = dir.join("test_roundtrip.shp");

        // Create a simple shapefile with one polygon
        let polygon = shapefile::Polygon::with_rings(vec![PolygonRing::Outer(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
            Point::new(0.0, 0.0),
        ])]);

        let table_builder =
            TableWriterBuilder::new().add_character_field(FieldName::try_from("name").unwrap(), 50);

        let mut writer = Writer::from_path(&shp_path, table_builder).unwrap();

        let mut record = shapefile::dbase::Record::default();
        record.insert(
            "name".to_string(),
            FieldValue::Character(Some("Basin1".to_string())),
        );
        writer.write_shape_and_record(&polygon, &record).unwrap();
        drop(writer);

        // Read back with our reader
        let fc = read_shapefile(&shp_path).unwrap();
        assert_eq!(fc.len(), 1);

        let feature = &fc.features[0];
        assert!(feature.geometry.is_some());
        match feature.get_property("name") {
            Some(AttributeValue::String(s)) => assert_eq!(s, "Basin1"),
            other => panic!("Expected String 'Basin1', got {:?}", other),
        }

        // No .prj sidecar written above: CRS must be honestly unknown.
        assert!(fc.crs().is_none());

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_shapefile_populates_crs_from_prj_sidecar() {
        use shapefile::dbase::TableWriterBuilder;
        use shapefile::{Point, PolygonRing, Writer};

        let dir = std::env::temp_dir().join("surtgis_shp_prj_test");
        std::fs::create_dir_all(&dir).unwrap();
        let shp_path = dir.join("test_prj.shp");

        let polygon = shapefile::Polygon::with_rings(vec![PolygonRing::Outer(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0),
        ])]);
        let table_builder = TableWriterBuilder::new();
        let mut writer = Writer::from_path(&shp_path, table_builder).unwrap();
        writer
            .write_shape_and_record(&polygon, &shapefile::dbase::Record::default())
            .unwrap();
        drop(writer);

        // Write a minimal ESRI WKT .prj sidecar next to the .shp
        let prj_path = shp_path.with_extension("prj");
        std::fs::write(
            &prj_path,
            r#"PROJCS["WGS_1984_UTM_Zone_19S",GEOGCS["GCS_WGS_1984"]]"#,
        )
        .unwrap();

        let fc = read_shapefile(&shp_path).unwrap();
        let crs = fc.crs().expect("CRS should be populated from .prj sidecar");
        assert!(crs.wkt().unwrap().contains("UTM_Zone_19S"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_shapefile_point() {
        use shapefile::dbase::{FieldName, FieldValue, TableWriterBuilder};
        use shapefile::{Point, Writer};

        let dir = std::env::temp_dir().join("surtgis_shp_point_test");
        std::fs::create_dir_all(&dir).unwrap();
        let shp_path = dir.join("test_points.shp");

        let table_builder =
            TableWriterBuilder::new().add_character_field(FieldName::try_from("id").unwrap(), 10);

        let mut writer = Writer::from_path(&shp_path, table_builder).unwrap();

        let mut record = shapefile::dbase::Record::default();
        record.insert(
            "id".to_string(),
            FieldValue::Character(Some("P1".to_string())),
        );
        writer
            .write_shape_and_record(&Point::new(100.0, -33.5), &record)
            .unwrap();
        drop(writer);

        let fc = read_shapefile(&shp_path).unwrap();
        assert_eq!(fc.len(), 1);
        match &fc.features[0].geometry {
            Some(Geometry::Point(p)) => {
                assert!((p.x() - 100.0).abs() < 1e-10);
                assert!((p.y() - (-33.5)).abs() < 1e-10);
            }
            other => panic!("Expected Point, got {:?}", other),
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
