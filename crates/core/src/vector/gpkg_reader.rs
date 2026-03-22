//! GeoPackage (.gpkg) reader
//!
//! Reads OGC GeoPackage files into [`FeatureCollection`] using `rusqlite` for
//! SQLite access and a custom WKB geometry parser.
//!
//! GeoPackage is an OGC standard based on SQLite that stores vector features
//! (and optionally raster tiles) in a portable, self-describing format.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};
use geo_types::{Coord, Geometry, LineString, Point, Polygon};

use super::{AttributeValue, Feature, FeatureCollection};

/// Read a GeoPackage file into a [`FeatureCollection`].
///
/// If `layer` is `None`, reads the first feature table found in the GeoPackage.
/// If `layer` is `Some(name)`, reads the specified table.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use surtgis_core::vector::read_gpkg;
///
/// // Read the first layer
/// let fc = read_gpkg(Path::new("data.gpkg"), None).unwrap();
///
/// // Read a specific layer
/// let fc = read_gpkg(Path::new("data.gpkg"), Some("basins")).unwrap();
/// ```
pub fn read_gpkg(path: &Path, layer: Option<&str>) -> Result<FeatureCollection> {
    let conn = rusqlite::Connection::open(path)
        .map_err(|e| Error::Other(format!("Cannot open GeoPackage '{}': {}", path.display(), e)))?;

    // Find the feature table name
    let table_name = if let Some(name) = layer {
        name.to_string()
    } else {
        conn.query_row(
            "SELECT table_name FROM gpkg_contents WHERE data_type = 'features' LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .map_err(|e| {
            Error::Other(format!(
                "No feature tables in GeoPackage '{}': {}",
                path.display(),
                e
            ))
        })?
    };

    // Find the geometry column name
    let geom_col: String = conn
        .query_row(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?1",
            [&table_name],
            |row| row.get(0),
        )
        .map_err(|e| {
            Error::Other(format!(
                "No geometry column found for table '{}': {}",
                table_name, e
            ))
        })?;

    // Get all column names (excluding geometry) via PRAGMA
    let pragma_query = format!("PRAGMA table_info(\"{}\")", table_name);
    let mut col_stmt = conn
        .prepare(&pragma_query)
        .map_err(|e| Error::Other(format!("Cannot query table info: {}", e)))?;
    let columns: Vec<String> = col_stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| Error::Other(format!("Cannot read columns: {}", e)))?
        .filter_map(|r| r.ok())
        .filter(|name| name != &geom_col)
        .collect();

    // Build SELECT with quoted column names
    let all_cols: Vec<String> = std::iter::once(format!("\"{}\"", geom_col))
        .chain(columns.iter().map(|c| format!("\"{}\"", c)))
        .collect();
    let select_query = format!("SELECT {} FROM \"{}\"", all_cols.join(", "), table_name);

    let mut stmt = conn
        .prepare(&select_query)
        .map_err(|e| Error::Other(format!("Cannot prepare query: {}", e)))?;

    let mut result_rows = stmt
        .query([])
        .map_err(|e| Error::Other(format!("Query failed: {}", e)))?;

    let mut features = FeatureCollection::new();

    while let Some(row) = result_rows
        .next()
        .map_err(|e| Error::Other(format!("Row error: {}", e)))?
    {
        // Parse geometry from GeoPackage WKB (column 0)
        let geometry = if let Ok(blob) = row.get_ref(0) {
            match blob {
                rusqlite::types::ValueRef::Blob(bytes) => parse_gpkg_wkb(bytes),
                _ => None,
            }
        } else {
            None
        };

        // Parse attribute columns (1..N)
        let mut properties = HashMap::new();
        for (i, col_name) in columns.iter().enumerate() {
            let col_idx = i + 1; // offset by geometry column
            let value = if let Ok(val_ref) = row.get_ref(col_idx) {
                match val_ref {
                    rusqlite::types::ValueRef::Null => AttributeValue::Null,
                    rusqlite::types::ValueRef::Integer(v) => AttributeValue::Int(v),
                    rusqlite::types::ValueRef::Real(v) => AttributeValue::Float(v),
                    rusqlite::types::ValueRef::Text(s) => {
                        AttributeValue::String(String::from_utf8_lossy(s).to_string())
                    }
                    rusqlite::types::ValueRef::Blob(_) => AttributeValue::Null,
                }
            } else {
                AttributeValue::Null
            };
            properties.insert(col_name.clone(), value);
        }

        let mut feature = match geometry {
            Some(geom) => Feature::new(geom),
            None => Feature::empty(),
        };
        feature.properties = properties;
        features.push(feature);
    }

    Ok(features)
}

/// List all feature table names in a GeoPackage.
///
/// Useful when the file contains multiple layers and the user needs to
/// choose which one to read.
pub fn list_gpkg_layers(path: &Path) -> Result<Vec<String>> {
    let conn = rusqlite::Connection::open(path)
        .map_err(|e| Error::Other(format!("Cannot open GeoPackage: {}", e)))?;

    let mut stmt = conn
        .prepare("SELECT table_name FROM gpkg_contents WHERE data_type = 'features'")
        .map_err(|e| Error::Other(format!("Cannot query gpkg_contents: {}", e)))?;

    let names: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .map_err(|e| Error::Other(format!("Query failed: {}", e)))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(names)
}

// ---------------------------------------------------------------------------
// GeoPackage WKB parser
// ---------------------------------------------------------------------------

/// Parse GeoPackage WKB geometry (standard WKB with GeoPackage binary header).
///
/// GeoPackage binary header format (GeoPackageBinary):
/// - 2 bytes: magic number `0x47 0x50` ("GP")
/// - 1 byte: version
/// - 1 byte: flags (bits 1-3 = envelope type, bit 0 = byte order)
/// - 4 bytes: SRS ID (int32)
/// - Variable: envelope (0/32/48/48/64 bytes depending on envelope type)
/// - Remainder: standard OGC WKB geometry
fn parse_gpkg_wkb(bytes: &[u8]) -> Option<Geometry<f64>> {
    if bytes.len() < 8 {
        return None;
    }

    // Check GP magic number
    if bytes[0] != 0x47 || bytes[1] != 0x50 {
        // Not a GeoPackage header — try parsing as plain WKB
        return parse_wkb(bytes);
    }

    let flags = bytes[3];
    let envelope_type = (flags >> 1) & 0x07;

    // Calculate envelope size in bytes
    let envelope_size = match envelope_type {
        0 => 0,  // no envelope
        1 => 32, // minx, maxx, miny, maxy (4 doubles)
        2 => 48, // + minz, maxz (6 doubles)
        3 => 48, // + minm, maxm (6 doubles)
        4 => 64, // + minz, maxz, minm, maxm (8 doubles)
        _ => return None,
    };

    let wkb_start = 8 + envelope_size;
    if wkb_start > bytes.len() {
        return None;
    }

    parse_wkb(&bytes[wkb_start..])
}

/// Parse standard OGC WKB geometry.
///
/// Supports: Point (1), LineString (2), Polygon (3), MultiPoint (4),
/// MultiLineString (5), MultiPolygon (6). Type codes 1000+/2000+/3000+
/// are handled by masking to the base 2D type.
fn parse_wkb(bytes: &[u8]) -> Option<Geometry<f64>> {
    if bytes.len() < 5 {
        return None;
    }

    let is_le = bytes[0] == 1;

    let geom_type = read_u32(bytes, &mut 1, is_le)?;

    // Mask out Z (1000), M (2000), ZM (3000) flags
    let base_type = geom_type % 1000;
    // Determine coordinate dimensionality
    let has_z = matches!(geom_type / 1000, 1 | 3);
    let has_m = matches!(geom_type / 1000, 2 | 3);
    let coord_size = 2 + if has_z { 1 } else { 0 } + if has_m { 1 } else { 0 };

    let mut cursor = 5;

    match base_type {
        1 => parse_wkb_point(bytes, &mut cursor, is_le, coord_size),
        2 => parse_wkb_linestring(bytes, &mut cursor, is_le, coord_size),
        3 => parse_wkb_polygon(bytes, &mut cursor, is_le, coord_size),
        4 => parse_wkb_multi(bytes, &mut cursor, is_le, 1), // MultiPoint
        5 => parse_wkb_multi(bytes, &mut cursor, is_le, 2), // MultiLineString
        6 => parse_wkb_multi(bytes, &mut cursor, is_le, 3), // MultiPolygon
        _ => None,
    }
}

fn read_f64(bytes: &[u8], cursor: &mut usize, is_le: bool) -> Option<f64> {
    if *cursor + 8 > bytes.len() {
        return None;
    }
    let val = if is_le {
        f64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().ok()?)
    } else {
        f64::from_be_bytes(bytes[*cursor..*cursor + 8].try_into().ok()?)
    };
    *cursor += 8;
    Some(val)
}

fn read_u32(bytes: &[u8], cursor: &mut usize, is_le: bool) -> Option<u32> {
    if *cursor + 4 > bytes.len() {
        return None;
    }
    let val = if is_le {
        u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().ok()?)
    } else {
        u32::from_be_bytes(bytes[*cursor..*cursor + 4].try_into().ok()?)
    };
    *cursor += 4;
    Some(val)
}

/// Read a 2D coordinate (x, y), skipping any Z/M ordinates.
fn read_coord(bytes: &[u8], cursor: &mut usize, is_le: bool, coord_size: usize) -> Option<Coord> {
    let x = read_f64(bytes, cursor, is_le)?;
    let y = read_f64(bytes, cursor, is_le)?;
    // Skip Z and/or M ordinates
    let extra = coord_size.saturating_sub(2);
    for _ in 0..extra {
        read_f64(bytes, cursor, is_le)?;
    }
    Some(Coord { x, y })
}

fn parse_wkb_point(
    bytes: &[u8],
    cursor: &mut usize,
    is_le: bool,
    coord_size: usize,
) -> Option<Geometry<f64>> {
    let c = read_coord(bytes, cursor, is_le, coord_size)?;
    Some(Geometry::Point(Point::new(c.x, c.y)))
}

fn parse_wkb_linestring(
    bytes: &[u8],
    cursor: &mut usize,
    is_le: bool,
    coord_size: usize,
) -> Option<Geometry<f64>> {
    let num_points = read_u32(bytes, cursor, is_le)? as usize;
    let mut coords = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        coords.push(read_coord(bytes, cursor, is_le, coord_size)?);
    }
    Some(Geometry::LineString(LineString::new(coords)))
}

fn parse_wkb_polygon(
    bytes: &[u8],
    cursor: &mut usize,
    is_le: bool,
    coord_size: usize,
) -> Option<Geometry<f64>> {
    let num_rings = read_u32(bytes, cursor, is_le)? as usize;
    let mut rings = Vec::with_capacity(num_rings);
    for _ in 0..num_rings {
        let num_points = read_u32(bytes, cursor, is_le)? as usize;
        let mut coords = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            coords.push(read_coord(bytes, cursor, is_le, coord_size)?);
        }
        rings.push(LineString::new(coords));
    }
    if rings.is_empty() {
        return None;
    }
    let exterior = rings.remove(0);
    Some(Geometry::Polygon(Polygon::new(exterior, rings)))
}

/// Parse a Multi* geometry (MultiPoint, MultiLineString, MultiPolygon).
/// Each sub-geometry has its own WKB header (byte order + type + data).
fn parse_wkb_multi(
    bytes: &[u8],
    cursor: &mut usize,
    _is_le: bool,
    expected_base_type: u32,
) -> Option<Geometry<f64>> {
    let num_geoms = read_u32(bytes, cursor, _is_le)? as usize;
    let mut sub_geoms = Vec::with_capacity(num_geoms);

    for _ in 0..num_geoms {
        if *cursor >= bytes.len() {
            break;
        }
        // Each sub-geometry has its own byte order
        let sub_le = bytes[*cursor] == 1;
        *cursor += 1;

        let sub_type = read_u32(bytes, cursor, sub_le)?;
        let sub_base = sub_type % 1000;
        let has_z = matches!(sub_type / 1000, 1 | 3);
        let has_m = matches!(sub_type / 1000, 2 | 3);
        let coord_size = 2 + if has_z { 1 } else { 0 } + if has_m { 1 } else { 0 };

        if sub_base != expected_base_type {
            return None; // Unexpected sub-geometry type
        }

        let geom = match sub_base {
            1 => parse_wkb_point(bytes, cursor, sub_le, coord_size)?,
            2 => parse_wkb_linestring(bytes, cursor, sub_le, coord_size)?,
            3 => parse_wkb_polygon(bytes, cursor, sub_le, coord_size)?,
            _ => return None,
        };
        sub_geoms.push(geom);
    }

    match expected_base_type {
        1 => {
            let points: Vec<geo_types::Point<f64>> = sub_geoms
                .into_iter()
                .filter_map(|g| match g {
                    Geometry::Point(p) => Some(p),
                    _ => None,
                })
                .collect();
            Some(Geometry::MultiPoint(geo_types::MultiPoint::new(points)))
        }
        2 => {
            let lines: Vec<LineString<f64>> = sub_geoms
                .into_iter()
                .filter_map(|g| match g {
                    Geometry::LineString(l) => Some(l),
                    _ => None,
                })
                .collect();
            Some(Geometry::MultiLineString(
                geo_types::MultiLineString::new(lines),
            ))
        }
        3 => {
            let polygons: Vec<Polygon<f64>> = sub_geoms
                .into_iter()
                .filter_map(|g| match g {
                    Geometry::Polygon(p) => Some(p),
                    _ => None,
                })
                .collect();
            Some(Geometry::MultiPolygon(geo_types::MultiPolygon::new(
                polygons,
            )))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_nonexistent_gpkg() {
        let result = read_gpkg(Path::new("/tmp/nonexistent_surtgis_test.gpkg"), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_wkb_point() {
        // Little-endian WKB point at (1.5, 2.5)
        let mut wkb = Vec::new();
        wkb.push(1u8); // LE
        wkb.extend_from_slice(&1u32.to_le_bytes()); // Point type
        wkb.extend_from_slice(&1.5f64.to_le_bytes());
        wkb.extend_from_slice(&2.5f64.to_le_bytes());

        let geom = parse_wkb(&wkb).unwrap();
        match geom {
            Geometry::Point(p) => {
                assert!((p.x() - 1.5).abs() < 1e-10);
                assert!((p.y() - 2.5).abs() < 1e-10);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_wkb_polygon() {
        // Little-endian WKB polygon: 1 ring, 5 points (square)
        let mut wkb = Vec::new();
        wkb.push(1u8); // LE
        wkb.extend_from_slice(&3u32.to_le_bytes()); // Polygon type
        wkb.extend_from_slice(&1u32.to_le_bytes()); // 1 ring
        wkb.extend_from_slice(&5u32.to_le_bytes()); // 5 points
        for &(x, y) in &[(0.0f64, 0.0f64), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)] {
            wkb.extend_from_slice(&x.to_le_bytes());
            wkb.extend_from_slice(&y.to_le_bytes());
        }

        let geom = parse_wkb(&wkb).unwrap();
        match geom {
            Geometry::Polygon(p) => {
                assert_eq!(p.exterior().0.len(), 5);
                assert_eq!(p.interiors().len(), 0);
            }
            other => panic!("Expected Polygon, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_gpkg_wkb_with_header() {
        // Build a GeoPackage binary header + WKB point
        let mut gpkg_wkb = Vec::new();

        // GP header: magic, version, flags, srs_id
        gpkg_wkb.extend_from_slice(&[0x47, 0x50]); // magic "GP"
        gpkg_wkb.push(0); // version
        gpkg_wkb.push(0b00000001); // flags: LE byte order, no envelope
        gpkg_wkb.extend_from_slice(&4326u32.to_le_bytes()); // SRS ID

        // WKB Point
        gpkg_wkb.push(1u8); // LE
        gpkg_wkb.extend_from_slice(&1u32.to_le_bytes()); // Point
        gpkg_wkb.extend_from_slice(&(-70.5f64).to_le_bytes());
        gpkg_wkb.extend_from_slice(&(-33.4f64).to_le_bytes());

        let geom = parse_gpkg_wkb(&gpkg_wkb).unwrap();
        match geom {
            Geometry::Point(p) => {
                assert!((p.x() - (-70.5)).abs() < 1e-10);
                assert!((p.y() - (-33.4)).abs() < 1e-10);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_gpkg_wkb_with_envelope() {
        // GP header with xy envelope (type 1, 32 bytes)
        let mut gpkg_wkb = Vec::new();
        gpkg_wkb.extend_from_slice(&[0x47, 0x50]); // magic
        gpkg_wkb.push(0); // version
        gpkg_wkb.push(0b00000011); // flags: LE, envelope type 1 (xy)
        gpkg_wkb.extend_from_slice(&4326u32.to_le_bytes()); // SRS ID

        // Envelope: minx, maxx, miny, maxy (4 doubles = 32 bytes)
        for &v in &[0.0f64, 10.0, 0.0, 10.0] {
            gpkg_wkb.extend_from_slice(&v.to_le_bytes());
        }

        // WKB Point
        gpkg_wkb.push(1u8);
        gpkg_wkb.extend_from_slice(&1u32.to_le_bytes());
        gpkg_wkb.extend_from_slice(&5.0f64.to_le_bytes());
        gpkg_wkb.extend_from_slice(&5.0f64.to_le_bytes());

        let geom = parse_gpkg_wkb(&gpkg_wkb).unwrap();
        match geom {
            Geometry::Point(p) => {
                assert!((p.x() - 5.0).abs() < 1e-10);
                assert!((p.y() - 5.0).abs() < 1e-10);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    #[test]
    fn test_gpkg_roundtrip() {
        let dir = std::env::temp_dir().join("surtgis_gpkg_test");
        std::fs::create_dir_all(&dir).unwrap();
        let gpkg_path = dir.join("test_roundtrip.gpkg");

        // Remove if exists from a previous test run
        let _ = std::fs::remove_file(&gpkg_path);

        // Create a minimal GeoPackage with rusqlite
        let conn = rusqlite::Connection::open(&gpkg_path).unwrap();

        conn.execute_batch(
            "
            CREATE TABLE gpkg_contents (
                table_name TEXT NOT NULL PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT,
                description TEXT DEFAULT '',
                last_change TEXT,
                min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
                srs_id INTEGER
            );
            CREATE TABLE gpkg_geometry_columns (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL
            );
            INSERT INTO gpkg_contents VALUES (
                'test_layer', 'features', 'test', '', '', 0, 0, 10, 10, 4326
            );
            INSERT INTO gpkg_geometry_columns VALUES (
                'test_layer', 'geom', 'POLYGON', 4326, 0, 0
            );
            CREATE TABLE test_layer (
                fid INTEGER PRIMARY KEY,
                geom BLOB,
                name TEXT,
                value REAL
            );
        ",
        )
        .unwrap();

        // Insert a polygon as plain WKB (little-endian)
        let mut wkb = Vec::new();
        wkb.push(1u8); // LE
        wkb.extend_from_slice(&3u32.to_le_bytes()); // Polygon
        wkb.extend_from_slice(&1u32.to_le_bytes()); // 1 ring
        wkb.extend_from_slice(&5u32.to_le_bytes()); // 5 points
        for &(x, y) in &[(0.0f64, 0.0f64), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)] {
            wkb.extend_from_slice(&x.to_le_bytes());
            wkb.extend_from_slice(&y.to_le_bytes());
        }

        conn.execute(
            "INSERT INTO test_layer (fid, geom, name, value) VALUES (1, ?1, 'Basin1', 42.5)",
            [&wkb],
        )
        .unwrap();
        drop(conn);

        // Read back
        let fc = read_gpkg(&gpkg_path, None).unwrap();
        assert_eq!(fc.len(), 1);

        let feature = &fc.features[0];
        assert!(feature.geometry.is_some());
        match feature.get_property("name") {
            Some(AttributeValue::String(s)) => assert_eq!(s, "Basin1"),
            other => panic!("Expected String 'Basin1', got {:?}", other),
        }
        match feature.get_property("value") {
            Some(AttributeValue::Float(v)) => assert!((*v - 42.5).abs() < 1e-10),
            other => panic!("Expected Float 42.5, got {:?}", other),
        }

        // Test list_gpkg_layers
        let layers = list_gpkg_layers(&gpkg_path).unwrap();
        assert_eq!(layers, vec!["test_layer"]);

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_gpkg_specific_layer() {
        let dir = std::env::temp_dir().join("surtgis_gpkg_layer_test");
        std::fs::create_dir_all(&dir).unwrap();
        let gpkg_path = dir.join("multi_layer.gpkg");
        let _ = std::fs::remove_file(&gpkg_path);

        let conn = rusqlite::Connection::open(&gpkg_path).unwrap();
        conn.execute_batch(
            "
            CREATE TABLE gpkg_contents (
                table_name TEXT NOT NULL PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT,
                description TEXT DEFAULT '',
                last_change TEXT,
                min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
                srs_id INTEGER
            );
            CREATE TABLE gpkg_geometry_columns (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL
            );
            INSERT INTO gpkg_contents VALUES ('layer_a', 'features', 'A', '', '', 0, 0, 10, 10, 4326);
            INSERT INTO gpkg_contents VALUES ('layer_b', 'features', 'B', '', '', 0, 0, 10, 10, 4326);
            INSERT INTO gpkg_geometry_columns VALUES ('layer_a', 'geom', 'POINT', 4326, 0, 0);
            INSERT INTO gpkg_geometry_columns VALUES ('layer_b', 'geom', 'POINT', 4326, 0, 0);

            CREATE TABLE layer_a (fid INTEGER PRIMARY KEY, geom BLOB, label TEXT);
            CREATE TABLE layer_b (fid INTEGER PRIMARY KEY, geom BLOB, label TEXT);

            INSERT INTO layer_a (fid, label) VALUES (1, 'from_a');
            INSERT INTO layer_b (fid, label) VALUES (1, 'from_b');
            INSERT INTO layer_b (fid, label) VALUES (2, 'from_b_2');
        ",
        )
        .unwrap();
        drop(conn);

        // Read specific layer
        let fc_b = read_gpkg(&gpkg_path, Some("layer_b")).unwrap();
        assert_eq!(fc_b.len(), 2);
        match fc_b.features[0].get_property("label") {
            Some(AttributeValue::String(s)) => assert_eq!(s, "from_b"),
            other => panic!("Expected 'from_b', got {:?}", other),
        }

        // Read default (first) layer
        let fc_default = read_gpkg(&gpkg_path, None).unwrap();
        assert_eq!(fc_default.len(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_parse_wkb_z_point() {
        // PointZ (type 1001) in LE
        let mut wkb = Vec::new();
        wkb.push(1u8);
        wkb.extend_from_slice(&1001u32.to_le_bytes()); // PointZ
        wkb.extend_from_slice(&1.0f64.to_le_bytes()); // x
        wkb.extend_from_slice(&2.0f64.to_le_bytes()); // y
        wkb.extend_from_slice(&3.0f64.to_le_bytes()); // z (skipped in 2D)

        let geom = parse_wkb(&wkb).unwrap();
        match geom {
            Geometry::Point(p) => {
                assert!((p.x() - 1.0).abs() < 1e-10);
                assert!((p.y() - 2.0).abs() < 1e-10);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }
}
