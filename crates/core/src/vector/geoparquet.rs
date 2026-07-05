//! GeoParquet point I/O (feature `parquet`).
//!
//! Pure-Rust reader/writer for **point tables with attributes** —
//! embedding matrices, training samples, extracted features — in
//! [GeoParquet 1.0](https://geoparquet.org/releases/v1.0.0/) layout:
//! WKB geometry column + `geo` file metadata, Snappy compression.
//! The output is directly queryable from DuckDB, GeoPandas and GDAL.
//!
//! The primary type is [`PointTable`], a columnar container that
//! avoids per-feature HashMaps for wide tables (e.g. one f32 column
//! per embedding dimension). [`FeatureCollection`] bridges exist for
//! interoperability with the rest of the vector module.
//!
//! Scope (v1):
//! - Point geometry only (2-D, WKB).
//! - Column types: f64, f32, i64, UTF-8 string. No nulls.
//! - CRS: GeoParquet `crs` is written as `null` (unknown) and the
//!   EPSG code is preserved in a `surtgis:epsg` metadata key, which
//!   the reader recovers. Readers other than SurtGIS will see
//!   correct geometry but must be told the CRS out-of-band.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use parquet::basic::{Compression, ConvertedType, LogicalType, Repetition, Type as PhysicalType};
use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, FloatType, Int64Type};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::record::Field;
use parquet::schema::types::Type as SchemaType;

use super::{AttributeValue, Feature, FeatureCollection};
use crate::crs::CRS;
use crate::error::{Error, Result};

/// A typed attribute column of a [`PointTable`].
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    /// 64-bit float values.
    F64(Vec<f64>),
    /// 32-bit float values.
    F32(Vec<f32>),
    /// 64-bit signed integer values.
    I64(Vec<i64>),
    /// UTF-8 string values.
    Str(Vec<String>),
}

impl ColumnData {
    fn len(&self) -> usize {
        match self {
            ColumnData::F64(v) => v.len(),
            ColumnData::F32(v) => v.len(),
            ColumnData::I64(v) => v.len(),
            ColumnData::Str(v) => v.len(),
        }
    }
}

/// A named attribute column.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    /// Column name.
    pub name: String,
    /// Column values.
    pub data: ColumnData,
}

/// Columnar table of 2-D points with attributes.
///
/// `x`, `y` and every column must have the same length. This is the
/// exchange type for embedding/feature tables: one row per point,
/// one column per attribute or embedding dimension.
#[derive(Debug, Clone, Default)]
pub struct PointTable {
    /// X coordinates, one per point.
    pub x: Vec<f64>,
    /// Y coordinates, one per point.
    pub y: Vec<f64>,
    /// EPSG code of the point coordinates, if known.
    pub epsg: Option<u32>,
    /// Attribute columns; each has the same length as `x` / `y`.
    pub columns: Vec<Column>,
}

impl PointTable {
    /// Number of points (rows).
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether the table has no points.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    fn validate(&self) -> Result<()> {
        if self.y.len() != self.x.len() {
            return Err(Error::Other(format!(
                "geoparquet: x has {} rows but y has {}",
                self.x.len(),
                self.y.len()
            )));
        }
        for col in &self.columns {
            if col.data.len() != self.x.len() {
                return Err(Error::Other(format!(
                    "geoparquet: column '{}' has {} rows but the table has {}",
                    col.name,
                    col.data.len(),
                    self.x.len()
                )));
            }
            if col.name == GEOMETRY_COLUMN {
                return Err(Error::Other(format!(
                    "geoparquet: column name '{}' is reserved",
                    GEOMETRY_COLUMN
                )));
            }
        }
        Ok(())
    }
}

const GEOMETRY_COLUMN: &str = "geometry";
const EPSG_METADATA_KEY: &str = "surtgis:epsg";

/// Little-endian ISO WKB for a 2-D point (21 bytes).
fn wkb_point(x: f64, y: f64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(21);
    buf.push(1u8); // little endian
    buf.extend_from_slice(&1u32.to_le_bytes()); // type = Point
    buf.extend_from_slice(&x.to_le_bytes());
    buf.extend_from_slice(&y.to_le_bytes());
    buf
}

fn parse_wkb_point(wkb: &[u8]) -> Result<(f64, f64)> {
    if wkb.len() < 21 {
        return Err(Error::Other(format!(
            "geoparquet: WKB geometry too short ({} bytes)",
            wkb.len()
        )));
    }
    let le = match wkb[0] {
        0 => false,
        1 => true,
        b => {
            return Err(Error::Other(format!(
                "geoparquet: invalid WKB byte order {}",
                b
            )));
        }
    };
    let u32_at = |off: usize| -> u32 {
        let b: [u8; 4] = wkb[off..off + 4].try_into().unwrap();
        if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        }
    };
    let f64_at = |off: usize| -> f64 {
        let b: [u8; 8] = wkb[off..off + 8].try_into().unwrap();
        if le {
            f64::from_le_bytes(b)
        } else {
            f64::from_be_bytes(b)
        }
    };
    let geom_type = u32_at(1);
    // Accept plain (1) and EWKB-with-SRID (0x20000001) points
    let (base, has_srid) = (geom_type & 0xFF, geom_type & 0x2000_0000 != 0);
    if base != 1 {
        return Err(Error::Other(format!(
            "geoparquet: only Point geometry is supported, got WKB type {}",
            geom_type
        )));
    }
    let coord_off = if has_srid { 9 } else { 5 };
    if wkb.len() < coord_off + 16 {
        return Err(Error::Other("geoparquet: truncated WKB point".into()));
    }
    Ok((f64_at(coord_off), f64_at(coord_off + 8)))
}

fn geo_metadata(table: &PointTable) -> String {
    let (mut minx, mut miny) = (f64::INFINITY, f64::INFINITY);
    let (mut maxx, mut maxy) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (&x, &y) in table.x.iter().zip(&table.y) {
        minx = minx.min(x);
        miny = miny.min(y);
        maxx = maxx.max(x);
        maxy = maxy.max(y);
    }
    let bbox = if table.is_empty() {
        serde_json::json!([0.0, 0.0, 0.0, 0.0])
    } else {
        serde_json::json!([minx, miny, maxx, maxy])
    };
    serde_json::json!({
        "version": "1.0.0",
        "primary_column": GEOMETRY_COLUMN,
        "columns": {
            GEOMETRY_COLUMN: {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "crs": null,
                "bbox": bbox,
            }
        }
    })
    .to_string()
}

fn build_schema(table: &PointTable) -> Result<Arc<SchemaType>> {
    let mut fields = Vec::with_capacity(table.columns.len() + 1);
    fields.push(Arc::new(
        SchemaType::primitive_type_builder(GEOMETRY_COLUMN, PhysicalType::BYTE_ARRAY)
            .with_repetition(Repetition::REQUIRED)
            .build()
            .map_err(|e| Error::Other(e.to_string()))?,
    ));
    for col in &table.columns {
        let builder = match &col.data {
            ColumnData::F64(_) => {
                SchemaType::primitive_type_builder(&col.name, PhysicalType::DOUBLE)
            }
            ColumnData::F32(_) => {
                SchemaType::primitive_type_builder(&col.name, PhysicalType::FLOAT)
            }
            ColumnData::I64(_) => {
                SchemaType::primitive_type_builder(&col.name, PhysicalType::INT64)
            }
            ColumnData::Str(_) => {
                SchemaType::primitive_type_builder(&col.name, PhysicalType::BYTE_ARRAY)
                    .with_logical_type(Some(LogicalType::String))
                    .with_converted_type(ConvertedType::UTF8)
            }
        };
        fields.push(Arc::new(
            builder
                .with_repetition(Repetition::REQUIRED)
                .build()
                .map_err(|e| Error::Other(e.to_string()))?,
        ));
    }
    SchemaType::group_type_builder("schema")
        .with_fields(fields)
        .build()
        .map(Arc::new)
        .map_err(|e| Error::Other(e.to_string()))
}

/// Write a [`PointTable`] as a GeoParquet file.
///
/// Geometry goes to a required `geometry` WKB column; attribute
/// columns keep their declared types (f64 → DOUBLE, f32 → FLOAT,
/// i64 → INT64, String → UTF-8). Snappy compression, single row
/// group.
pub fn write_geoparquet_points<P: AsRef<Path>>(table: &PointTable, path: P) -> Result<()> {
    table.validate()?;
    let schema = build_schema(table)?;

    let mut metadata = vec![KeyValue::new("geo".to_string(), geo_metadata(table))];
    if let Some(epsg) = table.epsg {
        metadata.push(KeyValue::new(
            EPSG_METADATA_KEY.to_string(),
            epsg.to_string(),
        ));
    }
    let props = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_key_value_metadata(Some(metadata))
            .build(),
    );

    let file = File::create(path.as_ref())
        .map_err(|e| Error::Other(format!("geoparquet: cannot create file: {}", e)))?;
    let mut writer =
        SerializedFileWriter::new(file, schema, props).map_err(|e| Error::Other(e.to_string()))?;

    let mut rg = writer
        .next_row_group()
        .map_err(|e| Error::Other(e.to_string()))?;

    // Column 0: geometry
    {
        let mut col = rg
            .next_column()
            .map_err(|e| Error::Other(e.to_string()))?
            .ok_or_else(|| Error::Other("geoparquet: missing geometry column writer".into()))?;
        let wkb: Vec<ByteArray> = table
            .x
            .iter()
            .zip(&table.y)
            .map(|(&x, &y)| ByteArray::from(wkb_point(x, y)))
            .collect();
        col.typed::<ByteArrayType>()
            .write_batch(&wkb, None, None)
            .map_err(|e| Error::Other(e.to_string()))?;
        col.close().map_err(|e| Error::Other(e.to_string()))?;
    }

    for column in &table.columns {
        let mut col = rg
            .next_column()
            .map_err(|e| Error::Other(e.to_string()))?
            .ok_or_else(|| {
                Error::Other(format!("geoparquet: missing writer for '{}'", column.name))
            })?;
        match &column.data {
            ColumnData::F64(v) => {
                col.typed::<DoubleType>()
                    .write_batch(v, None, None)
                    .map_err(|e| Error::Other(e.to_string()))?;
            }
            ColumnData::F32(v) => {
                col.typed::<FloatType>()
                    .write_batch(v, None, None)
                    .map_err(|e| Error::Other(e.to_string()))?;
            }
            ColumnData::I64(v) => {
                col.typed::<Int64Type>()
                    .write_batch(v, None, None)
                    .map_err(|e| Error::Other(e.to_string()))?;
            }
            ColumnData::Str(v) => {
                let bytes: Vec<ByteArray> = v.iter().map(|s| ByteArray::from(s.as_str())).collect();
                col.typed::<ByteArrayType>()
                    .write_batch(&bytes, None, None)
                    .map_err(|e| Error::Other(e.to_string()))?;
            }
        }
        col.close().map_err(|e| Error::Other(e.to_string()))?;
    }

    rg.close().map_err(|e| Error::Other(e.to_string()))?;
    writer.close().map_err(|e| Error::Other(e.to_string()))?;
    Ok(())
}

/// Read a GeoParquet point file into a [`PointTable`].
///
/// The geometry column is located through the `geo` metadata
/// (`primary_column`), falling back to a column named `geometry`.
/// Supported attribute types: DOUBLE, FLOAT, INT64/INT32 (widened to
/// i64), UTF-8 strings. Null values are rejected.
pub fn read_geoparquet_points<P: AsRef<Path>>(path: P) -> Result<PointTable> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::Other(format!("geoparquet: cannot open file: {}", e)))?;
    let reader = SerializedFileReader::new(file).map_err(|e| Error::Other(e.to_string()))?;
    let meta = reader.metadata().file_metadata();

    let mut geometry_column = GEOMETRY_COLUMN.to_string();
    let mut epsg = None;
    if let Some(kvs) = meta.key_value_metadata() {
        for kv in kvs {
            match (kv.key.as_str(), kv.value.as_deref()) {
                ("geo", Some(json)) => {
                    if let Ok(geo) = serde_json::from_str::<serde_json::Value>(json)
                        && let Some(primary) = geo["primary_column"].as_str()
                    {
                        geometry_column = primary.to_string();
                    }
                }
                (EPSG_METADATA_KEY, Some(code)) => {
                    epsg = code.parse::<u32>().ok();
                }
                _ => {}
            }
        }
    }

    let mut table = PointTable {
        epsg,
        ..Default::default()
    };
    let mut columns_init = false;

    let rows = reader
        .get_row_iter(None)
        .map_err(|e| Error::Other(e.to_string()))?;
    for row in rows {
        let row = row.map_err(|e| Error::Other(e.to_string()))?;
        let mut col_idx = 0usize;
        for (name, field) in row.get_column_iter() {
            if name == &geometry_column {
                let Field::Bytes(wkb) = field else {
                    return Err(Error::Other(format!(
                        "geoparquet: geometry column '{}' is not BYTE_ARRAY",
                        geometry_column
                    )));
                };
                let (x, y) = parse_wkb_point(wkb.data())?;
                table.x.push(x);
                table.y.push(y);
                continue;
            }

            if !columns_init {
                let data = match field {
                    Field::Double(_) => ColumnData::F64(Vec::new()),
                    Field::Float(_) => ColumnData::F32(Vec::new()),
                    Field::Long(_) | Field::Int(_) | Field::Short(_) | Field::Byte(_) => {
                        ColumnData::I64(Vec::new())
                    }
                    Field::Str(_) => ColumnData::Str(Vec::new()),
                    other => {
                        return Err(Error::Other(format!(
                            "geoparquet: unsupported type {:?} in column '{}'",
                            other, name
                        )));
                    }
                };
                table.columns.push(Column {
                    name: name.clone(),
                    data,
                });
            }

            let col = &mut table.columns[col_idx];
            match (&mut col.data, field) {
                (ColumnData::F64(v), Field::Double(x)) => v.push(*x),
                (ColumnData::F32(v), Field::Float(x)) => v.push(*x),
                (ColumnData::I64(v), Field::Long(x)) => v.push(*x),
                (ColumnData::I64(v), Field::Int(x)) => v.push(*x as i64),
                (ColumnData::I64(v), Field::Short(x)) => v.push(*x as i64),
                (ColumnData::I64(v), Field::Byte(x)) => v.push(*x as i64),
                (ColumnData::Str(v), Field::Str(s)) => v.push(s.clone()),
                (_, other) => {
                    return Err(Error::Other(format!(
                        "geoparquet: inconsistent or null value {:?} in column '{}'",
                        other, name
                    )));
                }
            }
            col_idx += 1;
        }
        columns_init = true;
    }

    Ok(table)
}

/// Write a [`FeatureCollection`] of points as GeoParquet.
///
/// Property keys become columns; the type of each column is taken
/// from its first occurrence (Int → i64, Float → f64, String → UTF-8,
/// Bool → i64 0/1). Every feature must be a point and carry every
/// property (no nulls in v1).
pub fn write_geoparquet<P: AsRef<Path>>(
    fc: &FeatureCollection,
    epsg: Option<u32>,
    path: P,
) -> Result<()> {
    let mut table = PointTable {
        epsg,
        ..Default::default()
    };

    // Stable column order: first-seen across features
    let mut names: Vec<String> = Vec::new();
    for feature in fc.iter() {
        for key in feature.properties.keys() {
            if !names.contains(key) {
                names.push(key.clone());
            }
        }
    }
    names.sort();

    for feature in fc.iter() {
        let Some(geo::Geometry::Point(p)) = &feature.geometry else {
            return Err(Error::Other(
                "geoparquet: every feature must have Point geometry".into(),
            ));
        };
        table.x.push(p.x());
        table.y.push(p.y());
    }

    for name in &names {
        let mut f64s = Vec::new();
        let mut i64s = Vec::new();
        let mut strs = Vec::new();
        let mut kind: Option<u8> = None; // 0=f64 1=i64 2=str
        for feature in fc.iter() {
            let value = feature.get_property(name).ok_or_else(|| {
                Error::Other(format!(
                    "geoparquet: feature missing property '{}' (nulls unsupported)",
                    name
                ))
            })?;
            let k = match value {
                AttributeValue::Float(_) => 0,
                AttributeValue::Int(_) | AttributeValue::Bool(_) => 1,
                AttributeValue::String(_) => 2,
                AttributeValue::Null => {
                    return Err(Error::Other(format!(
                        "geoparquet: null value in property '{}' (nulls unsupported)",
                        name
                    )));
                }
            };
            match kind {
                None => kind = Some(k),
                Some(existing) if existing != k => {
                    return Err(Error::Other(format!(
                        "geoparquet: mixed types in property '{}'",
                        name
                    )));
                }
                _ => {}
            }
            match value {
                AttributeValue::Float(v) => f64s.push(*v),
                AttributeValue::Int(v) => i64s.push(*v),
                AttributeValue::Bool(v) => i64s.push(*v as i64),
                AttributeValue::String(v) => strs.push(v.clone()),
                AttributeValue::Null => unreachable!(),
            }
        }
        let data = match kind {
            Some(0) => ColumnData::F64(f64s),
            Some(1) => ColumnData::I64(i64s),
            Some(2) => ColumnData::Str(strs),
            _ => continue, // empty collection
        };
        table.columns.push(Column {
            name: name.clone(),
            data,
        });
    }

    write_geoparquet_points(&table, path)
}

/// Read a GeoParquet point file as a [`FeatureCollection`].
///
/// The GeoParquet `geo` metadata's `crs` field is written as `null` by this
/// crate's own writer (see module docs), so the CRS is recovered from the
/// `surtgis:epsg` key-value metadata instead. Files from other GeoParquet
/// writers that don't set that key yield `crs: None` — honestly unknown
/// rather than assumed.
pub fn read_geoparquet<P: AsRef<Path>>(path: P) -> Result<FeatureCollection> {
    let table = read_geoparquet_points(path)?;
    let mut fc = FeatureCollection::with_crs(table.epsg.map(CRS::from_epsg));
    for i in 0..table.len() {
        let mut feature = Feature::new(geo::Geometry::Point(geo::Point::new(
            table.x[i], table.y[i],
        )));
        for col in &table.columns {
            let value = match &col.data {
                ColumnData::F64(v) => AttributeValue::Float(v[i]),
                ColumnData::F32(v) => AttributeValue::Float(v[i] as f64),
                ColumnData::I64(v) => AttributeValue::Int(v[i]),
                ColumnData::Str(v) => AttributeValue::String(v[i].clone()),
            };
            feature.set_property(col.name.clone(), value);
        }
        fc.push(feature);
    }
    Ok(fc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_table() -> PointTable {
        PointTable {
            x: vec![350_000.0, 350_010.0, 350_020.0],
            y: vec![6_300_000.0, 6_300_010.0, 6_300_020.0],
            epsg: Some(32719),
            columns: vec![
                Column {
                    name: "label".into(),
                    data: ColumnData::I64(vec![0, 1, 2]),
                },
                Column {
                    name: "ndvi".into(),
                    data: ColumnData::F64(vec![0.1, 0.5, 0.9]),
                },
                Column {
                    name: "e0".into(),
                    data: ColumnData::F32(vec![0.25, -0.5, 1.75]),
                },
                Column {
                    name: "site".into(),
                    data: ColumnData::Str(vec!["a".into(), "b".into(), "c".into()]),
                },
            ],
        }
    }

    #[test]
    fn point_table_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pts.parquet");
        let table = sample_table();
        write_geoparquet_points(&table, &path).unwrap();

        let back = read_geoparquet_points(&path).unwrap();
        assert_eq!(back.x, table.x);
        assert_eq!(back.y, table.y);
        assert_eq!(back.epsg, Some(32719));
        assert_eq!(back.columns.len(), 4);
        for (a, b) in back.columns.iter().zip(&table.columns) {
            assert_eq!(a, b, "column {} drifted", b.name);
        }
    }

    #[test]
    fn geo_metadata_is_valid_geoparquet() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pts.parquet");
        write_geoparquet_points(&sample_table(), &path).unwrap();

        let file = File::open(&path).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let kvs = reader
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .unwrap()
            .clone();
        let geo = kvs.iter().find(|kv| kv.key == "geo").expect("geo key");
        let parsed: serde_json::Value =
            serde_json::from_str(geo.value.as_deref().unwrap()).unwrap();
        assert_eq!(parsed["version"], "1.0.0");
        assert_eq!(parsed["primary_column"], "geometry");
        assert_eq!(parsed["columns"]["geometry"]["encoding"], "WKB");
        let bbox = parsed["columns"]["geometry"]["bbox"].as_array().unwrap();
        assert_eq!(bbox[0].as_f64().unwrap(), 350_000.0);
        assert_eq!(bbox[3].as_f64().unwrap(), 6_300_020.0);
    }

    #[test]
    fn wkb_point_roundtrip_and_ewkb() {
        let buf = wkb_point(-70.5, -33.4);
        let (x, y) = parse_wkb_point(&buf).unwrap();
        assert_eq!((x, y), (-70.5, -33.4));

        // EWKB with SRID flag
        let mut ewkb = vec![1u8];
        ewkb.extend_from_slice(&0x2000_0001u32.to_le_bytes());
        ewkb.extend_from_slice(&4326u32.to_le_bytes());
        ewkb.extend_from_slice(&10.0f64.to_le_bytes());
        ewkb.extend_from_slice(&20.0f64.to_le_bytes());
        let (x, y) = parse_wkb_point(&ewkb).unwrap();
        assert_eq!((x, y), (10.0, 20.0));

        // Non-point rejected
        let mut line = vec![1u8];
        line.extend_from_slice(&2u32.to_le_bytes());
        line.extend_from_slice(&[0u8; 16]);
        assert!(parse_wkb_point(&line).is_err());
    }

    #[test]
    fn feature_collection_roundtrip() {
        let mut fc = FeatureCollection::new();
        for i in 0..5 {
            let mut f = Feature::new(geo::Geometry::Point(geo::Point::new(
                i as f64 * 10.0,
                i as f64 * -5.0,
            )));
            f.set_property("clase", AttributeValue::Int(i));
            f.set_property("peso", AttributeValue::Float(i as f64 / 2.0));
            fc.push(f);
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fc.parquet");
        write_geoparquet(&fc, Some(4326), &path).unwrap();

        let back = read_geoparquet(&path).unwrap();
        assert_eq!(back.len(), 5);
        assert_eq!(back.crs().and_then(|c| c.epsg()), Some(4326));
        let f2 = &back.features[2];
        let Some(geo::Geometry::Point(p)) = &f2.geometry else {
            panic!("expected point");
        };
        assert_eq!((p.x(), p.y()), (20.0, -10.0));
        assert!(matches!(
            f2.get_property("clase"),
            Some(AttributeValue::Int(2))
        ));
        assert!(matches!(
            f2.get_property("peso"),
            Some(AttributeValue::Float(v)) if *v == 1.0
        ));
    }

    #[test]
    fn feature_collection_without_epsg_yields_none_crs() {
        let mut fc = FeatureCollection::new();
        fc.push(Feature::new(geo::Geometry::Point(geo::Point::new(
            1.0, 2.0,
        ))));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_epsg.parquet");
        write_geoparquet(&fc, None, &path).unwrap();

        let back = read_geoparquet(&path).unwrap();
        assert!(back.crs().is_none());
    }

    #[test]
    fn rejects_mismatched_lengths_and_reserved_name() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = sample_table();
        t.y.pop();
        assert!(write_geoparquet_points(&t, dir.path().join("bad.parquet")).is_err());

        let mut t = sample_table();
        t.columns.push(Column {
            name: "geometry".into(),
            data: ColumnData::I64(vec![0, 0, 0]),
        });
        assert!(write_geoparquet_points(&t, dir.path().join("bad2.parquet")).is_err());
    }

    #[test]
    fn wide_embedding_table() {
        // geoembed shape: 100 points × 64 dims
        let n = 100usize;
        let dims = 64usize;
        let mut table = PointTable {
            x: (0..n).map(|i| i as f64).collect(),
            y: (0..n).map(|i| -(i as f64)).collect(),
            epsg: Some(3857),
            columns: Vec::new(),
        };
        for d in 0..dims {
            table.columns.push(Column {
                name: format!("e{}", d),
                data: ColumnData::F32((0..n).map(|i| (i * d) as f32 * 0.01).collect()),
            });
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("emb.parquet");
        write_geoparquet_points(&table, &path).unwrap();
        let back = read_geoparquet_points(&path).unwrap();
        assert_eq!(back.len(), n);
        assert_eq!(back.columns.len(), dims);
        assert_eq!(back.columns[63].data, table.columns[63].data);
    }
}
