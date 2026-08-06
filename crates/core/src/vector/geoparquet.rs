//! GeoParquet I/O (feature `parquet`).
//!
//! Pure-Rust reader/writer in
//! [GeoParquet 1.0](https://geoparquet.org/releases/v1.0.0/) layout:
//! WKB geometry column + `geo` file metadata, Snappy compression.
//! The output is directly queryable from DuckDB, GeoPandas and GDAL.
//!
//! Two entry points:
//!
//! - [`PointTable`] — columnar point tables with attributes (embedding
//!   matrices, training samples, extracted features) that avoid
//!   per-feature HashMaps for wide tables.
//! - [`read_geoparquet`] — any geometry type into a [`FeatureCollection`]
//!   (Point, LineString, Polygon and Multi\* variants via the WKB parser),
//!   for tiling, analysis and format conversion.
//!
//! Scope (v1):
//! - Geometry: full 2-D WKB read (all standard types), point-only write.
//! - Column types: f64, f32, i64, bool, UTF-8 string.
//! - Nulls: supported by [`read_geoparquet`] — nullable (OPTIONAL) columns,
//!   as written by GeoPandas/pyarrow for non-uniform properties, map to
//!   [`AttributeValue::Null`] per feature. The columnar
//!   [`read_geoparquet_points`] path stays strict (no nulls) by design.
//! - CRS: GeoParquet `crs` is written as `null` (unknown) and the
//!   EPSG code is preserved in a `surtgis:epsg` metadata key, which
//!   the reader recovers. The reader also understands the standard
//!   GeoParquet `crs` field (PROJJSON id or OGC URI string) written by
//!   other tools. Readers other than SurtGIS see correct geometry but
//!   must be told the CRS out-of-band for SurtGIS-written files.

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

/// Cursor over a WKB/EWKB byte buffer.
struct WkbReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> WkbReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(Error::Other(format!(
                "geoparquet: truncated WKB geometry (need {} bytes at offset {})",
                n, self.pos
            )));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self, le: bool) -> Result<u32> {
        let b: [u8; 4] = self.take(4)?.try_into().unwrap();
        Ok(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }

    fn f64(&mut self, le: bool) -> Result<f64> {
        let b: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if le {
            f64::from_le_bytes(b)
        } else {
            f64::from_be_bytes(b)
        })
    }

    /// A coordinate tuple of `dims` doubles; returns the first two (x, y)
    /// and skips any Z/M extras.
    fn coord(&mut self, le: bool, dims: u32) -> Result<(f64, f64)> {
        let x = self.f64(le)?;
        let y = self.f64(le)?;
        for _ in 2..dims {
            self.f64(le)?;
        }
        Ok((x, y))
    }
}

/// Parse a full 2-D WKB geometry (any standard type, either byte order).
///
/// Handles EWKB dimension flags (Z/M are read and dropped) and the SRID
/// flag (the SRID word is consumed). Returns a `geo` geometry in source
/// coordinates — reprojection, if any, is the caller's job.
fn parse_wkb(wkb: &[u8]) -> Result<geo::Geometry<f64>> {
    let mut r = WkbReader::new(wkb);
    read_geometry(&mut r)
}

/// Read one geometry (a full WKB geometry, including its byte-order byte).
fn read_geometry(r: &mut WkbReader) -> Result<geo::Geometry<f64>> {
    let le = match r.u8()? {
        0 => false,
        1 => true,
        b => {
            return Err(Error::Other(format!(
                "geoparquet: invalid WKB byte order {}",
                b
            )));
        }
    };
    let ty = r.u32(le)?;
    let base = ty & 0xFF;
    let has_srid = ty & 0x2000_0000 != 0;
    let dims = 2 + ((ty & 0x8000_0000 != 0) as u32) + ((ty & 0x4000_0000 != 0) as u32);
    if has_srid {
        r.u32(le)?; // discard SRID word
    }

    match base {
        1 => {
            let (x, y) = r.coord(le, dims)?;
            Ok(geo::Geometry::Point(geo::Point::new(x, y)))
        }
        2 => {
            let n = r.u32(le)? as usize;
            let mut coords = Vec::with_capacity(n);
            for _ in 0..n {
                let (x, y) = r.coord(le, dims)?;
                coords.push(geo::Coord { x, y });
            }
            Ok(geo::Geometry::LineString(geo::LineString::new(coords)))
        }
        3 => {
            let n_rings = r.u32(le)? as usize;
            let mut rings = Vec::with_capacity(n_rings);
            for _ in 0..n_rings {
                let n = r.u32(le)? as usize;
                let mut ring = Vec::with_capacity(n);
                for _ in 0..n {
                    let (x, y) = r.coord(le, dims)?;
                    ring.push(geo::Coord { x, y });
                }
                rings.push(geo::LineString::new(ring));
            }
            let mut it = rings.into_iter();
            let exterior = it
                .next()
                .ok_or_else(|| Error::Other("geoparquet: polygon with no rings".into()))?;
            Ok(geo::Geometry::Polygon(geo::Polygon::new(
                exterior,
                it.collect(),
            )))
        }
        4 => {
            let n = r.u32(le)? as usize;
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                match read_geometry(r)? {
                    geo::Geometry::Point(p) => pts.push(p),
                    _ => {
                        return Err(Error::Other("geoparquet: non-point in MultiPoint".into()));
                    }
                }
            }
            Ok(geo::Geometry::MultiPoint(geo::MultiPoint::new(pts)))
        }
        5 => {
            let n = r.u32(le)? as usize;
            let mut lines = Vec::with_capacity(n);
            for _ in 0..n {
                match read_geometry(r)? {
                    geo::Geometry::LineString(l) => lines.push(l),
                    _ => {
                        return Err(Error::Other(
                            "geoparquet: non-line in MultiLineString".into(),
                        ));
                    }
                }
            }
            Ok(geo::Geometry::MultiLineString(geo::MultiLineString::new(
                lines,
            )))
        }
        6 => {
            let n = r.u32(le)? as usize;
            let mut polys = Vec::with_capacity(n);
            for _ in 0..n {
                match read_geometry(r)? {
                    geo::Geometry::Polygon(p) => polys.push(p),
                    _ => {
                        return Err(Error::Other(
                            "geoparquet: non-polygon in MultiPolygon".into(),
                        ));
                    }
                }
            }
            Ok(geo::Geometry::MultiPolygon(geo::MultiPolygon::new(polys)))
        }
        other => Err(Error::Other(format!(
            "geoparquet: unsupported WKB geometry type {}",
            other
        ))),
    }
}

/// Parse a WKB geometry that must be a point, returning `(x, y)`.
fn parse_wkb_point(wkb: &[u8]) -> Result<(f64, f64)> {
    match parse_wkb(wkb)? {
        geo::Geometry::Point(p) => Ok((p.x(), p.y())),
        _ => Err(Error::Other("geoparquet: expected a point geometry".into())),
    }
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
/// i64), UTF-8 strings.
///
/// **Null values are rejected**: [`PointTable`] is strictly columnar
/// (embedding matrices, training samples) and has no representation for
/// missing values. For nullable data — the GeoPandas/pyarrow default for
/// non-uniform properties — use [`read_geoparquet`], which maps nulls to
/// [`AttributeValue::Null`].
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
                    Field::Null => {
                        return Err(Error::Other(format!(
                            "geoparquet: null value in column '{}'; PointTable is \
                             strictly columnar — use read_geoparquet for nullable data",
                            name
                        )));
                    }
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
                (_, Field::Null) => {
                    return Err(Error::Other(format!(
                        "geoparquet: null value in column '{}'; PointTable is \
                         strictly columnar — use read_geoparquet for nullable data",
                        name
                    )));
                }
                (_, other) => {
                    return Err(Error::Other(format!(
                        "geoparquet: inconsistent value {:?} in column '{}'",
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

/// Read a GeoParquet file as a [`FeatureCollection`].
///
/// Any standard geometry type is supported (Point, LineString, Polygon
/// and Multi\* variants) via the WKB parser. CRS recovery order:
/// 1. `surtgis:epsg` key-value metadata (this crate's writer),
/// 2. the standard GeoParquet `geo.columns.<geom>.crs` field — a PROJJSON
///    object with an `id` (`{authority, code}`) or an OGC URI string
///    (`http://www.opengis.net/def/crs/EPSG/0/4326`).
///
/// Files with no recoverable CRS yield `crs: None` — honestly unknown
/// rather than assumed.
///
/// Nullable (OPTIONAL) attribute columns — the GeoPandas/pyarrow default
/// when properties are not uniform across features — are supported: a
/// missing value becomes [`AttributeValue::Null`] on that feature.
pub fn read_geoparquet<P: AsRef<Path>>(path: P) -> Result<FeatureCollection> {
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
                    if let Ok(geo) = serde_json::from_str::<serde_json::Value>(json) {
                        if let Some(primary) = geo["primary_column"].as_str() {
                            geometry_column = primary.to_string();
                        }
                        if let Some(col) = geo["columns"].get(&geometry_column)
                            && epsg.is_none()
                            && let Some(code) = crs_to_epsg(&col["crs"])
                        {
                            epsg = Some(code);
                        }
                    }
                }
                (EPSG_METADATA_KEY, Some(code)) => {
                    epsg = code.parse::<u32>().ok().or(epsg);
                }
                _ => {}
            }
        }
    }

    let mut fc = FeatureCollection::with_crs(epsg.map(CRS::from_epsg));

    let rows = reader
        .get_row_iter(None)
        .map_err(|e| Error::Other(e.to_string()))?;
    for row in rows {
        let row = row.map_err(|e| Error::Other(e.to_string()))?;
        let mut geometry = None;
        let mut props: Vec<(String, AttributeValue)> = Vec::new();
        for (name, field) in row.get_column_iter() {
            if name == &geometry_column {
                let Field::Bytes(wkb) = field else {
                    return Err(Error::Other(format!(
                        "geoparquet: geometry column '{}' is not BYTE_ARRAY",
                        geometry_column
                    )));
                };
                geometry = Some(parse_wkb(wkb.data())?);
                continue;
            }

            // Direct per-field mapping. Parquet's schema already guarantees
            // each column decodes to one consistent `Field` variant across
            // rows (or `Null` for OPTIONAL columns — the GeoPandas/pyarrow
            // default for non-uniform properties), so no cross-row type
            // bookkeeping is needed here.
            let value = match field {
                Field::Null => AttributeValue::Null,
                Field::Bool(b) => AttributeValue::Bool(*b),
                Field::Double(x) => AttributeValue::Float(*x),
                Field::Float(x) => AttributeValue::Float(*x as f64),
                Field::Long(x) => AttributeValue::Int(*x),
                Field::Int(x) => AttributeValue::Int(*x as i64),
                Field::Short(x) => AttributeValue::Int(*x as i64),
                Field::Byte(x) => AttributeValue::Int(*x as i64),
                Field::Str(s) => AttributeValue::String(s.clone()),
                other => {
                    return Err(Error::Other(format!(
                        "geoparquet: unsupported type {:?} in column '{}'",
                        other, name
                    )));
                }
            };
            props.push((name.clone(), value));
        }

        let Some(geometry) = geometry else {
            return Err(Error::Other(format!(
                "geoparquet: missing geometry column '{}'",
                geometry_column
            )));
        };
        let mut f = Feature::new(geometry);
        for (name, value) in props {
            f.set_property(name, value);
        }
        fc.push(f);
    }

    Ok(fc)
}

/// Extract an EPSG code from a GeoParquet `crs` metadata value.
///
/// Accepts a PROJJSON object with an `id` (`{authority, code}`), an OGC
/// URI string (`http://www.opengis.net/def/crs/EPSG/0/4326`,
/// `urn:ogc:def:crs:EPSG::4326`) or `null`/missing (→ `None`).
fn crs_to_epsg(crs: &serde_json::Value) -> Option<u32> {
    match crs {
        serde_json::Value::Object(obj) => obj
            .get("id")
            .and_then(|id| id.get("code"))
            .and_then(|c| c.as_u64())
            .map(|c| c as u32),
        serde_json::Value::String(s) => {
            // Last numeric token of the URI/URN is the EPSG code. The
            // split rev()s a DoubleEndedIterator so `next` is O(1).
            s.split(|c: char| !c.is_ascii_digit())
                .rev()
                .find_map(|tok| tok.parse::<u32>().ok())
        }
        _ => None,
    }
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

    // --- WKB helpers for the full-geometry tests ---

    fn wkb_linestring(coords: &[(f64, f64)]) -> Vec<u8> {
        let mut buf = vec![1u8, 2, 0, 0, 0];
        buf.extend_from_slice(&(coords.len() as u32).to_le_bytes());
        for (x, y) in coords {
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
        }
        buf
    }

    fn wkb_polygon(rings: &[Vec<(f64, f64)>]) -> Vec<u8> {
        let mut buf = vec![1u8, 3, 0, 0, 0];
        buf.extend_from_slice(&(rings.len() as u32).to_le_bytes());
        for ring in rings {
            buf.extend_from_slice(&(ring.len() as u32).to_le_bytes());
            for (x, y) in ring {
                buf.extend_from_slice(&x.to_le_bytes());
                buf.extend_from_slice(&y.to_le_bytes());
            }
        }
        buf
    }

    fn wkb_multipoint(pts: &[(f64, f64)]) -> Vec<u8> {
        let mut buf = vec![1u8, 4, 0, 0, 0];
        buf.extend_from_slice(&(pts.len() as u32).to_le_bytes());
        for (x, y) in pts {
            let mut p = vec![1u8, 1, 0, 0, 0];
            p.extend_from_slice(&x.to_le_bytes());
            p.extend_from_slice(&y.to_le_bytes());
            buf.extend_from_slice(&p);
        }
        buf
    }

    fn wkb_multilinestring(lines: &[Vec<(f64, f64)>]) -> Vec<u8> {
        let mut buf = vec![1u8, 5, 0, 0, 0];
        buf.extend_from_slice(&(lines.len() as u32).to_le_bytes());
        for coords in lines {
            buf.extend_from_slice(&wkb_linestring(coords));
        }
        buf
    }

    fn wkb_multipolygon(polys: &[Vec<Vec<(f64, f64)>>]) -> Vec<u8> {
        let mut buf = vec![1u8, 6, 0, 0, 0];
        buf.extend_from_slice(&(polys.len() as u32).to_le_bytes());
        for rings in polys {
            buf.extend_from_slice(&wkb_polygon(rings));
        }
        buf
    }

    #[test]
    fn wkb_parses_all_geometry_types() {
        // Point
        let (x, y) = parse_wkb_point(&wkb_point(1.5, -2.5)).unwrap();
        assert_eq!((x, y), (1.5, -2.5));

        // LineString
        let g = parse_wkb(&wkb_linestring(&[(0.0, 0.0), (1.0, 1.0)])).unwrap();
        match g {
            geo::Geometry::LineString(l) => {
                assert_eq!(l.0.len(), 2);
                assert_eq!(l.0[1], geo::Coord { x: 1.0, y: 1.0 });
            }
            _ => panic!("expected LineString"),
        }

        // Polygon with a hole
        let g = parse_wkb(&wkb_polygon(&[
            vec![(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (0.0, 0.0)],
            vec![(2.0, 2.0), (2.0, 4.0), (4.0, 4.0), (2.0, 2.0)],
        ]))
        .unwrap();
        match g {
            geo::Geometry::Polygon(p) => {
                assert_eq!(p.exterior().0.len(), 4);
                assert_eq!(p.interiors().len(), 1);
            }
            _ => panic!("expected Polygon"),
        }

        // MultiPoint
        let g = parse_wkb(&wkb_multipoint(&[(0.0, 0.0), (5.0, 5.0)])).unwrap();
        match g {
            geo::Geometry::MultiPoint(mp) => assert_eq!(mp.0.len(), 2),
            _ => panic!("expected MultiPoint"),
        }

        // MultiLineString
        let g = parse_wkb(&wkb_multilinestring(&[
            vec![(0.0, 0.0), (1.0, 1.0)],
            vec![(2.0, 2.0), (3.0, 3.0)],
        ]))
        .unwrap();
        match g {
            geo::Geometry::MultiLineString(ml) => assert_eq!(ml.0.len(), 2),
            _ => panic!("expected MultiLineString"),
        }

        // MultiPolygon
        let g = parse_wkb(&wkb_multipolygon(&[vec![vec![
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.0, 0.0),
        ]]]))
        .unwrap();
        match g {
            geo::Geometry::MultiPolygon(mp) => assert_eq!(mp.0.len(), 1),
            _ => panic!("expected MultiPolygon"),
        }
    }

    #[test]
    fn wkb_big_endian_and_ewkb_srid() {
        // Big-endian point
        let mut be = vec![0u8, 0, 0, 0, 1];
        be.extend_from_slice(&7.0f64.to_be_bytes());
        be.extend_from_slice(&8.0f64.to_be_bytes());
        let (x, y) = parse_wkb_point(&be).unwrap();
        assert_eq!((x, y), (7.0, 8.0));

        // EWKB point with SRID (still just a point to parse)
        let mut ewkb = vec![1u8];
        ewkb.extend_from_slice(&0x2000_0001u32.to_le_bytes());
        ewkb.extend_from_slice(&4326u32.to_le_bytes());
        ewkb.extend_from_slice(&10.0f64.to_le_bytes());
        ewkb.extend_from_slice(&20.0f64.to_le_bytes());
        let (x, y) = parse_wkb_point(&ewkb).unwrap();
        assert_eq!((x, y), (10.0, 20.0));
    }

    #[test]
    fn wkb_rejects_truncated_and_unknown() {
        assert!(parse_wkb(&[]).is_err());
        assert!(parse_wkb(&[1u8, 2, 0, 0, 0]).is_err()); // LineString, no count
        let mut unknown = vec![1u8, 9, 0, 0, 0];
        unknown.extend_from_slice(&1u32.to_le_bytes());
        assert!(parse_wkb(&unknown).is_err());
    }

    #[test]
    fn crs_to_epsg_understands_uris_and_projjson() {
        assert_eq!(
            crs_to_epsg(&serde_json::json!(
                "http://www.opengis.net/def/crs/EPSG/0/4326"
            )),
            Some(4326)
        );
        assert_eq!(
            crs_to_epsg(&serde_json::json!("urn:ogc:def:crs:EPSG::32719")),
            Some(32719)
        );
        assert_eq!(
            crs_to_epsg(
                &serde_json::json!({"type": "GeographicCRS", "id": {"authority": "EPSG", "code": 4326}})
            ),
            Some(4326)
        );
        assert_eq!(crs_to_epsg(&serde_json::json!(null)), None);
        assert_eq!(crs_to_epsg(&serde_json::json!({})), None);
    }

    #[test]
    fn read_geoparquet_recovers_epsg_from_geo_crs_field() {
        // Write a point file, then rewrite its `geo` metadata to carry a
        // standard PROJJSON crs (as other writers would) instead of the
        // surtgis:epsg key.
        let mut fc = FeatureCollection::new();
        fc.push(Feature::new(geo::Geometry::Point(geo::Point::new(
            1.0, 2.0,
        ))));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("projjson.parquet");
        write_geoparquet(&fc, None, &path).unwrap();

        let file = File::open(&path).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let mut kvs = reader
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .unwrap()
            .clone();
        let geo = kvs.iter_mut().find(|kv| kv.key == "geo").unwrap();
        let mut parsed: serde_json::Value =
            serde_json::from_str(geo.value.as_deref().unwrap()).unwrap();
        parsed["columns"]["geometry"]["crs"] =
            serde_json::json!({"id": {"authority": "EPSG", "code": 3857}});
        geo.value = Some(parsed.to_string());

        // Rewrite the parquet with the modified metadata.
        let schema = reader.metadata().file_metadata().schema().clone();
        let props = Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .set_key_value_metadata(Some(kvs))
                .build(),
        );
        let f = File::create(&path).unwrap();
        let mut w = SerializedFileWriter::new(f, schema.into(), props).unwrap();
        let mut rg = w.next_row_group().unwrap();
        // Single column: geometry (WKB points).
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<ByteArrayType>()
            .write_batch(&[ByteArray::from(wkb_point(1.0, 2.0))], None, None)
            .unwrap();
        col.close().unwrap();
        rg.close().unwrap();
        w.close().unwrap();

        let back = read_geoparquet(&path).unwrap();
        assert_eq!(back.crs().and_then(|c| c.epsg()), Some(3857));
    }

    /// The GeoPandas/pyarrow shape that motivated null support: mixed
    /// geometry types where a property only exists on some features, so
    /// the column is written OPTIONAL and carries nulls — including in
    /// the very first row, and one column that is null throughout.
    fn write_nullable_fixture(path: &std::path::Path, points_only: bool) {
        let geometry = SchemaType::primitive_type_builder("geometry", PhysicalType::BYTE_ARRAY)
            .with_repetition(Repetition::REQUIRED)
            .build()
            .unwrap();
        let estacion = SchemaType::primitive_type_builder("estacion", PhysicalType::BYTE_ARRAY)
            .with_converted_type(ConvertedType::UTF8)
            .with_repetition(Repetition::OPTIONAL)
            .build()
            .unwrap();
        let valor = SchemaType::primitive_type_builder("valor", PhysicalType::DOUBLE)
            .with_repetition(Repetition::OPTIONAL)
            .build()
            .unwrap();
        let vacia = SchemaType::primitive_type_builder("vacia", PhysicalType::INT64)
            .with_repetition(Repetition::OPTIONAL)
            .build()
            .unwrap();
        let schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![
                Arc::new(geometry),
                Arc::new(estacion),
                Arc::new(valor),
                Arc::new(vacia),
            ])
            .build()
            .unwrap();

        let props = Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build(),
        );
        let f = File::create(path).unwrap();
        let mut w = SerializedFileWriter::new(f, Arc::new(schema), props).unwrap();
        let mut rg = w.next_row_group().unwrap();

        // geometry: linestring, point, point (mixed types) — or points only,
        // so the strict PointTable reader reaches the null instead of
        // rejecting on geometry first.
        let first = if points_only {
            wkb_point(0.0, 0.0)
        } else {
            wkb_linestring(&[(0.0, 0.0), (1.0, 1.0)])
        };
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<ByteArrayType>()
            .write_batch(
                &[
                    ByteArray::from(first),
                    ByteArray::from(wkb_point(2.0, 2.0)),
                    ByteArray::from(wkb_point(3.0, 3.0)),
                ],
                None,
                None,
            )
            .unwrap();
        col.close().unwrap();

        // estacion: null on the linestring (FIRST row), present on points.
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<ByteArrayType>()
            .write_batch(
                &[ByteArray::from("EST-7"), ByteArray::from("EST-9")],
                Some(&[0, 1, 1]),
                None,
            )
            .unwrap();
        col.close().unwrap();

        // valor: present, null, present.
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<DoubleType>()
            .write_batch(&[1.5, 9.9], Some(&[1, 0, 1]), None)
            .unwrap();
        col.close().unwrap();

        // vacia: null in every row.
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<Int64Type>()
            .write_batch(&[], Some(&[0, 0, 0]), None)
            .unwrap();
        col.close().unwrap();

        rg.close().unwrap();
        w.close().unwrap();
    }

    #[test]
    fn read_geoparquet_reads_nullable_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nullable.parquet");
        write_nullable_fixture(&path, false);

        let fc = read_geoparquet(&path).unwrap();
        assert_eq!(fc.len(), 3);

        // Geometry types survive alongside the nullable columns.
        assert!(matches!(
            fc.features[0].geometry,
            Some(geo::Geometry::LineString(_))
        ));
        assert!(matches!(
            fc.features[1].geometry,
            Some(geo::Geometry::Point(_))
        ));

        let est = |i: usize| fc.features[i].get_property("estacion").unwrap();
        assert_eq!(est(0), &AttributeValue::Null); // null in the FIRST row
        assert_eq!(est(1), &AttributeValue::String("EST-7".into()));
        assert_eq!(est(2), &AttributeValue::String("EST-9".into()));

        let val = |i: usize| fc.features[i].get_property("valor").unwrap();
        assert_eq!(val(0), &AttributeValue::Float(1.5));
        assert_eq!(val(1), &AttributeValue::Null);
        assert_eq!(val(2), &AttributeValue::Float(9.9));

        // An all-null column still yields the property, as Null everywhere.
        for i in 0..3 {
            assert_eq!(
                fc.features[i].get_property("vacia").unwrap(),
                &AttributeValue::Null
            );
        }
    }

    #[test]
    fn read_geoparquet_points_rejects_nulls_pointing_to_the_nullable_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nullable_pts.parquet");
        write_nullable_fixture(&path, true);

        let err = read_geoparquet_points(&path).unwrap_err().to_string();
        assert!(
            err.contains("use read_geoparquet"),
            "error should point to the nullable-capable reader, got: {err}"
        );
    }
}
