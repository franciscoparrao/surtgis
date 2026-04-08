//! NetCDF reader for climate datasets (CMIP6, etc.).
//!
//! Reads local NetCDF files with CF Conventions support,
//! extracting 2D `Raster<f64>` slices from 3D (time × lat × lon) datasets.
//!
//! Unlike the [`ZarrReader`](crate::zarr_reader) which reads directly from
//! cloud stores, NetCDF files must be downloaded first and read locally.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use ndarray::Array2;

use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::CRS;

use crate::error::{CloudError, Result};
use crate::tile_index::BBox;
use crate::zarr_cf::CfMetadata;

// Re-use the time types from zarr_reader when zarr feature is also enabled,
// otherwise define locally.
#[cfg(feature = "zarr")]
pub use crate::zarr_reader::{AggMethod, TimeReduction, TimeSelector};

#[cfg(not(feature = "zarr"))]
pub use self::time_types::*;

#[cfg(not(feature = "zarr"))]
mod time_types {
    use chrono::{DateTime, Utc};

    #[derive(Debug, Clone)]
    pub enum TimeReduction {
        Single(TimeSelector),
        Aggregate {
            start: DateTime<Utc>,
            end: DateTime<Utc>,
            method: AggMethod,
        },
    }

    #[derive(Debug, Clone)]
    pub enum TimeSelector {
        Index(usize),
        Nearest(DateTime<Utc>),
        First,
        Last,
    }

    #[derive(Debug, Clone, Copy)]
    pub enum AggMethod {
        Mean,
        Sum,
        Min,
        Max,
    }
}

/// Metadata about an opened NetCDF file.
#[derive(Debug, Clone)]
pub struct NetCdfMetadata {
    pub path: PathBuf,
    pub variable: String,
    pub shape: Vec<usize>,
    pub dimension_names: Vec<String>,
    pub geo_transform: GeoTransform,
    pub crs: Option<CRS>,
    pub nodata: Option<f64>,
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub available_variables: Vec<String>,
}

/// Reader for local NetCDF files with CF Conventions support.
pub struct NetCdfReader {
    path: PathBuf,
    variable: String,
    cf: CfMetadata,
    lat_coords: Vec<f64>,
    lon_coords: Vec<f64>,
    time_coords: Vec<DateTime<Utc>>,
    lat_descending: bool,
    metadata: NetCdfMetadata,
}

impl NetCdfReader {
    /// Open a NetCDF file and select a variable for reading.
    pub fn open(path: &Path, variable: &str) -> Result<Self> {
        let file = netcdf::open(path)
            .map_err(|e| CloudError::NetCdf(format!("failed to open {}: {e}", path.display())))?;

        // List available variables
        let available_variables: Vec<String> = file
            .variables()
            .map(|v| v.name().to_string())
            .collect();

        // Open target variable
        let var = file.variable(variable).ok_or_else(|| {
            CloudError::NetCdf(format!(
                "variable '{}' not found. Available: [{}]",
                variable,
                available_variables.join(", ")
            ))
        })?;

        // Get dimensions
        let dimension_names: Vec<String> = var.dimensions().iter().map(|d| d.name().to_string()).collect();
        let shape: Vec<usize> = var.dimensions().iter().map(|d| d.len()).collect();

        // Build CF metadata from variable attributes
        let array_attrs = extract_attributes(&var);
        let group_attrs = extract_global_attributes(&file);
        let cf = CfMetadata::from_zarr_attributes(&array_attrs, &group_attrs, &dimension_names);

        // Identify coordinate dimensions
        let lat_dim = cf.lat_dim.ok_or_else(|| {
            CloudError::NetCdf("cannot identify latitude dimension".into())
        })?;
        let lon_dim = cf.lon_dim.ok_or_else(|| {
            CloudError::NetCdf("cannot identify longitude dimension".into())
        })?;

        // Read coordinate arrays
        let raw_lat = read_coord(&file, &dimension_names[lat_dim], &["latitude", "lat", "y"])?;
        let raw_lon = read_coord(&file, &dimension_names[lon_dim], &["longitude", "lon", "x"])?;

        let lat_descending = raw_lat.len() >= 2 && raw_lat[0] > raw_lat[raw_lat.len() - 1];
        let lat_coords = if lat_descending {
            raw_lat.iter().rev().copied().collect()
        } else {
            raw_lat
        };
        let lon_coords = normalise_longitude(raw_lon);

        // Read time coordinates
        let time_coords = if let Some(time_dim_idx) = cf.time_dim {
            let time_name = &dimension_names[time_dim_idx];
            let raw_time = read_coord(&file, time_name, &["time", "t"]).unwrap_or_default();
            if !raw_time.is_empty() {
                // Get time CF metadata from the time variable itself
                let time_cf = file.variable(time_name)
                    .or_else(|| file.variable("time"))
                    .map(|tv| {
                        let ta = extract_attributes(&tv);
                        CfMetadata::from_zarr_attributes(&ta, &serde_json::json!({}), &[])
                    });
                let decoder = time_cf.as_ref().unwrap_or(&cf);
                decoder.decode_time(&raw_time).unwrap_or_default()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let geo_transform = build_geotransform(&lat_coords, &lon_coords);
        let time_range = match time_coords.len() {
            0 => None,
            1 => Some((time_coords[0], time_coords[0])),
            n => Some((time_coords[0], time_coords[n - 1])),
        };

        let metadata = NetCdfMetadata {
            path: path.to_path_buf(),
            variable: variable.to_string(),
            shape,
            dimension_names: dimension_names.clone(),
            geo_transform,
            crs: Some(CRS::wgs84()),
            nodata: cf.fill_value,
            time_range,
            available_variables,
        };

        Ok(Self {
            path: path.to_path_buf(),
            variable: variable.to_string(),
            cf,
            lat_coords,
            lon_coords,
            time_coords,
            lat_descending,
            metadata,
        })
    }

    /// Read a geographic bounding box at a specific time.
    pub fn read_bbox(&self, bbox: &BBox, time: &TimeReduction) -> Result<Raster<f64>> {
        let file = netcdf::open(&self.path)
            .map_err(|e| CloudError::NetCdf(format!("failed to reopen: {e}")))?;
        let var = file.variable(&self.variable).ok_or_else(|| {
            CloudError::NetCdf(format!("variable '{}' gone", self.variable))
        })?;

        let (lat_start, lat_end) = self.lat_range_for_bbox(bbox)?;
        let (lon_start, lon_end) = self.lon_range_for_bbox(bbox)?;
        let lat_size = lat_end - lat_start;
        let lon_size = lon_end - lon_start;

        let lat_dim = self.cf.lat_dim.unwrap();
        let lon_dim = self.cf.lon_dim.unwrap();
        let ndim = self.metadata.shape.len();

        // Build indices and sizes for the subset read
        let (time_start, time_count) = if let Some(time_dim) = self.cf.time_dim {
            self.time_indices(time)?
        } else {
            (0, 1)
        };

        // Read the subset - handle 3D [time, lat, lon] case
        let values = if ndim == 3 && self.cf.time_dim == Some(0) && lat_dim == 1 && lon_dim == 2 {
            // Fast path: [time, lat, lon]
            let mut buf = vec![0f32; time_count * lat_size * lon_size];
            var.get_values_into(
                buf.as_mut_slice(),
                ([time_start, lat_start, lon_start], [time_count, lat_size, lon_size]),
            )
            .map_err(|e| CloudError::NetCdf(format!("read failed: {e}")))?;
            buf.iter().map(|&v| v as f64).collect::<Vec<f64>>()
        } else if ndim == 2 {
            // 2D [lat, lon] — no time dimension
            let mut buf = vec![0f32; lat_size * lon_size];
            var.get_values_into(
                buf.as_mut_slice(),
                ([lat_start, lon_start], [lat_size, lon_size]),
            )
            .map_err(|e| CloudError::NetCdf(format!("read failed: {e}")))?;
            buf.iter().map(|&v| v as f64).collect::<Vec<f64>>()
        } else {
            return Err(CloudError::NetCdf(format!(
                "unsupported dimensionality: {} dims {:?}",
                ndim, self.metadata.dimension_names
            )));
        };

        // Reduce 3D to 2D
        let data_2d = reduce_to_2d(&values, time_count, lat_size, lon_size, time, self.cf.fill_value)?;

        // Flip if lat was originally descending
        let data_2d = if self.lat_descending {
            flip_rows(data_2d)
        } else {
            data_2d
        };

        // Unpack scale/offset and replace fill values
        let data_2d = unpack_data(data_2d, &self.cf);

        // Build GeoTransform for subset
        let sub_lat = &self.lat_coords[lat_start..lat_end];
        let sub_lon = &self.lon_coords[lon_start..lon_end];
        let geo_transform = build_geotransform(sub_lat, sub_lon);

        let mut raster = Raster::from_array(data_2d);
        raster.set_transform(geo_transform);
        raster.set_crs(Some(CRS::wgs84()));
        raster.set_nodata(self.cf.fill_value);

        Ok(raster)
    }

    /// Return metadata.
    pub fn metadata(&self) -> &NetCdfMetadata {
        &self.metadata
    }

    /// List all variables in a NetCDF file.
    pub fn list_variables(path: &Path) -> Result<Vec<String>> {
        let file = netcdf::open(path)
            .map_err(|e| CloudError::NetCdf(format!("failed to open: {e}")))?;
        Ok(file.variables().map(|v| v.name().to_string()).collect())
    }

    // ── Internals ────────────────────────────────────────────────────

    fn lat_range_for_bbox(&self, bbox: &BBox) -> Result<(usize, usize)> {
        let start = find_nearest(&self.lat_coords, bbox.min_y);
        let end = (find_nearest(&self.lat_coords, bbox.max_y) + 1).min(self.lat_coords.len());
        if start >= end {
            return Err(CloudError::BBoxOutside);
        }
        if self.lat_descending {
            let n = self.lat_coords.len();
            Ok((n - end, n - start))
        } else {
            Ok((start, end))
        }
    }

    fn lon_range_for_bbox(&self, bbox: &BBox) -> Result<(usize, usize)> {
        let start = find_nearest(&self.lon_coords, bbox.min_x);
        let end = (find_nearest(&self.lon_coords, bbox.max_x) + 1).min(self.lon_coords.len());
        if start >= end {
            return Err(CloudError::BBoxOutside);
        }
        Ok((start, end))
    }

    fn time_indices(&self, time: &TimeReduction) -> Result<(usize, usize)> {
        match time {
            TimeReduction::Single(sel) => {
                let idx = self.resolve_time_selector(sel)?;
                Ok((idx, 1))
            }
            TimeReduction::Aggregate { start, end, .. } => {
                let si = find_nearest_time(&self.time_coords, start);
                let ei = find_nearest_time(&self.time_coords, end);
                if si > ei {
                    return Err(CloudError::NetCdf(format!(
                        "time range {start} to {end} out of range"
                    )));
                }
                Ok((si, ei - si + 1))
            }
        }
    }

    fn resolve_time_selector(&self, sel: &TimeSelector) -> Result<usize> {
        if self.time_coords.is_empty() {
            return Ok(0);
        }
        match sel {
            TimeSelector::Index(i) => {
                if *i >= self.time_coords.len() {
                    Err(CloudError::NetCdf(format!("time index {} out of range", i)))
                } else {
                    Ok(*i)
                }
            }
            TimeSelector::Nearest(dt) => Ok(find_nearest_time(&self.time_coords, dt)),
            TimeSelector::First => Ok(0),
            TimeSelector::Last => Ok(self.time_coords.len() - 1),
        }
    }
}

// ─── Free functions ──────────────────────────────────────────────────

/// Extract variable attributes as serde_json::Value for CfMetadata parsing.
fn extract_attributes(var: &netcdf::Variable) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for attr in var.attributes() {
        if let Ok(val) = attr.value() {
            let json_val = attr_value_to_json(&val);
            map.insert(attr.name().to_string(), json_val);
        }
    }
    serde_json::Value::Object(map)
}

/// Extract global file attributes.
fn extract_global_attributes(file: &netcdf::File) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for attr in file.attributes() {
        if let Ok(val) = attr.value() {
            let json_val = attr_value_to_json(&val);
            map.insert(attr.name().to_string(), json_val);
        }
    }
    serde_json::Value::Object(map)
}

/// Convert a netcdf attribute value to serde_json::Value.
fn attr_value_to_json(val: &netcdf::AttributeValue) -> serde_json::Value {
    use netcdf::AttributeValue;
    match val {
        AttributeValue::Str(s) => serde_json::Value::String(s.to_string()),
        AttributeValue::Double(v) => serde_json::json!(v),
        AttributeValue::Float(v) => serde_json::json!(v),
        AttributeValue::Int(v) => serde_json::json!(v),
        AttributeValue::Short(v) => serde_json::json!(v),
        AttributeValue::Schar(v) => serde_json::json!(v),
        AttributeValue::Uint(v) => serde_json::json!(v),
        AttributeValue::Ushort(v) => serde_json::json!(v),
        AttributeValue::Uchar(v) => serde_json::json!(v),
        AttributeValue::Longlong(v) => serde_json::json!(v),
        AttributeValue::Ulonglong(v) => serde_json::json!(v),
        AttributeValue::Doubles(v) => serde_json::json!(v),
        AttributeValue::Floats(v) => serde_json::json!(v),
        AttributeValue::Ints(v) => serde_json::json!(v),
        AttributeValue::Shorts(v) => serde_json::json!(v),
        AttributeValue::Schars(v) => serde_json::json!(v),
        AttributeValue::Uints(v) => serde_json::json!(v),
        AttributeValue::Ushorts(v) => serde_json::json!(v),
        AttributeValue::Uchars(v) => serde_json::json!(v),
        AttributeValue::Longlongs(v) => serde_json::json!(v),
        AttributeValue::Ulonglongs(v) => serde_json::json!(v),
        AttributeValue::Strs(v) => {
            let strs: Vec<String> = v.iter().map(|s| s.to_string()).collect();
            serde_json::json!(strs)
        }
    }
}

/// Read a coordinate variable, trying primary name then fallbacks.
fn read_coord(file: &netcdf::File, primary: &str, fallbacks: &[&str]) -> Result<Vec<f64>> {
    let names: Vec<&str> = std::iter::once(primary)
        .chain(fallbacks.iter().copied())
        .collect();

    for name in &names {
        if let Some(var) = file.variable(name) {
            let len: usize = var.dimensions().iter().map(|d| d.len()).product();
            // Try f64 first
            let mut buf_f64 = vec![0f64; len];
            if var.get_values_into(buf_f64.as_mut_slice(), ..).is_ok() {
                return Ok(buf_f64);
            }
            // Try f32
            let mut buf_f32 = vec![0f32; len];
            if var.get_values_into(buf_f32.as_mut_slice(), ..).is_ok() {
                return Ok(buf_f32.iter().map(|&v| v as f64).collect());
            }
        }
    }
    Err(CloudError::NetCdf(format!(
        "coordinate not found: tried {:?}",
        names
    )))
}

fn normalise_longitude(lon: Vec<f64>) -> Vec<f64> {
    if lon.is_empty() || !lon.iter().any(|&v| v > 180.0) {
        return lon;
    }
    let mut out: Vec<f64> = lon.iter().map(|&v| if v > 180.0 { v - 360.0 } else { v }).collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

fn build_geotransform(lat: &[f64], lon: &[f64]) -> GeoTransform {
    if lat.len() < 2 || lon.len() < 2 {
        return GeoTransform::new(
            lon.first().copied().unwrap_or(0.0),
            lat.last().copied().unwrap_or(0.0),
            1.0,
            -1.0,
        );
    }
    let pixel_width = (lon[lon.len() - 1] - lon[0]) / (lon.len() - 1) as f64;
    let lat_step = (lat[lat.len() - 1] - lat[0]) / (lat.len() - 1) as f64;
    let origin_x = lon[0] - pixel_width / 2.0;
    let origin_y = lat[lat.len() - 1] + lat_step / 2.0;
    GeoTransform::new(origin_x, origin_y, pixel_width, -lat_step)
}

fn find_nearest(coords: &[f64], value: f64) -> usize {
    match coords.binary_search_by(|c| c.partial_cmp(&value).unwrap()) {
        Ok(i) => i,
        Err(i) => {
            if i == 0 { 0 }
            else if i >= coords.len() { coords.len() - 1 }
            else if (coords[i] - value).abs() < (coords[i - 1] - value).abs() { i }
            else { i - 1 }
        }
    }
}

fn find_nearest_time(coords: &[DateTime<Utc>], target: &DateTime<Utc>) -> usize {
    coords
        .iter()
        .enumerate()
        .min_by_key(|(_, t)| (*target - **t).num_seconds().abs())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn flip_rows(data: Array2<f64>) -> Array2<f64> {
    let nrows = data.nrows();
    let ncols = data.ncols();
    let mut flipped = Array2::zeros((nrows, ncols));
    for r in 0..nrows {
        flipped.row_mut(r).assign(&data.row(nrows - 1 - r));
    }
    flipped
}

fn unpack_data(mut data: Array2<f64>, cf: &CfMetadata) -> Array2<f64> {
    let has_transform = cf.scale_factor.is_some() || cf.add_offset.is_some();
    if !has_transform && cf.fill_value.is_none() {
        return data;
    }
    data.mapv_inplace(|v| {
        if cf.is_fill(v) { return f64::NAN; }
        cf.unpack_value(v)
    });
    data
}

/// Reduce [time × lat × lon] to [lat × lon] via aggregation or single-step selection.
fn reduce_to_2d(
    values: &[f64],
    time_count: usize,
    lat_size: usize,
    lon_size: usize,
    time: &TimeReduction,
    fill_value: Option<f64>,
) -> Result<Array2<f64>> {
    let spatial = lat_size * lon_size;

    if time_count == 1 {
        let slice = &values[..spatial];
        return Array2::from_shape_vec((lat_size, lon_size), slice.to_vec())
            .map_err(|e| CloudError::NetCdf(format!("reshape: {e}")));
    }

    match time {
        TimeReduction::Single(_) => {
            let slice = &values[..spatial];
            Array2::from_shape_vec((lat_size, lon_size), slice.to_vec())
                .map_err(|e| CloudError::NetCdf(format!("reshape: {e}")))
        }
        TimeReduction::Aggregate { method, .. } => {
            let mut out = Array2::zeros((lat_size, lon_size));
            for lat_i in 0..lat_size {
                for lon_i in 0..lon_size {
                    let pix = lat_i * lon_size + lon_i;
                    let mut valid = Vec::with_capacity(time_count);
                    for t in 0..time_count {
                        let v = values[t * spatial + pix];
                        if !is_fill(v, fill_value) {
                            valid.push(v);
                        }
                    }
                    out[[lat_i, lon_i]] = if valid.is_empty() {
                        f64::NAN
                    } else {
                        match method {
                            AggMethod::Mean => valid.iter().sum::<f64>() / valid.len() as f64,
                            AggMethod::Sum => valid.iter().sum(),
                            AggMethod::Min => valid.iter().copied().fold(f64::INFINITY, f64::min),
                            AggMethod::Max => valid.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        }
                    };
                }
            }
            Ok(out)
        }
    }
}

fn is_fill(v: f64, fill_value: Option<f64>) -> bool {
    match fill_value {
        Some(fv) => v == fv || (fv.is_nan() && v.is_nan()),
        None => false,
    }
}
