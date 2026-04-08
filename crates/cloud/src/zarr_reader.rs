//! Zarr dataset reader for climate data (ERA5, TerraClimate, etc.).
//!
//! Reads multi-dimensional Zarr stores via HTTP or Azure Blob Storage,
//! slicing time/lat/lon dimensions and returning 2D `Raster<f64>`.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use ndarray::Array2;
use zarrs::array::{Array, ArraySubset};
use zarrs::config::MetadataRetrieveVersion;
use zarrs::group::Group;
use zarrs_storage::AsyncReadableListableStorageTraits;

use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::CRS;

use crate::error::{CloudError, Result};
use crate::tile_index::BBox;
use crate::zarr_auth;
use crate::zarr_cf::CfMetadata;

// ─── Public types ────────────────────────────────────────────────────

/// Options for configuring a [`ZarrReader`].
#[derive(Debug, Clone, Default)]
pub struct ZarrReaderOptions {
    /// SAS token query string for Azure Blob (Planetary Computer).
    pub sas_token: Option<String>,
}

/// How to select / reduce the time dimension to produce a 2D raster.
#[derive(Debug, Clone)]
pub enum TimeReduction {
    /// Pick a single time step.
    Single(TimeSelector),
    /// Aggregate a time range with a function.
    Aggregate {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        method: AggMethod,
    },
}

/// Selector for a single time step.
#[derive(Debug, Clone)]
pub enum TimeSelector {
    /// Absolute index into the time dimension.
    Index(usize),
    /// Nearest time step to the given datetime.
    Nearest(DateTime<Utc>),
    /// First time step.
    First,
    /// Last time step.
    Last,
}

/// Aggregation method for time ranges.
#[derive(Debug, Clone, Copy)]
pub enum AggMethod {
    Mean,
    Sum,
    Min,
    Max,
}

/// Metadata about a Zarr store and the selected variable.
#[derive(Debug, Clone)]
pub struct ZarrMetadata {
    pub store_url: String,
    pub variable: String,
    /// Array shape, e.g. `[8760, 721, 1440]` for `[time, lat, lon]`.
    pub shape: Vec<u64>,
    /// Dimension names, e.g. `["time", "latitude", "longitude"]`.
    pub dimension_names: Vec<String>,
    pub geo_transform: GeoTransform,
    pub crs: Option<CRS>,
    pub nodata: Option<f64>,
    /// First and last decoded time step (if time dimension exists).
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// All variable names found in the store.
    pub available_variables: Vec<String>,
}

// ─── ZarrReader ──────────────────────────────────────────────────────

/// Async reader for Zarr climate datasets.
pub struct ZarrReader {
    store: Arc<dyn AsyncReadableListableStorageTraits>,
    array: Array<dyn AsyncReadableListableStorageTraits>,
    cf: CfMetadata,
    /// Latitude coordinate values (always sorted ascending internally).
    lat_coords: Vec<f64>,
    /// Longitude coordinate values (normalised to -180..180).
    lon_coords: Vec<f64>,
    /// Decoded time coordinate values.
    time_coords: Vec<DateTime<Utc>>,
    /// True if the original lat array was descending (N → S, e.g. ERA5).
    lat_descending: bool,
    metadata: ZarrMetadata,
}

impl ZarrReader {
    /// Open a Zarr store and select a variable for reading.
    pub async fn open(
        store_url: &str,
        variable: &str,
        options: ZarrReaderOptions,
    ) -> Result<Self> {
        let store = zarr_auth::build_zarr_store(
            store_url,
            options.sas_token.as_deref(),
        )
        .await?;

        // Open root group (try default → V2 fallback)
        let group = match Group::async_open(store.clone(), "/").await {
            Ok(g) => g,
            Err(_) => Group::async_open_opt(store.clone(), "/", &MetadataRetrieveVersion::V2)
                .await
                .map_err(|e| CloudError::Zarr(format!("failed to open Zarr group: {e}")))?,
        };

        let group_attrs = serde_json::Value::Object(group.attributes().clone());

        // List available variables (may be empty if store doesn't support listing)
        let available_variables = Self::list_arrays(&store).await.unwrap_or_default();

        // Open data array (try default → V2 fallback)
        let array_path = format!("/{variable}");
        let array = match Array::async_open(store.clone(), &array_path).await {
            Ok(a) => a,
            Err(_) => {
                // Retry with explicit V2 (common for climate Zarr stores)
                Array::async_open_opt(store.clone(), &array_path, &MetadataRetrieveVersion::V2)
                    .await
                    .map_err(|e| CloudError::Zarr(format!(
                        "failed to open array '/{variable}': {e}. Available: [{}]",
                        available_variables.join(", ")
                    )))?
            }
        };

        // Dimension names
        // DimensionName = Option<String>, so we unwrap each name
        let dimension_names: Vec<String> = array
            .dimension_names()
            .as_ref()
            .map(|dn| {
                dn.iter()
                    .enumerate()
                    .map(|(i, d)| {
                        d.as_ref()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("dim_{i}"))
                    })
                    .collect()
            })
            .unwrap_or_else(|| default_dimension_names(array.dimensionality()));

        // CF metadata
        let array_attrs = serde_json::Value::Object(array.attributes().clone());
        let cf = CfMetadata::from_zarr_attributes(&array_attrs, &group_attrs, &dimension_names);

        // Read coordinate arrays
        let lat_dim = cf.lat_dim.ok_or_else(|| {
            CloudError::ZarrCfError("cannot identify latitude dimension".into())
        })?;
        let lon_dim = cf.lon_dim.ok_or_else(|| {
            CloudError::ZarrCfError("cannot identify longitude dimension".into())
        })?;

        // Read coordinate arrays (try dimension name, then common aliases)
        let raw_lat = read_coord_with_fallbacks(
            &store,
            &dimension_names[lat_dim],
            &["latitude", "lat", "y"],
        )
        .await?;
        let raw_lon = read_coord_with_fallbacks(
            &store,
            &dimension_names[lon_dim],
            &["longitude", "lon", "x"],
        )
        .await?;

        let lat_descending = raw_lat.len() >= 2 && raw_lat[0] > raw_lat[raw_lat.len() - 1];
        let lat_coords = if lat_descending {
            raw_lat.iter().rev().copied().collect()
        } else {
            raw_lat
        };
        let lon_coords = normalise_longitude(raw_lon);

        // Time coordinates: read time array and its own CF attributes
        let time_coords = if let Some(time_dim_idx) = cf.time_dim {
            let time_names = &["time", "t", "datetime"];
            let time_name = &dimension_names[time_dim_idx];

            match read_coord_with_fallbacks(&store, time_name, time_names).await {
                Ok(raw) => {
                    // Read time array's own attributes for units/calendar
                    let time_cf = read_time_cf_metadata(&store, time_name, time_names).await;
                    let decoder = time_cf.as_ref().unwrap_or(&cf);
                    match decoder.decode_time(&raw) {
                        Ok(decoded) => decoded,
                        Err(_) => Vec::new(),
                    }
                }
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        };

        let geo_transform = build_geotransform(&lat_coords, &lon_coords);
        let nodata = cf.fill_value;
        let time_range = match time_coords.len() {
            0 => None,
            1 => Some((time_coords[0], time_coords[0])),
            n => Some((time_coords[0], time_coords[n - 1])),
        };

        let metadata = ZarrMetadata {
            store_url: store_url.to_string(),
            variable: variable.to_string(),
            shape: array.shape().to_vec(),
            dimension_names: dimension_names.clone(),
            geo_transform,
            crs: Some(CRS::wgs84()),
            nodata,
            time_range,
            available_variables,
        };

        Ok(Self {
            store,
            array,
            cf,
            lat_coords,
            lon_coords,
            time_coords,
            lat_descending,
            metadata,
        })
    }

    /// Read a geographic bounding box at a specific time.
    pub async fn read_bbox(&self, bbox: &BBox, time: &TimeReduction) -> Result<Raster<f64>> {
        let (lat_start, lat_end) = self.lat_range_for_bbox(bbox)?;
        let (lon_start, lon_end) = self.lon_range_for_bbox(bbox)?;

        let lat_dim = self.cf.lat_dim.unwrap();
        let lon_dim = self.cf.lon_dim.unwrap();
        let ndim = self.array.dimensionality();

        let mut starts = vec![0u64; ndim];
        let mut sizes = vec![0u64; ndim];

        starts[lat_dim] = lat_start as u64;
        sizes[lat_dim] = (lat_end - lat_start) as u64;
        starts[lon_dim] = lon_start as u64;
        sizes[lon_dim] = (lon_end - lon_start) as u64;

        let time_count = if let Some(time_dim) = self.cf.time_dim {
            let (ts, tc) = self.time_indices(time)?;
            starts[time_dim] = ts as u64;
            sizes[time_dim] = tc as u64;
            tc
        } else {
            1
        };

        // Fill remaining dims with full extent
        for i in 0..ndim {
            if sizes[i] == 0 {
                sizes[i] = self.array.shape()[i];
            }
        }

        let subset = ArraySubset::new_with_start_shape(starts.clone(), sizes.clone())
            .map_err(|e| CloudError::Zarr(format!("invalid subset: {e}")))?;

        // Fetch values — try f64 first, then f32 (climate data often uses float32)
        let values: Vec<f64> = match self
            .array
            .async_retrieve_array_subset::<Vec<f64>>(&subset)
            .await
        {
            Ok(v) => v,
            Err(_) => {
                let v32: Vec<f32> = self
                    .array
                    .async_retrieve_array_subset(&subset)
                    .await
                    .map_err(|e| CloudError::Zarr(format!("failed to read subset: {e}")))?;
                v32.iter().map(|&v| v as f64).collect()
            }
        };

        // Build shape array for manual indexing
        let shape: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();

        // Reduce to 2D
        let data_2d = reduce_to_2d(
            &values,
            &shape,
            lat_dim,
            lon_dim,
            self.cf.time_dim,
            time,
            time_count,
            self.cf.fill_value,
        )?;

        // Flip if lat was originally descending
        let data_2d = if self.lat_descending {
            flip_rows(data_2d)
        } else {
            data_2d
        };

        // Apply scale/offset and replace fill values
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

    /// Read the full spatial extent at a specific time.
    pub async fn read_full(&self, time: &TimeReduction) -> Result<Raster<f64>> {
        let full_bbox = BBox {
            min_x: *self.lon_coords.first().unwrap_or(&-180.0),
            max_x: *self.lon_coords.last().unwrap_or(&180.0),
            min_y: *self.lat_coords.first().unwrap_or(&-90.0),
            max_y: *self.lat_coords.last().unwrap_or(&90.0),
        };
        self.read_bbox(&full_bbox, time).await
    }

    /// Return metadata.
    pub fn metadata(&self) -> &ZarrMetadata {
        &self.metadata
    }

    /// List all variables in a store (static, no open needed).
    pub async fn list_variables(
        store_url: &str,
        options: ZarrReaderOptions,
    ) -> Result<Vec<String>> {
        let store =
            zarr_auth::build_zarr_store(store_url, options.sas_token.as_deref()).await?;
        Self::list_arrays(&store).await
    }

    // ── Internals ────────────────────────────────────────────────────

    async fn list_arrays(
        store: &Arc<dyn AsyncReadableListableStorageTraits>,
    ) -> Result<Vec<String>> {
        let group = Group::async_open(store.clone(), "/")
            .await
            .map_err(|e| CloudError::Zarr(format!("failed to open root group: {e}")))?;

        let children = group
            .async_children(false)
            .await
            .map_err(|e| CloudError::Zarr(format!("failed to list children: {e}")))?;

        let mut arrays = Vec::new();
        for node in &children {
            let path = node.path().as_str();
            let name = path.trim_start_matches('/');
            if Array::async_open(store.clone(), path).await.is_ok() {
                arrays.push(name.to_string());
            }
        }
        Ok(arrays)
    }

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
                    return Err(CloudError::ZarrTimeOutOfRange {
                        requested: format!("{start} to {end}"),
                        available: format!(
                            "{} to {}",
                            self.time_coords.first().map(|t| t.to_string()).unwrap_or_default(),
                            self.time_coords.last().map(|t| t.to_string()).unwrap_or_default(),
                        ),
                    });
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
                    Err(CloudError::ZarrTimeOutOfRange {
                        requested: format!("index {i}"),
                        available: format!("0..{}", self.time_coords.len()),
                    })
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

/// Read the CF metadata from a time coordinate array's attributes.
async fn read_time_cf_metadata(
    store: &Arc<dyn AsyncReadableListableStorageTraits>,
    primary: &str,
    fallbacks: &[&str],
) -> Option<CfMetadata> {
    let names: Vec<&str> = std::iter::once(primary)
        .chain(fallbacks.iter().copied())
        .collect();

    for name in names {
        let path = format!("/{name}");
        let arr = Array::async_open(store.clone(), &path)
            .await
            .or_else(|_| {
                // V2 fallback in a sync-compatible way — try sync approach
                futures::executor::block_on(Array::async_open_opt(
                    store.clone(),
                    &path,
                    &MetadataRetrieveVersion::V2,
                ))
            })
            .ok()?;

        let attrs = serde_json::Value::Object(arr.attributes().clone());
        // Check if this array has time units
        if attrs.get("units").and_then(|u| u.as_str()).is_some_and(|u| u.contains("since")) {
            let cf = CfMetadata::from_zarr_attributes(&attrs, &serde_json::json!({}), &[]);
            return Some(cf);
        }
    }
    None
}

/// Read a coordinate array, trying the given name and then fallback aliases.
async fn read_coord_with_fallbacks(
    store: &Arc<dyn AsyncReadableListableStorageTraits>,
    primary: &str,
    fallbacks: &[&str],
) -> Result<Vec<f64>> {
    if let Ok(v) = read_coord_array(store, primary).await {
        return Ok(v);
    }
    for name in fallbacks {
        if *name != primary {
            if let Ok(v) = read_coord_array(store, name).await {
                return Ok(v);
            }
        }
    }
    Err(CloudError::ZarrCfError(format!(
        "coordinate array not found: tried '{}' and {:?}",
        primary, fallbacks
    )))
}

/// Read a 1D coordinate array by name. Handles both f32 and f64 dtypes.
async fn read_coord_array(
    store: &Arc<dyn AsyncReadableListableStorageTraits>,
    name: &str,
) -> Result<Vec<f64>> {
    let path = format!("/{name}");
    let arr = match Array::async_open(store.clone(), &path).await {
        Ok(a) => a,
        Err(_) => Array::async_open_opt(store.clone(), &path, &MetadataRetrieveVersion::V2)
            .await
            .map_err(|e| CloudError::ZarrCfError(format!("failed to open coord '{name}': {e}")))?,
    };

    let subset = arr.subset_all();

    // Try f64 first, then f32 (climate data often uses float32 for coordinates)
    if let Ok(values) = arr.async_retrieve_array_subset::<Vec<f64>>(&subset).await {
        return Ok(values);
    }

    let values_f32: Vec<f32> = arr
        .async_retrieve_array_subset(&subset)
        .await
        .map_err(|e| CloudError::ZarrCfError(format!("failed to read coord '{name}': {e}")))?;

    Ok(values_f32.iter().map(|&v| v as f64).collect())
}

/// Normalise longitude from 0-360 to -180..180.
fn normalise_longitude(lon: Vec<f64>) -> Vec<f64> {
    if lon.is_empty() || !lon.iter().any(|&v| v > 180.0) {
        return lon;
    }
    let mut out: Vec<f64> = lon
        .iter()
        .map(|&v| if v > 180.0 { v - 360.0 } else { v })
        .collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

/// Build a GeoTransform from sorted lat/lon coordinate arrays.
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

    // Origin: top-left corner (half pixel offset from first center)
    let origin_x = lon[0] - pixel_width / 2.0;
    let origin_y = lat[lat.len() - 1] + lat_step / 2.0;

    GeoTransform::new(origin_x, origin_y, pixel_width, -lat_step)
}

fn find_nearest(coords: &[f64], value: f64) -> usize {
    match coords.binary_search_by(|c| c.partial_cmp(&value).unwrap()) {
        Ok(i) => i,
        Err(i) => {
            if i == 0 {
                0
            } else if i >= coords.len() {
                coords.len() - 1
            } else if (coords[i] - value).abs() < (coords[i - 1] - value).abs() {
                i
            } else {
                i - 1
            }
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

fn default_dimension_names(ndim: usize) -> Vec<String> {
    match ndim {
        3 => vec!["time".into(), "latitude".into(), "longitude".into()],
        2 => vec!["latitude".into(), "longitude".into()],
        _ => (0..ndim).map(|i| format!("dim_{i}")).collect(),
    }
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

/// Apply scale_factor/add_offset and replace fill values with NaN.
fn unpack_data(mut data: Array2<f64>, cf: &CfMetadata) -> Array2<f64> {
    let has_transform = cf.scale_factor.is_some() || cf.add_offset.is_some();
    if !has_transform && cf.fill_value.is_none() {
        return data;
    }

    data.mapv_inplace(|v| {
        if cf.is_fill(v) {
            return f64::NAN;
        }
        cf.unpack_value(v)
    });
    data
}

/// Reduce a flat buffer with known shape to a 2D lat×lon Array2.
///
/// For a 3D array `[time, lat, lon]`, either picks a single time step or
/// aggregates across time with the given method.
fn reduce_to_2d(
    values: &[f64],
    shape: &[usize],
    lat_dim: usize,
    lon_dim: usize,
    time_dim: Option<usize>,
    time: &TimeReduction,
    time_count: usize,
    fill_value: Option<f64>,
) -> Result<Array2<f64>> {
    let lat_size = shape[lat_dim];
    let lon_size = shape[lon_dim];

    // For 3D [time, lat, lon] (most common climate data layout)
    if shape.len() == 3 && time_dim == Some(0) && lat_dim == 1 && lon_dim == 2 {
        return reduce_3d_tlatlat(values, time_count, lat_size, lon_size, time, fill_value);
    }

    // Generic case: single time step — extract lat×lon plane
    if time_count == 1 || time_dim.is_none() {
        // Compute strides for row-major layout
        let strides = compute_strides(shape);
        let mut out = Array2::zeros((lat_size, lon_size));

        for lat_i in 0..lat_size {
            for lon_i in 0..lon_size {
                let mut idx = 0;
                for d in 0..shape.len() {
                    let coord = if d == lat_dim {
                        lat_i
                    } else if d == lon_dim {
                        lon_i
                    } else {
                        0 // first index for other dims
                    };
                    idx += coord * strides[d];
                }
                out[[lat_i, lon_i]] = values[idx];
            }
        }
        return Ok(out);
    }

    // Fallback: should not reach here for standard climate data
    Err(CloudError::Zarr(
        "unsupported dimension layout for time aggregation".into(),
    ))
}

/// Fast path for [time, lat, lon] layout.
fn reduce_3d_tlatlat(
    values: &[f64],
    time_count: usize,
    lat_size: usize,
    lon_size: usize,
    time: &TimeReduction,
    fill_value: Option<f64>,
) -> Result<Array2<f64>> {
    let spatial = lat_size * lon_size;

    match time {
        TimeReduction::Single(_) => {
            // First (and only) time step
            let slice = &values[0..spatial];
            Array2::from_shape_vec((lat_size, lon_size), slice.to_vec())
                .map_err(|e| CloudError::Zarr(format!("reshape error: {e}")))
        }
        TimeReduction::Aggregate { method, .. } => {
            let mut out = Array2::zeros((lat_size, lon_size));

            for lat_i in 0..lat_size {
                for lon_i in 0..lon_size {
                    let pix = lat_i * lon_size + lon_i;

                    // Gather values across time for this pixel
                    let mut valid_vals = Vec::with_capacity(time_count);
                    for t in 0..time_count {
                        let v = values[t * spatial + pix];
                        if !is_fill(v, fill_value) {
                            valid_vals.push(v);
                        }
                    }

                    out[[lat_i, lon_i]] = if valid_vals.is_empty() {
                        f64::NAN
                    } else {
                        match method {
                            AggMethod::Mean => {
                                valid_vals.iter().sum::<f64>() / valid_vals.len() as f64
                            }
                            AggMethod::Sum => valid_vals.iter().sum(),
                            AggMethod::Min => valid_vals.iter().copied().fold(f64::INFINITY, f64::min),
                            AggMethod::Max => {
                                valid_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                            }
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

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![1; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
