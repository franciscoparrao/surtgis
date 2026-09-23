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

use surtgis_core::CRS;
use surtgis_core::raster::{GeoTransform, Raster};

use crate::error::{CloudError, Result};
use crate::latlon_grid::{
    build_geotransform, find_nearest, flip_rows, lat_range_ascending, lat_range_raw,
    needs_north_up_flip,
};
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
        /// Inclusive start of the time range.
        start: DateTime<Utc>,
        /// Inclusive end of the time range.
        end: DateTime<Utc>,
        /// Reduction applied across the selected time steps.
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
    /// Arithmetic mean over the time range.
    Mean,
    /// Sum over the time range.
    Sum,
    /// Minimum over the time range.
    Min,
    /// Maximum over the time range.
    Max,
}

/// Per-pixel partial statistics of one time range, mergeable across stores.
///
/// Climate archives are often split into one store per period (ERA5 on
/// Planetary Computer publishes one Zarr store per month), so a single
/// aggregate such as an annual sum spans several stores. Each store yields
/// one partial through [`ZarrReader::read_bbox_partial`]; [`merge`] combines
/// them and [`finish`] derives the requested [`AggMethod`]. Sums and counts
/// merge exactly, so `yearly-sum == Σ monthly-sum` holds by construction,
/// and a mean is always the mean over every time step actually read, never
/// a mean of per-store means.
///
/// Values are already unpacked (`scale_factor`/`add_offset` applied) and
/// fill values excluded, so the accumulators are in physical units.
///
/// [`merge`]: TimeAggPartial::merge
/// [`finish`]: TimeAggPartial::finish
#[derive(Debug, Clone)]
pub struct TimeAggPartial {
    /// Sum of the valid values per pixel (`0.0` where none); carries the
    /// georeferencing shared by every accumulator.
    pub sum: Raster<f64>,
    /// Number of valid time steps per pixel.
    pub count: Array2<f64>,
    /// Minimum valid value per pixel (`+inf` where none).
    pub min: Array2<f64>,
    /// Maximum valid value per pixel (`-inf` where none).
    pub max: Array2<f64>,
    /// Total number of time steps read (valid or not).
    pub time_steps: usize,
}

impl TimeAggPartial {
    /// Fold another partial (same grid, disjoint time steps) into this one.
    ///
    /// Only the grid shape is checked: the caller is responsible for reading
    /// every partial with the same bounding box from stores on the same grid.
    pub fn merge(&mut self, other: &TimeAggPartial) -> Result<()> {
        if self.sum.shape() != other.sum.shape() {
            return Err(CloudError::Zarr(format!(
                "cannot merge time partials of different shapes: {:?} vs {:?}",
                self.sum.shape(),
                other.sum.shape()
            )));
        }
        let (rows, cols) = self.sum.shape();
        for r in 0..rows {
            for c in 0..cols {
                self.sum.data_mut()[[r, c]] += other.sum.data()[[r, c]];
                self.count[[r, c]] += other.count[[r, c]];
                self.min[[r, c]] = self.min[[r, c]].min(other.min[[r, c]]);
                self.max[[r, c]] = self.max[[r, c]].max(other.max[[r, c]]);
            }
        }
        self.time_steps += other.time_steps;
        Ok(())
    }

    /// Derive the final raster for `method`; pixels with no valid step are NaN.
    pub fn finish(&self, method: AggMethod) -> Raster<f64> {
        let (rows, cols) = self.sum.shape();
        let mut out = Array2::from_elem((rows, cols), f64::NAN);
        for r in 0..rows {
            for c in 0..cols {
                let n = self.count[[r, c]];
                if n == 0.0 {
                    continue;
                }
                out[[r, c]] = match method {
                    AggMethod::Mean => self.sum.data()[[r, c]] / n,
                    AggMethod::Sum => self.sum.data()[[r, c]],
                    AggMethod::Min => self.min[[r, c]],
                    AggMethod::Max => self.max[[r, c]],
                };
            }
        }
        let mut raster = Raster::from_array(out);
        raster.set_transform(*self.sum.transform());
        raster.set_crs(self.sum.crs().cloned());
        raster.set_nodata(self.sum.nodata());
        raster
    }
}

/// Metadata about a Zarr store and the selected variable.
#[derive(Debug, Clone)]
pub struct ZarrMetadata {
    /// URL of the Zarr store.
    pub store_url: String,
    /// Name of the selected variable.
    pub variable: String,
    /// Array shape, e.g. `[8760, 721, 1440]` for `[time, lat, lon]`.
    pub shape: Vec<u64>,
    /// Dimension names, e.g. `["time", "latitude", "longitude"]`.
    pub dimension_names: Vec<String>,
    /// Affine transform mapping pixel coordinates to world coordinates.
    pub geo_transform: GeoTransform,
    /// Coordinate reference system, if resolvable.
    pub crs: Option<CRS>,
    /// Nodata / fill value, if declared.
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
    pub async fn open(store_url: &str, variable: &str, options: ZarrReaderOptions) -> Result<Self> {
        let store = zarr_auth::build_zarr_store(store_url, options.sas_token.as_deref()).await?;

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
                    .map_err(|e| {
                        CloudError::Zarr(format!(
                            "failed to open array '/{variable}': {e}. Available: [{}]",
                            available_variables.join(", ")
                        ))
                    })?
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
        let lat_dim = cf
            .lat_dim
            .ok_or_else(|| CloudError::ZarrCfError("cannot identify latitude dimension".into()))?;
        let lon_dim = cf
            .lon_dim
            .ok_or_else(|| CloudError::ZarrCfError("cannot identify longitude dimension".into()))?;

        // Read coordinate arrays (try dimension name, then common aliases)
        let raw_lat =
            read_coord_with_fallbacks(&store, &dimension_names[lat_dim], &["latitude", "lat", "y"])
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
        // Keep longitudes in the dataset's physical order (0-360 for
        // ERA5-style grids): sorting them here would break the mapping
        // between coordinate positions and physical array indices, so a
        // western-hemisphere bbox would read the wrong columns.
        let lon_coords = raw_lon;

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
        if let TimeReduction::Aggregate { start, end, method } = time {
            return Ok(self
                .read_bbox_partial(bbox, start, end)
                .await?
                .finish(*method));
        }

        let sub = self.fetch_subset(bbox, time).await?;
        let data_2d = reduce_to_2d(
            &sub.values,
            &sub.shape,
            sub.lat_dim,
            sub.lon_dim,
            sub.time_count,
        )?;
        let data_2d = self.orient_north_up(data_2d);
        // Apply scale/offset and replace fill values
        let data_2d = unpack_data(data_2d, &self.cf);
        Ok(self.build_raster(data_2d, &sub))
    }

    /// Read a bounding box over `[start, end]` (inclusive) and return the
    /// per-pixel partial statistics instead of a finished raster.
    ///
    /// Use this when one aggregate spans several stores: read one partial
    /// per store, [`TimeAggPartial::merge`] them, then
    /// [`TimeAggPartial::finish`]. Only the time steps whose coordinate
    /// falls inside the range are read; fails with
    /// [`CloudError::ZarrTimeOutOfRange`] when the store has none.
    pub async fn read_bbox_partial(
        &self,
        bbox: &BBox,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
    ) -> Result<TimeAggPartial> {
        let time = TimeReduction::Aggregate {
            start: *start,
            end: *end,
            method: AggMethod::Sum,
        };
        let sub = self.fetch_subset(bbox, &time).await?;

        let layout_ok = (sub.shape.len() == 3
            && self.cf.time_dim == Some(0)
            && sub.lat_dim == 1
            && sub.lon_dim == 2)
            || (sub.shape.len() == 2
                && self.cf.time_dim.is_none()
                && sub.lat_dim == 0
                && sub.lon_dim == 1);
        if !layout_ok {
            return Err(CloudError::Zarr(
                "unsupported dimension layout for time aggregation".into(),
            ));
        }

        let lat_size = sub.shape[sub.lat_dim];
        let lon_size = sub.shape[sub.lon_dim];
        let stats =
            accumulate_time_stats(&sub.values, sub.time_count, lat_size, lon_size, &self.cf);

        let sum = self.build_raster(self.orient_north_up(stats.sum), &sub);
        Ok(TimeAggPartial {
            sum,
            count: self.orient_north_up(stats.count),
            min: self.orient_north_up(stats.min),
            max: self.orient_north_up(stats.max),
            time_steps: sub.time_count,
        })
    }

    /// Fetch the raw array subset for `bbox` and the time steps selected by
    /// `time`. Values are still packed (no scale/offset, fills untouched).
    async fn fetch_subset(&self, bbox: &BBox, time: &TimeReduction) -> Result<Subset> {
        let (lat_start, lat_end) = self.lat_range_for_bbox(bbox)?;
        let (lon_start, lon_end) = self.lon_range_for_bbox(bbox)?;

        let lat_dim = self.cf.lat_dim.unwrap();
        let lon_dim = self.cf.lon_dim.unwrap();
        let ndim = self.array.dimensionality();

        let mut starts = vec![0u64; ndim];
        let mut sizes = vec![0u64; ndim];

        // (lat_start, lat_end) is in ascending `self.lat_coords` space; the
        // on-disk array may store latitude descending (N→S, e.g. ERA5), so
        // map to raw index space only for the array read. Using one pair for
        // both spaces mirrored the output to the opposite hemisphere
        // (BUG_ZARR_CLIMATE_LAT_MIRROR).
        let (raw_lat_start, raw_lat_end) = lat_range_raw(
            (lat_start, lat_end),
            self.lat_coords.len(),
            self.lat_descending,
        );
        starts[lat_dim] = raw_lat_start as u64;
        sizes[lat_dim] = (raw_lat_end - raw_lat_start) as u64;
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
        for (size, &full) in sizes.iter_mut().zip(self.array.shape()) {
            if *size == 0 {
                *size = full;
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

        let shape: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();

        Ok(Subset {
            values,
            shape,
            lat_dim,
            lon_dim,
            lat_start,
            lat_end,
            lon_start,
            lon_end,
            time_count,
        })
    }

    /// The GeoTransform is always north-up (row 0 = northernmost). A
    /// descending source is already north-up in raw order; an ascending one
    /// arrives south-up and needs the flip.
    fn orient_north_up(&self, data_2d: Array2<f64>) -> Array2<f64> {
        if needs_north_up_flip(self.lat_descending) {
            flip_rows(data_2d)
        } else {
            data_2d
        }
    }

    /// Wrap an already north-up, unpacked plane as a georeferenced raster.
    fn build_raster(&self, data_2d: Array2<f64>, sub: &Subset) -> Raster<f64> {
        let sub_lat = &self.lat_coords[sub.lat_start..sub.lat_end];
        // Present western-hemisphere subsets of 0-360 grids in -180..180.
        // Safe only when the whole subset is > 180 (shifting keeps it
        // monotonic); subsets spanning 180 keep the 0-360 convention.
        let sub_lon_raw = &self.lon_coords[sub.lon_start..sub.lon_end];
        let sub_lon: Vec<f64> = if sub_lon_raw.iter().all(|&v| v > 180.0) {
            sub_lon_raw.iter().map(|&v| v - 360.0).collect()
        } else {
            sub_lon_raw.to_vec()
        };
        let geo_transform = build_geotransform(sub_lat, &sub_lon);

        let mut raster = Raster::from_array(data_2d);
        raster.set_transform(geo_transform);
        raster.set_crs(Some(CRS::wgs84()));
        // Fill pixels were replaced with NaN in `unpack_data`, so the
        // declared nodata must be NaN too — declaring the original finite
        // fill would make a subsequent write stamp GDAL_NODATA=<fill> over
        // NaN pixels, and external tools would read the NaNs as valid data.
        raster.set_nodata(self.cf.fill_value.map(|_| f64::NAN));
        raster
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
        let store = zarr_auth::build_zarr_store(store_url, options.sas_token.as_deref()).await?;
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

    /// Latitude index range in **ascending** `self.lat_coords` space.
    /// Map through [`lat_range_raw`] before indexing the raw array.
    fn lat_range_for_bbox(&self, bbox: &BBox) -> Result<(usize, usize)> {
        lat_range_ascending(&self.lat_coords, bbox.min_y, bbox.max_y).ok_or(CloudError::BBoxOutside)
    }

    fn lon_range_for_bbox(&self, bbox: &BBox) -> Result<(usize, usize)> {
        lon_range_physical(&self.lon_coords, bbox.min_x, bbox.max_x)
    }

    fn time_indices(&self, time: &TimeReduction) -> Result<(usize, usize)> {
        match time {
            TimeReduction::Single(sel) => {
                let idx = self.resolve_time_selector(sel)?;
                Ok((idx, 1))
            }
            TimeReduction::Aggregate { start, end, .. } => {
                time_index_range(&self.time_coords, start, end).ok_or_else(|| {
                    CloudError::ZarrTimeOutOfRange {
                        requested: format!("{start} to {end}"),
                        available: format!(
                            "{} to {}",
                            self.time_coords
                                .first()
                                .map(|t| t.to_string())
                                .unwrap_or_default(),
                            self.time_coords
                                .last()
                                .map(|t| t.to_string())
                                .unwrap_or_default(),
                        ),
                    }
                })
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
        let arr = match Array::async_open(store.clone(), &path).await {
            Ok(a) => a,
            // V2 metadata fallback. Awaited normally — a `block_on` here
            // would park the worker on a future that may need this very
            // runtime to make progress (deadlock under the 2-worker
            // blocking wrapper).
            Err(_) => Array::async_open_opt(store.clone(), &path, &MetadataRetrieveVersion::V2)
                .await
                .ok()?,
        };

        let attrs = serde_json::Value::Object(arr.attributes().clone());
        // Check if this array has time units
        if attrs
            .get("units")
            .and_then(|u| u.as_str())
            .is_some_and(|u| u.contains("since"))
        {
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

/// Map a bbox longitude range to physical array indices.
///
/// For 0-360 grids (ERA5 style), negative bbox longitudes are converted
/// to the dataset convention (x + 360) so `find_nearest` runs against the
/// physical coordinate array — indices then address the correct columns.
/// A bbox that crosses the 0/360 seam after conversion (e.g. -10..30 on a
/// 0-360 grid) would need two discontiguous reads; that is rejected with
/// an explicit error rather than returning wrong columns silently.
fn lon_range_physical(lon_coords: &[f64], min_x: f64, max_x: f64) -> Result<(usize, usize)> {
    let is_0_360 = lon_coords.iter().any(|&v| v > 180.0);
    let (mut lo, mut hi) = (min_x, max_x);
    if is_0_360 {
        if lo < 0.0 {
            lo += 360.0;
        }
        if hi < 0.0 {
            hi += 360.0;
        }
        if lo > hi {
            return Err(CloudError::Zarr(format!(
                "bbox longitude range {min_x}..{max_x} crosses the 0/360 seam \
                 of this 0-360 grid; split the request at longitude 0"
            )));
        }
    }
    let start = find_nearest(lon_coords, lo);
    let end = (find_nearest(lon_coords, hi) + 1).min(lon_coords.len());
    if start >= end {
        return Err(CloudError::BBoxOutside);
    }
    Ok((start, end))
}

fn find_nearest_time(coords: &[DateTime<Utc>], target: &DateTime<Utc>) -> usize {
    coords
        .iter()
        .enumerate()
        .min_by_key(|(_, t)| (*target - **t).num_seconds().abs())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Contiguous index range `(first, count)` of the time steps whose
/// coordinate falls inside `[start, end]` (both inclusive), or `None` when no
/// step does. Assumes ascending coordinates, as CF climate stores provide.
///
/// Membership, not nearest-neighbour clamping: a range that starts before the
/// store's first step or ends after its last one picks up only the steps the
/// store actually holds, so several stores covering consecutive periods can
/// be read with the same range and never double count a boundary step.
fn time_index_range(
    coords: &[DateTime<Utc>],
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
) -> Option<(usize, usize)> {
    if coords.is_empty() {
        // No decodable time coordinate: read the single step that exists.
        return Some((0, 1));
    }
    let first = coords.partition_point(|t| t < start);
    let past = coords.partition_point(|t| t <= end);
    if first >= past {
        None
    } else {
        Some((first, past - first))
    }
}

fn default_dimension_names(ndim: usize) -> Vec<String> {
    match ndim {
        3 => vec!["time".into(), "latitude".into(), "longitude".into()],
        2 => vec!["latitude".into(), "longitude".into()],
        _ => (0..ndim).map(|i| format!("dim_{i}")).collect(),
    }
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

/// Raw array subset for one bounding box plus the index ranges that
/// georeference it (ascending-latitude space).
struct Subset {
    values: Vec<f64>,
    shape: Vec<usize>,
    lat_dim: usize,
    lon_dim: usize,
    lat_start: usize,
    lat_end: usize,
    lon_start: usize,
    lon_end: usize,
    time_count: usize,
}

/// Per-pixel accumulators over the time axis, still in raw array row order.
struct TimeStats {
    sum: Array2<f64>,
    count: Array2<f64>,
    min: Array2<f64>,
    max: Array2<f64>,
}

/// Accumulate sum/count/min/max across `time_count` planes of a
/// `[time, lat, lon]` buffer. Fill values are skipped and every value is
/// unpacked (`scale_factor`/`add_offset`) *before* accumulating, so the sum
/// of a packed variable is the sum of physical values, not
/// `scale · Σraw + n · offset`.
fn accumulate_time_stats(
    values: &[f64],
    time_count: usize,
    lat_size: usize,
    lon_size: usize,
    cf: &CfMetadata,
) -> TimeStats {
    let spatial = lat_size * lon_size;
    let mut sum = Array2::zeros((lat_size, lon_size));
    let mut count = Array2::zeros((lat_size, lon_size));
    let mut min = Array2::from_elem((lat_size, lon_size), f64::INFINITY);
    let mut max = Array2::from_elem((lat_size, lon_size), f64::NEG_INFINITY);

    for t in 0..time_count {
        let plane = &values[t * spatial..(t + 1) * spatial];
        for lat_i in 0..lat_size {
            for lon_i in 0..lon_size {
                let raw = plane[lat_i * lon_size + lon_i];
                if cf.is_fill(raw) || raw.is_nan() {
                    continue;
                }
                let v = cf.unpack_value(raw);
                sum[[lat_i, lon_i]] += v;
                count[[lat_i, lon_i]] += 1.0;
                if v < min[[lat_i, lon_i]] {
                    min[[lat_i, lon_i]] = v;
                }
                if v > max[[lat_i, lon_i]] {
                    max[[lat_i, lon_i]] = v;
                }
            }
        }
    }
    TimeStats {
        sum,
        count,
        min,
        max,
    }
}

/// Extract the single lat×lon plane of a flat buffer with known shape.
///
/// Time aggregation never comes through here: [`ZarrReader::read_bbox`]
/// routes every [`TimeReduction::Aggregate`] through
/// [`ZarrReader::read_bbox_partial`].
fn reduce_to_2d(
    values: &[f64],
    shape: &[usize],
    lat_dim: usize,
    lon_dim: usize,
    time_count: usize,
) -> Result<Array2<f64>> {
    let lat_size = shape[lat_dim];
    let lon_size = shape[lon_dim];

    if time_count != 1 {
        return Err(CloudError::Zarr(
            "internal: single-step reduction called with several time steps".into(),
        ));
    }

    // Fast path for the common [time, lat, lon] and [lat, lon] layouts.
    if (shape.len() == 3 && lat_dim == 1 && lon_dim == 2)
        || (shape.len() == 2 && lat_dim == 0 && lon_dim == 1)
    {
        let spatial = lat_size * lon_size;
        return Array2::from_shape_vec((lat_size, lon_size), values[0..spatial].to_vec())
            .map_err(|e| CloudError::Zarr(format!("reshape error: {e}")));
    }

    // Generic layout: index the lat×lon plane through the row-major strides,
    // taking the first index on every other dimension.
    let strides = compute_strides(shape);
    let mut out = Array2::zeros((lat_size, lon_size));
    for lat_i in 0..lat_size {
        for lon_i in 0..lon_size {
            let mut idx = 0;
            for (d, stride) in strides.iter().enumerate() {
                let coord = if d == lat_dim {
                    lat_i
                } else if d == lon_dim {
                    lon_i
                } else {
                    0
                };
                idx += coord * stride;
            }
            out[[lat_i, lon_i]] = values[idx];
        }
    }
    Ok(out)
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![1; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ERA5-style physical grid: 0, 0.25, ..., 359.75
    fn era5_lons() -> Vec<f64> {
        (0..1440).map(|i| i as f64 * 0.25).collect()
    }

    /// Regression: a western-hemisphere bbox (Chile) on a 0-360 grid must
    /// address the physical columns of the converted longitudes, not the
    /// columns that a sorted -180..180 copy of the array would suggest.
    #[test]
    fn test_lon_range_western_hemisphere_on_0_360_grid() {
        let lons = era5_lons();
        // Chile: lon -76..-66  →  284..294 in dataset convention
        let (start, end) = lon_range_physical(&lons, -76.0, -66.0).unwrap();
        assert_eq!(start, 1136, "physical index of 284.0"); // 284 / 0.25
        assert_eq!(end, 1177, "physical index just past 294.0");
        // The buggy sorted-array logic mapped -76 near logical index 416
        assert!(
            start > 720,
            "western longitudes live in the second half of a 0-360 grid"
        );
    }

    #[test]
    fn test_lon_range_eastern_hemisphere_unchanged() {
        let lons = era5_lons();
        let (start, end) = lon_range_physical(&lons, 10.0, 20.0).unwrap();
        assert_eq!(start, 40); // 10 / 0.25
        assert_eq!(end, 81);
    }

    #[test]
    fn test_lon_range_seam_crossing_rejected() {
        let lons = era5_lons();
        // -10..30 crosses the 0/360 seam on this grid: must be an explicit
        // error, never a silent wrong-columns read.
        let err = lon_range_physical(&lons, -10.0, 30.0);
        assert!(err.is_err(), "seam-crossing bbox must be rejected");
    }

    #[test]
    fn test_lon_range_regular_grid_minus180_180() {
        // Grids already in -180..180 are untouched by the conversion
        let lons: Vec<f64> = (0..360).map(|i| -180.0 + i as f64).collect();
        let (start, end) = lon_range_physical(&lons, -76.0, -66.0).unwrap();
        assert_eq!(start, 104);
        assert_eq!(end, 115);
    }

    // ─── Time aggregation across stores ───────────────────────────────

    fn hours(from: &str, n: usize) -> Vec<DateTime<Utc>> {
        let t0: DateTime<Utc> = from.parse().unwrap();
        (0..n)
            .map(|h| t0 + chrono::TimeDelta::hours(h as i64))
            .collect()
    }

    fn dt(s: &str) -> DateTime<Utc> {
        s.parse().unwrap()
    }

    fn cf_plain() -> CfMetadata {
        CfMetadata {
            time_dim: Some(0),
            lat_dim: Some(1),
            lon_dim: Some(2),
            time_units: None,
            time_calendar: None,
            fill_value: None,
            scale_factor: None,
            add_offset: None,
            units: None,
            long_name: None,
        }
    }

    /// Partial over a 1×2 grid with the given per-step planes.
    fn partial_from_planes(planes: &[[f64; 2]], cf: &CfMetadata) -> TimeAggPartial {
        let values: Vec<f64> = planes.iter().flatten().copied().collect();
        let stats = accumulate_time_stats(&values, planes.len(), 1, 2, cf);
        TimeAggPartial {
            sum: Raster::from_array(stats.sum),
            count: stats.count,
            min: stats.min,
            max: stats.max,
            time_steps: planes.len(),
        }
    }

    /// Membership, not clamping: a January store asked for the whole year
    /// contributes exactly its 744 hourly steps, and a request for a period
    /// the store does not cover contributes nothing.
    #[test]
    fn test_time_index_range_is_inclusive_membership() {
        let jan = hours("2020-01-01T00:00:00Z", 31 * 24);
        let y0 = dt("2020-01-01T00:00:00Z");
        let y1 = dt("2020-12-31T23:59:59Z");
        assert_eq!(time_index_range(&jan, &y0, &y1), Some((0, 744)));

        // Exact monthly window: both boundary steps included.
        let m1 = dt("2020-01-31T23:59:59Z");
        assert_eq!(time_index_range(&jan, &y0, &m1), Some((0, 744)));

        // A single day in the middle of the store.
        let d0 = dt("2020-01-10T00:00:00Z");
        let d1 = dt("2020-01-10T23:59:59Z");
        assert_eq!(time_index_range(&jan, &d0, &d1), Some((9 * 24, 24)));

        // February asked of the January store: nothing, not "nearest".
        let f0 = dt("2020-02-01T00:00:00Z");
        let f1 = dt("2020-02-29T23:59:59Z");
        assert_eq!(time_index_range(&jan, &f0, &f1), None);
    }

    /// Consecutive stores never share a step: the last hour of January and
    /// the first of February land in different monthly windows.
    #[test]
    fn test_time_index_range_no_double_count_at_boundary() {
        let jan = hours("2020-01-01T00:00:00Z", 31 * 24);
        let feb = hours("2020-02-01T00:00:00Z", 29 * 24);
        let jan_win = (dt("2020-01-01T00:00:00Z"), dt("2020-01-31T23:59:59Z"));
        let feb_win = (dt("2020-02-01T00:00:00Z"), dt("2020-02-29T23:59:59Z"));
        assert_eq!(
            time_index_range(&jan, &jan_win.0, &jan_win.1),
            Some((0, 744))
        );
        assert_eq!(time_index_range(&jan, &feb_win.0, &feb_win.1), None);
        assert_eq!(time_index_range(&feb, &jan_win.0, &jan_win.1), None);
        assert_eq!(
            time_index_range(&feb, &feb_win.0, &feb_win.1),
            Some((0, 696))
        );
    }

    /// The identity behind `yearly-sum == Σ monthly-sum`: merging the
    /// partials of two stores equals the partial of their concatenation,
    /// for every method.
    #[test]
    fn test_partial_merge_equals_single_pass() {
        let cf = cf_plain();
        let a = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let b = [[4.0, 40.0], [5.0, 50.0]];
        let all: Vec<[f64; 2]> = a.iter().chain(b.iter()).copied().collect();

        let mut merged = partial_from_planes(&a, &cf);
        merged.merge(&partial_from_planes(&b, &cf)).unwrap();
        let single = partial_from_planes(&all, &cf);

        assert_eq!(merged.time_steps, 5);
        for m in [
            AggMethod::Sum,
            AggMethod::Mean,
            AggMethod::Min,
            AggMethod::Max,
        ] {
            let x = merged.finish(m);
            let y = single.finish(m);
            assert_eq!(x.data(), y.data(), "{m:?}");
        }
        let sum = merged.finish(AggMethod::Sum);
        assert_eq!(sum.data()[[0, 0]], 15.0);
        assert_eq!(sum.data()[[0, 1]], 150.0);
        // Mean over every step read, not a mean of per-store means
        // ((2+4.5)/2 = 3.25 would be wrong).
        assert_eq!(merged.finish(AggMethod::Mean).data()[[0, 0]], 3.0);
    }

    /// Fill values are excluded from every accumulator; a pixel with no
    /// valid step finishes as NaN.
    #[test]
    fn test_partial_skips_fill_values() {
        let mut cf = cf_plain();
        cf.fill_value = Some(-9999.0);
        let planes = [[1.0, -9999.0], [-9999.0, -9999.0], [3.0, -9999.0]];
        let p = partial_from_planes(&planes, &cf);
        assert_eq!(p.count[[0, 0]], 2.0);
        assert_eq!(p.count[[0, 1]], 0.0);
        assert_eq!(p.finish(AggMethod::Sum).data()[[0, 0]], 4.0);
        assert_eq!(p.finish(AggMethod::Mean).data()[[0, 0]], 2.0);
        assert!(p.finish(AggMethod::Sum).data()[[0, 1]].is_nan());
        assert!(p.finish(AggMethod::Min).data()[[0, 1]].is_nan());
    }

    /// Packed data is unpacked per value before summing: the offset must
    /// be applied once per step, not once per sum.
    #[test]
    fn test_partial_unpacks_before_accumulating() {
        let mut cf = cf_plain();
        cf.scale_factor = Some(0.5);
        cf.add_offset = Some(100.0);
        let planes = [[2.0, 4.0], [6.0, 8.0]];
        let p = partial_from_planes(&planes, &cf);
        // (2·0.5+100) + (6·0.5+100) = 204
        assert_eq!(p.finish(AggMethod::Sum).data()[[0, 0]], 204.0);
        assert_eq!(p.finish(AggMethod::Mean).data()[[0, 0]], 102.0);
        assert_eq!(p.finish(AggMethod::Min).data()[[0, 1]], 102.0);
        assert_eq!(p.finish(AggMethod::Max).data()[[0, 1]], 104.0);
    }

    #[test]
    fn test_partial_merge_rejects_shape_mismatch() {
        let cf = cf_plain();
        let mut a = partial_from_planes(&[[1.0, 2.0]], &cf);
        let values = vec![1.0, 2.0, 3.0];
        let stats = accumulate_time_stats(&values, 1, 1, 3, &cf);
        let b = TimeAggPartial {
            sum: Raster::from_array(stats.sum),
            count: stats.count,
            min: stats.min,
            max: stats.max,
            time_steps: 1,
        };
        assert!(a.merge(&b).is_err());
    }
}
