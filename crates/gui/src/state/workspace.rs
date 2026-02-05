//! Workspace state: datasets, grid systems, layer management.

use std::collections::HashMap;
use std::path::PathBuf;

use surtgis_colormap::ColorScheme;
use surtgis_core::raster::Raster;

/// Unique identifier for a dataset in the workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DatasetId(pub u64);

/// A loaded raster dataset with metadata for display.
pub struct Dataset {
    pub id: DatasetId,
    pub name: String,
    pub source_path: Option<PathBuf>,
    pub raster: DatasetRaster,
    pub colormap: ColorScheme,
    pub visible: bool,
    /// Layer opacity [0.0, 1.0].
    pub opacity: f32,
    /// Cached RGBA pixels (invalidated when colormap changes).
    pub rgba_cache: Option<Vec<u8>>,
    /// Which algorithm produced this (None if loaded from file).
    pub provenance: Option<String>,
}

/// Raster data in different numeric types.
pub enum DatasetRaster {
    F64(Raster<f64>),
    U8(Raster<u8>),
    I32(Raster<i32>),
}

impl DatasetRaster {
    pub fn rows(&self) -> usize {
        match self {
            Self::F64(r) => r.rows(),
            Self::U8(r) => r.rows(),
            Self::I32(r) => r.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::F64(r) => r.cols(),
            Self::U8(r) => r.cols(),
            Self::I32(r) => r.cols(),
        }
    }

    /// Get geographic bounds (min_x, min_y, max_x, max_y).
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        match self {
            Self::F64(r) => r.bounds(),
            Self::U8(r) => r.bounds(),
            Self::I32(r) => r.bounds(),
        }
    }

    /// Convert pixel coordinates to geographic coordinates.
    pub fn pixel_to_geo(&self, col: usize, row: usize) -> (f64, f64) {
        match self {
            Self::F64(r) => r.pixel_to_geo(col, row),
            Self::U8(r) => r.pixel_to_geo(col, row),
            Self::I32(r) => r.pixel_to_geo(col, row),
        }
    }
}

/// The workspace holds all loaded datasets and display state.
pub struct Workspace {
    datasets: HashMap<DatasetId, Dataset>,
    /// Display order (front to back).
    layer_order: Vec<DatasetId>,
    /// The currently active/selected dataset.
    pub active_dataset: Option<DatasetId>,
    /// Counter for generating unique IDs.
    next_id: u64,
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

impl Workspace {
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
            layer_order: Vec::new(),
            active_dataset: None,
            next_id: 1,
        }
    }

    /// Add a dataset and return its ID. Automatically becomes the active dataset.
    pub fn add_dataset(&mut self, mut dataset: Dataset) -> DatasetId {
        let id = DatasetId(self.next_id);
        self.next_id += 1;
        dataset.id = id;
        self.layer_order.push(id);
        self.active_dataset = Some(id);
        self.datasets.insert(id, dataset);
        id
    }

    /// Get a dataset by ID.
    pub fn get(&self, id: DatasetId) -> Option<&Dataset> {
        self.datasets.get(&id)
    }

    /// Get a mutable dataset by ID.
    pub fn get_mut(&mut self, id: DatasetId) -> Option<&mut Dataset> {
        self.datasets.get_mut(&id)
    }

    /// Get the active dataset.
    pub fn active(&self) -> Option<&Dataset> {
        self.active_dataset.and_then(|id| self.datasets.get(&id))
    }

    /// Get the active dataset mutably.
    pub fn active_mut(&mut self) -> Option<&mut Dataset> {
        self.active_dataset.and_then(|id| self.datasets.get_mut(&id))
    }

    /// Remove a dataset.
    pub fn remove(&mut self, id: DatasetId) {
        self.datasets.remove(&id);
        self.layer_order.retain(|&i| i != id);
        if self.active_dataset == Some(id) {
            self.active_dataset = self.layer_order.last().copied();
        }
    }

    /// Iterate datasets in display order.
    pub fn datasets_ordered(&self) -> impl Iterator<Item = &Dataset> {
        self.layer_order
            .iter()
            .filter_map(|id| self.datasets.get(id))
    }

    /// Number of loaded datasets.
    pub fn len(&self) -> usize {
        self.datasets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }

    /// All dataset IDs in display order.
    pub fn layer_order(&self) -> &[DatasetId] {
        &self.layer_order
    }

    /// Get list of dataset names for combo boxes (input selection).
    pub fn dataset_names(&self) -> Vec<(DatasetId, String)> {
        self.layer_order
            .iter()
            .filter_map(|id| {
                self.datasets
                    .get(id)
                    .map(|d| (*id, d.name.clone()))
            })
            .collect()
    }
}
