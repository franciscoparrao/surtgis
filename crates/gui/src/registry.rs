//! Algorithm registry with declarative parameter definitions.
//!
//! Each algorithm entry describes its name, category, parameters, and output type.
//! The tool dialog uses `ParamDef` to auto-generate UI controls.

use crate::state::DatasetId;

/// Category of algorithms (maps to tree structure).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgoCategory {
    Terrain,
    Hydrology,
    Imagery,
    Morphology,
    Statistics,
}

impl AlgoCategory {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Terrain => "Terrain",
            Self::Hydrology => "Hydrology",
            Self::Imagery => "Imagery",
            Self::Morphology => "Morphology",
            Self::Statistics => "Statistics",
        }
    }

    pub const ALL: &[AlgoCategory] = &[
        Self::Terrain,
        Self::Hydrology,
        Self::Imagery,
        Self::Morphology,
        Self::Statistics,
    ];
}

/// Definition of a single parameter for an algorithm.
#[derive(Debug, Clone)]
pub struct ParamDef {
    pub name: &'static str,
    pub label: &'static str,
    pub kind: ParamKind,
}

/// The kind/type of a parameter, determining which UI widget is shown.
#[derive(Debug, Clone)]
pub enum ParamKind {
    /// Floating point value with range and default.
    Float {
        default: f64,
        min: f64,
        max: f64,
        speed: f64,
    },
    /// Integer value with range and default.
    Int {
        default: i64,
        min: i64,
        max: i64,
    },
    /// Boolean toggle.
    Bool { default: bool },
    /// Selection from a list of string options.
    Choice {
        options: &'static [&'static str],
        default: usize,
    },
    /// Reference to an input dataset (combo box of loaded datasets).
    InputRaster,
}

/// Runtime parameter value (set by user in the dialog).
#[derive(Debug, Clone)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Choice(usize),
    InputRaster(Option<DatasetId>),
}

impl ParamValue {
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Float(v) => *v,
            Self::Int(v) => *v as f64,
            _ => 0.0,
        }
    }

    pub fn as_i64(&self) -> i64 {
        match self {
            Self::Int(v) => *v,
            Self::Float(v) => *v as i64,
            _ => 0,
        }
    }

    pub fn as_usize(&self) -> usize {
        match self {
            Self::Int(v) => *v as usize,
            Self::Choice(v) => *v,
            _ => 0,
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(v) => *v,
            _ => false,
        }
    }

    pub fn as_choice_index(&self) -> usize {
        match self {
            Self::Choice(v) => *v,
            _ => 0,
        }
    }

    pub fn as_dataset_id(&self) -> Option<DatasetId> {
        match self {
            Self::InputRaster(id) => *id,
            _ => None,
        }
    }
}

/// An algorithm entry in the registry.
#[derive(Debug, Clone)]
pub struct AlgorithmEntry {
    pub id: &'static str,
    pub name: &'static str,
    pub category: AlgoCategory,
    pub description: &'static str,
    pub params: Vec<ParamDef>,
    /// Number of input rasters required (1 for most, 2 for imagery indices).
    pub input_count: usize,
}

/// Build the default algorithm registry (~15 algorithms for MVP).
pub fn build_registry() -> Vec<AlgorithmEntry> {
    vec![
        // ── Terrain ────────────────────────────────────────────
        AlgorithmEntry {
            id: "slope",
            name: "Slope",
            category: AlgoCategory::Terrain,
            description: "Calculate slope from DEM",
            input_count: 1,
            params: vec![
                ParamDef {
                    name: "units",
                    label: "Units",
                    kind: ParamKind::Choice {
                        options: &["Degrees", "Percent", "Radians"],
                        default: 0,
                    },
                },
                ParamDef {
                    name: "z_factor",
                    label: "Z Factor",
                    kind: ParamKind::Float {
                        default: 1.0,
                        min: 0.001,
                        max: 1000.0,
                        speed: 0.1,
                    },
                },
            ],
        },
        AlgorithmEntry {
            id: "aspect",
            name: "Aspect",
            category: AlgoCategory::Terrain,
            description: "Calculate aspect (direction of steepest slope)",
            input_count: 1,
            params: vec![ParamDef {
                name: "format",
                label: "Output Format",
                kind: ParamKind::Choice {
                    options: &["Degrees", "Radians"],
                    default: 0,
                },
            }],
        },
        AlgorithmEntry {
            id: "hillshade",
            name: "Hillshade",
            category: AlgoCategory::Terrain,
            description: "Calculate hillshade (shaded relief)",
            input_count: 1,
            params: vec![
                ParamDef {
                    name: "azimuth",
                    label: "Azimuth (°)",
                    kind: ParamKind::Float {
                        default: 315.0,
                        min: 0.0,
                        max: 360.0,
                        speed: 1.0,
                    },
                },
                ParamDef {
                    name: "altitude",
                    label: "Altitude (°)",
                    kind: ParamKind::Float {
                        default: 45.0,
                        min: 0.0,
                        max: 90.0,
                        speed: 1.0,
                    },
                },
                ParamDef {
                    name: "z_factor",
                    label: "Z Factor",
                    kind: ParamKind::Float {
                        default: 1.0,
                        min: 0.001,
                        max: 1000.0,
                        speed: 0.1,
                    },
                },
            ],
        },
        AlgorithmEntry {
            id: "curvature",
            name: "Curvature",
            category: AlgoCategory::Terrain,
            description: "Calculate surface curvature",
            input_count: 1,
            params: vec![
                ParamDef {
                    name: "type",
                    label: "Curvature Type",
                    kind: ParamKind::Choice {
                        options: &["General", "Profile", "Plan"],
                        default: 0,
                    },
                },
                ParamDef {
                    name: "z_factor",
                    label: "Z Factor",
                    kind: ParamKind::Float {
                        default: 1.0,
                        min: 0.001,
                        max: 1000.0,
                        speed: 0.1,
                    },
                },
            ],
        },
        AlgorithmEntry {
            id: "tpi",
            name: "TPI (Topographic Position Index)",
            category: AlgoCategory::Terrain,
            description: "Calculate topographic position index",
            input_count: 1,
            params: vec![ParamDef {
                name: "radius",
                label: "Radius (cells)",
                kind: ParamKind::Int {
                    default: 3,
                    min: 1,
                    max: 100,
                },
            }],
        },
        AlgorithmEntry {
            id: "tri",
            name: "TRI (Terrain Ruggedness Index)",
            category: AlgoCategory::Terrain,
            description: "Calculate terrain ruggedness index",
            input_count: 1,
            params: vec![ParamDef {
                name: "radius",
                label: "Radius (cells)",
                kind: ParamKind::Int {
                    default: 1,
                    min: 1,
                    max: 100,
                },
            }],
        },
        AlgorithmEntry {
            id: "twi",
            name: "TWI (Topographic Wetness Index)",
            category: AlgoCategory::Terrain,
            description: "Calculate topographic wetness index from DEM",
            input_count: 1,
            params: vec![],
        },
        AlgorithmEntry {
            id: "geomorphons",
            name: "Geomorphons",
            category: AlgoCategory::Terrain,
            description: "Landform classification using geomorphon approach",
            input_count: 1,
            params: vec![
                ParamDef {
                    name: "flatness",
                    label: "Flatness Threshold (°)",
                    kind: ParamKind::Float {
                        default: 1.0,
                        min: 0.0,
                        max: 10.0,
                        speed: 0.1,
                    },
                },
                ParamDef {
                    name: "radius",
                    label: "Search Radius (cells)",
                    kind: ParamKind::Int {
                        default: 10,
                        min: 1,
                        max: 100,
                    },
                },
            ],
        },
        AlgorithmEntry {
            id: "multidirectional_hillshade",
            name: "Multidirectional Hillshade",
            category: AlgoCategory::Terrain,
            description: "Hillshade from multiple light directions",
            input_count: 1,
            params: vec![],
        },
        AlgorithmEntry {
            id: "dev",
            name: "DEV (Deviation from Mean Elevation)",
            category: AlgoCategory::Terrain,
            description: "Calculate deviation from mean elevation",
            input_count: 1,
            params: vec![ParamDef {
                name: "radius",
                label: "Radius (cells)",
                kind: ParamKind::Int {
                    default: 10,
                    min: 1,
                    max: 200,
                },
            }],
        },
        // ── Hydrology ──────────────────────────────────────────
        AlgorithmEntry {
            id: "fill_sinks",
            name: "Fill Sinks",
            category: AlgoCategory::Hydrology,
            description: "Fill depressions in DEM for hydrological analysis",
            input_count: 1,
            params: vec![],
        },
        AlgorithmEntry {
            id: "flow_direction",
            name: "Flow Direction (D8)",
            category: AlgoCategory::Hydrology,
            description: "Calculate D8 flow direction",
            input_count: 1,
            params: vec![],
        },
        AlgorithmEntry {
            id: "flow_accumulation",
            name: "Flow Accumulation",
            category: AlgoCategory::Hydrology,
            description: "Calculate flow accumulation from D8 flow direction",
            input_count: 1,
            params: vec![],
        },
        // ── Imagery ────────────────────────────────────────────
        AlgorithmEntry {
            id: "ndvi",
            name: "NDVI",
            category: AlgoCategory::Imagery,
            description: "Normalized Difference Vegetation Index (NIR - Red) / (NIR + Red)",
            input_count: 2,
            params: vec![
                ParamDef {
                    name: "nir",
                    label: "NIR Band",
                    kind: ParamKind::InputRaster,
                },
                ParamDef {
                    name: "red",
                    label: "Red Band",
                    kind: ParamKind::InputRaster,
                },
            ],
        },
        AlgorithmEntry {
            id: "ndwi",
            name: "NDWI",
            category: AlgoCategory::Imagery,
            description: "Normalized Difference Water Index (Green - NIR) / (Green + NIR)",
            input_count: 2,
            params: vec![
                ParamDef {
                    name: "green",
                    label: "Green Band",
                    kind: ParamKind::InputRaster,
                },
                ParamDef {
                    name: "nir",
                    label: "NIR Band",
                    kind: ParamKind::InputRaster,
                },
            ],
        },
        // ── Morphology ─────────────────────────────────────────
        AlgorithmEntry {
            id: "focal_mean",
            name: "Focal Mean",
            category: AlgoCategory::Statistics,
            description: "Moving average (focal mean) filter",
            input_count: 1,
            params: vec![ParamDef {
                name: "radius",
                label: "Radius (cells)",
                kind: ParamKind::Int {
                    default: 3,
                    min: 1,
                    max: 50,
                },
            }],
        },
    ]
}

/// Get algorithms filtered by category.
pub fn algorithms_by_category(
    registry: &[AlgorithmEntry],
    category: AlgoCategory,
) -> Vec<&AlgorithmEntry> {
    registry.iter().filter(|a| a.category == category).collect()
}
