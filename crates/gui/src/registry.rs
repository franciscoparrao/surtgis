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

/// Helper: radius parameter common to many algorithms.
fn param_radius(default: i64, max: i64) -> ParamDef {
    ParamDef {
        name: "radius",
        label: "Radius (cells)",
        kind: ParamKind::Int { default, min: 1, max },
    }
}

/// Helper: z_factor parameter.
fn param_z_factor() -> ParamDef {
    ParamDef {
        name: "z_factor",
        label: "Z Factor",
        kind: ParamKind::Float { default: 1.0, min: 0.001, max: 1000.0, speed: 0.1 },
    }
}

/// Helper: structuring element shape parameter.
fn param_se_shape() -> ParamDef {
    ParamDef {
        name: "shape",
        label: "Shape",
        kind: ParamKind::Choice {
            options: &["Square", "Cross", "Disk"],
            default: 0,
        },
    }
}

/// Build the algorithm registry (~40 algorithms).
pub fn build_registry() -> Vec<AlgorithmEntry> {
    vec![
        // ═══════════════════════════════════════════════════════
        // TERRAIN (20)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "slope", name: "Slope", category: AlgoCategory::Terrain,
            description: "Calculate slope from DEM",
            input_count: 1,
            params: vec![
                ParamDef { name: "units", label: "Units", kind: ParamKind::Choice {
                    options: &["Degrees", "Percent", "Radians"], default: 0,
                }},
                param_z_factor(),
            ],
        },
        AlgorithmEntry {
            id: "aspect", name: "Aspect", category: AlgoCategory::Terrain,
            description: "Calculate aspect (direction of steepest slope)",
            input_count: 1,
            params: vec![ParamDef { name: "format", label: "Output Format", kind: ParamKind::Choice {
                options: &["Degrees", "Radians"], default: 0,
            }}],
        },
        AlgorithmEntry {
            id: "hillshade", name: "Hillshade", category: AlgoCategory::Terrain,
            description: "Calculate hillshade (shaded relief)",
            input_count: 1,
            params: vec![
                ParamDef { name: "azimuth", label: "Azimuth (\u{00b0})", kind: ParamKind::Float {
                    default: 315.0, min: 0.0, max: 360.0, speed: 1.0,
                }},
                ParamDef { name: "altitude", label: "Altitude (\u{00b0})", kind: ParamKind::Float {
                    default: 45.0, min: 0.0, max: 90.0, speed: 1.0,
                }},
                param_z_factor(),
            ],
        },
        AlgorithmEntry {
            id: "multidirectional_hillshade", name: "Multidirectional Hillshade",
            category: AlgoCategory::Terrain,
            description: "Hillshade from multiple light directions",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "curvature", name: "Curvature", category: AlgoCategory::Terrain,
            description: "Calculate surface curvature",
            input_count: 1,
            params: vec![
                ParamDef { name: "type", label: "Curvature Type", kind: ParamKind::Choice {
                    options: &["General", "Profile", "Plan"], default: 0,
                }},
                param_z_factor(),
            ],
        },
        AlgorithmEntry {
            id: "tpi", name: "TPI (Topographic Position Index)", category: AlgoCategory::Terrain,
            description: "Calculate topographic position index",
            input_count: 1, params: vec![param_radius(3, 100)],
        },
        AlgorithmEntry {
            id: "tri", name: "TRI (Terrain Ruggedness Index)", category: AlgoCategory::Terrain,
            description: "Calculate terrain ruggedness index",
            input_count: 1, params: vec![param_radius(1, 100)],
        },
        AlgorithmEntry {
            id: "twi", name: "TWI (Topographic Wetness Index)", category: AlgoCategory::Terrain,
            description: "Calculate topographic wetness index from DEM (auto fill+flow+slope)",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "geomorphons", name: "Geomorphons", category: AlgoCategory::Terrain,
            description: "Landform classification using geomorphon approach (10 classes)",
            input_count: 1,
            params: vec![
                ParamDef { name: "flatness", label: "Flatness Threshold (\u{00b0})", kind: ParamKind::Float {
                    default: 1.0, min: 0.0, max: 10.0, speed: 0.1,
                }},
                param_radius(10, 100),
            ],
        },
        AlgorithmEntry {
            id: "dev", name: "DEV (Deviation from Mean Elevation)", category: AlgoCategory::Terrain,
            description: "Calculate deviation from mean elevation",
            input_count: 1, params: vec![param_radius(10, 200)],
        },
        AlgorithmEntry {
            id: "landform", name: "Landform Classification", category: AlgoCategory::Terrain,
            description: "Multi-scale TPI landform classification (11 classes)",
            input_count: 1,
            params: vec![
                ParamDef { name: "small_radius", label: "Small Radius", kind: ParamKind::Int {
                    default: 3, min: 1, max: 50,
                }},
                ParamDef { name: "large_radius", label: "Large Radius", kind: ParamKind::Int {
                    default: 10, min: 2, max: 100,
                }},
                ParamDef { name: "threshold", label: "TPI Threshold", kind: ParamKind::Float {
                    default: 1.0, min: 0.01, max: 10.0, speed: 0.1,
                }},
                ParamDef { name: "slope_threshold", label: "Slope Threshold (\u{00b0})", kind: ParamKind::Float {
                    default: 6.0, min: 0.0, max: 45.0, speed: 0.5,
                }},
            ],
        },
        AlgorithmEntry {
            id: "sky_view_factor", name: "Sky View Factor", category: AlgoCategory::Terrain,
            description: "Fraction of visible sky hemisphere [0-1]",
            input_count: 1,
            params: vec![
                param_radius(10, 200),
                ParamDef { name: "directions", label: "Directions", kind: ParamKind::Int {
                    default: 16, min: 4, max: 64,
                }},
            ],
        },
        AlgorithmEntry {
            id: "positive_openness", name: "Positive Openness", category: AlgoCategory::Terrain,
            description: "Mean of zenith angles in all directions (convexity measure)",
            input_count: 1,
            params: vec![
                param_radius(10, 200),
                ParamDef { name: "directions", label: "Directions", kind: ParamKind::Int {
                    default: 8, min: 4, max: 64,
                }},
            ],
        },
        AlgorithmEntry {
            id: "negative_openness", name: "Negative Openness", category: AlgoCategory::Terrain,
            description: "Mean of nadir angles in all directions (concavity measure)",
            input_count: 1,
            params: vec![
                param_radius(10, 200),
                ParamDef { name: "directions", label: "Directions", kind: ParamKind::Int {
                    default: 8, min: 4, max: 64,
                }},
            ],
        },
        AlgorithmEntry {
            id: "convergence_index", name: "Convergence Index", category: AlgoCategory::Terrain,
            description: "Flow convergence/divergence indicator",
            input_count: 1, params: vec![param_radius(1, 50)],
        },
        AlgorithmEntry {
            id: "vrm", name: "VRM (Vector Ruggedness Measure)", category: AlgoCategory::Terrain,
            description: "Surface roughness from normal vector dispersion [0-1]",
            input_count: 1, params: vec![param_radius(1, 50)],
        },
        AlgorithmEntry {
            id: "shape_index", name: "Shape Index", category: AlgoCategory::Terrain,
            description: "Surface shape from principal curvatures [-1 to 1]",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "curvedness", name: "Curvedness", category: AlgoCategory::Terrain,
            description: "Magnitude of surface bending (always >= 0)",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "northness", name: "Northness", category: AlgoCategory::Terrain,
            description: "Cosine of aspect (north-facing = 1, south = -1)",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "eastness", name: "Eastness", category: AlgoCategory::Terrain,
            description: "Sine of aspect (east-facing = 1, west = -1)",
            input_count: 1, params: vec![],
        },

        // ═══════════════════════════════════════════════════════
        // HYDROLOGY (5)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "fill_sinks", name: "Fill Sinks", category: AlgoCategory::Hydrology,
            description: "Fill depressions in DEM (Planchon-Darboux)",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "priority_flood", name: "Priority Flood", category: AlgoCategory::Hydrology,
            description: "Fill depressions using priority queue (faster for large DEMs)",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "flow_direction", name: "Flow Direction (D8)", category: AlgoCategory::Hydrology,
            description: "Calculate D8 flow direction codes",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "flow_accumulation", name: "Flow Accumulation", category: AlgoCategory::Hydrology,
            description: "Calculate flow accumulation (auto fill + flow direction from DEM)",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "hand", name: "HAND (Height Above Nearest Drainage)", category: AlgoCategory::Hydrology,
            description: "Height above nearest stream (auto fill + flow + acc from DEM)",
            input_count: 1,
            params: vec![ParamDef { name: "stream_threshold", label: "Stream Threshold (cells)",
                kind: ParamKind::Float { default: 1000.0, min: 10.0, max: 100000.0, speed: 100.0 },
            }],
        },

        // ═══════════════════════════════════════════════════════
        // IMAGERY (8)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "ndvi", name: "NDVI", category: AlgoCategory::Imagery,
            description: "Normalized Difference Vegetation Index",
            input_count: 2,
            params: vec![
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "red", label: "Red Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "ndwi", name: "NDWI", category: AlgoCategory::Imagery,
            description: "Normalized Difference Water Index",
            input_count: 2,
            params: vec![
                ParamDef { name: "green", label: "Green Band", kind: ParamKind::InputRaster },
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "mndwi", name: "MNDWI", category: AlgoCategory::Imagery,
            description: "Modified Normalized Difference Water Index",
            input_count: 2,
            params: vec![
                ParamDef { name: "green", label: "Green Band", kind: ParamKind::InputRaster },
                ParamDef { name: "swir", label: "SWIR Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "nbr", name: "NBR", category: AlgoCategory::Imagery,
            description: "Normalized Burn Ratio",
            input_count: 2,
            params: vec![
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "swir", label: "SWIR Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "savi", name: "SAVI", category: AlgoCategory::Imagery,
            description: "Soil-Adjusted Vegetation Index",
            input_count: 2,
            params: vec![
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "red", label: "Red Band", kind: ParamKind::InputRaster },
                ParamDef { name: "l_factor", label: "L Factor", kind: ParamKind::Float {
                    default: 0.5, min: 0.0, max: 1.0, speed: 0.05,
                }},
            ],
        },
        AlgorithmEntry {
            id: "evi", name: "EVI", category: AlgoCategory::Imagery,
            description: "Enhanced Vegetation Index (3 bands)",
            input_count: 3,
            params: vec![
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "red", label: "Red Band", kind: ParamKind::InputRaster },
                ParamDef { name: "blue", label: "Blue Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "bsi", name: "BSI", category: AlgoCategory::Imagery,
            description: "Bare Soil Index (4 bands)",
            input_count: 4,
            params: vec![
                ParamDef { name: "swir", label: "SWIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "red", label: "Red Band", kind: ParamKind::InputRaster },
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "blue", label: "Blue Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "band_math", name: "Band Math", category: AlgoCategory::Imagery,
            description: "Binary arithmetic between two rasters",
            input_count: 2,
            params: vec![
                ParamDef { name: "a", label: "Raster A", kind: ParamKind::InputRaster },
                ParamDef { name: "b", label: "Raster B", kind: ParamKind::InputRaster },
                ParamDef { name: "op", label: "Operation", kind: ParamKind::Choice {
                    options: &["Add", "Subtract", "Multiply", "Divide", "Min", "Max"],
                    default: 0,
                }},
            ],
        },

        // ═══════════════════════════════════════════════════════
        // MORPHOLOGY (7)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "erode", name: "Erosion", category: AlgoCategory::Morphology,
            description: "Morphological erosion (minimum filter)",
            input_count: 1, params: vec![param_se_shape(), param_radius(1, 20)],
        },
        AlgorithmEntry {
            id: "dilate", name: "Dilation", category: AlgoCategory::Morphology,
            description: "Morphological dilation (maximum filter)",
            input_count: 1, params: vec![param_se_shape(), param_radius(1, 20)],
        },
        AlgorithmEntry {
            id: "morph_opening", name: "Opening", category: AlgoCategory::Morphology,
            description: "Erosion then dilation (removes small bright features)",
            input_count: 1, params: vec![param_se_shape(), param_radius(1, 20)],
        },
        AlgorithmEntry {
            id: "morph_closing", name: "Closing", category: AlgoCategory::Morphology,
            description: "Dilation then erosion (removes small dark features)",
            input_count: 1, params: vec![param_se_shape(), param_radius(1, 20)],
        },
        AlgorithmEntry {
            id: "morph_gradient", name: "Morphological Gradient", category: AlgoCategory::Morphology,
            description: "Dilation minus erosion (edge detection)",
            input_count: 1, params: vec![param_se_shape(), param_radius(1, 20)],
        },
        AlgorithmEntry {
            id: "top_hat", name: "Top Hat", category: AlgoCategory::Morphology,
            description: "Original minus opening (bright feature extraction)",
            input_count: 1, params: vec![param_se_shape(), param_radius(1, 20)],
        },
        AlgorithmEntry {
            id: "black_hat", name: "Black Hat", category: AlgoCategory::Morphology,
            description: "Closing minus original (dark feature extraction)",
            input_count: 1, params: vec![param_se_shape(), param_radius(1, 20)],
        },

        // ═══════════════════════════════════════════════════════
        // STATISTICS (4)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "focal_mean", name: "Focal Mean", category: AlgoCategory::Statistics,
            description: "Moving average filter",
            input_count: 1, params: vec![param_radius(3, 50)],
        },
        AlgorithmEntry {
            id: "focal_std", name: "Focal Std Dev", category: AlgoCategory::Statistics,
            description: "Moving standard deviation filter",
            input_count: 1, params: vec![param_radius(3, 50)],
        },
        AlgorithmEntry {
            id: "focal_range", name: "Focal Range", category: AlgoCategory::Statistics,
            description: "Moving range filter (local max - min)",
            input_count: 1, params: vec![param_radius(3, 50)],
        },
        AlgorithmEntry {
            id: "focal_median", name: "Focal Median", category: AlgoCategory::Statistics,
            description: "Moving median filter",
            input_count: 1, params: vec![param_radius(3, 50)],
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
