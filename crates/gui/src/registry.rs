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
    Interpolation,
    Landscape,
    Classification,
    Texture,
}

impl AlgoCategory {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Terrain => "Terrain",
            Self::Hydrology => "Hydrology",
            Self::Imagery => "Imagery",
            Self::Morphology => "Morphology",
            Self::Statistics => "Statistics",
            Self::Interpolation => "Interpolation",
            Self::Landscape => "Landscape",
            Self::Classification => "Classification",
            Self::Texture => "Texture",
        }
    }

    pub const ALL: &[AlgoCategory] = &[
        Self::Terrain,
        Self::Hydrology,
        Self::Imagery,
        Self::Morphology,
        Self::Statistics,
        Self::Interpolation,
        Self::Landscape,
        Self::Classification,
        Self::Texture,
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
        AlgorithmEntry {
            id: "focal_min", name: "Focal Min", category: AlgoCategory::Statistics,
            description: "Moving minimum filter",
            input_count: 1, params: vec![param_radius(3, 50)],
        },
        AlgorithmEntry {
            id: "focal_max", name: "Focal Max", category: AlgoCategory::Statistics,
            description: "Moving maximum filter",
            input_count: 1, params: vec![param_radius(3, 50)],
        },
        AlgorithmEntry {
            id: "focal_sum", name: "Focal Sum", category: AlgoCategory::Statistics,
            description: "Moving sum filter",
            input_count: 1, params: vec![param_radius(3, 50)],
        },
        AlgorithmEntry {
            id: "zonal_statistics", name: "Zonal Statistics", category: AlgoCategory::Statistics,
            description: "Statistics for each zone of a categorical raster",
            input_count: 2,
            params: vec![
                ParamDef { name: "zones", label: "Zones Raster", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "morans_i", name: "Moran's I (Global)", category: AlgoCategory::Statistics,
            description: "Global spatial autocorrelation index",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "getis_ord", name: "Getis-Ord Gi* (Hot Spots)", category: AlgoCategory::Statistics,
            description: "Local hot/cold spot analysis (z-scores)",
            input_count: 1, params: vec![param_radius(3, 50)],
        },

        // ═══════════════════════════════════════════════════════
        // HYDROLOGY — new (+9)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "breach_depressions", name: "Breach Depressions", category: AlgoCategory::Hydrology,
            description: "Breach depressions in DEM (alternative to fill sinks)",
            input_count: 1,
            params: vec![
                ParamDef { name: "max_depth", label: "Max Depth (m)", kind: ParamKind::Float {
                    default: f64::INFINITY, min: 0.0, max: 10000.0, speed: 1.0,
                }},
                ParamDef { name: "max_length", label: "Max Length (cells)", kind: ParamKind::Int {
                    default: 0, min: 0, max: 10000,
                }},
            ],
        },
        AlgorithmEntry {
            id: "flow_direction_dinf", name: "Flow Direction (D-inf)", category: AlgoCategory::Hydrology,
            description: "D-infinity flow direction (continuous angle in radians)",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "flow_accumulation_mfd", name: "Flow Accumulation (MFD)", category: AlgoCategory::Hydrology,
            description: "Multiple Flow Direction accumulation (auto fill from DEM)",
            input_count: 1,
            params: vec![
                ParamDef { name: "exponent", label: "Exponent", kind: ParamKind::Float {
                    default: 1.1, min: 0.1, max: 10.0, speed: 0.1,
                }},
            ],
        },
        AlgorithmEntry {
            id: "flow_accumulation_mfd_adaptive", name: "Flow Accumulation (Adaptive MFD)",
            category: AlgoCategory::Hydrology,
            description: "Adaptive Multiple Flow Direction accumulation (auto fill from DEM)",
            input_count: 1,
            params: vec![
                ParamDef { name: "scale_factor", label: "Scale Factor", kind: ParamKind::Float {
                    default: 1.0, min: 0.1, max: 10.0, speed: 0.1,
                }},
            ],
        },
        AlgorithmEntry {
            id: "flow_accumulation_tfga", name: "Flow Accumulation (TFGA)",
            category: AlgoCategory::Hydrology,
            description: "TFGA flow accumulation (auto fill from DEM)",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "watershed", name: "Watershed Basins", category: AlgoCategory::Hydrology,
            description: "Delineate watershed basins (auto fill + flow direction from DEM)",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "stream_network", name: "Stream Network", category: AlgoCategory::Hydrology,
            description: "Extract stream network (auto fill + flow + acc from DEM)",
            input_count: 1,
            params: vec![ParamDef { name: "stream_threshold", label: "Stream Threshold (cells)",
                kind: ParamKind::Float { default: 1000.0, min: 10.0, max: 100000.0, speed: 100.0 },
            }],
        },
        AlgorithmEntry {
            id: "nested_depressions", name: "Nested Depressions", category: AlgoCategory::Hydrology,
            description: "Analyze nested depression hierarchy in DEM",
            input_count: 1,
            params: vec![
                ParamDef { name: "min_depth", label: "Min Depth (m)", kind: ParamKind::Float {
                    default: 0.1, min: 0.0, max: 100.0, speed: 0.1,
                }},
                ParamDef { name: "min_area", label: "Min Area (cells)", kind: ParamKind::Int {
                    default: 10, min: 1, max: 100000,
                }},
            ],
        },
        AlgorithmEntry {
            id: "flow_dinf", name: "D-inf Flow (Direction + Accumulation)",
            category: AlgoCategory::Hydrology,
            description: "D-infinity flow direction and accumulation in one pass",
            input_count: 1, params: vec![],
        },

        // ═══════════════════════════════════════════════════════
        // IMAGERY — new (+7)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "ndre", name: "NDRE", category: AlgoCategory::Imagery,
            description: "Normalized Difference Red Edge (vegetation health/stress)",
            input_count: 2,
            params: vec![
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "red_edge", label: "Red Edge Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "gndvi", name: "GNDVI", category: AlgoCategory::Imagery,
            description: "Green NDVI (chlorophyll-sensitive vegetation index)",
            input_count: 2,
            params: vec![
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "green", label: "Green Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "ngrdi", name: "NGRDI", category: AlgoCategory::Imagery,
            description: "Normalized Green Red Difference Index (visible-only vegetation proxy)",
            input_count: 2,
            params: vec![
                ParamDef { name: "green", label: "Green Band", kind: ParamKind::InputRaster },
                ParamDef { name: "red", label: "Red Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "reci", name: "RECI", category: AlgoCategory::Imagery,
            description: "Red Edge Chlorophyll Index",
            input_count: 2,
            params: vec![
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "red_edge", label: "Red Edge Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "normalized_difference", name: "Normalized Difference (custom)", category: AlgoCategory::Imagery,
            description: "Generic (A - B) / (A + B) for any two bands",
            input_count: 2,
            params: vec![
                ParamDef { name: "a", label: "Band A", kind: ParamKind::InputRaster },
                ParamDef { name: "b", label: "Band B", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "reclassify", name: "Reclassify", category: AlgoCategory::Imagery,
            description: "Reclassify raster values into discrete classes (e.g. 0-0.2->1, 0.2-0.5->2)",
            input_count: 1,
            params: vec![
                ParamDef { name: "n_classes", label: "Number of Classes", kind: ParamKind::Int {
                    default: 5, min: 2, max: 50,
                }},
            ],
        },
        AlgorithmEntry {
            id: "band_math_expr", name: "Band Math (Expression)", category: AlgoCategory::Imagery,
            description: "Apply formula to each pixel: pow, sqrt, log, abs, clamp (single raster)",
            input_count: 1,
            params: vec![
                ParamDef { name: "operation", label: "Operation", kind: ParamKind::Choice {
                    options: &["sqrt", "log", "log10", "abs", "negate", "square", "normalize 0-1"],
                    default: 0,
                }},
            ],
        },

        // ═══════════════════════════════════════════════════════
        // TERRAIN — new (+8)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "spi", name: "SPI (Stream Power Index)", category: AlgoCategory::Terrain,
            description: "Erosive power of flowing water (auto fill+flow+acc+slope from DEM)",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "sti", name: "STI (Sediment Transport Index)", category: AlgoCategory::Terrain,
            description: "Sediment transport capacity (USLE-based, auto fill+flow+acc+slope from DEM)",
            input_count: 1,
            params: vec![
                ParamDef { name: "m", label: "m (slope-length exponent)", kind: ParamKind::Float {
                    default: 0.4, min: 0.0, max: 5.0, speed: 0.05,
                }},
                ParamDef { name: "n", label: "n (slope steepness exponent)", kind: ParamKind::Float {
                    default: 1.3, min: 0.0, max: 5.0, speed: 0.05,
                }},
            ],
        },
        AlgorithmEntry {
            id: "viewshed", name: "Viewshed", category: AlgoCategory::Terrain,
            description: "Line-of-sight visibility from an observer point",
            input_count: 1,
            params: vec![
                ParamDef { name: "observer_row", label: "Observer Row", kind: ParamKind::Int {
                    default: 0, min: 0, max: 100000,
                }},
                ParamDef { name: "observer_col", label: "Observer Col", kind: ParamKind::Int {
                    default: 0, min: 0, max: 100000,
                }},
                ParamDef { name: "observer_height", label: "Observer Height (m)", kind: ParamKind::Float {
                    default: 1.7, min: 0.0, max: 1000.0, speed: 0.5,
                }},
                ParamDef { name: "target_height", label: "Target Height (m)", kind: ParamKind::Float {
                    default: 0.0, min: 0.0, max: 1000.0, speed: 0.5,
                }},
                ParamDef { name: "max_radius", label: "Max Radius (cells, 0=unlimited)", kind: ParamKind::Int {
                    default: 0, min: 0, max: 100000,
                }},
            ],
        },
        AlgorithmEntry {
            id: "mrvbf", name: "MRVBF / MRRTF", category: AlgoCategory::Terrain,
            description: "Multi-resolution Valley Bottom / Ridge Top Flatness",
            input_count: 1,
            params: vec![
                ParamDef { name: "t_slope", label: "Initial Slope Threshold", kind: ParamKind::Float {
                    default: 16.0, min: 1.0, max: 90.0, speed: 1.0,
                }},
                ParamDef { name: "steps", label: "Resolution Steps", kind: ParamKind::Int {
                    default: 3, min: 1, max: 10,
                }},
            ],
        },
        AlgorithmEntry {
            id: "wind_exposure", name: "Wind Exposure", category: AlgoCategory::Terrain,
            description: "Topographic wind shelter/exposure index",
            input_count: 1,
            params: vec![
                param_radius(30, 500),
                ParamDef { name: "directions", label: "Directions", kind: ParamKind::Int {
                    default: 8, min: 4, max: 64,
                }},
            ],
        },
        AlgorithmEntry {
            id: "solar_radiation", name: "Solar Radiation", category: AlgoCategory::Terrain,
            description: "Daily beam + diffuse solar radiation (auto slope+aspect from DEM)",
            input_count: 1,
            params: vec![
                ParamDef { name: "day", label: "Day of Year (1-365)", kind: ParamKind::Int {
                    default: 172, min: 1, max: 365,
                }},
                ParamDef { name: "latitude", label: "Latitude (\u{00b0})", kind: ParamKind::Float {
                    default: 40.0, min: -90.0, max: 90.0, speed: 0.5,
                }},
                ParamDef { name: "transmittance", label: "Atmospheric Transmittance", kind: ParamKind::Float {
                    default: 0.7, min: 0.1, max: 1.0, speed: 0.05,
                }},
            ],
        },
        AlgorithmEntry {
            id: "lineament_detection", name: "Lineament Detection", category: AlgoCategory::Terrain,
            description: "Detect linear features from curvature (auto plan+profile curvature from DEM)",
            input_count: 1,
            params: vec![
                ParamDef { name: "min_length", label: "Min Length (cells)", kind: ParamKind::Int {
                    default: 5, min: 2, max: 100,
                }},
            ],
        },
        AlgorithmEntry {
            id: "advanced_curvatures", name: "Advanced Curvatures (Florinsky)", category: AlgoCategory::Terrain,
            description: "Florinsky curvature: mean, Gaussian, horizontal, vertical, accumulation, etc.",
            input_count: 1,
            params: vec![
                ParamDef { name: "curv_type", label: "Curvature Type", kind: ParamKind::Choice {
                    options: &[
                        "Mean (H)", "Gaussian (K)", "Unsphericity (M)", "Difference (E)",
                        "Minimal (Kmin)", "Maximal (Kmax)", "Horizontal (Kh)", "Vertical (Kv)",
                        "Horiz. Excess (Khe)", "Vert. Excess (Kve)", "Accumulation (Ka)",
                        "Ring (Kr)", "Rotor", "Laplacian",
                    ],
                    default: 0,
                }},
            ],
        },

        // ═══════════════════════════════════════════════════════
        // INTERPOLATION (6) — new category
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "interp_idw", name: "IDW Interpolation", category: AlgoCategory::Interpolation,
            description: "Inverse Distance Weighting — fill NoData gaps in the active raster",
            input_count: 1,
            params: vec![
                ParamDef { name: "power", label: "Power", kind: ParamKind::Float {
                    default: 2.0, min: 0.1, max: 10.0, speed: 0.1,
                }},
                ParamDef { name: "max_points", label: "Max Points", kind: ParamKind::Int {
                    default: 12, min: 1, max: 100,
                }},
            ],
        },
        AlgorithmEntry {
            id: "interp_nearest", name: "Nearest Neighbor Interpolation", category: AlgoCategory::Interpolation,
            description: "Fill NoData with nearest valid pixel value",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "interp_natural", name: "Natural Neighbor Interpolation", category: AlgoCategory::Interpolation,
            description: "Sibson natural neighbor — fill NoData gaps",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "interp_tin", name: "TIN Interpolation", category: AlgoCategory::Interpolation,
            description: "Triangulated Irregular Network — fill NoData gaps",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "interp_tps", name: "Thin Plate Spline", category: AlgoCategory::Interpolation,
            description: "Thin Plate Spline — smooth interpolation of NoData gaps",
            input_count: 1,
            params: vec![
                ParamDef { name: "smoothing", label: "Smoothing", kind: ParamKind::Float {
                    default: 0.0, min: 0.0, max: 100.0, speed: 0.1,
                }},
            ],
        },
        AlgorithmEntry {
            id: "interp_kriging", name: "Ordinary Kriging", category: AlgoCategory::Interpolation,
            description: "Geostatistical interpolation — fill NoData gaps with variance estimation",
            input_count: 1,
            params: vec![
                ParamDef { name: "max_points", label: "Max Points", kind: ParamKind::Int {
                    default: 16, min: 4, max: 64,
                }},
            ],
        },

        // ═══════════════════════════════════════════════════════
        // IMAGERY — Fase B (+5)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "ndsi", name: "NDSI", category: AlgoCategory::Imagery,
            description: "Normalized Difference Snow Index (Green - SWIR) / (Green + SWIR)",
            input_count: 2,
            params: vec![
                ParamDef { name: "green", label: "Green Band", kind: ParamKind::InputRaster },
                ParamDef { name: "swir", label: "SWIR Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "ndbi", name: "NDBI", category: AlgoCategory::Imagery,
            description: "Normalized Difference Built-up Index (SWIR - NIR) / (SWIR + NIR)",
            input_count: 2,
            params: vec![
                ParamDef { name: "swir", label: "SWIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "ndmi", name: "NDMI", category: AlgoCategory::Imagery,
            description: "Normalized Difference Moisture Index (NIR - SWIR) / (NIR + SWIR)",
            input_count: 2,
            params: vec![
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "swir", label: "SWIR Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "msavi", name: "MSAVI", category: AlgoCategory::Imagery,
            description: "Modified SAVI — self-adjusting soil brightness correction",
            input_count: 2,
            params: vec![
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "red", label: "Red Band", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "evi2", name: "EVI2", category: AlgoCategory::Imagery,
            description: "Two-band Enhanced Vegetation Index (no blue band needed)",
            input_count: 2,
            params: vec![
                ParamDef { name: "nir", label: "NIR Band", kind: ParamKind::InputRaster },
                ParamDef { name: "red", label: "Red Band", kind: ParamKind::InputRaster },
            ],
        },

        // ═══════════════════════════════════════════════════════
        // STATISTICS — Fase B (+1)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "focal_majority", name: "Focal Majority", category: AlgoCategory::Statistics,
            description: "Moving majority (mode) filter — most frequent value in window",
            input_count: 1, params: vec![param_radius(3, 50)],
        },

        // ═══════════════════════════════════════════════════════
        // TERRAIN — Fase B (+2)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "contour_lines", name: "Contour Lines", category: AlgoCategory::Terrain,
            description: "Generate contour lines at regular intervals (raster overlay)",
            input_count: 1,
            params: vec![
                ParamDef { name: "interval", label: "Contour Interval (m)", kind: ParamKind::Float {
                    default: 10.0, min: 0.1, max: 10000.0, speed: 1.0,
                }},
                ParamDef { name: "base", label: "Base Value (m)", kind: ParamKind::Float {
                    default: 0.0, min: -10000.0, max: 10000.0, speed: 1.0,
                }},
            ],
        },
        AlgorithmEntry {
            id: "cost_distance", name: "Cost Distance", category: AlgoCategory::Terrain,
            description: "Accumulated cost of travel from a source cell (Dijkstra 8-connected)",
            input_count: 1,
            params: vec![
                ParamDef { name: "source_row", label: "Source Row", kind: ParamKind::Int {
                    default: 0, min: 0, max: 100000,
                }},
                ParamDef { name: "source_col", label: "Source Col", kind: ParamKind::Int {
                    default: 0, min: 0, max: 100000,
                }},
            ],
        },

        // ═══════════════════════════════════════════════════════
        // LANDSCAPE ECOLOGY — Fase B (+3, new category)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "shannon_diversity", name: "Shannon Diversity Index", category: AlgoCategory::Landscape,
            description: "Landscape information entropy H' = -Σ(pi·ln(pi)) per moving window",
            input_count: 1, params: vec![param_radius(3, 50)],
        },
        AlgorithmEntry {
            id: "simpson_diversity", name: "Simpson Diversity Index", category: AlgoCategory::Landscape,
            description: "Probability of interspecific encounter 1-Σ(pi²) per moving window",
            input_count: 1, params: vec![param_radius(3, 50)],
        },
        AlgorithmEntry {
            id: "patch_density", name: "Patch Density", category: AlgoCategory::Landscape,
            description: "Number of distinct patches per unit area in moving window",
            input_count: 1, params: vec![param_radius(3, 50)],
        },

        // ═══════════════════════════════════════════════════════
        // CLASSIFICATION — Fase C (+5, new category)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "kmeans", name: "K-Means Clustering", category: AlgoCategory::Classification,
            description: "Unsupervised K-means pixel clustering",
            input_count: 1,
            params: vec![
                ParamDef { name: "k", label: "Number of Clusters", kind: ParamKind::Int {
                    default: 5, min: 2, max: 50,
                }},
                ParamDef { name: "max_iterations", label: "Max Iterations", kind: ParamKind::Int {
                    default: 100, min: 1, max: 1000,
                }},
            ],
        },
        AlgorithmEntry {
            id: "isodata", name: "ISODATA Clustering", category: AlgoCategory::Classification,
            description: "Iterative self-organizing clustering with split/merge",
            input_count: 1,
            params: vec![
                ParamDef { name: "initial_k", label: "Initial Clusters", kind: ParamKind::Int {
                    default: 5, min: 2, max: 50,
                }},
                ParamDef { name: "min_k", label: "Min Clusters", kind: ParamKind::Int {
                    default: 2, min: 1, max: 20,
                }},
                ParamDef { name: "max_k", label: "Max Clusters", kind: ParamKind::Int {
                    default: 10, min: 2, max: 100,
                }},
                ParamDef { name: "max_std_dev", label: "Max Std Dev (split)", kind: ParamKind::Float {
                    default: 10.0, min: 0.1, max: 1000.0, speed: 1.0,
                }},
                ParamDef { name: "min_merge_dist", label: "Min Merge Distance", kind: ParamKind::Float {
                    default: 5.0, min: 0.01, max: 1000.0, speed: 1.0,
                }},
            ],
        },
        AlgorithmEntry {
            id: "minimum_distance", name: "Minimum Distance Classification",
            category: AlgoCategory::Classification,
            description: "Supervised classification: assign each pixel to nearest training class centroid",
            input_count: 2,
            params: vec![
                ParamDef { name: "training", label: "Training Raster (classified)", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "maximum_likelihood", name: "Maximum Likelihood Classification",
            category: AlgoCategory::Classification,
            description: "Supervised classification: Gaussian MLE from training data",
            input_count: 2,
            params: vec![
                ParamDef { name: "training", label: "Training Raster (classified)", kind: ParamKind::InputRaster },
            ],
        },
        AlgorithmEntry {
            id: "pca", name: "PCA (first component)", category: AlgoCategory::Classification,
            description: "Principal Component Analysis — returns first principal component",
            input_count: 1,
            params: vec![],
        },

        // ═══════════════════════════════════════════════════════
        // TEXTURE — Fase C (+3, new category)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "haralick_glcm", name: "GLCM Texture (Haralick)", category: AlgoCategory::Texture,
            description: "Gray-Level Co-occurrence Matrix texture features",
            input_count: 1,
            params: vec![
                param_radius(3, 20),
                ParamDef { name: "n_levels", label: "Gray Levels", kind: ParamKind::Int {
                    default: 32, min: 4, max: 256,
                }},
                ParamDef { name: "texture", label: "Texture Measure", kind: ParamKind::Choice {
                    options: &["Contrast", "Energy", "Homogeneity", "Correlation", "Entropy", "Dissimilarity"],
                    default: 0,
                }},
            ],
        },
        AlgorithmEntry {
            id: "sobel_edge", name: "Sobel Edge Detection", category: AlgoCategory::Texture,
            description: "Gradient magnitude using 3x3 Sobel operators",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "laplacian", name: "Laplacian Filter", category: AlgoCategory::Texture,
            description: "Second-derivative edge detection (3x3 Laplacian kernel)",
            input_count: 1, params: vec![],
        },

        // ═══════════════════════════════════════════════════════
        // IMAGERY — Fase C (+2: change detection)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "raster_difference", name: "Raster Difference", category: AlgoCategory::Imagery,
            description: "Change detection: difference and classification (decrease/no change/increase)",
            input_count: 2,
            params: vec![
                ParamDef { name: "after", label: "After Raster", kind: ParamKind::InputRaster },
                ParamDef { name: "threshold", label: "Change Threshold", kind: ParamKind::Float {
                    default: 0.1, min: 0.0, max: 1000.0, speed: 0.01,
                }},
            ],
        },
        AlgorithmEntry {
            id: "change_vector_analysis", name: "Change Vector Analysis (CVA)",
            category: AlgoCategory::Imagery,
            description: "Multi-band change detection: magnitude + direction from two band pairs",
            input_count: 4,
            params: vec![
                ParamDef { name: "b1_after", label: "Band 1 After", kind: ParamKind::InputRaster },
                ParamDef { name: "b2_before", label: "Band 2 Before", kind: ParamKind::InputRaster },
                ParamDef { name: "b2_after", label: "Band 2 After", kind: ParamKind::InputRaster },
            ],
        },

        // ═══════════════════════════════════════════════════════
        // HYDROLOGY — Fase C (+4: advanced)
        // ═══════════════════════════════════════════════════════
        AlgorithmEntry {
            id: "strahler_order", name: "Strahler Stream Order", category: AlgoCategory::Hydrology,
            description: "Stream ordering (auto fill + flow + stream from DEM)",
            input_count: 1,
            params: vec![
                ParamDef { name: "stream_threshold", label: "Stream Threshold (cells)",
                    kind: ParamKind::Float { default: 1000.0, min: 10.0, max: 100000.0, speed: 100.0 },
                },
            ],
        },
        AlgorithmEntry {
            id: "flow_path_length", name: "Flow Path Length", category: AlgoCategory::Hydrology,
            description: "Downstream distance along D8 flow path (auto fill + flow from DEM)",
            input_count: 1, params: vec![],
        },
        AlgorithmEntry {
            id: "isobasins", name: "Isobasins", category: AlgoCategory::Hydrology,
            description: "Equal-area watershed subdivision (auto fill + flow + acc from DEM)",
            input_count: 1,
            params: vec![
                ParamDef { name: "target_area", label: "Target Area (cells)", kind: ParamKind::Int {
                    default: 1000, min: 10, max: 1000000,
                }},
            ],
        },
        AlgorithmEntry {
            id: "flood_fill_simulation", name: "Flood Simulation", category: AlgoCategory::Hydrology,
            description: "Simulate flooding at a given water level (BFS from boundaries)",
            input_count: 1,
            params: vec![
                ParamDef { name: "water_level", label: "Water Level (m)", kind: ParamKind::Float {
                    default: 10.0, min: -10000.0, max: 100000.0, speed: 1.0,
                }},
            ],
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
