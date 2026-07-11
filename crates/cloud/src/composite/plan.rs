//! Composite planning: search sizing, scene-date selection, the calibrated
//! memory-budget model that caps `strip_rows`, and per-strip geometry.
//!
//! The budget model is the empirically calibrated one from the v0.6.22 →
//! v0.7.1 RAM saga (see `docs/postmortems/2026-04-stac-composite-ram.md`);
//! it is expected to be *replaced* by `MemoryBudget` (audit R9), at which
//! point the calibration constants disappear. Extracting it here first
//! gives the current behaviour a seam and a test suite.

use crate::{BBox, StacCatalog};
use surtgis_core::{GeoTransform, Raster};

/// Result of [`estimate_search_limit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchEstimate {
    /// Bounding-box width in degrees.
    pub width_deg: f64,
    /// Bounding-box height in degrees.
    pub height_deg: f64,
    /// Rough number of ~60 km spatial tiles covering the bbox.
    pub spatial_tiles: usize,
    /// STAC search limit: `spatial_tiles × max_scenes × 5`, floored at
    /// 1 000 and capped at 10 000.
    pub limit: u32,
}

/// Estimate how many STAC items a composite search needs to page through.
///
/// A Sentinel-2 tile is ~110 × 110 km with ~73 acquisition dates per year
/// per tile; the ×5 safety margin keeps dense time ranges from truncating
/// the scene pool before date selection happens.
pub fn estimate_search_limit(bbox: &BBox, max_scenes: usize) -> SearchEstimate {
    let width_deg = (bbox.max_x - bbox.min_x).abs();
    let height_deg = (bbox.max_y - bbox.min_y).abs();
    let tiles_x = ((width_deg * 111.0) / 60.0).ceil().max(1.0) as usize;
    let tiles_y = ((height_deg * 111.0) / 60.0).ceil().max(1.0) as usize;
    let spatial_tiles = tiles_x * tiles_y;
    let limit = ((spatial_tiles * max_scenes * 5).max(1000) as u32).min(10000);
    SearchEstimate {
        width_deg,
        height_deg,
        spatial_tiles,
        limit,
    }
}

/// Pick the scene dates to composite, preferring dates with more spatial
/// tiles (better bbox coverage — e.g. both Sentinel-2 orbit columns
/// represented), tie-broken by ascending date for determinism.
///
/// `date_coverage` pairs each date with its item count; the function sorts
/// a copy and returns up to `max_scenes` dates.
pub fn select_dates_by_coverage(
    mut date_coverage: Vec<(String, usize)>,
    max_scenes: usize,
) -> Vec<String> {
    date_coverage.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    date_coverage
        .into_iter()
        .take(max_scenes)
        .map(|(d, _)| d)
        .collect()
}

/// Catalog-specific memory-model constants.
///
/// Earth Search serves 1024² COG tiles and mixes UTM zones (reprojection
/// doubles the working set); Planetary Computer serves 512² tiles in a
/// single grid. `tile_concurrency` reflects HTTP/2 multiplexing limits.
#[derive(Debug, Clone, Copy)]
pub struct BudgetProfile {
    /// Working-set inflation of the active band mosaic, in multiples of the
    /// strip buffer.
    pub band_inflation: usize,
    /// Working-set inflation of a scene's cloud-mask mosaic.
    pub mask_inflation: usize,
    /// Concurrent tile downloads/decodes.
    pub tile_concurrency: usize,
    /// Decoded size of one tile in bytes (tile_px² × 8).
    pub tile_internal_bytes: usize,
    /// Human-readable label for budget reports.
    pub label: &'static str,
}

/// The per-catalog constants of the calibrated budget model.
pub fn budget_profile(catalog: &StacCatalog) -> BudgetProfile {
    match catalog {
        StacCatalog::EarthSearch => BudgetProfile {
            band_inflation: 14,
            mask_inflation: 4,
            tile_concurrency: 32,
            tile_internal_bytes: 1024 * 1024 * 8,
            label: "Earth Search (1024² COGs, multi-UTM reprojection)",
        },
        StacCatalog::PlanetaryComputer => BudgetProfile {
            band_inflation: 8,
            mask_inflation: 2,
            tile_concurrency: 48,
            tile_internal_bytes: 512 * 512 * 8,
            label: "Planetary Computer (512² COGs)",
        },
        StacCatalog::Custom(_) => BudgetProfile {
            band_inflation: 14,
            mask_inflation: 4,
            tile_concurrency: 32,
            tile_internal_bytes: 1024 * 1024 * 8,
            label: "custom catalog",
        },
    }
}

/// Mask-cache calibration: observed footprint is ~1.8× the nominal term,
/// from strip→strip double-tenancy and decoded-tile staging
/// (BUG_RAM_V070_BUDGET_VS_REAL_MAULE_PC, v0.7.1).
pub const MASK_INFLATION_CALIB: f64 = 1.8;
/// mimalloc retains pages between strips: ~10% over the live working set.
pub const ALLOC_OVERHEAD_FRAC: f64 = 0.10;

/// Inputs of [`plan_strips`].
#[derive(Debug, Clone)]
pub struct StripPlanInput {
    /// Target STAC catalog (selects the [`BudgetProfile`]).
    pub catalog: StacCatalog,
    /// Number of output bands.
    pub n_bands: usize,
    /// Number of scenes (dates) being composited.
    pub n_scenes: usize,
    /// Output grid rows.
    pub out_rows: usize,
    /// Output grid columns.
    pub out_cols: usize,
    /// Requested `--band-chunk-size` (clamped into `[1, n_bands]`).
    pub band_chunk_size: usize,
    /// Requested strip height in rows (upper bound; the model may cap it).
    pub strip_rows_cfg: usize,
    /// RAM budget in GB (caller resolves `SURTGIS_RAM_BUDGET_GB`).
    pub budget_gb: f64,
}

/// GB breakdown of the predicted peak, for budget reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetBreakdown {
    /// Persistent output buffers: `n_bands × cells × 8`.
    pub output_gb: f64,
    /// Cloud-mask mosaics held for the whole strip (calibrated ×1.8).
    pub mask_cache_gb: f64,
    /// Per-band scene strips for the active band chunk.
    pub scene_strips_gb: f64,
    /// Active-band mosaic working set.
    pub band_working_gb: f64,
    /// Concurrent tile decode.
    pub decode_gb: f64,
    /// Allocator retention (~10% of the variable set).
    pub alloc_overhead_gb: f64,
    /// Sum of all components.
    pub estimated_total_gb: f64,
}

/// Result of [`plan_strips`].
#[derive(Debug, Clone)]
pub struct StripPlan {
    /// Strip height in rows after applying the budget cap.
    pub strip_rows: usize,
    /// Number of strips covering `out_rows`.
    pub num_strips: usize,
    /// Effective band chunk size K (clamped).
    pub band_chunk: usize,
    /// Concurrent tile downloads for this catalog.
    pub tile_concurrency: usize,
    /// Catalog label for reports.
    pub catalog_label: &'static str,
    /// `true` when the budget capped `strip_rows` below the requested value.
    pub capped: bool,
    /// Predicted peak-RAM breakdown at the planned strip size.
    pub breakdown: BudgetBreakdown,
}

/// Solve for the strip height that keeps the predicted peak RSS within the
/// budget (the calibrated v0.7.1 model; outer-band structure, so only one
/// band chunk's tile cache is resident at a time).
///
/// Components: output buffers (persistent), per-scene mask mosaics
/// (calibrated ×[`MASK_INFLATION_CALIB`]), per-scene band strips and the
/// active mosaic working set (both scaled by the band chunk K), fixed
/// concurrent-decode headroom, and ~10% allocator retention
/// ([`ALLOC_OVERHEAD_FRAC`]). The solved strip height is clamped to
/// `[8, 512]` rows and never exceeds `strip_rows_cfg`.
pub fn plan_strips(input: &StripPlanInput) -> StripPlan {
    let profile = budget_profile(&input.catalog);
    let n_bands = input.n_bands;
    let n_scenes = input.n_scenes;
    let out_cols = input.out_cols;
    let total_cells = input.out_rows * out_cols;

    let k = input.band_chunk_size.clamp(1, n_bands.max(1));
    let concurrent_decode_bytes = profile.tile_concurrency * (k + 1) * profile.tile_internal_bytes;

    let global_budget_bytes = (input.budget_gb * 1e9) as usize;
    let output_buffer_bytes = n_bands.max(1) * total_cells.max(1) * 8;

    let per_row_calibrated_bytes = {
        let mask_term = (n_scenes.max(1) * profile.mask_inflation) as f64 * MASK_INFLATION_CALIB;
        let scene_term = (k * n_scenes.max(1)) as f64;
        let band_term = (k * profile.band_inflation) as f64;
        ((mask_term + scene_term + band_term) * (out_cols.max(1) * 8) as f64) as usize
    };

    let variable_budget = global_budget_bytes
        .saturating_sub(output_buffer_bytes)
        .saturating_sub(concurrent_decode_bytes)
        .max(64 * 1024 * 1024);
    let headroom_for_variable_bytes =
        (variable_budget as f64 / (1.0 + ALLOC_OVERHEAD_FRAC)) as usize;
    let auto_strip_rows =
        (headroom_for_variable_bytes / per_row_calibrated_bytes.max(1)).clamp(8, 512);
    let strip_rows = input.strip_rows_cfg.min(auto_strip_rows);

    let output_gb = output_buffer_bytes as f64 / 1e9;
    let decode_gb = concurrent_decode_bytes as f64 / 1e9;
    let mask_cache_gb = (strip_rows * n_scenes * out_cols * 8 * profile.mask_inflation) as f64
        * MASK_INFLATION_CALIB
        / 1e9;
    let scene_strips_gb = (strip_rows * n_scenes * out_cols * 8 * k) as f64 / 1e9;
    let band_working_gb = (strip_rows * out_cols * 8 * profile.band_inflation * k) as f64 / 1e9;
    let variable_gb = mask_cache_gb + scene_strips_gb + band_working_gb;
    let alloc_overhead_gb = variable_gb * ALLOC_OVERHEAD_FRAC;
    let estimated_total_gb = output_gb + decode_gb + variable_gb + alloc_overhead_gb;

    StripPlan {
        strip_rows,
        num_strips: input.out_rows.div_ceil(strip_rows.max(1)),
        band_chunk: k,
        tile_concurrency: profile.tile_concurrency,
        catalog_label: profile.label,
        capped: strip_rows < input.strip_rows_cfg,
        breakdown: BudgetBreakdown {
            output_gb,
            mask_cache_gb,
            scene_strips_gb,
            band_working_gb,
            decode_gb,
            alloc_overhead_gb,
            estimated_total_gb,
        },
    }
}

/// Geometry of one horizontal strip of the output grid.
#[derive(Debug, Clone, Copy)]
pub struct StripBounds {
    /// First output row of the strip (inclusive).
    pub row_start: usize,
    /// One past the last output row (exclusive; last strip may be short).
    pub row_end: usize,
    /// `row_end - row_start`.
    pub rows: usize,
    /// Strip bounding box on the output grid.
    pub bbox: BBox,
    /// Strip bbox padded outward for tile overlap.
    pub padded_bbox: BBox,
}

/// Compute the row range and bounding boxes of strip `strip_idx`.
///
/// `grid_bbox` is the full output-grid bbox (its `max_y` anchors row 0);
/// `pad` (grid CRS units) expands the tile-intersection bbox so border
/// pixels get contributions from neighboring tiles.
pub fn strip_bounds(
    strip_idx: usize,
    strip_rows: usize,
    out_rows: usize,
    out_transform: &GeoTransform,
    grid_bbox: &BBox,
    pad: f64,
) -> StripBounds {
    let row_start = strip_idx * strip_rows;
    let row_end = (row_start + strip_rows).min(out_rows);
    let rows = row_end - row_start;

    let ph = out_transform.pixel_height.abs();
    let strip_min_y = grid_bbox.max_y - (row_end as f64 * ph);
    let strip_max_y = grid_bbox.max_y - (row_start as f64 * ph);
    let bbox = BBox::new(grid_bbox.min_x, strip_min_y, grid_bbox.max_x, strip_max_y);
    let padded_bbox = BBox::new(
        bbox.min_x - pad,
        bbox.min_y - pad,
        bbox.max_x + pad,
        bbox.max_y + pad,
    );

    StripBounds {
        row_start,
        row_end,
        rows,
        bbox,
        padded_bbox,
    }
}

impl StripBounds {
    /// Empty reference raster spanning this strip on the output grid, used
    /// as the resampling target for scene mosaics.
    pub fn reference_raster(&self, out_transform: &GeoTransform, out_cols: usize) -> Raster<f64> {
        let mut r = Raster::<f64>::new(self.rows, out_cols);
        r.set_transform(GeoTransform::new(
            self.bbox.min_x,
            self.bbox.max_y,
            out_transform.pixel_width,
            out_transform.pixel_height,
        ));
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_limit_floor_and_cap() {
        // Tiny bbox, 1 scene → floor of 1000.
        let bb = BBox::new(0.0, 0.0, 0.1, 0.1);
        assert_eq!(estimate_search_limit(&bb, 1).limit, 1000);
        // Huge bbox, many scenes → capped at 10000.
        let bb = BBox::new(-75.0, -45.0, -65.0, -20.0);
        let est = estimate_search_limit(&bb, 50);
        assert_eq!(est.limit, 10000);
        assert!(est.spatial_tiles > 100);
    }

    #[test]
    fn date_selection_prefers_coverage_then_earlier_date() {
        let coverage = vec![
            ("2026-01-05".to_string(), 2),
            ("2026-01-01".to_string(), 4),
            ("2026-01-03".to_string(), 4),
            ("2026-01-02".to_string(), 1),
        ];
        let dates = select_dates_by_coverage(coverage, 3);
        // 4-tile dates first (tie → ascending date), then the 2-tile date.
        assert_eq!(dates, vec!["2026-01-01", "2026-01-03", "2026-01-05"]);
    }

    #[test]
    fn plan_caps_strip_rows_under_budget_pressure() {
        // Maule-like: 10 bands, 8 scenes, wide grid, small budget.
        let input = StripPlanInput {
            catalog: StacCatalog::EarthSearch,
            n_bands: 10,
            n_scenes: 8,
            out_rows: 20_000,
            out_cols: 20_000,
            band_chunk_size: 1,
            strip_rows_cfg: 512,
            budget_gb: 12.0,
        };
        let plan = plan_strips(&input);
        assert!(plan.capped, "512-row strips cannot fit a 12 GB budget here");
        assert!(plan.strip_rows >= 8, "clamped to at least 8 rows");
        assert!(plan.strip_rows < 512);
        assert_eq!(plan.num_strips, input.out_rows.div_ceil(plan.strip_rows));
        // Predicted peak must respect the budget within the model's own
        // ±10% tolerance (the output term is fixed, so allow the fixed
        // floor of 8 rows to exceed it only when strip_rows hit the floor).
        if plan.strip_rows > 8 {
            assert!(
                plan.breakdown.estimated_total_gb <= input.budget_gb * 1.1,
                "estimated {:.1} GB exceeds budget {:.1} GB",
                plan.breakdown.estimated_total_gb,
                input.budget_gb
            );
        }
    }

    #[test]
    fn plan_respects_requested_strip_rows_when_budget_allows() {
        let input = StripPlanInput {
            catalog: StacCatalog::PlanetaryComputer,
            n_bands: 2,
            n_scenes: 3,
            out_rows: 2_000,
            out_cols: 2_000,
            band_chunk_size: 1,
            strip_rows_cfg: 256,
            budget_gb: 16.0,
        };
        let plan = plan_strips(&input);
        assert!(!plan.capped);
        assert_eq!(plan.strip_rows, 256);
        assert_eq!(plan.num_strips, 8);
    }

    #[test]
    fn band_chunk_is_clamped_and_scales_decode() {
        let mut input = StripPlanInput {
            catalog: StacCatalog::PlanetaryComputer,
            n_bands: 4,
            n_scenes: 2,
            out_rows: 1_000,
            out_cols: 1_000,
            band_chunk_size: 99,
            strip_rows_cfg: 128,
            budget_gb: 16.0,
        };
        let plan_big = plan_strips(&input);
        assert_eq!(plan_big.band_chunk, 4, "K clamps to n_bands");
        input.band_chunk_size = 1;
        let plan_small = plan_strips(&input);
        assert!(
            plan_big.breakdown.decode_gb > plan_small.breakdown.decode_gb,
            "concurrent decode must scale with K"
        );
    }

    #[test]
    fn catalog_profiles_differ() {
        let es = budget_profile(&StacCatalog::EarthSearch);
        let pc = budget_profile(&StacCatalog::PlanetaryComputer);
        assert!(es.band_inflation > pc.band_inflation);
        assert!(es.tile_internal_bytes > pc.tile_internal_bytes);
        assert!(pc.tile_concurrency > es.tile_concurrency);
    }

    #[test]
    fn strip_bounds_geometry_and_last_partial_strip() {
        let gt = GeoTransform::new(500_000.0, 6_000_000.0, 10.0, -10.0);
        let grid = BBox::new(
            500_000.0,
            6_000_000.0 - 250.0 * 10.0,
            502_000.0,
            6_000_000.0,
        );

        let s0 = strip_bounds(0, 100, 250, &gt, &grid, 100.0);
        assert_eq!((s0.row_start, s0.row_end, s0.rows), (0, 100, 100));
        assert_eq!(s0.bbox.max_y, 6_000_000.0);
        assert_eq!(s0.bbox.min_y, 6_000_000.0 - 100.0 * 10.0);
        assert_eq!(s0.padded_bbox.max_y, s0.bbox.max_y + 100.0);

        let s2 = strip_bounds(2, 100, 250, &gt, &grid, 100.0);
        assert_eq!((s2.row_start, s2.row_end, s2.rows), (200, 250, 50));
        assert_eq!(s2.bbox.min_y, grid.min_y);

        let r = s2.reference_raster(&gt, 200);
        assert_eq!(r.shape(), (50, 200));
        assert_eq!(r.transform().origin_y, s2.bbox.max_y);
    }
}
