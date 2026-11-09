//! Composite planning: search sizing, scene-date selection, the memory
//! model that caps `strip_rows`, and per-strip geometry.
//!
//! The calibrated model from the v0.6.22 → v0.7.1 RAM saga (see
//! `docs/postmortems/2026-04-stac-composite-ram.md`) was retired in audit
//! R9-PR2: [`plan_strips`] now sizes strips from plain arithmetic
//! (output + held + decode bytes) with no calibration constants, and the
//! decode phase is additionally bounded at runtime by `MemoryBudget`.

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

/// Nominal decoded size of one COG tile for `catalog`, in bytes
/// (`tile_px² × 8`). This is the servers' real internal tile dimension —
/// Earth Search 1024², Planetary Computer 512² — **not** an empirical fudge
/// factor. Used only as the [`MemoryBudget`](super::MemoryBudget) fallback
/// for a window whose real size can't be computed (bbox outside the granule);
/// the engine otherwise reserves the true `rows×cols×8` of the read.
pub fn nominal_tile_bytes(catalog: &StacCatalog) -> usize {
    match catalog {
        StacCatalog::PlanetaryComputer => 512 * 512 * 8,
        StacCatalog::EarthSearch | StacCatalog::Custom(_) => 1024 * 1024 * 8,
    }
}

/// Fraction of the non-output budget reserved for concurrent tile decode.
const DECODE_BUDGET_FRAC: f64 = 0.25;
/// Decode-budget floor and ceiling.
const DECODE_BUDGET_MIN: usize = 128 * 1024 * 1024;
const DECODE_BUDGET_MAX: usize = 1024 * 1024 * 1024;
/// Strip-height clamp (rows).
const STRIP_ROWS_MIN: usize = 8;
const STRIP_ROWS_MAX: usize = 512;

/// Inputs of [`plan_strips`].
#[derive(Debug, Clone)]
pub struct StripPlanInput {
    /// Target STAC catalog (selects the nominal tile size).
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
    /// Requested strip height in rows (upper bound; the plan may cap it).
    pub strip_rows_cfg: usize,
    /// RAM budget in GB (caller resolves `SURTGIS_RAM_BUDGET_GB`).
    pub budget_gb: f64,
    /// Whether the sink holds the whole output in RAM (`true`) or streams it
    /// to disk (`false`). When streaming, the output buffers don't exist, so
    /// their bytes are not reserved and strips aren't needlessly shrunk.
    pub output_in_ram: bool,
}

/// GB breakdown of the planned working set, for budget reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetBreakdown {
    /// Persistent output buffers: `n_bands × cells × 8`.
    pub output_gb: f64,
    /// Per-strip held set: `strip_rows × n_scenes × (1 mask + k band strips)
    /// × out_cols × 8`.
    pub held_gb: f64,
    /// Concurrent tile decode (the [`MemoryBudget`](super::MemoryBudget) size).
    pub decode_gb: f64,
    /// Sum of the three (excludes allocator retention, which the byte budget
    /// bounds at runtime rather than modelling).
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
    /// Bytes reserved for concurrent tile decode — the size of the
    /// [`MemoryBudget`](super::MemoryBudget) the engine enforces at runtime.
    pub decode_budget_bytes: usize,
    /// Nominal decoded size of one tile — the budget-acquisition fallback
    /// when a window's real size can't be computed (see [`nominal_tile_bytes`]).
    pub tile_bytes: usize,
    /// `true` when the honest sizing capped `strip_rows` below the request.
    pub capped: bool,
    /// GB breakdown of the planned working set.
    pub breakdown: BudgetBreakdown,
}

/// Size the strips and the concurrent-decode budget from honest byte
/// accounting — no empirical inflation constants.
///
/// The v0.7.1 calibrated model (mask ×1.8, +10% allocator, per-catalog
/// band/mask inflation, fixed tile concurrency) is retired: concurrent tile
/// decode is now bounded at runtime by a real [`MemoryBudget`](super::MemoryBudget)
/// of [`decode_budget_bytes`](StripPlan::decode_budget_bytes) (so RAM can't
/// overshoot however wrong an estimate is), and the strip height is chosen so
/// the *held* working set — one mask mosaic per scene plus `k` band strips
/// per scene, at their true `out_cols × 8` bytes per row — fits the remaining
/// budget after the persistent output buffers and the decode reservation. The
/// height is clamped to `[8, 512]` rows and never exceeds `strip_rows_cfg`.
pub fn plan_strips(input: &StripPlanInput) -> StripPlan {
    let n_bands = input.n_bands.max(1);
    let n_scenes = input.n_scenes.max(1);
    let out_cols = input.out_cols.max(1);
    let total_cells = input.out_rows.max(1) * out_cols;
    let k = input.band_chunk_size.clamp(1, n_bands);

    let total_budget = (input.budget_gb * 1e9) as usize;
    let tile_bytes = nominal_tile_bytes(&input.catalog);

    // Persistent output buffers, only when the sink holds them in RAM. A
    // streaming sink writes each strip to disk on arrival and holds nothing,
    // so reserving the full output (tens of GB for a large grid) would drive
    // strips to the floor and cause a storm of tiny reads.
    let output_bytes = n_bands * total_cells * 8;
    let output_held = if input.output_in_ram { output_bytes } else { 0 };
    let after_output = total_budget
        .saturating_sub(output_held)
        .max(64 * 1024 * 1024);

    // Reserve a slice for concurrent tile decode; the byte budget enforces it.
    let decode_budget_bytes = ((after_output as f64 * DECODE_BUDGET_FRAC) as usize)
        .clamp(DECODE_BUDGET_MIN.min(after_output), DECODE_BUDGET_MAX);

    // Honest per-row held set: 1 mask mosaic + k band strips, per scene.
    let per_row_held = n_scenes * (1 + k) * out_cols * 8;
    let held_budget = after_output
        .saturating_sub(decode_budget_bytes)
        .max(STRIP_ROWS_MIN * per_row_held);
    let auto_strip_rows = (held_budget / per_row_held.max(1)).clamp(STRIP_ROWS_MIN, STRIP_ROWS_MAX);
    let strip_rows = input.strip_rows_cfg.min(auto_strip_rows).max(1);

    let output_gb = output_held as f64 / 1e9;
    let held_gb = (strip_rows * per_row_held) as f64 / 1e9;
    let decode_gb = decode_budget_bytes as f64 / 1e9;

    StripPlan {
        strip_rows,
        num_strips: input.out_rows.div_ceil(strip_rows.max(1)),
        band_chunk: k,
        decode_budget_bytes,
        tile_bytes,
        capped: strip_rows < input.strip_rows_cfg,
        breakdown: BudgetBreakdown {
            output_gb,
            held_gb,
            decode_gb,
            estimated_total_gb: output_gb + held_gb + decode_gb,
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
            n_bands: 1,
            n_scenes: 8,
            out_rows: 20_000,
            out_cols: 20_000,
            band_chunk_size: 1,
            strip_rows_cfg: 512,
            budget_gb: 4.5,
            output_in_ram: true,
        };
        let plan = plan_strips(&input);
        assert!(
            plan.capped,
            "512-row strips cannot fit the held set in 4.5 GB here"
        );
        assert!(plan.strip_rows >= 8, "clamped to at least 8 rows");
        assert!(plan.strip_rows < 512);
        assert_eq!(plan.num_strips, input.out_rows.div_ceil(plan.strip_rows));
        // The decode reservation is a real byte budget the engine enforces.
        assert!(plan.decode_budget_bytes >= 128 * 1024 * 1024);
        assert!(plan.decode_budget_bytes <= 1024 * 1024 * 1024);
        // The held set fits the budget after output + decode (above the floor).
        if plan.strip_rows > 8 {
            assert!(
                plan.breakdown.estimated_total_gb <= input.budget_gb * 1.05,
                "planned {:.1} GB exceeds budget {:.1} GB",
                plan.breakdown.estimated_total_gb,
                input.budget_gb
            );
        }
    }

    #[test]
    fn streaming_sink_does_not_reserve_phantom_output_bytes() {
        // A 20k×20k×10-band output is ~32 GB. In RAM it swamps a 4 GB budget
        // and drives strips to the floor; streamed to disk it holds nothing,
        // so the same budget must plan taller strips.
        let base = StripPlanInput {
            catalog: StacCatalog::EarthSearch,
            n_bands: 10,
            n_scenes: 8,
            out_rows: 20_000,
            out_cols: 20_000,
            band_chunk_size: 1,
            strip_rows_cfg: 512,
            budget_gb: 4.0,
            output_in_ram: true,
        };
        let in_ram = plan_strips(&base);
        let streaming = plan_strips(&StripPlanInput {
            output_in_ram: false,
            ..base
        });
        assert_eq!(
            in_ram.breakdown.output_gb,
            0.0_f64.max(in_ram.breakdown.output_gb),
            "sanity"
        );
        assert!(
            streaming.strip_rows > in_ram.strip_rows,
            "streaming ({}) should plan taller strips than in-RAM ({})",
            streaming.strip_rows,
            in_ram.strip_rows
        );
        assert_eq!(
            streaming.breakdown.output_gb, 0.0,
            "streaming sink reserves no output bytes"
        );
        assert!(in_ram.breakdown.output_gb > 0.0);
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
            output_in_ram: true,
        };
        let plan = plan_strips(&input);
        assert!(!plan.capped);
        assert_eq!(plan.strip_rows, 256);
        assert_eq!(plan.num_strips, 8);
    }

    #[test]
    fn band_chunk_is_clamped_and_shrinks_held_strip() {
        let mut input = StripPlanInput {
            catalog: StacCatalog::PlanetaryComputer,
            n_bands: 4,
            n_scenes: 2,
            out_rows: 1_000,
            out_cols: 1_000,
            band_chunk_size: 99,
            strip_rows_cfg: 512,
            budget_gb: 2.0,
            output_in_ram: true,
        };
        let plan_big = plan_strips(&input);
        assert_eq!(plan_big.band_chunk, 4, "K clamps to n_bands");
        input.band_chunk_size = 1;
        let plan_small = plan_strips(&input);
        // More bands per chunk = a larger held set per row, so under budget
        // pressure the strip height must shrink (not exceed) vs K=1.
        assert!(
            plan_big.strip_rows <= plan_small.strip_rows,
            "larger band chunk should not give taller strips under budget pressure"
        );
    }

    #[test]
    fn nominal_tile_size_differs_by_catalog() {
        assert_eq!(
            nominal_tile_bytes(&StacCatalog::PlanetaryComputer),
            512 * 512 * 8
        );
        assert_eq!(
            nominal_tile_bytes(&StacCatalog::EarthSearch),
            1024 * 1024 * 8
        );
    }

    #[test]
    fn decode_budget_reserved_within_bounds() {
        let plan = plan_strips(&StripPlanInput {
            catalog: StacCatalog::EarthSearch,
            n_bands: 4,
            n_scenes: 4,
            out_rows: 4_000,
            out_cols: 4_000,
            band_chunk_size: 1,
            strip_rows_cfg: 256,
            budget_gb: 16.0,
            output_in_ram: true,
        });
        assert!((128 * 1024 * 1024..=1024 * 1024 * 1024).contains(&plan.decode_budget_bytes));
        assert_eq!(plan.tile_bytes, 1024 * 1024 * 8);
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
