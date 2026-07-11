//! The composite orchestrator: search → asset resolution → output-grid
//! determination → strip loop → sink.
//!
//! [`CompositeEngine`] owns the STAC client and download runtime and drives
//! the whole pipeline described by a [`CompositeSpec`]. Everything that is
//! *not* pure cloud/STAC work is injected as a trait so the engine stays
//! decoupled from `surtgis-algorithms` (cloud masking), from the CLI (asset
//! naming, progress reporting) and from the output medium (in-memory buffers
//! today, streaming COG tiles tomorrow):
//!
//! - [`AssetResolver`] — map a band/mask key to a STAC asset href.
//! - [`MaskApplier`] — apply a scene's cloud mask to a data raster.
//! - [`StripSink`] — receive each composited band strip.
//! - [`CompositeProgress`] — receive progress and RAM-diagnostic hooks.
//!
//! The RAM-critical structure is preserved verbatim from the CLI pipeline it
//! replaces (outer-band loop, per-scene mask cache dropped at strip end,
//! SAS-token refresh, bounded-concurrency chunked download); allocator
//! management (e.g. `mi_collect`) lives in the caller's
//! [`CompositeProgress::strip_finished`] hook, since a library must not pin
//! a global allocator.

use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};

use ndarray::Array2;
use surtgis_core::{CRS, GeoTransform, Raster};

use super::budget::MemoryBudget;
use super::plan::{StripPlan, StripPlanInput, plan_strips, strip_bounds};
use super::reduce::composite_scene_strips;
use super::spec::{CompositeSpec, OutputGrid};
use super::tiles::{
    TileOutcome, classify_benign_tile_error, cog_cache_key, mosaic_tile_rasters,
    overview_for_target_resolution, reproject_bbox_between_crs, retry_jitter_ms,
};
use crate::blocking::{CogReaderBlocking, StacClientBlocking};
use crate::stac_models::StacItem;
use crate::{
    BBox, CloudError, CogReaderOptions, Result, StacCatalog, StacClientOptions, StacSearchParams,
    reproject,
};

/// Resolve a band/mask key to the original (unsigned) asset href in a STAC
/// item. The CLI implements this over its band-alias table; the engine then
/// signs the returned href via the STAC client.
pub trait AssetResolver: Send + Sync {
    /// Return the original href of asset `key` in `item`, or `None` if the
    /// item has no matching asset.
    fn resolve(&self, item: &StacItem, key: &str) -> Option<String>;
}

/// Apply a scene's mosaicked cloud mask to a data raster, returning the
/// masked data. Implemented by the caller over its cloud-mask strategy
/// (e.g. `surtgis_algorithms::imagery::CloudMaskStrategy`).
pub trait MaskApplier: Send + Sync {
    /// Mask `data` using `mask` (both on the same grid after mosaicking).
    fn apply(&self, data: &Raster<f64>, mask: &Raster<f64>) -> Raster<f64>;
}

/// Receive composited band strips as the engine produces them. Called once
/// per (band, strip) with the band's composited values for that strip.
pub trait StripSink {
    /// Called once, after the output grid is determined and before the first
    /// strip, so the sink can size its storage. Default: no-op.
    fn begin(&mut self, grid: &OutputGrid) -> Result<()> {
        let _ = grid;
        Ok(())
    }
    /// Accept the composited `strip` for band `band_idx`, whose top row is
    /// `strip_row_start` on the output grid.
    fn accept(&mut self, band_idx: usize, strip_row_start: usize, strip: Array2<f64>)
    -> Result<()>;
}

/// Progress and RAM-diagnostic callbacks. Every method has a no-op default,
/// so a headless caller can pass [`NoProgress`].
#[allow(unused_variables)]
pub trait CompositeProgress {
    /// STAC search finished: total items, distinct dates, dates selected.
    fn search_done(&mut self, items: usize, dates_total: usize, dates_used: usize) {}
    /// Scenes resolved (all requested bands present).
    fn scenes_resolved(&mut self, n_scenes: usize, n_bands: usize) {}
    /// Output grid determined.
    fn grid_ready(&mut self, grid: &OutputGrid, n_scenes: usize) {}
    /// Strip-sizing plan computed.
    fn plan_ready(&mut self, plan: &StripPlan, budget_gb: f64) {}
    /// A strip started (0-based `idx` of `num`).
    fn strip_started(&mut self, idx: usize, num: usize, row_start: usize, row_end: usize) {}
    /// A strip finished, after its mask cache was dropped. The caller may
    /// reclaim allocator pages here (e.g. `mi_collect`).
    fn strip_finished(&mut self, idx: usize, num: usize) {}
    /// How many scenes contributed data to a band (first/last strip only).
    fn band_contribution(&mut self, band_idx: usize, contributed: usize, total: usize) {}
    /// A RAM-diagnostic checkpoint with a preformatted label.
    fn ram_checkpoint(&mut self, label: &str) {}
    /// A tile download failed after retries.
    fn tile_failed(&mut self, msg: &str) {}
}

/// A [`CompositeProgress`] that does nothing.
pub struct NoProgress;
impl CompositeProgress for NoProgress {}

/// Summary returned by [`CompositeEngine::run`].
#[derive(Debug, Clone)]
pub struct CompositeReport {
    /// Number of scene dates composited.
    pub scenes_used: usize,
    /// Total tiles across all scenes.
    pub total_tiles: usize,
    /// Tiles that failed after retries (real gaps, not benign misses).
    pub failed_tiles: usize,
    /// Distinct scene dates touched by a failure.
    pub failed_dates: usize,
    /// Last failure message, if any.
    pub last_error: Option<String>,
    /// The output grid the composite was written onto.
    pub grid: OutputGrid,
}

/// How long a SAS token is trusted before proactive re-signing.
const TOKEN_REFRESH_THRESHOLD: Duration = Duration::from_secs(30 * 60);
/// Strip-bbox padding (grid CRS units) for tile overlap.
const STRIP_PAD: f64 = 100.0;

/// Per-tile resolved+signed asset references for one STAC item.
struct TileRef {
    /// `(signed_href, original_href)` per band, in `spec.band_keys` order.
    band_hrefs: Vec<(String, String)>,
    mask_href: String,
    original_mask_href: String,
    epsg: Option<u32>,
    signed_at: Instant,
}

/// One scene date with its resolved tiles.
struct Scene {
    date: String,
    tiles: Vec<TileRef>,
    epsg: Option<u32>,
}

/// Drives a [`CompositeSpec`] to completion.
pub struct CompositeEngine {
    spec: CompositeSpec,
    client: StacClientBlocking,
    runtime: tokio::runtime::Runtime,
}

impl CompositeEngine {
    /// Build an engine for `spec`: opens the STAC client for the catalog and
    /// creates the shared 8-worker download runtime.
    pub fn new(spec: CompositeSpec) -> Result<Self> {
        let cat = StacCatalog::from_str_or_url(&spec.catalog);
        let search_limit =
            super::plan::estimate_search_limit(&spec.bbox_wgs84, spec.max_scenes).limit as usize;
        let client_opts = StacClientOptions {
            max_items: search_limit,
            ..StacClientOptions::default()
        };
        let client = StacClientBlocking::new(cat, client_opts)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(8)
            .enable_all()
            .build()
            .map_err(|e| CloudError::Composite(format!("failed to build download runtime: {e}")))?;
        Ok(Self {
            spec,
            client,
            runtime,
        })
    }

    /// Run the composite, streaming each band strip to `sink`.
    ///
    /// On success the sink has received every `(band, strip)`; on an abort
    /// (tile-failure threshold exceeded) an [`CloudError::Composite`] is
    /// returned and the caller should discard any partial sink state.
    pub fn run(
        &mut self,
        resolver: &dyn AssetResolver,
        mask: &dyn MaskApplier,
        sink: &mut dyn StripSink,
        progress: &mut dyn CompositeProgress,
    ) -> Result<CompositeReport> {
        let (dates, grouped) = self.search_dates(progress)?;
        let mut scenes = self.resolve_scenes(&dates, &grouped, resolver, progress)?;
        let grid = self.determine_grid(&scenes)?;
        progress.grid_ready(&grid, scenes.len());
        sink.begin(&grid)?;

        let report = self.run_strips(&mut scenes, &grid, mask, sink, progress)?;
        Ok(report)
    }

    /// STAC search + coverage-based date selection. Returns the chosen dates
    /// and the searched items grouped by date (for resolution).
    #[allow(clippy::type_complexity)]
    fn search_dates(
        &self,
        progress: &mut dyn CompositeProgress,
    ) -> Result<(Vec<String>, BTreeMap<String, Vec<StacItem>>)> {
        let cat = StacCatalog::from_str_or_url(&self.spec.catalog);
        let bb = &self.spec.bbox_wgs84;
        let est = super::plan::estimate_search_limit(bb, self.spec.max_scenes);
        let params = StacSearchParams::new()
            .bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y)
            .datetime(&self.spec.datetime)
            .collections(&[&self.spec.collection])
            .limit(est.limit.min(cat.max_page_size()));
        let items = self.client.search_all(&params)?;
        if items.is_empty() {
            return Err(CloudError::Composite(
                "no items found matching the search criteria".into(),
            ));
        }

        let mut by_date: BTreeMap<String, usize> = BTreeMap::new();
        for item in &items {
            let date = item
                .properties
                .datetime
                .as_deref()
                .unwrap_or("")
                .get(..10)
                .unwrap_or("unknown")
                .to_string();
            *by_date.entry(date).or_default() += 1;
        }
        let dates_total = by_date.len();
        let coverage: Vec<(String, usize)> = by_date.into_iter().collect();
        let dates = super::plan::select_dates_by_coverage(coverage, self.spec.max_scenes);
        progress.search_done(items.len(), dates_total, dates.len());
        Ok((dates, group_items_by_date(items)))
    }

    /// Resolve every requested band + the mask asset for each item of each
    /// selected date, signing hrefs. Items missing any band are skipped.
    fn resolve_scenes(
        &self,
        dates: &[String],
        grouped: &BTreeMap<String, Vec<StacItem>>,
        resolver: &dyn AssetResolver,
        progress: &mut dyn CompositeProgress,
    ) -> Result<Vec<Scene>> {
        let collection = &self.spec.collection;
        let mut scenes: Vec<Scene> = Vec::new();

        for date in dates {
            let Some(items) = grouped.get(date) else {
                continue;
            };
            let mut tiles = Vec::new();
            let mut scene_epsg = None;

            for item in items {
                let tile_epsg = item.epsg();

                let mut band_hrefs: Vec<(String, String)> = Vec::with_capacity(self.spec.n_bands());
                let mut all_ok = true;
                for band_key in &self.spec.band_keys {
                    match resolver.resolve(item, band_key) {
                        Some(original) => match self.client.sign_asset_href(
                            &original,
                            item.collection.as_deref().unwrap_or(collection),
                        ) {
                            Ok(signed) => band_hrefs.push((signed, original)),
                            Err(_) => {
                                all_ok = false;
                                break;
                            }
                        },
                        None => {
                            all_ok = false;
                            break;
                        }
                    }
                }
                if !all_ok {
                    continue; // spatial consistency: need every band
                }

                let (mask_href, original_mask_href) = self
                    .spec
                    .mask_key
                    .as_deref()
                    .and_then(|mk| {
                        resolver.resolve(item, mk).and_then(|orig| {
                            self.client
                                .sign_asset_href(
                                    &orig,
                                    item.collection.as_deref().unwrap_or(collection),
                                )
                                .ok()
                                .map(|signed| (signed, orig))
                        })
                    })
                    .unwrap_or_default();

                if scene_epsg.is_none() {
                    scene_epsg = tile_epsg;
                }
                tiles.push(TileRef {
                    band_hrefs,
                    mask_href,
                    original_mask_href,
                    epsg: tile_epsg,
                    signed_at: Instant::now(),
                });
            }

            if !tiles.is_empty() {
                scenes.push(Scene {
                    date: date.clone(),
                    tiles,
                    epsg: scene_epsg,
                });
            }
        }

        if scenes.is_empty() {
            return Err(CloudError::Composite(
                "no valid scenes found (all bands must be present in each item)".into(),
            ));
        }
        progress.scenes_resolved(scenes.len(), self.spec.n_bands());
        Ok(scenes)
    }

    /// Use the aligned grid if given, else probe the first COG and derive
    /// the output grid from the AOI + native pixel size.
    fn determine_grid(&self, scenes: &[Scene]) -> Result<OutputGrid> {
        if let Some(grid) = &self.spec.align_grid {
            return Ok(grid.clone());
        }
        let first_href = &scenes[0].tiles[0].band_hrefs[0].0;
        let probe = CogReaderBlocking::open(first_href, CogReaderOptions::default())?;
        let meta = probe.metadata();

        let bb = &self.spec.bbox_wgs84;
        let cog_bb = match scenes[0].epsg {
            Some(epsg) if !reproject::is_wgs84(epsg) => reproject::reproject_bbox_to_cog(bb, epsg),
            _ => *bb,
        };
        let pw = meta.geo_transform.pixel_width.abs();
        let ph = meta.geo_transform.pixel_height.abs();
        let cols = ((cog_bb.max_x - cog_bb.min_x) / pw).round() as usize;
        let rows = ((cog_bb.max_y - cog_bb.min_y) / ph).round() as usize;
        Ok(OutputGrid {
            cols,
            rows,
            transform: GeoTransform::new(cog_bb.min_x, cog_bb.max_y, pw, -ph),
            crs: scenes[0].epsg.map(CRS::from_epsg),
            bbox: cog_bb,
        })
    }

    /// The strip loop: per strip, precompute per-scene masks (Phase A), then
    /// per band-chunk download → mask → resample → composite → sink (Phase B).
    fn run_strips(
        &self,
        scenes: &mut [Scene],
        grid: &OutputGrid,
        mask: &dyn MaskApplier,
        sink: &mut dyn StripSink,
        progress: &mut dyn CompositeProgress,
    ) -> Result<CompositeReport> {
        let spec = &self.spec;
        let n_bands = spec.n_bands();
        let n_scenes = scenes.len();
        let out_cols = grid.cols;
        let out_rows = grid.rows;
        let out_epsg = grid.epsg();
        let out_pixel_size = grid.pixel_size();

        let plan = plan_strips(&StripPlanInput {
            catalog: StacCatalog::from_str_or_url(&spec.catalog),
            n_bands,
            n_scenes,
            out_rows,
            out_cols,
            band_chunk_size: spec.band_chunk_size,
            strip_rows_cfg: spec.strip_rows,
            budget_gb: spec.budget_gb,
        });
        progress.plan_ready(&plan, spec.budget_gb);
        let (strip_rows, num_strips, k) = (plan.strip_rows, plan.num_strips, plan.band_chunk);
        // Bound concurrent tile decode by bytes, not a fixed count: each tile
        // acquires `tile_bytes` before decoding, so the number decoding at
        // once emerges from the budget (RAM can't overshoot regardless of any
        // size estimate) and the fixed-count chunk convoy is gone.
        let decode_budget = MemoryBudget::new(plan.decode_budget_bytes);
        let tile_bytes = plan.tile_bytes;

        let mut failed_tiles = 0usize;
        let mut failed_dates: HashSet<String> = HashSet::new();
        let mut last_error: Option<String> = None;
        let mut cumulative_tiles = 0usize;

        for strip_idx in 0..num_strips {
            let bounds = strip_bounds(
                strip_idx,
                strip_rows,
                out_rows,
                &grid.transform,
                &grid.bbox,
                STRIP_PAD,
            );
            let actual_rows = bounds.rows;
            let tile_bb = bounds.padded_bbox;
            progress.strip_started(strip_idx, num_strips, bounds.row_start, bounds.row_end);
            let strip_ref = bounds.reference_raster(&grid.transform, out_cols);

            // --- Phase A: mosaicked cloud masks, one per scene ---
            let mut scene_masks: Vec<Option<Raster<f64>>> = Vec::with_capacity(n_scenes);
            let mut phase_a_tiles = 0usize;
            for scene in scenes.iter_mut() {
                self.refresh_tokens_if_stale(scene);
                let mask_tasks: Vec<(String, BBox)> = scene
                    .tiles
                    .iter()
                    .filter(|t| !t.mask_href.is_empty())
                    .map(|t| {
                        (
                            t.mask_href.clone(),
                            tile_task_bbox(t.epsg, out_epsg, &tile_bb),
                        )
                    })
                    .collect();
                let outcomes = self.download_tiles(
                    &mask_tasks,
                    &decode_budget,
                    tile_bytes,
                    /* zero_to_nan */ false,
                    out_pixel_size,
                    progress,
                );
                phase_a_tiles += mask_tasks.len();
                cumulative_tiles += mask_tasks.len();
                let mut mask_tiles = Vec::with_capacity(outcomes.len());
                for outcome in outcomes {
                    match outcome {
                        TileOutcome::Data(r) => mask_tiles.push(r),
                        TileOutcome::Failed(msg) => {
                            failed_tiles += 1;
                            failed_dates.insert(scene.date.clone());
                            last_error = Some(msg);
                        }
                        TileOutcome::OutsideOrMissing => {}
                    }
                }
                scene_masks.push(mosaic_tile_rasters(mask_tiles));
            }
            progress.ram_checkpoint(&format!(
                "strip {}/{} phase A masks loaded (phase_a_tiles={}, cumulative={})",
                strip_idx + 1,
                num_strips,
                phase_a_tiles,
                cumulative_tiles
            ));

            // --- Phase B: outer band-chunk loop ---
            let mut chunk_start = 0usize;
            while chunk_start < n_bands {
                let chunk_end = (chunk_start + k).min(n_bands);
                let chunk_bands: Vec<usize> = (chunk_start..chunk_end).collect();
                let chunk_k = chunk_bands.len();
                let mut chunk_scene_strips: Vec<Vec<Array2<f64>>> =
                    (0..chunk_k).map(|_| Vec::with_capacity(n_scenes)).collect();

                for (si, scene) in scenes.iter_mut().enumerate() {
                    self.refresh_tokens_if_stale(scene);
                    let mut all_tasks: Vec<(String, BBox)> = Vec::new();
                    let mut task_band_local: Vec<usize> = Vec::new();
                    for (bi_local, &bi) in chunk_bands.iter().enumerate() {
                        for tile in &scene.tiles {
                            let Some((signed, _)) = tile.band_hrefs.get(bi) else {
                                continue;
                            };
                            if signed.is_empty() {
                                continue;
                            }
                            all_tasks.push((
                                signed.clone(),
                                tile_task_bbox(tile.epsg, out_epsg, &tile_bb),
                            ));
                            task_band_local.push(bi_local);
                        }
                    }

                    let all_rasters = self.download_tiles(
                        &all_tasks,
                        &decode_budget,
                        tile_bytes,
                        /* zero_to_nan */ true,
                        out_pixel_size,
                        progress,
                    );
                    cumulative_tiles += all_tasks.len();

                    let mut per_band: Vec<Vec<Raster<f64>>> =
                        (0..chunk_k).map(|_| Vec::new()).collect();
                    for (outcome, &bi_local) in all_rasters.into_iter().zip(task_band_local.iter())
                    {
                        match outcome {
                            TileOutcome::Data(r) => per_band[bi_local].push(r),
                            TileOutcome::Failed(msg) => {
                                failed_tiles += 1;
                                failed_dates.insert(scene.date.clone());
                                last_error = Some(msg);
                            }
                            TileOutcome::OutsideOrMissing => {}
                        }
                    }

                    for (bi_local, data_tiles) in per_band.into_iter().enumerate() {
                        if data_tiles.is_empty() {
                            continue;
                        }
                        let Some(data_m) = mosaic_tile_rasters(data_tiles) else {
                            continue;
                        };
                        let clean = match &scene_masks[si] {
                            Some(m) => mask.apply(&data_m, m),
                            None => data_m,
                        };
                        let resampled = surtgis_core::resample_to_grid(
                            &clean,
                            &strip_ref,
                            surtgis_core::ResampleMethod::Bilinear,
                        )
                        .unwrap_or(clean);
                        if resampled.data().iter().any(|v| v.is_finite()) {
                            chunk_scene_strips[bi_local].push(resampled.data().to_owned());
                        }
                    }
                }

                for (bi_local, &bi) in chunk_bands.iter().enumerate() {
                    let scene_strips = std::mem::take(&mut chunk_scene_strips[bi_local]);
                    if strip_idx == 0 {
                        progress.band_contribution(bi, scene_strips.len(), n_scenes);
                    }
                    let strip_out = composite_scene_strips(&scene_strips, actual_rows, out_cols);
                    sink.accept(bi, bounds.row_start, strip_out)?;
                }

                progress.ram_checkpoint(&format!(
                    "strip {}/{} chunk bands [{}..{}] end (cumulative={})",
                    strip_idx + 1,
                    num_strips,
                    chunk_start,
                    chunk_end,
                    cumulative_tiles
                ));
                chunk_start = chunk_end;
            }

            drop(scene_masks);
            progress.strip_finished(strip_idx, num_strips);

            if spec.max_tile_failures > 0 && failed_tiles > spec.max_tile_failures {
                return Err(CloudError::Composite(format!(
                    "aborting: {} tiles failed after retries (> max_tile_failures={}), \
                     {} scenes affected. Partial output was NOT written. Last error: {}",
                    failed_tiles,
                    spec.max_tile_failures,
                    failed_dates.len(),
                    last_error.as_deref().unwrap_or("(none)")
                )));
            }
        }

        let total_tiles = scenes.iter().map(|s| s.tiles.len()).sum();
        Ok(CompositeReport {
            scenes_used: n_scenes,
            total_tiles,
            failed_tiles,
            failed_dates: failed_dates.len(),
            last_error,
            grid: grid.clone(),
        })
    }

    /// Re-sign a scene's SAS tokens if they are older than the refresh
    /// threshold (cheap no-op when fresh).
    fn refresh_tokens_if_stale(&self, scene: &mut Scene) {
        let stale = scene
            .tiles
            .first()
            .map(|t| t.signed_at.elapsed() > TOKEN_REFRESH_THRESHOLD)
            .unwrap_or(false);
        if !stale {
            return;
        }
        for tile in &mut scene.tiles {
            for (signed, original) in &mut tile.band_hrefs {
                if let Ok(h) = self.client.sign_asset_href(original, "") {
                    *signed = h;
                }
            }
            if !tile.original_mask_href.is_empty()
                && let Ok(h) = self.client.sign_asset_href(&tile.original_mask_href, "")
            {
                tile.mask_href = h;
            }
            tile.signed_at = Instant::now();
        }
    }

    /// Download a batch of COG tiles, bounding concurrent decode by a byte
    /// budget instead of a fixed count, with an on-disk cache and one bounded
    /// retry per tile (the cloud HTTP client has already exhausted its own
    /// retries by the time an error surfaces).
    ///
    /// Every tile task acquires `tile_bytes` of `budget` before opening the
    /// COG and holds it until the decoded raster is handed back, so the number
    /// of tiles decoding (and fetching) at once emerges from the budget — a
    /// bigger budget decodes more in parallel, and RAM can never overshoot.
    /// All tasks are spawned at once; those beyond the budget park on
    /// `acquire`, so there is no fixed-size chunk barrier (the old "convoy").
    fn download_tiles(
        &self,
        tasks: &[(String, BBox)],
        budget: &MemoryBudget,
        tile_bytes: usize,
        zero_to_nan: bool,
        out_pixel_size: f64,
        progress: &mut dyn CompositeProgress,
    ) -> Vec<TileOutcome<Raster<f64>>> {
        if tasks.is_empty() {
            return Vec::new();
        }
        let use_cache = self.spec.use_cache;
        let out_px = out_pixel_size;
        let outcomes = self.runtime.block_on(async {
            use crate::CogReader;

            let mut results: Vec<TileOutcome<Raster<f64>>> = (0..tasks.len())
                .map(|_| TileOutcome::OutsideOrMissing)
                .collect();

            let mut needs_download: Vec<usize> = Vec::with_capacity(tasks.len());
            for (idx, (href, bb)) in tasks.iter().enumerate() {
                if use_cache && let Some(r) = cache_read(&cache_path(href, bb)) {
                    results[idx] = TileOutcome::Data(r);
                    continue;
                }
                needs_download.push(idx);
            }

            // Spawn every tile at once; the byte budget throttles how many
            // actually decode concurrently (no fixed-count chunk barrier).
            let mut handles: Vec<(usize, tokio::task::JoinHandle<TileOutcome<Raster<f64>>>)> =
                Vec::with_capacity(needs_download.len());
            {
                for &idx in &needs_download {
                    let (href, bb) = tasks[idx].clone();
                    let do_cache = use_cache;
                    let zero_nan = zero_to_nan;
                    let href_cache = href.clone();
                    let task_salt = idx as u64;
                    let budget = budget.clone();
                    handles.push((
                        idx,
                        tokio::spawn(async move {
                            // Hold decode budget for this tile's whole
                            // open+read; released when the task returns.
                            let _permit = budget.acquire(tile_bytes).await;
                            const MAX_ATTEMPTS: u8 = 2;
                            let mut last_err: Option<String> = None;
                            for attempt in 0..MAX_ATTEMPTS {
                                if attempt > 0 {
                                    let base_ms: u64 = 500;
                                    let jitter = retry_jitter_ms(task_salt, base_ms);
                                    tokio::time::sleep(Duration::from_millis(base_ms + jitter))
                                        .await;
                                }
                                let mut reader =
                                    match CogReader::open(&href, CogReaderOptions::default()).await
                                    {
                                        Ok(r) => r,
                                        Err(e) => {
                                            if let Some(o) = classify_benign_tile_error(&e) {
                                                return o;
                                            }
                                            last_err = Some(e.to_string());
                                            continue;
                                        }
                                    };
                                let nodata_val = reader.metadata().nodata.unwrap_or(0.0);
                                let overview = {
                                    let meta = reader.metadata();
                                    overview_for_target_resolution(
                                        meta.width,
                                        meta.height,
                                        meta.geo_transform.pixel_width.abs(),
                                        &reader.overviews(),
                                        out_px,
                                    )
                                };
                                match reader.read_bbox::<f64>(&bb, overview).await {
                                    Ok(mut r) => {
                                        if zero_nan {
                                            for val in r.data_mut().iter_mut() {
                                                if *val == nodata_val || *val == 0.0 {
                                                    *val = f64::NAN;
                                                }
                                            }
                                        }
                                        if do_cache {
                                            cache_write(&cache_path(&href_cache, &bb), &r);
                                        }
                                        return TileOutcome::Data(r);
                                    }
                                    Err(e) => {
                                        if let Some(o) = classify_benign_tile_error(&e) {
                                            return o;
                                        }
                                        last_err = Some(e.to_string());
                                    }
                                }
                            }
                            TileOutcome::Failed(
                                last_err.unwrap_or_else(|| "unknown error".to_string()),
                            )
                        }),
                    ));
                }
                for (idx, h) in handles {
                    results[idx] = h
                        .await
                        .unwrap_or_else(|e| TileOutcome::Failed(format!("task panicked: {e}")));
                }
            }
            results
        });
        for o in &outcomes {
            if let TileOutcome::Failed(msg) = o {
                progress.tile_failed(msg);
            }
        }
        outcomes
    }
}

/// Group STAC items by their `YYYY-MM-DD` acquisition date.
fn group_items_by_date(items: Vec<StacItem>) -> BTreeMap<String, Vec<StacItem>> {
    let mut by_date: BTreeMap<String, Vec<StacItem>> = BTreeMap::new();
    for item in items {
        let date = item
            .properties
            .datetime
            .as_deref()
            .unwrap_or("")
            .get(..10)
            .unwrap_or("unknown")
            .to_string();
        by_date.entry(date).or_default().push(item);
    }
    by_date
}

/// Translate the output-grid strip bbox into a tile's CRS when it lives in a
/// different UTM zone.
fn tile_task_bbox(tile_epsg: Option<u32>, out_epsg: Option<u32>, tile_bb: &BBox) -> BBox {
    match (tile_epsg, out_epsg) {
        (Some(te), Some(oe)) if te != oe => reproject_bbox_between_crs(tile_bb, oe, te),
        _ => *tile_bb,
    }
}

/// On-disk cache path for a tile read, mirroring the CLI cache layout.
fn cache_path(href: &str, bb: &BBox) -> std::path::PathBuf {
    let hex = cog_cache_key(href, bb);
    let cache_dir = std::env::var("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
        .join("surtgis")
        .join("cog");
    cache_dir
        .join(&hex[..2])
        .join(&hex[2..4])
        .join(format!("{}.tif", &hex[4..]))
}

fn cache_read(path: &std::path::Path) -> Option<Raster<f64>> {
    if !path.exists() {
        return None;
    }
    surtgis_core::io::read_geotiff::<f64, _>(path, None).ok()
}

fn cache_write(path: &std::path::Path, raster: &Raster<f64>) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = surtgis_core::io::write_geotiff(
        raster,
        path,
        Some(surtgis_core::io::GeoTiffOptions {
            compression: "DEFLATE".into(),
            ..Default::default()
        }),
    );
}
