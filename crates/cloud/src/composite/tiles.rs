//! Per-tile building blocks: fetch-outcome classification, retry jitter,
//! overview selection, CRS unification and mosaicking, cache keys.

use crate::error::CloudError;
use crate::tile_index::select_overview_by_ratio;
use crate::{BBox, OverviewInfo};
use surtgis_core::Raster;

/// Outcome of attempting to fetch and decode a single COG tile.
///
/// Distinguishes benign "nothing here" results — the tile is outside the
/// bbox of interest, or the asset legitimately doesn't exist (404), both
/// expected in mosaics built from irregular STAC tile grids — from a real
/// failure that survived retries. Historically both collapsed into the same
/// `None`, so a run of real download failures (rate limiting, transient
/// network errors, auth issues) looked identical to "this tile just isn't
/// part of the scene" and silently degraded the composite (gap-filled with
/// no diagnostic). Callers should count `Failed` separately and report it.
#[derive(Debug)]
pub enum TileOutcome<T> {
    /// Tile fetched and decoded successfully.
    Data(T),
    /// Expected, benign absence: bbox doesn't intersect this tile, or the
    /// server returned 404 for the asset.
    OutsideOrMissing,
    /// The download failed after exhausting retries. Carries a short error
    /// description for diagnostics.
    Failed(String),
}

/// Classify a [`CloudError`] from a tile open/read as benign ("no data
/// here") or not, matching on the *typed* error variants instead of
/// substring-matching `.to_string()` output.
///
/// Returns `Some(TileOutcome::OutsideOrMissing)` for a benign miss, or
/// `None` if the error is a real failure the caller should count/report
/// (and, on the first attempt, retry once).
pub fn classify_benign_tile_error<T>(e: &CloudError) -> Option<TileOutcome<T>> {
    match e {
        CloudError::BBoxOutside => Some(TileOutcome::OutsideOrMissing),
        CloudError::HttpStatus { status, .. } if *status == 404 => {
            Some(TileOutcome::OutsideOrMissing)
        }
        _ => None,
    }
}

/// Compute the jitter (in ms, `< base_ms`) to add to a retry backoff.
///
/// Uses wall-clock nanoseconds XORed with a per-task salt (e.g. the tile's
/// index within the current chunk) so concurrently retrying tasks disperse
/// instead of synchronizing after a shared 429.
pub fn retry_jitter_ms(task_salt: u64, base_ms: u64) -> u64 {
    let base_ms = base_ms.max(1);
    let now_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    (now_nanos ^ task_salt) % base_ms
}

/// Decide which overview level (if any) to read a COG at, given its own
/// native pixel size and a target output-grid pixel size.
///
/// Returns `None` (meaning: read at full native resolution) unless the
/// output grid is substantially coarser than native (`ratio =
/// out_pixel_size / native_pixel_size > 1.5`) — reading full-res tiles just
/// to immediately downsample them to a much coarser output grid wastes
/// network transfer and decode time proportional to `ratio²`. When an
/// overview is used, [`select_overview_by_ratio`] picks the *coarsest*
/// overview whose scale still does not exceed `ratio`.
pub fn overview_for_target_resolution(
    native_width: u32,
    native_height: u32,
    native_pixel_size: f64,
    overviews: &[OverviewInfo],
    out_pixel_size: f64,
) -> Option<usize> {
    if native_pixel_size <= 0.0 || out_pixel_size <= 0.0 {
        return None;
    }
    let ratio = out_pixel_size / native_pixel_size;
    if ratio <= 1.5 {
        return None;
    }
    let mut ifd_dims = vec![(native_width, native_height)];
    ifd_dims.extend(overviews.iter().map(|o| (o.width, o.height)));
    let idx = select_overview_by_ratio(&ifd_dims, ratio);
    if idx > 0 { Some(idx) } else { None }
}

/// Reproject every tile to the CRS of the first tile (in place), so a
/// multi-UTM scene can be mosaicked on a single grid. Tiles whose CRS is
/// unknown or already matches are left untouched.
pub fn unify_tile_crs(tiles: &mut [Raster<f64>]) {
    if tiles.len() <= 1 {
        return;
    }
    let target_epsg = tiles[0].crs().and_then(|c| c.epsg());
    if let Some(target) = target_epsg {
        for tile in tiles.iter_mut().skip(1) {
            let Some(src_epsg) = tile.crs().and_then(|c| c.epsg()) else {
                continue;
            };
            if src_epsg != target
                && let Some(reprojected) =
                    crate::reproject::reproject_raster_utm(tile, src_epsg, target)
            {
                *tile = reprojected;
            }
        }
    }
}

/// Mosaic tiles into a single raster; `None` if the input is empty.
/// Unifies CRS first (see [`unify_tile_crs`]).
pub fn mosaic_tile_rasters(mut tiles: Vec<Raster<f64>>) -> Option<Raster<f64>> {
    if tiles.is_empty() {
        return None;
    }
    if tiles.len() == 1 {
        return Some(tiles.into_iter().next().unwrap());
    }
    unify_tile_crs(&mut tiles);
    let refs: Vec<&Raster<f64>> = tiles.iter().collect();
    surtgis_core::mosaic(&refs, None).ok()
}

/// Stable cache key (hex) for a COG tile read: hashes the URL with its
/// query string stripped (SAS tokens change on every signing, the object
/// doesn't) together with the requested bbox (same COG, different strip =
/// different cache entry).
pub fn cog_cache_key(href: &str, bb: &BBox) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let base_url = href.split('?').next().unwrap_or(href);
    let key = format!(
        "{}__{:.2}_{:.2}_{:.2}_{:.2}",
        base_url, bb.min_x, bb.min_y, bb.max_x, bb.max_y
    );
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Reproject a bbox between two projected CRSs.
///
/// With the `projections` feature, uses proj4rs corner-transform (accurate
/// across UTM zones). Otherwise falls back to a WGS84 round-trip via the
/// native UTM helpers. Used to translate an output-grid strip bbox into the
/// CRS of a tile that lives in a different UTM zone.
pub fn reproject_bbox_between_crs(bbox: &BBox, from_epsg: u32, to_epsg: u32) -> BBox {
    #[cfg(feature = "projections")]
    {
        use proj4rs::Proj;
        if let (Ok(src), Ok(dst)) = (
            Proj::from_epsg_code(from_epsg as u16),
            Proj::from_epsg_code(to_epsg as u16),
        ) {
            let corners = [
                (bbox.min_x, bbox.min_y),
                (bbox.min_x, bbox.max_y),
                (bbox.max_x, bbox.min_y),
                (bbox.max_x, bbox.max_y),
            ];
            let (mut min_x, mut min_y) = (f64::MAX, f64::MAX);
            let (mut max_x, mut max_y) = (f64::MIN, f64::MIN);
            for &(x, y) in &corners {
                if let Ok((rx, ry)) = proj4rs::adaptors::transform_xy(&src, &dst, x, y) {
                    min_x = min_x.min(rx);
                    min_y = min_y.min(ry);
                    max_x = max_x.max(rx);
                    max_y = max_y.max(ry);
                }
            }
            if min_x < max_x && min_y < max_y {
                return BBox::new(min_x, min_y, max_x, max_y);
            }
        }
    }
    let wgs84 = crate::reproject::reproject_bbox_from_utm(bbox, from_epsg);
    crate::reproject::reproject_bbox_to_cog(&wgs84, to_epsg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the jitter bug: the previous implementation used
    /// `Instant::now().elapsed().subsec_nanos()`, which is ~0ns essentially
    /// always, so distinct tasks always computed the same "jitter".
    #[test]
    fn jitter_varies_across_tasks_not_always_zero() {
        let base_ms = 500;
        let samples: Vec<u64> = (0..64u64)
            .map(|salt| retry_jitter_ms(salt, base_ms))
            .collect();
        let distinct: std::collections::HashSet<u64> = samples.iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "expected jitter to differ across tasks, got all-identical values: {:?}",
            samples
        );
        assert!(
            samples.iter().any(|&j| j != 0),
            "jitter was always 0 — reproduces the Instant::now().elapsed() bug"
        );
    }

    #[test]
    fn jitter_is_bounded_by_base_ms() {
        for salt in 0..32u64 {
            let j = retry_jitter_ms(salt, 500);
            assert!(j < 500, "jitter {} not < base_ms 500", j);
        }
    }

    #[test]
    fn jitter_handles_zero_base_ms_without_panicking() {
        let _ = retry_jitter_ms(42, 0);
    }

    #[test]
    fn classify_benign_tile_error_treats_404_and_bbox_outside_as_benign() {
        let bbox_outside = CloudError::BBoxOutside;
        assert!(matches!(
            classify_benign_tile_error::<()>(&bbox_outside),
            Some(TileOutcome::OutsideOrMissing)
        ));

        let not_found = CloudError::HttpStatus {
            status: 404,
            url: "https://example.com/cog.tif".into(),
        };
        assert!(matches!(
            classify_benign_tile_error::<()>(&not_found),
            Some(TileOutcome::OutsideOrMissing)
        ));
    }

    #[test]
    fn classify_benign_tile_error_treats_persistent_errors_as_not_benign() {
        let rate_limited = CloudError::HttpStatus {
            status: 429,
            url: "https://example.com/cog.tif".into(),
        };
        assert!(classify_benign_tile_error::<()>(&rate_limited).is_none());

        let server_error = CloudError::HttpStatus {
            status: 500,
            url: "https://example.com/cog.tif".into(),
        };
        assert!(classify_benign_tile_error::<()>(&server_error).is_none());

        let network = CloudError::Network("connection reset".into());
        assert!(classify_benign_tile_error::<()>(&network).is_none());
    }

    #[test]
    fn overview_stays_full_res_when_ratio_at_or_below_threshold() {
        let overviews = vec![
            OverviewInfo {
                index: 1,
                width: 5000,
                height: 5000,
            },
            OverviewInfo {
                index: 2,
                width: 2500,
                height: 2500,
            },
        ];
        assert_eq!(
            overview_for_target_resolution(10_000, 10_000, 10.0, &overviews, 10.0),
            None
        );
        // threshold is "> 1.5", not ">="
        assert_eq!(
            overview_for_target_resolution(10_000, 10_000, 10.0, &overviews, 15.0),
            None
        );
    }

    #[test]
    fn overview_uses_coarsest_overview_above_threshold() {
        let overviews = vec![
            OverviewInfo {
                index: 1,
                width: 5000,
                height: 5000,
            }, // scale 2
            OverviewInfo {
                index: 2,
                width: 1250,
                height: 1250,
            }, // scale 8
        ];
        // ratio 9 → coarsest overview whose scale (8) doesn't exceed 9.
        assert_eq!(
            overview_for_target_resolution(10_000, 10_000, 10.0, &overviews, 90.0),
            Some(2)
        );
    }

    #[test]
    fn overview_handles_degenerate_native_pixel_size() {
        let overviews: Vec<OverviewInfo> = Vec::new();
        assert_eq!(
            overview_for_target_resolution(100, 100, 0.0, &overviews, 10.0),
            None
        );
    }

    #[test]
    fn cache_key_ignores_query_string_but_not_bbox() {
        let bb = BBox::new(0.0, 0.0, 10.0, 10.0);
        let a = cog_cache_key("https://x.blob.core.windows.net/c/t.tif?sig=AAA", &bb);
        let b = cog_cache_key("https://x.blob.core.windows.net/c/t.tif?sig=BBB", &bb);
        assert_eq!(a, b, "SAS re-signing must not change the cache key");

        let bb2 = BBox::new(0.0, 0.0, 10.0, 20.0);
        let c = cog_cache_key("https://x.blob.core.windows.net/c/t.tif?sig=AAA", &bb2);
        assert_ne!(a, c, "different strips of the same COG must not collide");
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn mosaic_passthrough_and_empty() {
        assert!(mosaic_tile_rasters(Vec::new()).is_none());
        let r = Raster::<f64>::filled(3, 3, 7.0);
        let out = mosaic_tile_rasters(vec![r]).expect("single tile passes through");
        assert_eq!(out.get(1, 1).unwrap(), 7.0);
    }
}
