//! Core COG reader: open remote COGs, read by bounding box or full extent.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use lru::LruCache;
use ndarray::Array2;
use surtgis_core::crs::CRS;
use surtgis_core::raster::{GeoTransform, Raster, RasterElement};

use crate::auth::{CloudAuth, NoAuth};
use crate::cache::{TileCache, TileKey};
use crate::decompress;
use crate::error::{CloudError, Result};
use crate::geotiff_keys::{self, GeoTiffMeta};
use crate::http::HttpClient;
use crate::ifd::{self, IfdInfo, RawTagEntry, TiffByteOrder, tags};
use crate::tile_index::{self, BBox, TileMapping};

/// Whether verbose tile-decode diagnostics should be emitted.
///
/// These were invaluable when tracking the `bps=15` striping bug, so they are
/// kept available but silenced by default — set `SURTGIS_COG_DEBUG=1` to
/// re-enable them. FATAL decode errors are always printed regardless.
fn cog_debug() -> bool {
    std::env::var_os("SURTGIS_COG_DEBUG").is_some()
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Options for configuring a [`CogReader`].
pub struct CogReaderOptions {
    /// Maximum number of concurrent HTTP fetches (default: 8).
    pub max_concurrent_fetches: usize,
    /// Number of tiles to keep in the LRU cache (default: 128).
    pub cache_capacity: usize,
    /// Timeout per HTTP request (default: 30 s).
    pub request_timeout: Duration,
    /// Maximum retries on transient failures (default: 3).
    pub max_retries: u32,
    /// Authentication provider (default: [`NoAuth`]).
    pub auth: Box<dyn CloudAuth>,
}

impl Default for CogReaderOptions {
    fn default() -> Self {
        Self {
            max_concurrent_fetches: 16,
            cache_capacity: 256,
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
            auth: Box::new(NoAuth),
        }
    }
}

/// Metadata exposed by [`CogReader::metadata`].
#[derive(Debug, Clone)]
pub struct CogMetadata {
    /// Source URL of the COG.
    pub url: String,
    /// Full-resolution image width in pixels.
    pub width: u32,
    /// Full-resolution image height in pixels.
    pub height: u32,
    /// Tile width in pixels.
    pub tile_width: u32,
    /// Tile height in pixels.
    pub tile_height: u32,
    /// Bits per sample.
    pub bits_per_sample: u16,
    /// Sample format: 1=uint, 2=int, 3=float.
    pub sample_format: u16,
    /// TIFF compression code.
    pub compression: u16,
    /// Affine transform mapping pixel coordinates to world coordinates.
    pub geo_transform: GeoTransform,
    /// Coordinate reference system, if it could be resolved from GeoKeys.
    pub crs: Option<CRS>,
    /// Nodata value, if declared.
    pub nodata: Option<f64>,
    /// Number of overview (reduced-resolution) levels.
    pub num_overviews: usize,
}

/// Overview level information.
#[derive(Debug, Clone)]
pub struct OverviewInfo {
    /// Index of the overview in the IFD chain (0 = full resolution).
    pub index: usize,
    /// Width of this overview level in pixels.
    pub width: u32,
    /// Height of this overview level in pixels.
    pub height: u32,
}

/// Cloud Optimized GeoTIFF reader.
///
/// Reads tiles on-demand via HTTP Range requests with LRU caching.
pub struct CogReader {
    url: String,
    client: Arc<HttpClient>,
    #[allow(dead_code)]
    byte_order: TiffByteOrder,
    ifds: Vec<IfdInfo>,
    geo_meta: GeoTiffMeta,
    cache: TileCache,
    options: CogReaderOptions,
}

// ---------------------------------------------------------------------------
// Structural metadata cache (IFDs + GeoTIFF meta), keyed by URL
// ---------------------------------------------------------------------------

/// Everything [`CogReader::open`] extracts from a COG's header bytes before
/// any tile data is read: TIFF byte order, the parsed IFD chain (full
/// resolution + overviews), and GeoTIFF metadata (transform/CRS/nodata).
///
/// This depends only on the remote file's *bytes*, not on any per-open
/// option (auth, timeouts, cache sizes, ...), so it is safe to share across
/// repeated `open()` calls for the same URL.
#[derive(Debug, Clone)]
struct CachedCogHeader {
    byte_order: TiffByteOrder,
    ifds: Vec<IfdInfo>,
    geo_meta: GeoTiffMeta,
}

/// Capacity of the process-wide IFD/header cache, in distinct COG URLs.
const IFD_CACHE_CAPACITY: usize = 256;

/// Process-wide cache of [`CachedCogHeader`], keyed by [`cache_key_for_url`].
static IFD_CACHE: OnceLock<Mutex<LruCache<String, Arc<CachedCogHeader>>>> = OnceLock::new();

fn ifd_cache() -> &'static Mutex<LruCache<String, Arc<CachedCogHeader>>> {
    IFD_CACHE.get_or_init(|| {
        Mutex::new(LruCache::new(
            NonZeroUsize::new(IFD_CACHE_CAPACITY).expect("capacity is a nonzero constant"),
        ))
    })
}

/// Cache key for a COG URL: the URL with any query string removed.
///
/// STAC asset hrefs are frequently re-signed with time-limited SAS tokens
/// or presigned-URL query parameters on every catalog fetch, even though
/// they point at the exact same underlying file. Keying the cache on the
/// full URL would make every re-signed href a guaranteed miss; stripping
/// the query string means repeat opens of the same asset — even through
/// differently-signed URLs — share one cache entry.
///
/// This is a plain `split('?')`, not full URL normalization: it does not
/// canonicalize scheme/host casing, trailing slashes, percent-encoding,
/// etc. That's an accepted limitation — STAC catalogs consistently emit
/// the same href casing/encoding for a given asset within one process run.
pub(crate) fn cache_key_for_url(url: &str) -> &str {
    url.split('?').next().unwrap_or(url)
}

fn cached_header(cache_key: &str) -> Option<Arc<CachedCogHeader>> {
    let mut guard = ifd_cache().lock().unwrap_or_else(|e| e.into_inner());
    guard.get(cache_key).cloned()
}

fn insert_cached_header(cache_key: String, header: CachedCogHeader) {
    let mut guard = ifd_cache().lock().unwrap_or_else(|e| e.into_inner());
    guard.put(cache_key, Arc::new(header));
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl CogReader {
    /// Open a remote COG by URL.
    ///
    /// Fetches the TIFF header and all IFD chains (full resolution + overviews)
    /// in as few HTTP Range requests as possible.
    ///
    /// The parsed IFD chain and GeoTIFF metadata (everything up to but not
    /// including tile pixel data) are cached process-wide, keyed by URL with
    /// the query string stripped (see [`cache_key_for_url`]) — so re-opening
    /// the same COG (e.g. once per strip/tile task in the STAC composite
    /// pipeline, or across differently-signed SAS URLs for the same asset)
    /// skips the header HEAD request, the header Range fetch, and IFD/GeoTIFF
    /// parsing entirely after the first open.
    ///
    /// # Cache staleness
    ///
    /// The cache is never invalidated or expired within a process. If the
    /// file at a cached URL/path is replaced with different content between
    /// two `open()` calls in the same process, the second call silently
    /// reuses the first file's stale structural metadata (tile layout,
    /// geotransform, nodata, ...) — though tile *pixel* reads are unaffected,
    /// since those still always go over the network via `client`. This is an
    /// accepted trade-off for the STAC composite use case, where the COGs
    /// backing a catalog item do not change mid-run; it is not safe to rely
    /// on for URLs whose content can mutate during a process's lifetime.
    pub async fn open(url: &str, options: CogReaderOptions) -> Result<Self> {
        let client = Self::client_for(&options)?;
        let auth = options.auth.as_ref();

        let cache_key = cache_key_for_url(url).to_string();

        let (byte_order, ifds, geo_meta) = if let Some(cached) = cached_header(&cache_key) {
            (
                cached.byte_order,
                cached.ifds.clone(),
                cached.geo_meta.clone(),
            )
        } else {
            // 1. Fetch first 64 KiB — usually contains header + first IFD + geotiff tags.
            let initial_size: u64 = 64 * 1024;
            let head_info = client.head(url, auth).await?;

            let file_size = head_info.content_length.unwrap_or(0);
            let fetch_size = if file_size > 0 {
                initial_size.min(file_size)
            } else {
                initial_size
            };

            let header_bytes = client.fetch_range(url, 0, fetch_size, auth).await?;

            // 2. Parse TIFF header.
            let tiff_header = ifd::parse_header(&header_bytes)?;
            let byte_order = tiff_header.byte_order;
            reject_big_endian(byte_order)?;

            // 3. Parse IFD chain.
            let mut ifds = Vec::new();
            let mut ifd_offset = tiff_header.first_ifd_offset as usize;

            while ifd_offset > 0 {
                let ifd_data = self::ensure_data(
                    client.as_ref(),
                    url,
                    auth,
                    &header_bytes,
                    ifd_offset,
                    file_size,
                )
                .await?;

                let raw_ifd = ifd::parse_ifd(byte_order, &ifd_data)?;

                // Resolve external tag values needed for this IFD.
                let ifd_info = resolve_ifd(
                    client.as_ref(),
                    url,
                    auth,
                    byte_order,
                    &raw_ifd.entries,
                    &header_bytes,
                    file_size,
                )
                .await?;

                ifds.push(ifd_info);
                ifd_offset = raw_ifd.next_ifd_offset as usize;
            }

            if ifds.is_empty() {
                return Err(CloudError::NoIfd);
            }

            validate_tiled(&ifds[0])?;

            // 4. Extract GeoTIFF metadata from the first (full-res) IFD.
            let geo_meta = resolve_geotiff_meta(
                client.as_ref(),
                url,
                auth,
                byte_order,
                &ifds[0].raw_entries,
                &header_bytes,
                file_size,
            )
            .await?;

            insert_cached_header(
                cache_key,
                CachedCogHeader {
                    byte_order,
                    ifds: ifds.clone(),
                    geo_meta: geo_meta.clone(),
                },
            );

            (byte_order, ifds, geo_meta)
        };

        let cache = TileCache::new(options.cache_capacity);

        Ok(Self {
            url: url.to_string(),
            client,
            byte_order,
            ifds,
            geo_meta,
            cache,
            options,
        })
    }

    /// Pick the [`HttpClient`] to use for this open: the process-wide shared
    /// client (see [`crate::http::shared_client`]) when the caller left
    /// timeout/retries at their defaults — the common case, and the one the
    /// shared connection pool is designed for — or a private client built to
    /// the caller's exact settings otherwise, preserving prior behaviour for
    /// callers that customize them.
    fn client_for(options: &CogReaderOptions) -> Result<Arc<HttpClient>> {
        let defaults = CogReaderOptions::default();
        if options.request_timeout == defaults.request_timeout
            && options.max_retries == defaults.max_retries
        {
            Ok(crate::http::shared_client())
        } else {
            Ok(Arc::new(HttpClient::new(
                options.request_timeout,
                options.max_retries,
            )?))
        }
    }

    /// Read a geographic bounding box into a `Raster<T>`.
    ///
    /// If `overview` is `None`, the full-resolution IFD is used.
    pub async fn read_bbox<T: RasterElement>(
        &mut self,
        bbox: &BBox,
        overview: Option<usize>,
    ) -> Result<Raster<T>> {
        let ifd_idx = overview.unwrap_or(0);
        let ifd = self.get_ifd(ifd_idx)?;
        let gt = self.geo_transform_for(ifd_idx);

        let mapping = tile_index::tiles_for_bbox(
            bbox,
            &gt,
            ifd.width,
            ifd.height,
            ifd.tile_width,
            ifd.tile_height,
        )
        .ok_or(CloudError::BBoxOutside)?;

        self.assemble_raster::<T>(ifd_idx, &ifd, &gt, &mapping)
            .await
    }

    /// Read the full raster extent.
    pub async fn read_full<T: RasterElement>(
        &mut self,
        overview: Option<usize>,
    ) -> Result<Raster<T>> {
        let ifd_idx = overview.unwrap_or(0);
        let ifd = self.get_ifd(ifd_idx)?;
        let gt = self.geo_transform_for(ifd_idx);

        let (min_x, min_y, max_x, max_y) = gt.bounds(ifd.width as usize, ifd.height as usize);
        let bbox = BBox::new(min_x, min_y, max_x, max_y);

        let mapping = tile_index::tiles_for_bbox(
            &bbox,
            &gt,
            ifd.width,
            ifd.height,
            ifd.tile_width,
            ifd.tile_height,
        )
        .ok_or(CloudError::BBoxOutside)?;

        self.assemble_raster::<T>(ifd_idx, &ifd, &gt, &mapping)
            .await
    }

    /// Return metadata about the COG.
    pub fn metadata(&self) -> CogMetadata {
        let ifd = &self.ifds[0];
        CogMetadata {
            url: self.url.clone(),
            width: ifd.width,
            height: ifd.height,
            tile_width: ifd.tile_width,
            tile_height: ifd.tile_height,
            bits_per_sample: ifd.bits_per_sample,
            sample_format: ifd.sample_format,
            compression: ifd.compression,
            geo_transform: self.geo_meta.geo_transform,
            crs: self.geo_meta.crs.clone(),
            nodata: self.geo_meta.nodata,
            num_overviews: self.ifds.len().saturating_sub(1),
        }
    }

    /// Return overview information.
    pub fn overviews(&self) -> Vec<OverviewInfo> {
        self.ifds
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, ifd)| OverviewInfo {
                index: i,
                width: ifd.width,
                height: ifd.height,
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Clone an IFD entry to break borrow on self.
    fn get_ifd(&self, idx: usize) -> Result<IfdInfo> {
        self.ifds.get(idx).cloned().ok_or(CloudError::InvalidTiff {
            reason: format!("IFD index {} out of range (have {})", idx, self.ifds.len()),
        })
    }

    /// Compute the GeoTransform for a given IFD level.
    fn geo_transform_for(&self, ifd_idx: usize) -> GeoTransform {
        if ifd_idx == 0 {
            self.geo_meta.geo_transform
        } else {
            let full = &self.ifds[0];
            let ovr = &self.ifds[ifd_idx];
            let sx = full.width as f64 / ovr.width as f64;
            let sy = full.height as f64 / ovr.height as f64;
            GeoTransform::new(
                self.geo_meta.geo_transform.origin_x,
                self.geo_meta.geo_transform.origin_y,
                self.geo_meta.geo_transform.pixel_width * sx,
                self.geo_meta.geo_transform.pixel_height * sy,
            )
        }
    }

    /// Fetch tiles, decompress, and assemble into a `Raster<T>`.
    async fn assemble_raster<T: RasterElement>(
        &mut self,
        ifd_idx: usize,
        ifd: &IfdInfo,
        gt: &GeoTransform,
        mapping: &TileMapping,
    ) -> Result<Raster<T>> {
        let tw = ifd.tile_width as usize;
        let th = ifd.tile_height as usize;
        let bps = ifd.bits_per_sample;
        let sf = ifd.sample_format;
        let compression = ifd.compression;
        let bytes_per_pixel = (bps as usize).div_ceil(8);
        let raw_tile_size = tw * th * bytes_per_pixel;

        let (px_min_col, px_min_row, px_max_col, px_max_row) = mapping.pixel_window;
        let (out_rows, out_cols) = mapping.output_shape;

        // Identify which tiles we need to fetch (not in cache).
        let mut to_fetch: Vec<(usize, u64, u64)> = Vec::new(); // (tile_list_idx, offset, length)
        for (i, tr) in mapping.tiles.iter().enumerate() {
            let key = TileKey {
                ifd_idx,
                tile_idx: tr.tile_idx,
            };
            if self.cache.get(&key).is_none() && tr.tile_idx < ifd.tile_offsets.len() {
                let offset = ifd.tile_offsets[tr.tile_idx];
                let length = ifd.tile_byte_counts[tr.tile_idx];
                if length > 0 {
                    to_fetch.push((i, offset, length));
                }
            }
        }

        // Fetch uncached tiles concurrently (in batches).
        let auth = self.options.auth.as_ref();
        let batch_size = self.options.max_concurrent_fetches;

        let mut fetch_count = 0usize;
        for chunk in to_fetch.chunks(batch_size) {
            let ranges: Vec<(u64, u64)> = chunk.iter().map(|&(_, o, l)| (o, l)).collect();
            let fetched = self.client.fetch_ranges(&self.url, &ranges, auth).await?;

            for (j, &(tile_list_idx, _, _)) in chunk.iter().enumerate() {
                let tr = &mapping.tiles[tile_list_idx];
                let mut raw = decompress::decompress_tile(&fetched[j], compression, raw_tile_size)?;
                // Apply predictor undo (TIFF tag 317):
                //   1 = no predictor
                //   2 = horizontal differencing (integer samples)
                //   3 = floating-point predictor (TIFF TN3 — used by Copernicus DEM)
                match ifd.predictor {
                    1 => {}
                    2 => {
                        decompress::undo_horizontal_differencing(&mut raw, tw, bytes_per_pixel);
                        if fetch_count == 0 && cog_debug() {
                            eprintln!(
                                "    [pred] undo horiz-diff: bps={} tw={} len={}",
                                bytes_per_pixel,
                                tw,
                                raw.len()
                            );
                        }
                    }
                    3 => {
                        decompress::undo_floating_point_predictor(&mut raw, tw, bytes_per_pixel);
                        if fetch_count == 0 && cog_debug() {
                            let sample0 = if raw.len() >= 4 {
                                f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
                            } else {
                                0.0
                            };
                            eprintln!(
                                "    [pred] undo fp-predictor: bps={} tw={} sample[0]={}",
                                bytes_per_pixel, tw, sample0
                            );
                        }
                    }
                    p => {
                        if fetch_count == 0 && cog_debug() {
                            eprintln!("    [pred] UNSUPPORTED: predictor={}", p);
                        }
                    }
                }
                fetch_count += 1;
                let key = TileKey {
                    ifd_idx,
                    tile_idx: tr.tile_idx,
                };
                self.cache.insert(key, raw);
            }
        }

        // Assemble output array from cached tiles.
        let mut output = Array2::<T>::from_elem((out_rows, out_cols), T::default_nodata());
        let mut tiles_written = 0usize;
        let mut tiles_skipped = 0usize;
        let tiles_fetched = mapping.tiles.len();

        for tr in &mapping.tiles {
            let key = TileKey {
                ifd_idx,
                tile_idx: tr.tile_idx,
            };

            let raw = match self.cache.get(&key) {
                Some(data) => {
                    tiles_written += 1;
                    data.clone()
                }
                None => {
                    // Sparse COGs mark intentionally-missing tiles with
                    // byte count 0: the window stays nodata, which is the
                    // correct fill. Any other cache miss means requested
                    // data was never fetched/decoded — that is data loss,
                    // not a skippable condition.
                    let sparse = ifd
                        .tile_byte_counts
                        .get(tr.tile_idx)
                        .is_some_and(|&c| c == 0);
                    if sparse {
                        tiles_skipped += 1;
                        continue;
                    }
                    return Err(CloudError::InvalidTiff {
                        reason: format!(
                            "tile {} (col {}, row {}) was requested but never \
                             fetched or decoded; refusing to return a \
                             partially-empty raster",
                            tr.tile_idx, tr.tile_col, tr.tile_row
                        ),
                    });
                }
            };

            // Debug: log first tile's raw bytes and interpreted values
            if tiles_written == 1 && raw.len() >= 10 && cog_debug() {
                let first_bytes: Vec<String> =
                    raw[..10].iter().map(|b| format!("{:02x}", b)).collect();
                let first_u16 = u16::from_le_bytes([raw[0], raw[1]]);
                let second_u16 = u16::from_le_bytes([raw[2], raw[3]]);
                eprintln!(
                    "    [cog] tile({},{}) idx={} raw[0..10]: {} → u16 LE: [{}, {}, ...]",
                    tr.tile_col,
                    tr.tile_row,
                    tr.tile_idx,
                    first_bytes.join(" "),
                    first_u16,
                    second_u16
                );
                eprintln!(
                    "    [cog] pred={} comp={} bps={} sf={} raw_len={} expected={}",
                    ifd.predictor,
                    compression,
                    bps,
                    sf,
                    raw.len(),
                    tw * th * bytes_per_pixel
                );
            }

            let typed: Vec<T> = decompress::bytes_to_typed(&raw, bps, sf)?;

            // Check if decompressed tile has expected size. Prior to v0.7.5
            // this was a warning and the partial buffer was silently copied
            // into the output, producing the striping artefacts described in
            // BUG_TILE_DECODE_BPS15_STRIPING.md. We now treat the mismatch as
            // a hard error so the run aborts loudly instead of producing
            // scientifically unusable composites.
            let expected_pixels = tw * th;
            if typed.len() != expected_pixels {
                eprintln!(
                    "    [cog] FATAL tile({},{}) idx={} decoded {} px, expected {} (raw={} bytes, bps={} sf={} tw={} th={}).\
                     \n          Aborting to prevent silent striping; see BUG_TILE_DECODE_BPS15_STRIPING.md.",
                    tr.tile_col,
                    tr.tile_row,
                    tr.tile_idx,
                    typed.len(),
                    expected_pixels,
                    raw.len(),
                    bps,
                    sf,
                    tw,
                    th,
                );
                return Err(crate::error::CloudError::Decompress(format!(
                    "tile decode produced {} samples, expected {} (bps={} sf={} tw={} th={} raw_len={})",
                    typed.len(),
                    expected_pixels,
                    bps,
                    sf,
                    tw,
                    th,
                    raw.len(),
                )));
            }

            // Pixel bounds of this tile in full-image coordinates.
            let tile_px_col = tr.tile_col * tw;
            let tile_px_row = tr.tile_row * th;

            // Copy the relevant portion into the output.
            for local_row in 0..th {
                let img_row = tile_px_row + local_row;
                if img_row < px_min_row || img_row >= px_max_row {
                    continue;
                }
                let out_row = img_row - px_min_row;

                for local_col in 0..tw {
                    let img_col = tile_px_col + local_col;
                    if img_col < px_min_col || img_col >= px_max_col {
                        continue;
                    }
                    let out_col = img_col - px_min_col;

                    let tile_linear = local_row * tw + local_col;
                    if tile_linear < typed.len() {
                        output[(out_row, out_col)] = typed[tile_linear];
                    }
                }
            }
        }

        // Tile assembly stats (verbose; gated behind SURTGIS_COG_DEBUG).
        let valid_pixels = output.iter().filter(|v| !v.is_nodata(None)).count();
        if cog_debug() {
            eprintln!(
                "    [cog] assembled: {} tiles ({}+{} skip), output={}x{}, tw={} th={}, valid={}/{} ({:.0}%)",
                tiles_fetched,
                tiles_written,
                tiles_skipped,
                out_cols,
                out_rows,
                tw,
                th,
                valid_pixels,
                out_rows * out_cols,
                if out_rows * out_cols > 0 {
                    valid_pixels as f64 / (out_rows * out_cols) as f64 * 100.0
                } else {
                    0.0
                }
            );
        }

        // Build Raster with correct geo metadata.
        let mut raster = Raster::from_array(output);

        // GeoTransform for the output window.
        let (corner_x, corner_y) = gt.pixel_to_geo_corner(px_min_col, px_min_row);
        let out_gt = GeoTransform::new(corner_x, corner_y, gt.pixel_width, gt.pixel_height);
        raster.set_transform(out_gt);
        raster.set_crs(self.geo_meta.crs.clone());

        if let Some(nd) = self.geo_meta.nodata
            && let Some(nd_t) = num_traits::cast(nd)
        {
            raster.set_nodata(Some(nd_t));
        }

        Ok(raster)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Ensure we have bytes starting at `offset` within the file.
/// If the initial header fetch covers it, slice from there;
/// otherwise do an additional Range request.
async fn ensure_data(
    client: &HttpClient,
    url: &str,
    auth: &dyn CloudAuth,
    header_bytes: &[u8],
    offset: usize,
    file_size: u64,
) -> Result<Vec<u8>> {
    // We need at least 2 bytes to read entry_count, but grab a generous chunk.
    let need = 4096usize; // enough for most IFDs
    let end = offset + need;

    if end <= header_bytes.len() {
        Ok(header_bytes[offset..end].to_vec())
    } else if offset < header_bytes.len() {
        // Partial overlap — fetch the rest.
        let have = header_bytes.len() - offset;
        let extra_offset = header_bytes.len() as u64;
        let extra_len = (need - have) as u64;
        let extra_len = if file_size > 0 {
            extra_len.min(file_size - extra_offset)
        } else {
            extra_len
        };
        let extra = client
            .fetch_range(url, extra_offset, extra_len, auth)
            .await?;
        let mut combined = header_bytes[offset..].to_vec();
        combined.extend_from_slice(&extra);
        Ok(combined)
    } else {
        // Entirely outside initial fetch.
        let fetch_len = need as u64;
        let fetch_len = if file_size > 0 {
            fetch_len.min(file_size - offset as u64)
        } else {
            fetch_len
        };
        client
            .fetch_range(url, offset as u64, fetch_len, auth)
            .await
    }
}

/// Resolve an IFD's tag entries by fetching external values.
async fn resolve_ifd(
    client: &HttpClient,
    url: &str,
    auth: &dyn CloudAuth,
    byte_order: TiffByteOrder,
    entries: &[RawTagEntry],
    header_bytes: &[u8],
    file_size: u64,
) -> Result<IfdInfo> {
    // Tags we need for raster structure.
    let needed_tags = [
        tags::IMAGE_WIDTH,
        tags::IMAGE_LENGTH,
        tags::BITS_PER_SAMPLE,
        tags::COMPRESSION,
        tags::SAMPLES_PER_PIXEL,
        tags::PLANAR_CONFIG,
        tags::TILE_WIDTH,
        tags::TILE_LENGTH,
        tags::TILE_OFFSETS,
        tags::TILE_BYTE_COUNTS,
        tags::SAMPLE_FORMAT,
    ];

    // Fetch external values for the needed tags.
    let mut resolved: Vec<(u16, Vec<u8>)> = Vec::new();

    for entry in entries {
        if !needed_tags.contains(&entry.tag) {
            continue;
        }
        if entry.inline {
            continue;
        }
        let size = ifd::external_value_size(entry);
        let offset = entry.value_or_offset as u64;
        let data = fetch_or_slice(client, url, auth, header_bytes, offset, size, file_size).await?;
        resolved.push((entry.tag, data));
    }

    // Extract values.
    let width = get_tag_u32(byte_order, entries, &resolved, tags::IMAGE_WIDTH).unwrap_or(0);
    let height = get_tag_u32(byte_order, entries, &resolved, tags::IMAGE_LENGTH).unwrap_or(0);
    let tile_width = get_tag_u32(byte_order, entries, &resolved, tags::TILE_WIDTH).unwrap_or(width);
    let tile_height =
        get_tag_u32(byte_order, entries, &resolved, tags::TILE_LENGTH).unwrap_or(height);
    let bits_per_sample =
        get_tag_u16(byte_order, entries, &resolved, tags::BITS_PER_SAMPLE).unwrap_or(8);
    let compression = get_tag_u16(byte_order, entries, &resolved, tags::COMPRESSION).unwrap_or(1);
    let samples_per_pixel =
        get_tag_u16(byte_order, entries, &resolved, tags::SAMPLES_PER_PIXEL).unwrap_or(1);
    let planar_config =
        get_tag_u16(byte_order, entries, &resolved, tags::PLANAR_CONFIG).unwrap_or(1);
    let sample_format =
        get_tag_u16(byte_order, entries, &resolved, tags::SAMPLE_FORMAT).unwrap_or(1);
    let predictor = get_tag_u16(byte_order, entries, &resolved, 317).unwrap_or(1); // Tag 317 = Predictor

    // Tile offsets & byte counts (arrays).
    let tile_offsets = get_tag_u64_array(byte_order, entries, &resolved, tags::TILE_OFFSETS);
    let tile_byte_counts =
        get_tag_u64_array(byte_order, entries, &resolved, tags::TILE_BYTE_COUNTS);

    Ok(IfdInfo {
        width,
        height,
        tile_width,
        tile_height,
        tile_offsets,
        tile_byte_counts,
        bits_per_sample,
        sample_format,
        compression,
        samples_per_pixel,
        planar_config,
        predictor,
        raw_entries: entries.to_vec(),
    })
}

/// Resolve GeoTIFF-specific tags from the first IFD.
async fn resolve_geotiff_meta(
    client: &HttpClient,
    url: &str,
    auth: &dyn CloudAuth,
    byte_order: TiffByteOrder,
    entries: &[RawTagEntry],
    header_bytes: &[u8],
    file_size: u64,
) -> Result<GeoTiffMeta> {
    let geo_tags = [
        tags::MODEL_PIXEL_SCALE,
        tags::MODEL_TIEPOINT,
        tags::MODEL_TRANSFORMATION,
        tags::GEO_KEY_DIRECTORY,
        tags::GEO_DOUBLE_PARAMS,
        tags::GEO_ASCII_PARAMS,
        tags::GDAL_NODATA,
    ];

    let mut resolved: Vec<(u16, Vec<u8>)> = Vec::new();

    for entry in entries {
        if !geo_tags.contains(&entry.tag) {
            continue;
        }
        if entry.inline {
            continue;
        }
        let size = ifd::external_value_size(entry);
        let offset = entry.value_or_offset as u64;
        let data = fetch_or_slice(client, url, auth, header_bytes, offset, size, file_size).await?;
        resolved.push((entry.tag, data));
    }

    Ok(geotiff_keys::extract_geotiff_meta(
        byte_order, entries, &resolved,
    ))
}

/// Fetch data or slice from the pre-fetched header bytes.
async fn fetch_or_slice(
    client: &HttpClient,
    url: &str,
    auth: &dyn CloudAuth,
    header_bytes: &[u8],
    offset: u64,
    size: u64,
    file_size: u64,
) -> Result<Vec<u8>> {
    let start = offset as usize;
    let end = start + size as usize;

    if end <= header_bytes.len() {
        Ok(header_bytes[start..end].to_vec())
    } else {
        let fetch_size = if file_size > 0 {
            size.min(file_size - offset)
        } else {
            size
        };
        client.fetch_range(url, offset, fetch_size, auth).await
    }
}

// ---------------------------------------------------------------------------
// Tag value extraction helpers
// ---------------------------------------------------------------------------

fn get_tag_u16(
    byte_order: TiffByteOrder,
    entries: &[RawTagEntry],
    resolved: &[(u16, Vec<u8>)],
    tag_id: u16,
) -> Option<u16> {
    let entry = entries.iter().find(|e| e.tag == tag_id)?;
    if entry.inline {
        Some(ifd::inline_u16(byte_order, entry))
    } else {
        let data = resolved.iter().find(|(t, _)| *t == tag_id)?.1.as_slice();
        let vals = ifd::read_offset_values_u64(byte_order, entry, data);
        vals.first().map(|&v| v as u16)
    }
}

fn get_tag_u32(
    byte_order: TiffByteOrder,
    entries: &[RawTagEntry],
    resolved: &[(u16, Vec<u8>)],
    tag_id: u16,
) -> Option<u32> {
    let entry = entries.iter().find(|e| e.tag == tag_id)?;
    if entry.inline {
        Some(ifd::inline_u32(byte_order, entry))
    } else {
        let data = resolved.iter().find(|(t, _)| *t == tag_id)?.1.as_slice();
        let vals = ifd::read_offset_values_u64(byte_order, entry, data);
        vals.first().map(|&v| v as u32)
    }
}

fn get_tag_u64_array(
    byte_order: TiffByteOrder,
    entries: &[RawTagEntry],
    resolved: &[(u16, Vec<u8>)],
    tag_id: u16,
) -> Vec<u64> {
    let entry = match entries.iter().find(|e| e.tag == tag_id) {
        Some(e) => e,
        None => return Vec::new(),
    };

    if entry.inline {
        vec![entry.value_or_offset as u64]
    } else {
        match resolved.iter().find(|(t, _)| *t == tag_id) {
            Some((_, data)) => ifd::read_offset_values_u64(byte_order, entry, data),
            None => Vec::new(),
        }
    }
}

/// Reject big-endian (`MM`) TIFFs before any tile decoding is attempted.
///
/// The IFD/tag parser (`ifd.rs`) reads `byte_order` correctly for both
/// orderings. But the tile decompression and predictor-undo paths
/// (`decompress.rs`: `bytes_to_typed`, `cast_from_le`, the horizontal and
/// floating-point predictor un-shufflers) hard-code little-endian sample
/// layout regardless of what the header declares. Proceeding on a real
/// big-endian TIFF would silently decode garbage pixel values instead of
/// failing — the same "silent wrong data" failure mode `validate_tiled`
/// guards against below. Full big-endian decode support is future work;
/// for now, fail loudly and early instead.
fn reject_big_endian(byte_order: TiffByteOrder) -> Result<()> {
    if byte_order == TiffByteOrder::BigEndian {
        return Err(CloudError::InvalidTiff {
            reason: "big-endian TIFF byte order ('MM') is not yet supported by the tile \
                      decoder (only little-endian 'II' COGs are supported); re-encode as \
                      little-endian, e.g. via `gdal_translate`"
                .into(),
        });
    }
    Ok(())
}

/// Reject TIFFs the tile path cannot read.
///
/// A TIFF without `TileOffsets` cannot be addressed by tile index. If it
/// has `StripOffsets` it is a striped (non-tiled) TIFF — common outside
/// strict COGs. Before this check, such files produced an all-nodata
/// raster returned as success (no tile was ever fetched, every tile was
/// "skipped"): silent wrong data.
fn validate_tiled(ifd_info: &IfdInfo) -> Result<()> {
    if !ifd_info.tile_offsets.is_empty() {
        return Ok(());
    }
    let has_strips = ifd_info
        .raw_entries
        .iter()
        .any(|e| e.tag == ifd::tags::STRIP_OFFSETS);
    let reason = if has_strips {
        "striped (non-tiled) TIFF is not supported by the COG reader; \
         convert to a tiled COG (e.g. gdal_translate -of COG)"
    } else {
        "no TileOffsets found: not a valid tiled TIFF"
    };
    Err(CloudError::InvalidTiff {
        reason: reason.into(),
    })
}

#[cfg(test)]
mod strip_rejection_tests {
    use super::*;

    fn base_ifd(tile_offsets: Vec<u64>, raw_entries: Vec<RawTagEntry>) -> IfdInfo {
        IfdInfo {
            width: 100,
            height: 100,
            tile_width: 100,
            tile_height: 100,
            tile_offsets,
            tile_byte_counts: vec![],
            bits_per_sample: 32,
            sample_format: 3,
            compression: 1,
            samples_per_pixel: 1,
            planar_config: 1,
            predictor: 1,
            raw_entries,
        }
    }

    fn entry(tag: u16) -> RawTagEntry {
        RawTagEntry {
            tag,
            type_id: 4,
            count: 1,
            value_or_offset: 8,
            inline: true,
        }
    }

    /// Regression: a striped TIFF (StripOffsets, no TileOffsets) must be
    /// rejected at open — not silently produce an all-nodata raster.
    #[test]
    fn striped_tiff_is_rejected_with_clear_error() {
        let ifd_info = base_ifd(vec![], vec![entry(ifd::tags::STRIP_OFFSETS)]);
        let err = validate_tiled(&ifd_info).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("striped"), "unhelpful error: {msg}");
    }

    #[test]
    fn tiff_without_any_offsets_is_rejected() {
        let ifd_info = base_ifd(vec![], vec![]);
        assert!(validate_tiled(&ifd_info).is_err());
    }

    #[test]
    fn tiled_tiff_passes() {
        let ifd_info = base_ifd(vec![8], vec![entry(ifd::tags::TILE_OFFSETS)]);
        assert!(validate_tiled(&ifd_info).is_ok());
    }
}

#[cfg(test)]
mod byte_order_tests {
    use super::*;

    /// Regression: a big-endian ("MM") TIFF header must be rejected with a
    /// clear, explicit error at open — not silently decoded as if it were
    /// little-endian (which would produce garbage pixel values, since
    /// `decompress.rs`'s sample/predictor code hard-codes little-endian).
    #[test]
    fn big_endian_header_is_rejected_with_clear_error() {
        let err = reject_big_endian(TiffByteOrder::BigEndian).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.to_lowercase().contains("big-endian"),
            "unhelpful error: {msg}"
        );
    }

    #[test]
    fn little_endian_header_passes() {
        assert!(reject_big_endian(TiffByteOrder::LittleEndian).is_ok());
    }

    /// Parsing an actual big-endian-marked header ("MM") end-to-end through
    /// `ifd::parse_header` and feeding the result into the same guard used
    /// by `open()` confirms the check fires on real header bytes, not just
    /// a hand-constructed enum value.
    #[test]
    fn real_big_endian_header_bytes_are_rejected() {
        // "MM" + magic 42 (big-endian u16) + first IFD offset 8 (big-endian u32).
        let header: [u8; 8] = [b'M', b'M', 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08];
        let parsed = ifd::parse_header(&header).unwrap();
        assert_eq!(parsed.byte_order, TiffByteOrder::BigEndian);
        let err = reject_big_endian(parsed.byte_order).unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("big-endian"));
    }
}

// ---------------------------------------------------------------------------
// IFD/header cache tests (P8)
//
// `surtgis-cloud` has no mock-HTTP test infrastructure (no `httpmock`/
// `wiremock`/local test server anywhere in this crate — verified by
// searching `crates/cloud` before writing these tests), and the existing
// network-dependent tests in `tests/integration.rs` are all `#[ignore]`d,
// hitting real public COG endpoints. Building one from scratch just for
// this change would mean hand-crafting a byte-exact minimal tiled COG
// (valid IFD chain + GeoTIFF keys) to drive `CogReader::open()`
// end-to-end, which is a much larger and riskier undertaking than the
// caching logic it would be testing.
//
// Instead, these tests exercise the cache key derivation and the
// insert/lookup primitives directly and in isolation (no network, no
// `CogReader::open()`), using unique per-test cache keys (via an atomic
// counter) so they don't interfere with each other despite sharing the
// process-wide `IFD_CACHE` static under parallel test execution.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod ifd_cache_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn sample_ifds() -> Vec<IfdInfo> {
        vec![IfdInfo {
            width: 10,
            height: 10,
            tile_width: 10,
            tile_height: 10,
            tile_offsets: vec![8],
            tile_byte_counts: vec![100],
            bits_per_sample: 32,
            sample_format: 3,
            compression: 1,
            samples_per_pixel: 1,
            planar_config: 1,
            predictor: 1,
            raw_entries: vec![],
        }]
    }

    fn sample_geo_meta() -> GeoTiffMeta {
        GeoTiffMeta {
            geo_transform: GeoTransform::new(0.0, 0.0, 1.0, -1.0),
            crs: None,
            nodata: Some(-9999.0),
        }
    }

    /// Generates a fresh `https://.../<label>-<n>.tif` base URL on every
    /// call, so tests running in parallel never collide on the shared
    /// process-wide `IFD_CACHE`.
    fn unique_base_url(label: &str) -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("https://example.blob.core.windows.net/container/{label}-{n}.tif")
    }

    #[test]
    fn cache_key_strips_query_string() {
        assert_eq!(
            cache_key_for_url("https://x/y.tif?sig=abc&exp=123"),
            "https://x/y.tif"
        );
        assert_eq!(cache_key_for_url("https://x/y.tif?"), "https://x/y.tif");
    }

    #[test]
    fn cache_key_passes_through_url_without_query() {
        let url = "https://x/y.tif";
        assert_eq!(cache_key_for_url(url), url);
    }

    /// Core regression for the P8 header cache: two URLs for the exact
    /// same file that differ ONLY in their query string — exactly what
    /// happens when a STAC catalog re-signs the same asset href with a new
    /// SAS token / expiry on every fetch — must resolve to the same cache
    /// key, and a value inserted under one must be visible under the
    /// other. This is what lets `CogReader::open()` skip the header
    /// HEAD+fetch+parse on the second open of the "same" (differently
    /// signed) URL.
    #[test]
    fn same_path_different_query_shares_one_cache_entry() {
        let base = unique_base_url("scene");
        let url_a = format!("{base}?sv=2024-01-01&sig=AAAAAAAA");
        let url_b = format!("{base}?sv=2024-01-01&sig=BBBBBBBB");

        let key_a = cache_key_for_url(&url_a).to_string();
        let key_b = cache_key_for_url(&url_b).to_string();
        assert_eq!(
            key_a, key_b,
            "same path with differing SAS-token-like query strings must share a cache key"
        );

        insert_cached_header(
            key_a,
            CachedCogHeader {
                byte_order: TiffByteOrder::LittleEndian,
                ifds: sample_ifds(),
                geo_meta: sample_geo_meta(),
            },
        );

        // Looking up via url_b's key (different query string, same path)
        // must hit the entry inserted under url_a's key — i.e. no second
        // fetch/parse would be needed for url_b in `open()`.
        let hit = cached_header(&key_b)
            .expect("expected a cache hit across differing query strings for the same path");
        assert_eq!(hit.ifds.len(), 1);
        assert_eq!(hit.ifds[0].width, 10);
        assert_eq!(hit.ifds[0].tile_offsets, vec![8]);
        assert_eq!(hit.geo_meta.nodata, Some(-9999.0));
    }

    /// Two genuinely different files (different path, not just different
    /// query string) must NOT share a cache entry.
    #[test]
    fn different_paths_do_not_share_a_cache_entry() {
        let url_a = format!("{}?sig=AAA", unique_base_url("scene-a"));
        let url_b = format!("{}?sig=AAA", unique_base_url("scene-b"));

        let key_a = cache_key_for_url(&url_a).to_string();
        let key_b = cache_key_for_url(&url_b).to_string();
        assert_ne!(key_a, key_b);

        insert_cached_header(
            key_a,
            CachedCogHeader {
                byte_order: TiffByteOrder::LittleEndian,
                ifds: sample_ifds(),
                geo_meta: sample_geo_meta(),
            },
        );

        assert!(
            cached_header(&key_b).is_none(),
            "a different path must not hit another path's cache entry"
        );
    }

    #[test]
    fn insert_then_lookup_round_trips() {
        let key = cache_key_for_url(&unique_base_url("roundtrip")).to_string();

        assert!(
            cached_header(&key).is_none(),
            "must be a clean miss before insert"
        );

        insert_cached_header(
            key.clone(),
            CachedCogHeader {
                byte_order: TiffByteOrder::LittleEndian,
                ifds: sample_ifds(),
                geo_meta: sample_geo_meta(),
            },
        );

        let hit = cached_header(&key).expect("just-inserted key must be a hit");
        assert_eq!(hit.ifds[0].tile_offsets, vec![8]);
        assert_eq!(hit.geo_meta.nodata, Some(-9999.0));
    }
}
