//! Core COG reader: open remote COGs, read by bounding box or full extent.

use std::time::Duration;

use ndarray::Array2;
use surtgis_core::crs::CRS;
use surtgis_core::raster::{GeoTransform, Raster, RasterElement};

use crate::auth::{CloudAuth, NoAuth};
use crate::cache::{TileCache, TileKey};
use crate::decompress;
use crate::error::{CloudError, Result};
use crate::geotiff_keys::{self, GeoTiffMeta};
use crate::http::HttpClient;
use crate::ifd::{
    self, IfdInfo, RawTagEntry, TiffByteOrder,
    tags,
};
use crate::tile_index::{self, BBox, TileMapping};

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
            max_concurrent_fetches: 8,
            cache_capacity: 128,
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
            auth: Box::new(NoAuth),
        }
    }
}

/// Metadata exposed by [`CogReader::metadata`].
#[derive(Debug, Clone)]
pub struct CogMetadata {
    pub url: String,
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub bits_per_sample: u16,
    pub sample_format: u16,
    pub compression: u16,
    pub geo_transform: GeoTransform,
    pub crs: Option<CRS>,
    pub nodata: Option<f64>,
    pub num_overviews: usize,
}

/// Overview level information.
#[derive(Debug, Clone)]
pub struct OverviewInfo {
    pub index: usize,
    pub width: u32,
    pub height: u32,
}

/// Cloud Optimized GeoTIFF reader.
///
/// Reads tiles on-demand via HTTP Range requests with LRU caching.
pub struct CogReader {
    url: String,
    client: HttpClient,
    #[allow(dead_code)]
    byte_order: TiffByteOrder,
    ifds: Vec<IfdInfo>,
    geo_meta: GeoTiffMeta,
    cache: TileCache,
    options: CogReaderOptions,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl CogReader {
    /// Open a remote COG by URL.
    ///
    /// Fetches the TIFF header and all IFD chains (full resolution + overviews)
    /// in as few HTTP Range requests as possible.
    pub async fn open(url: &str, options: CogReaderOptions) -> Result<Self> {
        let client = HttpClient::new(options.request_timeout, options.max_retries)?;
        let auth = options.auth.as_ref();

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

        // 3. Parse IFD chain.
        let mut ifds = Vec::new();
        let mut ifd_offset = tiff_header.first_ifd_offset as usize;

        while ifd_offset > 0 {
            let ifd_data = self::ensure_data(
                &client, url, auth, &header_bytes, ifd_offset, file_size,
            ).await?;

            let raw_ifd = ifd::parse_ifd(byte_order, &ifd_data)?;

            // Resolve external tag values needed for this IFD.
            let ifd_info = resolve_ifd(
                &client, url, auth, byte_order,
                &raw_ifd.entries, &header_bytes, file_size,
            ).await?;

            ifds.push(ifd_info);
            ifd_offset = raw_ifd.next_ifd_offset as usize;
        }

        if ifds.is_empty() {
            return Err(CloudError::NoIfd);
        }

        // 4. Extract GeoTIFF metadata from the first (full-res) IFD.
        let geo_meta = resolve_geotiff_meta(
            &client, url, auth, byte_order,
            &ifds[0].raw_entries, &header_bytes, file_size,
        ).await?;

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
            bbox, &gt, ifd.width, ifd.height, ifd.tile_width, ifd.tile_height,
        ).ok_or(CloudError::BBoxOutside)?;

        self.assemble_raster::<T>(ifd_idx, &ifd, &gt, &mapping).await
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
            &bbox, &gt, ifd.width, ifd.height, ifd.tile_width, ifd.tile_height,
        ).ok_or(CloudError::BBoxOutside)?;

        self.assemble_raster::<T>(ifd_idx, &ifd, &gt, &mapping).await
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

    /// Fetch tiles, decompress, and assemble into a Raster<T>.
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
        let bytes_per_pixel = (bps as usize + 7) / 8;
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
            if self.cache.get(&key).is_none() {
                if tr.tile_idx < ifd.tile_offsets.len() {
                    let offset = ifd.tile_offsets[tr.tile_idx];
                    let length = ifd.tile_byte_counts[tr.tile_idx];
                    if length > 0 {
                        to_fetch.push((i, offset, length));
                    }
                }
            }
        }

        // Fetch uncached tiles concurrently (in batches).
        let auth = self.options.auth.as_ref();
        let batch_size = self.options.max_concurrent_fetches;

        for chunk in to_fetch.chunks(batch_size) {
            let ranges: Vec<(u64, u64)> = chunk.iter().map(|&(_, o, l)| (o, l)).collect();
            let fetched = self.client.fetch_ranges(&self.url, &ranges, auth).await?;

            for (j, &(tile_list_idx, _, _)) in chunk.iter().enumerate() {
                let tr = &mapping.tiles[tile_list_idx];
                let raw = decompress::decompress_tile(
                    &fetched[j], compression, raw_tile_size,
                )?;
                let key = TileKey {
                    ifd_idx,
                    tile_idx: tr.tile_idx,
                };
                self.cache.insert(key, raw);
            }
        }

        // Assemble output array from cached tiles.
        let mut output = Array2::<T>::from_elem((out_rows, out_cols), T::default_nodata());

        for tr in &mapping.tiles {
            let key = TileKey {
                ifd_idx,
                tile_idx: tr.tile_idx,
            };

            let raw = match self.cache.get(&key) {
                Some(data) => data.clone(),
                None => continue, // empty tile (byte_count == 0)
            };

            let typed: Vec<T> = decompress::bytes_to_typed(&raw, bps, sf)?;

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

        // Build Raster with correct geo metadata.
        let mut raster = Raster::from_array(output);

        // GeoTransform for the output window.
        let (corner_x, corner_y) = gt.pixel_to_geo_corner(px_min_col, px_min_row);
        let out_gt = GeoTransform::new(corner_x, corner_y, gt.pixel_width, gt.pixel_height);
        raster.set_transform(out_gt);
        raster.set_crs(self.geo_meta.crs.clone());

        if let Some(nd) = self.geo_meta.nodata {
            if let Some(nd_t) = num_traits::cast(nd) {
                raster.set_nodata(Some(nd_t));
            }
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
        let extra = client.fetch_range(url, extra_offset, extra_len, auth).await?;
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
        client.fetch_range(url, offset as u64, fetch_len, auth).await
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
    let tile_height = get_tag_u32(byte_order, entries, &resolved, tags::TILE_LENGTH).unwrap_or(height);
    let bits_per_sample = get_tag_u16(byte_order, entries, &resolved, tags::BITS_PER_SAMPLE).unwrap_or(8);
    let compression = get_tag_u16(byte_order, entries, &resolved, tags::COMPRESSION).unwrap_or(1);
    let samples_per_pixel = get_tag_u16(byte_order, entries, &resolved, tags::SAMPLES_PER_PIXEL).unwrap_or(1);
    let planar_config = get_tag_u16(byte_order, entries, &resolved, tags::PLANAR_CONFIG).unwrap_or(1);
    let sample_format = get_tag_u16(byte_order, entries, &resolved, tags::SAMPLE_FORMAT).unwrap_or(1);

    // Tile offsets & byte counts (arrays).
    let tile_offsets = get_tag_u64_array(byte_order, entries, &resolved, tags::TILE_OFFSETS);
    let tile_byte_counts = get_tag_u64_array(byte_order, entries, &resolved, tags::TILE_BYTE_COUNTS);

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

    Ok(geotiff_keys::extract_geotiff_meta(byte_order, entries, &resolved))
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
