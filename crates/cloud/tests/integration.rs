//! Integration tests for the COG reader.
//!
//! Tests marked `#[ignore]` require network access and a real COG endpoint.
//! Run with: `cargo test -p surtgis-cloud --features native -- --ignored`

use surtgis_cloud::cog_reader::{CogReader, CogReaderOptions};
use surtgis_cloud::tile_index::BBox;

/// Read metadata from a public Copernicus DEM COG on AWS.
///
/// This file is small (1-degree tile, ~26 MB) and publicly accessible
/// without authentication.
#[tokio::test]
#[ignore]
async fn test_read_public_cog_metadata() {
    // Copernicus GLO-30 DEM — public COG on AWS Open Data
    let url = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N00_00_W080_00_DEM/Copernicus_DSM_COG_10_N00_00_W080_00_DEM.tif";

    let opts = CogReaderOptions::default();
    let reader = CogReader::open(url, opts).await.expect("failed to open COG");

    let meta = reader.metadata();
    assert!(meta.width > 0, "width should be positive");
    assert!(meta.height > 0, "height should be positive");
    assert!(meta.tile_width > 0, "tile_width should be positive");
    assert!(meta.tile_height > 0, "tile_height should be positive");

    println!("COG metadata: {}x{}, tiles {}x{}, compression={}, bps={}, sf={}",
        meta.width, meta.height,
        meta.tile_width, meta.tile_height,
        meta.compression, meta.bits_per_sample, meta.sample_format);
    println!("GeoTransform: {:?}", meta.geo_transform);
    println!("CRS: {:?}", meta.crs);
    println!("NoData: {:?}", meta.nodata);
    println!("Overviews: {}", meta.num_overviews);
}

/// Read a small bounding box from a public COG.
#[tokio::test]
#[ignore]
async fn test_read_public_cog_bbox() {
    let url = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N00_00_W080_00_DEM/Copernicus_DSM_COG_10_N00_00_W080_00_DEM.tif";

    let opts = CogReaderOptions::default();
    let mut reader = CogReader::open(url, opts).await.expect("failed to open COG");

    // Read a small region (~0.1 degree square)
    let bbox = BBox::new(-79.95, 0.05, -79.85, 0.15);
    let raster: surtgis_core::Raster<f32> = reader
        .read_bbox(&bbox, None)
        .await
        .expect("failed to read bbox");

    let (rows, cols) = raster.shape();
    println!("Raster: {}x{}", rows, cols);
    assert!(rows > 0);
    assert!(cols > 0);

    let stats = raster.statistics();
    println!("Stats: min={:?}, max={:?}, mean={:?}", stats.min, stats.max, stats.mean);
}

/// Unit-level: verify IFD parsing on a synthetic TIFF header.
#[test]
fn test_ifd_roundtrip() {
    use surtgis_cloud::ifd::{self, TiffByteOrder};

    // Build a minimal little-endian TIFF header + 1 IFD
    let mut data = Vec::new();

    // Header: II, 42, offset to IFD = 8
    data.extend_from_slice(b"II");
    data.extend_from_slice(&42u16.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());

    // IFD at offset 8: 2 entries
    data.extend_from_slice(&2u16.to_le_bytes());

    // Entry 1: ImageWidth (256) = SHORT, count=1, value=1024
    data.extend_from_slice(&256u16.to_le_bytes());
    data.extend_from_slice(&3u16.to_le_bytes()); // SHORT
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&1024u32.to_le_bytes());

    // Entry 2: ImageLength (257) = SHORT, count=1, value=768
    data.extend_from_slice(&257u16.to_le_bytes());
    data.extend_from_slice(&3u16.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&768u32.to_le_bytes());

    // Next IFD = 0
    data.extend_from_slice(&0u32.to_le_bytes());

    let header = ifd::parse_header(&data).unwrap();
    assert_eq!(header.byte_order, TiffByteOrder::LittleEndian);
    assert_eq!(header.first_ifd_offset, 8);

    let ifd_data = &data[8..];
    let raw_ifd = ifd::parse_ifd(TiffByteOrder::LittleEndian, ifd_data).unwrap();
    assert_eq!(raw_ifd.entries.len(), 2);
    assert_eq!(raw_ifd.next_ifd_offset, 0);

    let w = ifd::inline_u16(TiffByteOrder::LittleEndian, &raw_ifd.entries[0]);
    let h = ifd::inline_u16(TiffByteOrder::LittleEndian, &raw_ifd.entries[1]);
    assert_eq!(w, 1024);
    assert_eq!(h, 768);
}

/// Unit-level: verify decompression round-trip with DEFLATE.
#[cfg(feature = "deflate")]
#[test]
fn test_deflate_roundtrip() {
    use std::io::Write;
    use surtgis_cloud::decompress;

    let original: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

    let mut encoder = flate2::write::DeflateEncoder::new(
        Vec::new(),
        flate2::Compression::default(),
    );
    encoder.write_all(&original).unwrap();
    let compressed = encoder.finish().unwrap();

    let decompressed = decompress::decompress_tile(
        &compressed,
        decompress::compression::DEFLATE,
        original.len(),
    )
    .unwrap();

    assert_eq!(decompressed, original);
}

/// Unit-level: tile index computation.
#[test]
fn test_tile_index_full_image() {
    use surtgis_cloud::tile_index::{self, BBox};
    use surtgis_core::GeoTransform;

    let gt = GeoTransform::new(0.0, 100.0, 0.1, -0.1);
    // Image: 1000x1000 pixels, 100x100 tiles
    let bbox = BBox::new(0.0, 0.0, 100.0, 100.0); // Full extent
    let mapping = tile_index::tiles_for_bbox(&bbox, &gt, 1000, 1000, 100, 100).unwrap();

    assert_eq!(mapping.output_shape, (1000, 1000));
    assert_eq!(mapping.tiles.len(), 100); // 10x10 tiles
}
