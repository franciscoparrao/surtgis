//! Decode an ECW file (optionally at reduced resolution) and write one
//! GeoTIFF per band.
//!
//! Usage: `cargo run -p surtgis-ecw --example ecw_to_tiff -- <input.ecw> <out_prefix> [reduction]`

use surtgis_core::io::write_geotiff;
use surtgis_ecw::EcwReader;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: ecw_to_tiff <input.ecw> <out_prefix> [reduction]");
        std::process::exit(2);
    }
    let reduction: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(0);

    let mut reader = EcwReader::open(&args[1]).expect("open ECW");
    let h = reader.header();
    eprintln!(
        "{}x{} px, {} bands, {:?}, {} levels, GSD ({:.4}, {:.4}) {:?}, {} / {} (EPSG {:?})",
        h.x_size,
        h.y_size,
        h.nr_bands,
        h.compress_format,
        h.num_levels,
        h.cell_increment_x,
        h.cell_increment_y,
        h.cell_units,
        h.datum,
        h.projection,
        h.epsg()
    );

    let start = std::time::Instant::now();
    let bands = reader.read_reduced(reduction).expect("decode");
    eprintln!(
        "decoded {} bands of {:?} at 1:{} in {:.2?}",
        bands.len(),
        bands[0].shape(),
        1u32 << reduction,
        start.elapsed()
    );

    for (i, band) in bands.iter().enumerate() {
        let path = format!("{}_b{}.tif", args[2], i + 1);
        write_geotiff(band, &path, None).expect("write GeoTIFF");
        eprintln!("wrote {path}");
    }
}
