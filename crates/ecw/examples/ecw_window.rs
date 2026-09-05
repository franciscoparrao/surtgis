//! Decode a full-resolution window of an ECW file to per-band GeoTIFFs.
//!
//! Usage: `ecw_window <input.ecw> <out_prefix> <start_x> <start_y> <width> <height>`

use surtgis_core::io::write_geotiff;
use surtgis_ecw::{EcwReader, RegionParams};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        eprintln!(
            "usage: ecw_window <input.ecw> <out_prefix> <start_x> <start_y> <width> <height>"
        );
        std::process::exit(2);
    }
    let (sx, sy, w, h): (u32, u32, u32, u32) = (
        args[3].parse().unwrap(),
        args[4].parse().unwrap(),
        args[5].parse().unwrap(),
        args[6].parse().unwrap(),
    );

    let mut reader = EcwReader::open(&args[1]).expect("open ECW");
    let start = std::time::Instant::now();
    let bands = reader
        .read_region(RegionParams {
            start_x: sx,
            start_y: sy,
            end_x: sx + w - 1,
            end_y: sy + h - 1,
            number_x: w,
            number_y: h,
        })
        .expect("decode window");
    eprintln!("decoded {}x{} window in {:.2?}", w, h, start.elapsed());

    for (i, band) in bands.iter().enumerate() {
        let path = format!("{}_b{}.tif", args[2], i + 1);
        write_geotiff(band, &path, None).expect("write GeoTIFF");
    }
    eprintln!("wrote {} bands to {}_b*.tif", bands.len(), args[2]);
}
