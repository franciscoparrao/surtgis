//! Morphology demo: synthetic image processing pipeline
//!
//! Generates a 200x200 synthetic "image" with:
//! - Uniform background (value 50)
//! - Large bright rectangle (value 200)
//! - Small bright spots (single-pixel, value 220) — "salt" noise
//! - Large dark ellipse (value 10)
//! - Small dark spots (single-pixel, value 5) — "pepper" noise
//!
//! Then applies the full morphology pipeline and writes each result to TIFF:
//!   1. original.tif       — the synthetic input
//!   2. eroded.tif         — erosion (min filter)
//!   3. dilated.tif        — dilation (max filter)
//!   4. opened.tif         — opening (removes salt noise)
//!   5. closed.tif         — closing (removes pepper noise)
//!   6. gradient.tif       — morphological gradient (edges)
//!   7. tophat.tif         — top-hat (bright feature extraction)
//!   8. blackhat.tif       — black-hat (dark feature extraction)
//!   9. cleaned.tif        — opening then closing (full denoise)
//!
//! Run:
//!   cargo run -p surtgis-algorithms --example morphology_demo

use std::fs;
use std::path::Path;

use surtgis_algorithms::morphology::{
    closing, dilate, erode, gradient, opening, top_hat, black_hat,
    StructuringElement,
};
use surtgis_core::io::{write_geotiff, GeoTiffOptions};
use surtgis_core::{GeoTransform, Raster};

const ROWS: usize = 200;
const COLS: usize = 200;

fn main() {
    let out_dir = Path::new("output/morphology_demo");
    fs::create_dir_all(out_dir).expect("Cannot create output directory");

    // --- 1. Build synthetic image ---
    let input = build_synthetic_image();
    println!("Synthetic image: {}x{}", COLS, ROWS);
    print_stats("  input", &input);
    save(out_dir, "original.tif", &input);

    // --- 2. Structuring element ---
    let se = StructuringElement::Square(1); // 3x3
    println!("\nStructuring element: Square(1) — 3x3 kernel");

    // --- 3. Erosion ---
    let eroded = erode(&input, &se).expect("erode failed");
    print_stats("  eroded", &eroded);
    save(out_dir, "eroded.tif", &eroded);

    // --- 4. Dilation ---
    let dilated = dilate(&input, &se).expect("dilate failed");
    print_stats("  dilated", &dilated);
    save(out_dir, "dilated.tif", &dilated);

    // --- 5. Opening (removes salt noise) ---
    let opened = opening(&input, &se).expect("opening failed");
    print_stats("  opened", &opened);
    save(out_dir, "opened.tif", &opened);

    // --- 6. Closing (removes pepper noise) ---
    let closed = closing(&input, &se).expect("closing failed");
    print_stats("  closed", &closed);
    save(out_dir, "closed.tif", &closed);

    // --- 7. Morphological gradient (edges) ---
    let grad = gradient(&input, &se).expect("gradient failed");
    print_stats("  gradient", &grad);
    save(out_dir, "gradient.tif", &grad);

    // --- 8. Top-hat (bright features on dark background) ---
    let th = top_hat(&input, &se).expect("top_hat failed");
    print_stats("  top-hat", &th);
    save(out_dir, "tophat.tif", &th);

    // --- 9. Black-hat (dark features on bright background) ---
    let bh = black_hat(&input, &se).expect("black_hat failed");
    print_stats("  black-hat", &bh);
    save(out_dir, "blackhat.tif", &bh);

    // --- 10. Full denoise: opening then closing ---
    let denoised = closing(&opened, &se).expect("closing(opened) failed");
    print_stats("  cleaned", &denoised);
    save(out_dir, "cleaned.tif", &denoised);

    println!("\n9 TIFF files written to {}/", out_dir.display());

    // --- 11. Verify noise removal ---
    verify_noise_removal(&input, &opened, &closed, &denoised);
}

/// Build a 200x200 synthetic raster with geometric objects and noise.
fn build_synthetic_image() -> Raster<f64> {
    let mut img = Raster::filled(ROWS, COLS, 50.0);
    // 10 m cells, origin at (500000, 4500000) — UTM-like
    img.set_transform(GeoTransform::new(500_000.0, 4_500_000.0, 10.0, -10.0));
    img.set_nodata(Some(f64::NAN));

    // Large bright rectangle: rows 30..70, cols 30..90 → value 200
    for r in 30..70 {
        for c in 30..90 {
            img.set(r, c, 200.0).unwrap();
        }
    }

    // Large dark ellipse: center (140, 100), semi-axes 30x20 → value 10
    for r in 0..ROWS {
        for c in 0..COLS {
            let dr = (r as f64 - 140.0) / 30.0;
            let dc = (c as f64 - 100.0) / 20.0;
            if dr * dr + dc * dc <= 1.0 {
                img.set(r, c, 10.0).unwrap();
            }
        }
    }

    // Salt noise: 80 bright single-pixel spots (value 220)
    // Deterministic positions using a simple LCG
    let mut seed: u64 = 42;
    for _ in 0..80 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = ((seed >> 33) as usize) % ROWS;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let c = ((seed >> 33) as usize) % COLS;
        img.set(r, c, 220.0).unwrap();
    }

    // Pepper noise: 80 dark single-pixel spots (value 5)
    seed = 137;
    for _ in 0..80 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = ((seed >> 33) as usize) % ROWS;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let c = ((seed >> 33) as usize) % COLS;
        img.set(r, c, 5.0).unwrap();
    }

    img
}

fn print_stats(label: &str, raster: &Raster<f64>) {
    let s = raster.statistics();
    println!(
        "{:<12} min={:>6.1}  max={:>6.1}  mean={:>6.1}  valid={:>5}  nodata={:>5}",
        label,
        s.min.unwrap_or(f64::NAN),
        s.max.unwrap_or(f64::NAN),
        s.mean.unwrap_or(f64::NAN),
        s.valid_count,
        s.nodata_count,
    );
}

fn save(dir: &Path, name: &str, raster: &Raster<f64>) {
    let path = dir.join(name);
    write_geotiff(raster, &path, Some(GeoTiffOptions::default()))
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", path.display(), e));
}

/// Verify that opening removes salt noise and closing removes pepper noise.
fn verify_noise_removal(
    original: &Raster<f64>,
    opened: &Raster<f64>,
    closed: &Raster<f64>,
    cleaned: &Raster<f64>,
) {
    println!("\n--- Verification ---");

    // Count pixels with specific values in the valid interior (skip NaN border)
    let border = 4; // opening+closing = 2*radius each = 4 total

    let mut orig_salt = 0usize;
    let mut orig_pepper = 0usize;
    let mut opened_salt = 0usize;
    let mut closed_pepper = 0usize;
    let mut cleaned_salt = 0usize;
    let mut cleaned_pepper = 0usize;

    for r in border..(ROWS - border) {
        for c in border..(COLS - border) {
            let ov = original.get(r, c).unwrap();
            if (ov - 220.0).abs() < 0.1 {
                orig_salt += 1;
            }
            if (ov - 5.0).abs() < 0.1 {
                orig_pepper += 1;
            }

            let opv = opened.get(r, c).unwrap();
            if !opv.is_nan() && (opv - 220.0).abs() < 0.1 {
                opened_salt += 1;
            }

            let clv = closed.get(r, c).unwrap();
            if !clv.is_nan() && (clv - 5.0).abs() < 0.1 {
                closed_pepper += 1;
            }

            let dv = cleaned.get(r, c).unwrap();
            if !dv.is_nan() && (dv - 220.0).abs() < 0.1 {
                cleaned_salt += 1;
            }
            if !dv.is_nan() && (dv - 5.0).abs() < 0.1 {
                cleaned_pepper += 1;
            }
        }
    }

    println!("  Original:  salt pixels = {}, pepper pixels = {}", orig_salt, orig_pepper);
    println!("  Opened:    salt pixels = {} (should be 0 — removed by opening)", opened_salt);
    println!("  Closed:    pepper pixels = {} (should be 0 — removed by closing)", closed_pepper);
    println!("  Cleaned:   salt = {}, pepper = {} (both should be 0)", cleaned_salt, cleaned_pepper);
}
