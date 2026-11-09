//! Integration tests against the real Geomag orthomosaic of the Maipú
//! corridor (ENAP demo). The file is not part of the repository; every
//! test skips silently when it is absent so CI stays green.
//!
//! Ground truth for the header assertions is the byte-level analysis of
//! the file recorded in `docs/ecw_format.md` (§ "El archivo de Maipú").

use surtgis_ecw::{CompressFormat, EcwReader, RegionParams};

const MAIPU: &str =
    "/home/franciscoparrao/proyectos/territorio-digital/data/ORTOOMOSAICO ENAP DEMO  MAIPU.ecw";

fn open_maipu() -> Option<EcwReader> {
    if !std::path::Path::new(MAIPU).exists() {
        eprintln!("skipping: Maipú ECW not present");
        return None;
    }
    Some(EcwReader::open(MAIPU).expect("Maipú ECW must parse"))
}

#[test]
fn header_matches_byte_level_analysis() {
    let Some(reader) = open_maipu() else { return };
    let h = reader.header();

    assert_eq!(h.version, 2);
    assert_eq!(h.compress_format, CompressFormat::Multiband);
    assert_eq!(h.num_levels, 8);
    assert_eq!((h.x_size, h.y_size), (31666, 41817));
    assert_eq!(h.nr_bands, 4);
    assert_eq!(h.scale_factor, 1);
    assert_eq!((h.x_block_size, h.y_block_size), (64, 64));
    assert_eq!(h.compression_rate, 30);
    assert_eq!(h.datum, "WGS84");
    assert_eq!(h.projection, "SUTM19");
    assert_eq!(h.epsg(), Some(32719));
    assert_eq!(h.total_blocks, 108_241);

    // GSD ~4.05 cm, north-up, origin in UTM 19S near Maipú.
    assert!((h.cell_increment_x - 0.0405).abs() < 0.001);
    assert!((h.cell_increment_y + 0.0405).abs() < 0.001);
    assert!(h.origin_x > 300_000.0 && h.origin_x < 360_000.0);
    assert!(h.origin_y > 6_250_000.0 && h.origin_y < 6_320_000.0);

    // Level chain doubles from 124x164 to 15833x20909.
    assert_eq!((h.levels[0].x_size, h.levels[0].y_size), (124, 164));
    assert_eq!((h.levels[7].x_size, h.levels[7].y_size), (15833, 20909));
    for w in h.levels.windows(2) {
        assert_eq!(w[1].x_size.div_ceil(2), w[0].x_size);
        assert_eq!(w[1].y_size.div_ceil(2), w[0].y_size);
    }
}

#[test]
fn coarse_overview_decodes_with_plausible_statistics() {
    let Some(mut reader) = open_maipu() else {
        return;
    };
    // 1:256 → 123x163 output, reconstructed from the smallest levels.
    let bands = reader.read_reduced(8).expect("overview decode");
    assert_eq!(bands.len(), 4);
    let (rows, cols) = bands[0].shape();
    assert_eq!((rows, cols), (163, 123));

    for (i, band) in bands.iter().enumerate() {
        let data = band.data();
        let n = data.len() as f64;
        let mean = data.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let sd = (data
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        // An orthomosaic overview is neither black, saturated nor flat.
        assert!(
            mean > 5.0 && mean < 250.0,
            "band {i}: implausible mean {mean}"
        );
        assert!(sd > 1.0, "band {i}: implausibly flat (sd {sd})");
    }

    // Georeference: 1:256 of a 4 cm GSD is ~10.4 m cells.
    let t = bands[0].transform();
    assert!((t.pixel_width - 0.0405 * 31666.0 / 123.0).abs() < 0.2);
    assert!(t.pixel_height < 0.0);
    assert_eq!(
        bands[0].crs().and_then(|c| c.epsg()),
        Some(32719),
        "CRS must map SUTM19/WGS84 to EPSG:32719"
    );
}

#[test]
fn windowed_full_resolution_read() {
    let Some(mut reader) = open_maipu() else {
        return;
    };
    // A 256x256 window at 1:1 somewhere inside the mosaic.
    let params = RegionParams {
        start_x: 15_000,
        start_y: 20_000,
        end_x: 15_255,
        end_y: 20_255,
        number_x: 256,
        number_y: 256,
    };
    let bands = reader.read_region(params).expect("windowed decode");
    assert_eq!(bands.len(), 4);
    assert_eq!(bands[0].shape(), (256, 256));

    // Full-resolution imagery has fine texture: neighbouring-pixel
    // differences must not all be zero.
    let flat: Vec<u8> = bands[0].data().iter().copied().collect();
    let diffs = flat.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(
        diffs > 100,
        "window is suspiciously uniform ({diffs} diffs)"
    );
}

#[test]
fn overviews_are_consistent_across_levels() {
    // The same scene decoded at 1:256 and at 1:128-downsampled-to-1:256
    // must agree strongly: they are reconstructions of the same wavelet
    // pyramid truncated one level apart.
    let Some(mut reader) = open_maipu() else {
        return;
    };
    let coarse = reader.read_reduced(8).expect("1:256");
    let fine = reader.read_reduced(7).expect("1:128");

    let (rows, cols) = coarse[0].shape();
    let mut sum_abs = 0.0f64;
    let mut count = 0u64;
    for r in 0..rows {
        for c in 0..cols {
            let a = f64::from(coarse[0].get(r, c).unwrap());
            let b = f64::from(fine[0].get(2 * r, 2 * c).unwrap());
            sum_abs += (a - b).abs();
            count += 1;
        }
    }
    let mae = sum_abs / count as f64;
    // Nearest-neighbour subsampling vs one more synthesis level: small
    // but nonzero disagreement. A broken decoder produces MAE > 50.
    assert!(mae < 20.0, "cross-level MAE too high: {mae}");
}
