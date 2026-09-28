//! Embedded provenance: written by every native writer, read back, and
//! verifiable against the decoded array.

use std::sync::{Arc, Mutex};

use surtgis_core::io::{
    CogOptions, GeoTiffOptions, StripWriterConfig, raster_data_hash, read_geotiff,
    read_geotiff_bands, read_provenance, write_cog_with_provenance, write_geotiff,
    write_geotiff_stack, write_geotiff_streaming, write_geotiff_with_provenance,
};
use surtgis_core::provenance::{self, InputRecord, Provenance};
use surtgis_core::{GeoTransform, Raster};

/// Process-wide hooks are shared by every test in this binary: serialise
/// the ones that install them.
static HOOKS: Mutex<()> = Mutex::new(());

fn dem(rows: usize, cols: usize) -> Raster<f64> {
    let mut r = Raster::<f64>::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            r.data_mut()[[i, j]] = 100.0 + (i * cols + j) as f64 * 0.5;
        }
    }
    r.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 10.0, -10.0));
    r.set_crs(Some(surtgis_core::CRS::from_epsg(32719)));
    r.set_nodata(Some(-9999.0));
    r
}

fn record() -> Provenance {
    let mut p = Provenance::new()
        .with_operation("test")
        .with_command(["surtgis", "test"].map(String::from))
        .with_parameters(serde_json::json!({"k": 1}))
        .with_threads(3);
    p.push_input(InputRecord {
        source: "memory".into(),
        blake3: None,
        bytes: None,
        role: Some("dem".into()),
    });
    p
}

#[test]
fn single_band_explicit_record_round_trips_and_verifies() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.tif");
    let r = dem(20, 30);
    write_geotiff_with_provenance(&r, &path, None, &record()).unwrap();

    let p = read_provenance(&path).unwrap().expect("record present");
    assert_eq!(p.operation.as_deref(), Some("test"));
    assert_eq!(p.threads, Some(3));
    assert_eq!(p.inputs.len(), 1);
    let out = p.output.expect("writer fills output");
    assert_eq!(out.shape, [20, 30]);
    assert_eq!(out.bands, 1);
    assert_eq!(out.dtype, "f64");

    // Verification = digest of the decoded array equals the recorded one.
    let back: Raster<f64> = read_geotiff(&path, None).unwrap();
    assert_eq!(raster_data_hash(&[&back]).data_blake3, out.data_blake3);

    // A different array gives a different digest.
    let mut other = back.clone();
    other.data_mut()[[3, 4]] += 1.0;
    assert_ne!(raster_data_hash(&[&other]).data_blake3, out.data_blake3);

    // The pixels themselves are untouched by the tag.
    assert_eq!(back.data(), r.data());
}

#[test]
fn no_record_without_hooks_or_explicit() {
    let _g = HOOKS.lock().unwrap_or_else(|e| e.into_inner());
    provenance::set_output_provider(None);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.tif");
    write_geotiff(&dem(4, 5), &path, None).unwrap();
    assert!(read_provenance(&path).unwrap().is_none());
}

#[test]
fn process_hooks_feed_every_writer() {
    let _g = HOOKS.lock().unwrap_or_else(|e| e.into_inner());
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let s2 = seen.clone();
    provenance::set_input_observer(Some(Arc::new(move |s: &str| {
        s2.lock().unwrap().push(s.to_string())
    })));
    provenance::set_output_provider(Some(Arc::new(|| {
        Some(Provenance::new().with_operation("hooked"))
    })));

    let dir = tempfile::tempdir().unwrap();
    let r = dem(16, 24);

    // Plain single-band writer picks the provider up.
    let single = dir.path().join("single.tif");
    write_geotiff(&r, &single, Some(GeoTiffOptions::default())).unwrap();
    let p = read_provenance(&single).unwrap().unwrap();
    assert_eq!(p.operation.as_deref(), Some("hooked"));
    let back: Raster<f64> = read_geotiff(&single, None).unwrap();
    assert_eq!(
        raster_data_hash(&[&back]).data_blake3,
        p.output.unwrap().data_blake3
    );
    // …and the reader reported the input.
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .any(|s| s.ends_with("single.tif"))
    );

    // Streaming writer (f32 strips) hashes what it writes.
    let streamed = dir.path().join("stream.tif");
    let cfg = StripWriterConfig {
        rows: 16,
        cols: 24,
        transform: *r.transform(),
        crs: None,
        nodata: Some(-9999.0),
        compress: true,
        rows_per_strip: 5,
    };
    let src = r.clone();
    write_geotiff_streaming(&streamed, &cfg, |idx, n| {
        let start = idx * 5;
        Ok(src
            .data()
            .slice(ndarray::s![start..start + n, ..])
            .to_owned())
    })
    .unwrap();
    let p = read_provenance(&streamed).unwrap().unwrap();
    let out = p.output.unwrap();
    assert_eq!((out.dtype.as_str(), out.shape), ("f32", [16, 24]));
    let back: Raster<f32> = read_geotiff(&streamed, None).unwrap();
    assert_eq!(raster_data_hash(&[&back]).data_blake3, out.data_blake3);

    // Stack writer keeps band names and adds the record.
    let stacked = dir.path().join("stack.tif");
    let b2 = {
        let mut b = r.clone();
        b.data_mut().mapv_inplace(|v| v * 2.0);
        b
    };
    write_geotiff_stack(
        &[&r, &b2],
        Some(&["a", "b"][..]),
        &stacked,
        &GeoTiffOptions::default(),
    )
    .unwrap();
    let items = surtgis_core::io::read_gdal_metadata(&stacked).unwrap();
    assert_eq!(items.iter().filter(|(k, _)| k == "DESCRIPTION").count(), 2);
    let p = read_provenance(&stacked).unwrap().unwrap();
    let out = p.output.unwrap();
    assert_eq!(out.bands, 2);
    let bands: Vec<Raster<f64>> = read_geotiff_bands(&stacked).unwrap();
    let refs: Vec<&Raster<f64>> = bands.iter().collect();
    assert_eq!(raster_data_hash(&refs).data_blake3, out.data_blake3);

    provenance::set_input_observer(None);
    provenance::set_output_provider(None);
}

#[test]
fn cog_main_ifd_carries_record() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out_cog.tif");
    let r = dem(700, 900);
    write_cog_with_provenance(&r, &path, &CogOptions::default(), &record()).unwrap();
    let p = read_provenance(&path).unwrap().expect("record in main IFD");
    let out = p.output.unwrap();
    assert_eq!(out.shape, [700, 900]);
    let back: Raster<f64> = read_geotiff(&path, None).unwrap();
    assert_eq!(raster_data_hash(&[&back]).data_blake3, out.data_blake3);
}

#[test]
fn json_in_xml_survives_escaping() {
    // The record is JSON inside XML: quotes, angle brackets and ampersands
    // in parameters must round-trip exactly.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("esc.tif");
    let p = Provenance::new()
        .with_operation("a<b>&\"c\"")
        .with_parameters(serde_json::json!({"formula": "(N-R)/(N+R) < 1 & \"x\""}));
    write_geotiff_with_provenance(&dem(3, 3), &path, None, &p).unwrap();
    let back = read_provenance(&path).unwrap().unwrap();
    assert_eq!(back.operation, p.operation);
    assert_eq!(back.parameters, p.parameters);
}
