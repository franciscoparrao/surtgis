//! Multi-band COG reads against small fixtures served by a local
//! range-capable HTTP server (no network, no GDAL).
//!
//! Fixtures (written with rasterio, pixel-interleaved, 96 × 64, 32 × 32
//! tiles, deflate):
//! - `rgb_u8_tiled.tif`: 3 × u8, predictor 2; band values are
//!   `(k % 256) % 251` (numpy's uint8 arange wraps before the modulo),
//!   `(k / 3) % 199` and the constant 7, with `k = i·W + j`.
//! - `f32_2band_tiled.tif`: 2 × f32, predictor 3, nodata −9999 in band 1
//!   cells `[0..4, 0..4]`, one overview (÷2).

#![cfg(feature = "native")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;

use surtgis_cloud::cog_reader::{CogReader, CogReaderOptions};
use surtgis_cloud::tile_index::BBox;

const W: usize = 96;
const H: usize = 64;

/// Serve every file under `crates/cloud/tests/fixtures` with `HEAD` and
/// single-range `GET` (what the COG reader issues). Leaks its thread.
fn spawn_range_server() -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let root = root.clone();
            std::thread::spawn(move || {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let mut lines = req.lines();
                let first = lines.next().unwrap_or("");
                let mut parts = first.split_whitespace();
                let method = parts.next().unwrap_or("GET");
                let path = parts.next().unwrap_or("/").trim_start_matches('/');
                let range = lines
                    .filter_map(|l| l.split_once(':'))
                    .find(|(k, _)| k.eq_ignore_ascii_case("range"))
                    .map(|(_, v)| v.trim().to_string());
                let Ok(bytes) = std::fs::read(root.join(path)) else {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    return;
                };
                let total = bytes.len();
                let resp = match (method, range) {
                    ("HEAD", _) => format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {total}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n"
                    )
                    .into_bytes(),
                    (_, Some(r)) => {
                        let spec = r.trim_start_matches("bytes=");
                        let (a, b) = spec.split_once('-').unwrap_or(("0", ""));
                        let start: usize = a.parse().unwrap_or(0);
                        let end: usize = b.parse::<usize>().map(|e| e.min(total - 1)).unwrap_or(total - 1);
                        let body = &bytes[start..=end];
                        let mut v = format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{total}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .into_bytes();
                        v.extend_from_slice(body);
                        v
                    }
                    _ => {
                        let mut v = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {total}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n"
                        )
                        .into_bytes();
                        v.extend_from_slice(&bytes);
                        v
                    }
                };
                let _ = stream.write_all(&resp);
                let _ = stream.flush();
            });
        }
    });
    format!("http://{addr}")
}

fn full_bbox() -> BBox {
    // Origin (500000, 6300000), 10 m cells, 96 × 64.
    BBox::new(
        500_000.0,
        6_300_000.0 - 640.0,
        500_000.0 + 960.0,
        6_300_000.0,
    )
}

#[tokio::test]
async fn rgb_u8_predictor2_all_bands_and_single_band_agree() {
    let base = spawn_range_server();
    let url = format!("{base}/rgb_u8_tiled.tif");
    let mut reader = CogReader::open(&url, CogReaderOptions::default())
        .await
        .expect("open rgb fixture");
    assert_eq!(reader.bands(), 3);
    let m = reader.metadata();
    assert_eq!((m.width, m.height, m.tile_width), (96, 64, 32));

    let bands = reader
        .read_bbox_bands::<u8>(&full_bbox(), None)
        .await
        .expect("read all bands");
    assert_eq!(bands.len(), 3);
    for b in &bands {
        assert_eq!(b.shape(), (H, W));
    }
    for i in 0..H {
        for j in 0..W {
            let k = i * W + j;
            assert_eq!(
                bands[0].data()[[i, j]],
                ((k % 256) % 251) as u8,
                "band 0 ({i},{j})"
            );
            assert_eq!(
                bands[1].data()[[i, j]],
                ((k / 3) % 199) as u8,
                "band 1 ({i},{j})"
            );
            assert_eq!(bands[2].data()[[i, j]], 7, "band 2 ({i},{j})");
        }
    }

    // read_bbox stays band 0; read_bbox_band picks any band.
    let b0 = reader.read_bbox::<u8>(&full_bbox(), None).await.unwrap();
    assert_eq!(b0.data(), bands[0].data());
    let b2 = reader
        .read_bbox_band::<u8>(&full_bbox(), None, 2)
        .await
        .unwrap();
    assert_eq!(b2.data(), bands[2].data());
    assert!(
        reader
            .read_bbox_band::<u8>(&full_bbox(), None, 3)
            .await
            .is_err()
    );

    // A window crossing tile borders, as f64.
    let win = BBox::new(
        500_000.0 + 250.0,
        6_300_000.0 - 400.0,
        500_000.0 + 650.0,
        6_300_000.0 - 150.0,
    );
    let sub = reader.read_bbox_bands::<f64>(&win, None).await.unwrap();
    let (rows, cols) = sub[0].shape();
    assert_eq!((rows, cols), (25, 40));
    for i in 0..rows {
        for j in 0..cols {
            let k = (15 + i) * W + (25 + j);
            assert_eq!(sub[0].data()[[i, j]], ((k % 256) % 251) as f64);
            assert_eq!(sub[1].data()[[i, j]], ((k / 3) % 199) as f64);
        }
    }
}

#[tokio::test]
async fn f32_two_bands_predictor3_nodata_and_overview() {
    let base = spawn_range_server();
    let url = format!("{base}/f32_2band_tiled.tif");
    let mut reader = CogReader::open(&url, CogReaderOptions::default())
        .await
        .expect("open f32 fixture");
    assert_eq!(reader.bands(), 2);
    assert_eq!(reader.metadata().num_overviews, 1);
    assert_eq!(reader.metadata().nodata, Some(-9999.0));

    let bands = reader
        .read_bbox_bands::<f32>(&full_bbox(), None)
        .await
        .expect("read both bands");
    assert_eq!(bands.len(), 2);
    let n = (H * W) as f32;
    for i in 0..H {
        for j in 0..W {
            let k = (i * W + j) as f32;
            let want0 = k * 1000.0 / (n - 1.0);
            let got0 = bands[0].data()[[i, j]];
            if i < 4 && j < 4 {
                assert!(got0.is_nan(), "nodata → NaN at ({i},{j}), got {got0}");
            } else {
                assert!(
                    (got0 - want0).abs() < 1e-3,
                    "band 0 ({i},{j}): {got0} vs {want0}"
                );
            }
            let got1 = bands[1].data()[[i, j]];
            assert!((got1 - k * 0.5).abs() < 1e-6, "band 1 ({i},{j}): {got1}");
        }
    }
    assert!(bands[0].nodata().unwrap().is_nan());

    // Overview ÷2: half the size, both bands.
    let ov = reader
        .read_bbox_bands::<f32>(&full_bbox(), Some(1))
        .await
        .expect("read overview");
    assert_eq!(ov.len(), 2);
    assert_eq!(ov[0].shape(), (H / 2, W / 2));
    assert_eq!(ov[1].shape(), (H / 2, W / 2));
    assert!(ov[1].data()[[10, 10]].is_finite());
}
