//! `StacClient::search_all` pagination against a local STAC server.
//!
//! Earth Search pages with a POST `next` link whose body carries a `next`
//! cursor. That body used to be parsed into `StacSearchParams`, which has
//! no `next` field, so the cursor was dropped and the first page was
//! fetched again until `max_items` (1000 copies of 6 items on a real
//! query).

#![cfg(feature = "native")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use surtgis_cloud::{StacCatalog, StacClient, StacClientOptions, StacSearchParams};

fn item(id: &str) -> String {
    format!(
        r#"{{"type":"Feature","id":"{id}","geometry":null,"properties":{{"datetime":"2024-01-03T00:00:00Z"}},"assets":{{}},"links":[]}}"#
    )
}

/// Serve `POST /search`: `page(body)` returns the response JSON for the
/// request body. Counts requests. Leaks its thread.
fn spawn_stac(page: fn(&str, &str) -> String) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let hits = Arc::new(AtomicUsize::new(0));
    let (b, h) = (base.clone(), hits.clone());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            h.fetch_add(1, Ordering::SeqCst);
            // Read headers, then the body up to Content-Length.
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let body = loop {
                let n = stream.read(&mut chunk).unwrap_or(0);
                if n == 0 {
                    break String::new();
                }
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(i) = text.find("\r\n\r\n") {
                    let len: usize = text[..i]
                        .lines()
                        .filter_map(|l| l.split_once(':'))
                        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                        .and_then(|(_, v)| v.trim().parse().ok())
                        .unwrap_or(0);
                    if buf.len() >= i + 4 + len {
                        break text[i + 4..i + 4 + len].to_string();
                    }
                }
            };
            let json = page(&b, &body);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/geo+json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
                json.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    (base, hits)
}

fn client(base: &str) -> StacClient {
    let opts = StacClientOptions {
        max_items: 1000,
        ..StacClientOptions::default()
    };
    StacClient::new(StacCatalog::Custom(base.to_string()), opts).unwrap()
}

fn params() -> StacSearchParams {
    StacSearchParams::new()
        .bbox(-71.5, -35.5, -71.4, -35.4)
        .collections(&["sentinel-2-l2a"])
}

/// Two pages, Earth Search style: the second is only returned when the
/// request body carries the `next` cursor of the first.
#[tokio::test]
async fn earth_search_next_cursor_reaches_the_second_page() {
    let (base, hits) = spawn_stac(|base, body| {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        if v.get("next").and_then(|n| n.as_str()) == Some("cursor-2") {
            format!(
                r#"{{"type":"FeatureCollection","features":[{},{}],"links":[]}}"#,
                item("c"),
                item("d")
            )
        } else {
            format!(
                r#"{{"type":"FeatureCollection","features":[{},{}],"links":[{{"rel":"next","method":"POST","href":"{base}/search","merge":false,"body":{{"collections":["sentinel-2-l2a"],"limit":2,"next":"cursor-2"}}}}]}}"#,
                item("a"),
                item("b")
            )
        }
    });
    let items = client(&base).search_all(&params()).await.unwrap();
    let ids: Vec<&str> = items.iter().map(|i| i.id.as_str()).collect();
    assert_eq!(ids, ["a", "b", "c", "d"]);
    assert_eq!(hits.load(Ordering::SeqCst), 2);
}

/// A catalog whose `next` link always returns the same page: stop after
/// the first repeat instead of collecting `max_items` copies.
#[tokio::test]
async fn repeating_page_stops_pagination() {
    let (base, hits) = spawn_stac(|base, _| {
        format!(
            r#"{{"type":"FeatureCollection","features":[{},{}],"links":[{{"rel":"next","method":"POST","href":"{base}/search","merge":true}}]}}"#,
            item("a"),
            item("b")
        )
    });
    let items = client(&base).search_all(&params()).await.unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(hits.load(Ordering::SeqCst), 2);
}
