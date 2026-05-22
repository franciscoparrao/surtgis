//! STAC ML-AOI / MLM metadata emission for `extract-patches`.
//!
//! When `--emit-stac` is passed, the handler writes a STAC Collection
//! plus one Item per chip into `<output>/stac/`. The collection uses the
//! [STAC MLM extension](https://github.com/stac-extensions/mlm) to record
//! the foundation-model target (when a `--profile` was set), and each item
//! uses the [ML-AOI extension](https://github.com/stac-extensions/ml-aoi)
//! to declare its role as labelled training data.
//!
//! Geometry handling: STAC v1.0 requires `bbox` and `geometry` in WGS84
//! (EPSG:4326). When the `projections` feature is compiled (default), we
//! reproject from the source CRS using proj4rs. When it isn't, we leave
//! coords in source-CRS units and stamp `proj:epsg` on the item so
//! downstream tools know what they're looking at.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

use surtgis_core::GeoTransform;

use super::gfm_profiles::GfmProfileSpec;

const STAC_VERSION: &str = "1.0.0";
const MLM_SCHEMA: &str = "https://stac-extensions.github.io/mlm/v1.4.0/schema.json";
const MLAOI_SCHEMA: &str = "https://stac-extensions.github.io/ml-aoi/v0.2.0/schema.json";
const PROJ_SCHEMA: &str = "https://stac-extensions.github.io/projection/v1.1.0/schema.json";

/// Per-chip info needed to build a STAC Item. Independent of the
/// PatchSpec type in `extract_patches.rs` to keep that module's
/// internal types private.
pub struct ChipInfo<'a> {
    pub index: usize,
    pub center_row: usize,
    pub center_col: usize,
    pub label_int: Option<i64>,
    pub label_float: Option<f64>,
    /// Asset path relative to the output dir (e.g. "patches.npy" or "patches.zarr")
    pub asset_path: &'a str,
    pub asset_role: &'a str, // "data" or "data-chunk"
}

/// Configuration for the collection-level metadata.
pub struct CollectionInfo<'a> {
    pub id: &'a str,
    pub description: &'a str,
    pub license: &'a str,
    pub source_mode: &'a str, // "points" or "polygons"
    pub patch_size: usize,
    pub n_patches: usize,
    pub n_bands: usize,
    pub n_timestamps: usize,
    pub band_names: &'a [String],
    pub timestamps: &'a [String],
    pub crs_epsg: Option<u32>,
    pub gt: &'a GeoTransform,
    pub grid_rows: usize,
    pub grid_cols: usize,
    pub profile_spec: Option<&'a GfmProfileSpec>,
}

/// Reproject a single (x, y) pair from source CRS to WGS84 (lon, lat).
/// Returns the input unchanged when projection is unavailable.
#[cfg(feature = "projections")]
fn to_wgs84(x: f64, y: f64, src_epsg: u32) -> Option<(f64, f64)> {
    use proj4rs::Proj;
    if src_epsg == 4326 {
        return Some((x, y));
    }
    let src = Proj::from_epsg_code(src_epsg as u16).ok()?;
    let dst = Proj::from_epsg_code(4326).ok()?;
    let mut pt = (x, y, 0.0_f64);
    proj4rs::transform::transform(&src, &dst, &mut pt).ok()?;
    // proj4rs returns lat/lon in radians for geographic CRS — convert
    if dst.is_latlong() {
        Some((pt.0.to_degrees(), pt.1.to_degrees()))
    } else {
        Some((pt.0, pt.1))
    }
}

#[cfg(not(feature = "projections"))]
fn to_wgs84(_x: f64, _y: f64, _src_epsg: u32) -> Option<(f64, f64)> {
    None
}

/// Build the WGS84 bbox + Polygon geometry for a chip. Returns
/// `(bbox_wgs84, polygon_coords_wgs84, in_source_crs)` — when reprojection
/// isn't available the coords are in source CRS and the bool is true.
fn chip_geometry(
    chip: &ChipInfo,
    patch_size: usize,
    gt: &GeoTransform,
    crs_epsg: Option<u32>,
) -> ([f64; 4], Vec<[f64; 2]>, bool) {
    let half = patch_size / 2;
    let r0 = (chip.center_row - half) as f64;
    let c0 = (chip.center_col - half) as f64;
    let r1 = r0 + patch_size as f64;
    let c1 = c0 + patch_size as f64;

    let pix_to_geo = |row: f64, col: f64| -> (f64, f64) {
        let x = gt.origin_x + col * gt.pixel_width;
        let y = gt.origin_y + row * gt.pixel_height;
        (x, y)
    };

    let corners_src = [
        pix_to_geo(r0, c0),
        pix_to_geo(r0, c1),
        pix_to_geo(r1, c1),
        pix_to_geo(r1, c0),
    ];

    let (corners, in_src) = match crs_epsg {
        Some(epsg) => {
            let mut reproj: Vec<(f64, f64)> = Vec::with_capacity(4);
            let mut all_ok = true;
            for (x, y) in &corners_src {
                match to_wgs84(*x, *y, epsg) {
                    Some(ll) => reproj.push(ll),
                    None => {
                        all_ok = false;
                        break;
                    }
                }
            }
            if all_ok && reproj.len() == 4 {
                (reproj, false)
            } else {
                (corners_src.to_vec(), true)
            }
        }
        None => (corners_src.to_vec(), true),
    };

    let xs: Vec<f64> = corners.iter().map(|(x, _)| *x).collect();
    let ys: Vec<f64> = corners.iter().map(|(_, y)| *y).collect();
    let bbox = [
        xs.iter().cloned().fold(f64::INFINITY, f64::min),
        ys.iter().cloned().fold(f64::INFINITY, f64::min),
        xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    ];
    // Close the polygon ring.
    let mut ring: Vec<[f64; 2]> = corners.iter().map(|(x, y)| [*x, *y]).collect();
    ring.push(ring[0]);
    (bbox, ring, in_src)
}

fn build_item(
    collection_id: &str,
    chip: &ChipInfo,
    bbox: [f64; 4],
    polygon: Vec<[f64; 2]>,
    in_source_crs: bool,
    crs_epsg: Option<u32>,
    timestamps: &[String],
) -> serde_json::Value {
    let mut props = serde_json::Map::new();
    // datetime: STAC requires either `datetime` or a `start_datetime`/`end_datetime`
    // pair. For training chips we typically don't have a single canonical
    // datetime; use null and emit the temporal extent on the collection.
    props.insert("datetime".to_string(), serde_json::Value::Null);
    if timestamps.len() == 1 {
        props.insert(
            "timestamp_label".to_string(),
            serde_json::Value::String(timestamps[0].clone()),
        );
    } else if timestamps.len() > 1 {
        let arr: Vec<serde_json::Value> = timestamps
            .iter()
            .map(|t| serde_json::Value::String(t.clone()))
            .collect();
        props.insert("timestamps".to_string(), serde_json::Value::Array(arr));
    }
    // ML-AOI extension: this chip is feature+label training data
    props.insert(
        "ml-aoi:role".to_string(),
        serde_json::Value::String("label".to_string()),
    );
    props.insert(
        "ml-aoi:reference-grid".to_string(),
        serde_json::Value::Bool(true),
    );
    if let Some(v) = chip.label_int {
        props.insert("ml-aoi:label_class".to_string(), serde_json::json!(v));
    } else if let Some(v) = chip.label_float {
        props.insert("ml-aoi:label_value".to_string(), serde_json::json!(v));
    }
    if in_source_crs {
        if let Some(epsg) = crs_epsg {
            props.insert("proj:epsg".to_string(), serde_json::json!(epsg));
        }
        props.insert(
            "ml-aoi:bbox_native_crs".to_string(),
            serde_json::Value::Bool(true),
        );
    }

    let mut extensions = vec![MLAOI_SCHEMA.to_string()];
    if in_source_crs && crs_epsg.is_some() {
        extensions.push(PROJ_SCHEMA.to_string());
    }

    serde_json::json!({
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": extensions,
        "id": format!("chip_{:06}", chip.index),
        "collection": collection_id,
        "bbox": bbox,
        "geometry": {
            "type": "Polygon",
            "coordinates": [polygon],
        },
        "properties": serde_json::Value::Object(props),
        "assets": {
            chip.asset_role: {
                "href": format!("../{}", chip.asset_path),
                "type": "application/octet-stream",
                "roles": [chip.asset_role, "data"],
                "title": format!("Patch tensor chip {}", chip.index),
            }
        },
        "links": [
            {"rel": "collection", "href": "../collection.json", "type": "application/json"},
            {"rel": "self", "href": format!("chip_{:06}.json", chip.index), "type": "application/json"},
        ],
    })
}

fn build_collection(info: &CollectionInfo, item_count: usize) -> serde_json::Value {
    let mut extensions = vec![MLAOI_SCHEMA.to_string()];
    if info.profile_spec.is_some() {
        extensions.push(MLM_SCHEMA.to_string());
    }

    // Collection-level spatial extent. We approximate by reprojecting the
    // four corners of the full raster grid — same heuristic as a per-chip
    // bbox, applied to the whole AOI.
    let dummy_chip = ChipInfo {
        index: 0,
        center_row: info.grid_rows / 2,
        center_col: info.grid_cols / 2,
        label_int: None,
        label_float: None,
        asset_path: "",
        asset_role: "",
    };
    let chip_full = ChipInfo {
        center_row: info.grid_rows.saturating_sub(1),
        center_col: info.grid_cols.saturating_sub(1),
        ..dummy_chip
    };
    let (bbox, _, _) = chip_geometry(
        &chip_full,
        info.grid_rows.max(info.grid_cols),
        info.gt,
        info.crs_epsg,
    );

    let mut props = serde_json::Map::new();
    if let Some(spec) = info.profile_spec {
        // MLM input descriptor — Prithvi-style: bands × T × H × W
        let input = serde_json::json!({
            "name": format!("{} input", spec.name),
            "bands": spec.bands_order,
            "input": {
                "shape": [-1, spec.bands_order.len(), info.n_timestamps.max(1), spec.tile_size, spec.tile_size],
                "dim_order": ["batch", "channel", "time", "height", "width"],
                "data_type": "float32",
            },
            "norm_by_channel": true,
            "norm_type": "z-score",
            "statistics": spec.band_norm.iter().map(|(m, s)| serde_json::json!({"mean": m, "stddev": s})).collect::<Vec<_>>(),
            "resize_type": "none",
        });
        props.insert(
            "mlm:framework".to_string(),
            serde_json::Value::String("pytorch".to_string()),
        );
        props.insert(
            "mlm:tasks".to_string(),
            serde_json::json!(["regression", "classification", "segmentation"]),
        );
        props.insert(
            "mlm:input".to_string(),
            serde_json::Value::Array(vec![input]),
        );
        props.insert(
            "mlm:model_target".to_string(),
            serde_json::Value::String(spec.model_target.to_string()),
        );
        props.insert(
            "mlm:source".to_string(),
            serde_json::Value::String(spec.source_url.to_string()),
        );
    }
    props.insert(
        "ml-aoi:tasks".to_string(),
        serde_json::json!([if info.source_mode == "points" {
            "patch-classification"
        } else {
            "patch-segmentation"
        }]),
    );

    serde_json::json!({
        "type": "Collection",
        "stac_version": STAC_VERSION,
        "stac_extensions": extensions,
        "id": info.id,
        "title": format!("SurtGIS extract-patches output: {} chips", info.n_patches),
        "description": info.description,
        "license": info.license,
        "extent": {
            "spatial": { "bbox": [bbox] },
            "temporal": {
                "interval": [[
                    info.timestamps.first().map(|t| serde_json::Value::String(t.clone())).unwrap_or(serde_json::Value::Null),
                    info.timestamps.last().map(|t| serde_json::Value::String(t.clone())).unwrap_or(serde_json::Value::Null),
                ]]
            }
        },
        "summaries": {
            "bands": info.band_names,
            "n_timestamps": info.n_timestamps,
            "patch_size": info.patch_size,
        },
        "properties": serde_json::Value::Object(props),
        "links": (0..item_count).map(|i| serde_json::json!({
            "rel": "item",
            "href": format!("items/chip_{:06}.json", i),
            "type": "application/json",
        })).collect::<Vec<_>>(),
    })
}

/// Write the full STAC dataset description to `<output_dir>/stac/`.
pub fn write_stac_output(
    output_dir: &Path,
    collection_info: &CollectionInfo,
    chips: &[ChipInfo],
) -> Result<()> {
    let stac_dir = output_dir.join("stac");
    let items_dir = stac_dir.join("items");
    fs::create_dir_all(&items_dir)
        .with_context(|| format!("Failed to create {}", items_dir.display()))?;

    let collection = build_collection(collection_info, chips.len());
    fs::write(
        stac_dir.join("collection.json"),
        serde_json::to_string_pretty(&collection)?,
    )?;

    let mut in_src_any = false;
    for chip in chips {
        let (bbox, polygon, in_src) = chip_geometry(
            chip,
            collection_info.patch_size,
            collection_info.gt,
            collection_info.crs_epsg,
        );
        if in_src {
            in_src_any = true;
        }
        let item = build_item(
            collection_info.id,
            chip,
            bbox,
            polygon,
            in_src,
            collection_info.crs_epsg,
            collection_info.timestamps,
        );
        let path = items_dir.join(format!("chip_{:06}.json", chip.index));
        fs::write(&path, serde_json::to_string_pretty(&item)?)?;
    }

    if in_src_any {
        eprintln!(
            "  WARNING: STAC items emitted in source CRS coords (proj:epsg). \
                   Build with --features projections for WGS84 reprojection."
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chip_geometry_no_crs_returns_source_coords() {
        let gt = GeoTransform::new(100.0, 200.0, 10.0, -10.0);
        let chip = ChipInfo {
            index: 0,
            center_row: 50,
            center_col: 50,
            label_int: Some(1),
            label_float: None,
            asset_path: "patches.npy",
            asset_role: "data",
        };
        let (bbox, ring, in_src) = chip_geometry(&chip, 20, &gt, None);
        assert!(in_src);
        // patch_size=20, so corners are rows 40..60, cols 40..60
        // x = 100 + 40*10 = 500 .. 100 + 60*10 = 700
        // y = 200 + 40*(-10) = -200 .. 200 + 60*(-10) = -400
        assert!((bbox[0] - 500.0).abs() < 1e-6);
        assert!((bbox[2] - 700.0).abs() < 1e-6);
        assert_eq!(ring.len(), 5); // closed ring
        assert_eq!(ring[0], ring[4]);
    }
}
