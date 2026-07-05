//! CSV / JSON writers for the fluvial-network data products.
//!
//! The core types in this module ([`super::LongProfile`],
//! [`super::SwathProfile`], [`super::KsnSegment`], and the per-cell χ
//! raster) are kept clean of serde derives so downstream consumers
//! don't transitively buy serde when they only want the in-memory
//! algorithm output. The writers here own the (de)serialisation
//! responsibility — they accept the core types and emit row-oriented
//! CSV or hierarchical JSON via local wrapper structs.
//!
//! ## Why row-oriented CSV
//!
//! Paper-figure plotting toolkits (matplotlib / ggplot2 / R) expect
//! tidy long-format tables: one row per observation, columns for the
//! categorical and continuous variables. The hierarchical structure
//! of a [`super::LongProfile`] (profile → nodes) collapses cleanly to
//! `(profile_id, node_index, ...)` so a single CSV file can hold all
//! profiles in a basin without forcing the user to merge files.
//!
//! JSON is the lossless format: it preserves the hierarchy and the
//! float `Option<f64>` semantics (chi/ksn `None` becomes JSON `null`
//! rather than an empty CSV cell).

use std::io::Write;

use serde::Serialize;
use surtgis_core::Raster;

use super::{KsnResult, KsnSegment, LongProfile, SwathProfile, SwathStats};

/// Errors raised by the writers.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// CSV serialization failed.
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
    /// JSON serialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Writing the output file failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

// ─── Long profile ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct LongProfileRow {
    /// Outlet index in the input `Vec<LongProfile>`.
    profile_id: usize,
    /// Position within the profile, 0 at outlet.
    node_index: usize,
    x: f64,
    y: f64,
    distance_m: f64,
    elevation: f64,
    area_m2: f64,
    /// `None` becomes empty in CSV, `null` in JSON.
    chi: Option<f64>,
    ksn: Option<f64>,
}

/// Write a flat CSV: one row per profile node. The `profile_id`
/// column groups rows belonging to the same outlet.
pub fn write_long_profiles_csv<W: Write>(
    profiles: &[LongProfile],
    w: W,
) -> Result<(), ExportError> {
    let mut wtr = csv::Writer::from_writer(w);
    for (pi, p) in profiles.iter().enumerate() {
        for (ni, n) in p.nodes.iter().enumerate() {
            wtr.serialize(LongProfileRow {
                profile_id: pi,
                node_index: ni,
                x: n.coord.0,
                y: n.coord.1,
                distance_m: n.distance_from_outlet_m,
                elevation: n.elevation,
                area_m2: n.area_m2,
                chi: n.chi,
                ksn: n.ksn,
            })?;
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(Serialize)]
struct LongProfileJson<'a> {
    outlet_x: f64,
    outlet_y: f64,
    nodes: Vec<LongProfileRow>,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<&'a ()>,
}

/// Write the hierarchical JSON: one object per profile with its
/// `outlet` coordinate and a `nodes` array of per-node observations.
pub fn write_long_profiles_json<W: Write>(
    profiles: &[LongProfile],
    w: W,
) -> Result<(), ExportError> {
    let docs: Vec<LongProfileJson> = profiles
        .iter()
        .enumerate()
        .map(|(pi, p)| LongProfileJson {
            outlet_x: p.outlet_coord.0,
            outlet_y: p.outlet_coord.1,
            nodes: p
                .nodes
                .iter()
                .enumerate()
                .map(|(ni, n)| LongProfileRow {
                    profile_id: pi,
                    node_index: ni,
                    x: n.coord.0,
                    y: n.coord.1,
                    distance_m: n.distance_from_outlet_m,
                    elevation: n.elevation,
                    area_m2: n.area_m2,
                    chi: n.chi,
                    ksn: n.ksn,
                })
                .collect(),
            _phantom: std::marker::PhantomData,
        })
        .collect();
    serde_json::to_writer_pretty(w, &docs)?;
    Ok(())
}

// ─── Swath profile ───────────────────────────────────────────────────────

#[derive(Serialize)]
struct SwathRow {
    bin_index: usize,
    baseline_x: f64,
    baseline_y: f64,
    distance_along_m: f64,
    min: f64,
    max: f64,
    mean: f64,
    median: f64,
    p25: f64,
    p75: f64,
    n_samples: usize,
}

/// Write a swath profile as CSV, one row per bin, to the given writer.
pub fn write_swath_csv<W: Write>(swath: &SwathProfile, w: W) -> Result<(), ExportError> {
    let mut wtr = csv::Writer::from_writer(w);
    for (i, b) in swath.bins.iter().enumerate() {
        let coord = swath
            .densified_baseline
            .get(i)
            .copied()
            .unwrap_or((f64::NAN, f64::NAN));
        wtr.serialize(SwathRow {
            bin_index: i,
            baseline_x: coord.0,
            baseline_y: coord.1,
            distance_along_m: b.distance_along_m,
            min: b.min,
            max: b.max,
            mean: b.mean,
            median: b.median,
            p25: b.p25,
            p75: b.p75,
            n_samples: b.n_samples,
        })?;
    }
    wtr.flush()?;
    Ok(())
}

/// Write a swath profile as a JSON array of per-bin records to the given writer.
pub fn write_swath_json<W: Write>(swath: &SwathProfile, w: W) -> Result<(), ExportError> {
    let rows: Vec<SwathRow> = swath
        .bins
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let coord = swath
                .densified_baseline
                .get(i)
                .copied()
                .unwrap_or((f64::NAN, f64::NAN));
            SwathRow {
                bin_index: i,
                baseline_x: coord.0,
                baseline_y: coord.1,
                distance_along_m: b.distance_along_m,
                min: b.min,
                max: b.max,
                mean: b.mean,
                median: b.median,
                p25: b.p25,
                p75: b.p75,
                n_samples: b.n_samples,
            }
        })
        .collect();
    serde_json::to_writer_pretty(w, &rows)?;
    Ok(())
}

// ─── Ksn segments ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct KsnSegmentRow {
    segment_id: usize,
    n_cells: usize,
    ksn_mean: f64,
    ksn_ci_low: f64,
    ksn_ci_high: f64,
    /// WKT-style 2D linestring of cell-centre coordinates,
    /// downstream → upstream. Convenient for ad-hoc CSV consumers
    /// (e.g. one-liner ogr2ogr import via `csv:WKT`).
    geometry_wkt: String,
}

fn segment_wkt(seg: &KsnSegment) -> String {
    let mut s = String::from("LINESTRING(");
    for (i, (x, y)) in seg.coordinates.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{x} {y}"));
    }
    s.push(')');
    s
}

/// CSV: one row per segment. Coordinates encoded as a WKT linestring
/// in `geometry_wkt` so the file remains compact and consumable by
/// ogr2ogr / GeoPandas via a single column.
pub fn write_ksn_segments_csv<W: Write>(result: &KsnResult, w: W) -> Result<(), ExportError> {
    let mut wtr = csv::Writer::from_writer(w);
    let Some(segments) = result.segments.as_ref() else {
        wtr.flush()?;
        return Ok(());
    };
    for (i, s) in segments.iter().enumerate() {
        wtr.serialize(KsnSegmentRow {
            segment_id: i,
            n_cells: s.n_cells,
            ksn_mean: s.ksn_mean,
            ksn_ci_low: s.ksn_ci.0,
            ksn_ci_high: s.ksn_ci.1,
            geometry_wkt: segment_wkt(s),
        })?;
    }
    wtr.flush()?;
    Ok(())
}

#[derive(Serialize)]
struct KsnSegmentJson<'a> {
    segment_id: usize,
    n_cells: usize,
    ksn_mean: f64,
    ksn_ci: (f64, f64),
    coordinates: &'a [(f64, f64)],
}

/// JSON: hierarchical — coordinates kept as a nested array of
/// `[x, y]` pairs.
pub fn write_ksn_segments_json<W: Write>(result: &KsnResult, w: W) -> Result<(), ExportError> {
    let docs: Vec<KsnSegmentJson> = result
        .segments
        .as_deref()
        .map(|segs| {
            segs.iter()
                .enumerate()
                .map(|(i, s)| KsnSegmentJson {
                    segment_id: i,
                    n_cells: s.n_cells,
                    ksn_mean: s.ksn_mean,
                    ksn_ci: s.ksn_ci,
                    coordinates: &s.coordinates,
                })
                .collect()
        })
        .unwrap_or_default();
    serde_json::to_writer_pretty(w, &docs)?;
    Ok(())
}

// ─── Chi raster ──────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChiRow {
    row: usize,
    col: usize,
    x: f64,
    y: f64,
    chi: f64,
}

/// CSV: one row per *finite* χ cell. Skips NaN so the table size
/// reflects the size of the channel network, not the whole raster.
/// Coordinates are emitted in raster CRS via the raster's transform.
pub fn write_chi_csv<W: Write>(chi: &Raster<f64>, w: W) -> Result<(), ExportError> {
    let mut wtr = csv::Writer::from_writer(w);
    let (rows, cols) = chi.shape();
    for r in 0..rows {
        for c in 0..cols {
            let v = chi.get(r, c).unwrap_or(f64::NAN);
            if !v.is_finite() {
                continue;
            }
            // `pixel_to_geo` already returns the pixel centre
            // (col + 0.5, row + 0.5 internally), so no further offset
            // is applied here.
            let (x, y) = chi.pixel_to_geo(c, r);
            wtr.serialize(ChiRow {
                row: r,
                col: c,
                x,
                y,
                chi: v,
            })?;
        }
    }
    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluvial::{LongProfile, LongProfileNode};

    #[test]
    fn long_profile_csv_has_one_row_per_node_with_grouping_column() {
        let p1 = LongProfile {
            outlet_coord: (10.0, 20.0),
            nodes: vec![
                LongProfileNode {
                    coord: (10.0, 20.0),
                    distance_from_outlet_m: 0.0,
                    elevation: 100.0,
                    area_m2: 1e6,
                    chi: Some(0.0),
                    ksn: Some(50.0),
                },
                LongProfileNode {
                    coord: (40.0, 20.0),
                    distance_from_outlet_m: 30.0,
                    elevation: 110.0,
                    area_m2: 8e5,
                    chi: Some(0.5),
                    ksn: None,
                },
            ],
        };
        let p2 = LongProfile {
            outlet_coord: (200.0, 300.0),
            nodes: vec![LongProfileNode {
                coord: (200.0, 300.0),
                distance_from_outlet_m: 0.0,
                elevation: 50.0,
                area_m2: 1e7,
                chi: None,
                ksn: None,
            }],
        };
        let mut buf: Vec<u8> = Vec::new();
        write_long_profiles_csv(&[p1, p2], &mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        // One header row + 3 data rows.
        assert_eq!(text.lines().count(), 4);
        assert!(text.contains("profile_id,node_index"));
        // Empty cells for None values.
        assert!(text.contains(",,")); // chi None or ksn None produces ',,'
        // The second profile should be labelled profile_id=1.
        assert!(text.contains("1,0,200"));
    }

    #[test]
    fn long_profile_json_roundtrips_to_serde_value() {
        let p = LongProfile {
            outlet_coord: (10.0, 20.0),
            nodes: vec![LongProfileNode {
                coord: (10.0, 20.0),
                distance_from_outlet_m: 0.0,
                elevation: 100.0,
                area_m2: 1e6,
                chi: Some(0.0),
                ksn: None,
            }],
        };
        let mut buf: Vec<u8> = Vec::new();
        write_long_profiles_json(&[p], &mut buf).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(v[0]["outlet_x"], 10.0);
        assert_eq!(v[0]["nodes"][0]["chi"], 0.0);
        assert!(v[0]["nodes"][0]["ksn"].is_null());
    }

    /// Regression test for the ½-pixel offset bug (CR-1): the exported
    /// χ CSV coordinate must equal `pixel_to_geo` exactly, with no
    /// extra half-cell added on top. `pixel_to_geo` already returns the
    /// pixel *centre* (col + 0.5, row + 0.5 internally).
    #[test]
    fn chi_csv_coord_matches_pixel_to_geo_exactly_no_extra_half_pixel() {
        use ndarray::Array2;
        use surtgis_core::GeoTransform;

        let arr = Array2::from_shape_vec((1, 1), vec![0.5_f64]).unwrap();
        let mut chi = Raster::from_array(arr);
        chi.set_transform(GeoTransform::new(0.0, 0.0, 10.0, -10.0));

        let mut buf: Vec<u8> = Vec::new();
        write_chi_csv(&chi, &mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();

        // Known coordinate: with origin (0,0) and pixel_size 10,
        // pixel_to_geo(0, 0) must be (5.0, -5.0), not (10.0, -10.0).
        let expected = chi.pixel_to_geo(0, 0);
        assert_eq!(expected, (5.0, -5.0));

        let data_line = text.lines().nth(1).expect("one data row");
        // CSV columns, in ChiRow field order: row,col,x,y,chi.
        let cols: Vec<&str> = data_line.split(',').collect();
        let x: f64 = cols[2].parse().unwrap();
        let y: f64 = cols[3].parse().unwrap();
        assert_eq!(
            (x, y),
            expected,
            "chi CSV coordinate must match pixel_to_geo exactly"
        );
    }
}
