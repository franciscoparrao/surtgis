//! Handler for extracting raster patches around points or within polygons.
//!
//! Produces a directory containing:
//!   - `patches.npy` : tensor [N, bands, H, W], dtype f32
//!   - `labels.npy`  : tensor [N], dtype i64 (class) or f32 (regression)
//!   - `manifest.csv`: one row per patch with label, center, source info
//!   - `meta.json`   : bands order, CRS, pixel size, patch size, skipped counts, seed
//!
//! Design notes:
//!   - NPY writing is hand-rolled (no new crate dep). Streaming: header written
//!     up front with N known, then patch rows appended sequentially.
//!   - Subsampling uses a seeded DefaultHasher — deterministic, no `rand` dep.
//!   - Polygons are sampled on a grid of stride `--stride` pixels (default =
//!     patch size, i.e. non-overlapping tiles); each candidate patch CENTER is
//!     tested for point-in-polygon. Holes inside polygons are NOT honoured in
//!     v1; patch centers in holes will still be accepted.

use anyhow::{Context, Result};
use geo::{Contains, BoundingRect};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use surtgis_core::vector::AttributeValue;

/// A patch center in pixel coordinates, with its label and origin metadata.
#[derive(Clone, Debug)]
struct PatchSpec {
    center_row: usize,
    center_col: usize,
    /// Label stored as i64 for classes or bit-reinterpreted f64 for regression.
    /// Kind is disambiguated by `LabelKind`.
    label_raw: LabelValue,
    /// Source feature index (point or polygon number in the input vector)
    source_idx: usize,
}

#[derive(Clone, Copy, Debug)]
enum LabelValue {
    Int(i64),
    Float(f64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum LabelKind {
    /// Integer labels → classification. NPY dtype = "<i8" (i64 little-endian).
    Int,
    /// Float labels → regression. NPY dtype = "<f4" (f32 little-endian).
    Float,
}

/// Recursively find all .tif/.tiff files under a directory.
fn find_tifs(dir: &Path) -> Vec<PathBuf> {
    let mut tifs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                tifs.extend(find_tifs(&path));
            } else if let Some(ext) = path.extension() {
                if ext.eq_ignore_ascii_case("tif") || ext.eq_ignore_ascii_case("tiff") {
                    tifs.push(path);
                }
            }
        }
    }
    tifs.sort();
    tifs
}

/// Load all feature rasters from a directory. Honours features.json if present
/// for explicit naming/order, then auto-discovers any unregistered .tif files.
fn load_feature_rasters(features_dir: &Path) -> Result<(Vec<String>, Vec<surtgis_core::Raster<f64>>)> {
    let mut feature_names: Vec<String> = Vec::new();
    let mut rasters: Vec<surtgis_core::Raster<f64>> = Vec::new();
    let mut loaded_paths: HashSet<PathBuf> = HashSet::new();

    let features_json_path = features_dir.join("features.json");
    if features_json_path.exists() {
        let s = std::fs::read_to_string(&features_json_path)
            .with_context(|| format!("Failed to read {}", features_json_path.display()))?;
        let meta: serde_json::Value = serde_json::from_str(&s)
            .context("Failed to parse features.json")?;
        if let Some(entries) = meta["features"].as_array() {
            for entry in entries {
                let name = entry["name"].as_str().context("Feature entry missing 'name'")?;
                let file = entry["file"].as_str().context("Feature entry missing 'file'")?;
                let p = features_dir.join(file);
                if !p.exists() {
                    eprintln!("  WARNING: skipping missing raster: {}", p.display());
                    continue;
                }
                let canonical = p.canonicalize().unwrap_or_else(|_| p.clone());
                let r = surtgis_core::io::read_geotiff::<f64, _>(&p, None)
                    .with_context(|| format!("Failed to read raster: {}", p.display()))?;
                feature_names.push(name.to_string());
                rasters.push(r);
                loaded_paths.insert(canonical);
            }
        }
    }

    for tif in find_tifs(features_dir) {
        let canonical = tif.canonicalize().unwrap_or_else(|_| tif.clone());
        if loaded_paths.contains(&canonical) { continue; }
        let rel = tif.strip_prefix(features_dir).unwrap_or(&tif);
        let name = rel.with_extension("").to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        match surtgis_core::io::read_geotiff::<f64, _>(&tif, None) {
            Ok(r) => {
                feature_names.push(name);
                rasters.push(r);
                loaded_paths.insert(canonical);
            }
            Err(e) => eprintln!("  WARNING: skipping {}: {}", tif.display(), e),
        }
    }

    if rasters.is_empty() {
        anyhow::bail!("No feature rasters found in {}", features_dir.display());
    }
    Ok((feature_names, rasters))
}

/// Verify all rasters share the same grid (rows × cols + transform). Returns
/// an error pointing at the first mismatch so the user can fix alignment.
fn validate_grid_alignment(rasters: &[surtgis_core::Raster<f64>]) -> Result<()> {
    if rasters.is_empty() { return Ok(()); }
    let (r0, c0) = rasters[0].shape();
    let gt0 = rasters[0].transform();
    for (i, r) in rasters.iter().enumerate().skip(1) {
        let (ri, ci) = r.shape();
        let gti = r.transform();
        if ri != r0 || ci != c0 {
            anyhow::bail!("Raster shape mismatch: raster 0 is {}x{}, raster {} is {}x{}",
                c0, r0, i, ci, ri);
        }
        let tol = 1e-6;
        if (gti.origin_x - gt0.origin_x).abs() > tol
            || (gti.origin_y - gt0.origin_y).abs() > tol
            || (gti.pixel_width - gt0.pixel_width).abs() > tol
            || (gti.pixel_height - gt0.pixel_height).abs() > tol
        {
            anyhow::bail!("Raster transform mismatch at raster {}. All rasters must share the same grid.", i);
        }
    }
    Ok(())
}

/// Extract a label value from a feature attribute. Returns None if the
/// attribute is missing, null, a bool, or a non-numeric string.
fn extract_label(feat: &surtgis_core::vector::Feature, col: &str) -> Option<LabelValue> {
    match feat.get_property(col)? {
        AttributeValue::Int(v) => Some(LabelValue::Int(*v)),
        AttributeValue::Float(v) => Some(LabelValue::Float(*v)),
        AttributeValue::Bool(v) => Some(LabelValue::Int(if *v { 1 } else { 0 })),
        AttributeValue::String(s) => {
            // Try parse as int, then float; otherwise fail
            s.parse::<i64>().ok().map(LabelValue::Int)
                .or_else(|| s.parse::<f64>().ok().map(LabelValue::Float))
        }
        AttributeValue::Null => None,
    }
}

/// Decide label storage kind from the set of labels observed. Any Float pushes
/// the whole dataset to f32; otherwise i64.
fn decide_label_kind(labels: &[LabelValue]) -> LabelKind {
    if labels.iter().any(|l| matches!(l, LabelValue::Float(_))) {
        LabelKind::Float
    } else {
        LabelKind::Int
    }
}

/// Hash-based deterministic subsample: keep the first `cap` patches after
/// sorting by hash(seed, spec.source_idx, spec.center_row, spec.center_col).
/// Equivalent in distribution to a seeded random subsample, no rand dep.
fn subsample_deterministic(specs: Vec<PatchSpec>, cap: usize, seed: u64) -> Vec<PatchSpec> {
    if specs.len() <= cap { return specs; }
    let mut keyed: Vec<(u64, PatchSpec)> = specs.into_iter().map(|s| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut h);
        s.source_idx.hash(&mut h);
        s.center_row.hash(&mut h);
        s.center_col.hash(&mut h);
        (h.finish(), s)
    }).collect();
    keyed.sort_unstable_by_key(|(k, _)| *k);
    keyed.into_iter().take(cap).map(|(_, s)| s).collect()
}

/// Write a NumPy .npy v1.0 header for the given shape and dtype. Hand-rolled
/// so we don't pull in a crate just for this — the format is stable and tiny.
fn write_npy_header(file: &mut File, shape: &[usize], dtype: &str) -> Result<()> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype, shape_str,
    );
    // Header must be padded so (10 + header_len) is a multiple of 64.
    // 10 = 6 magic + 2 version + 2 header_len (u16 LE for v1.0)
    let base_len = 10 + dict.len() + 1; // +1 for trailing \n
    let pad = (64 - (base_len % 64)) % 64;
    let padded = format!("{}{}\n", dict, " ".repeat(pad));
    let header_len = padded.len() as u16;

    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1u8, 0u8])?;                    // version 1.0
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(padded.as_bytes())?;
    Ok(())
}

/// Main entry point.
#[allow(clippy::too_many_arguments)]
pub fn handle(
    features_dir: &Path,
    points: Option<&Path>,
    polygons: Option<&Path>,
    label_col: &str,
    size: usize,
    stride: Option<usize>,
    skip_nan_threshold: f64,
    max_patches: Option<usize>,
    seed: u64,
    output: &Path,
) -> Result<()> {
    let start = Instant::now();

    if points.is_none() && polygons.is_none() {
        anyhow::bail!("Either --points or --polygons must be provided");
    }
    if size == 0 { anyhow::bail!("--size must be > 0"); }

    println!("SurtGIS Extract Patches");
    println!("=========================================");
    println!("  Features dir:  {}", features_dir.display());
    if let Some(p) = points { println!("  Points:        {}", p.display()); }
    if let Some(p) = polygons { println!("  Polygons:      {}", p.display()); }
    println!("  Label column:  {}", label_col);
    println!("  Patch size:    {}x{}", size, size);
    println!("  Output dir:    {}", output.display());
    println!();

    // --- 1. Load + validate rasters ---
    let (feature_names, rasters) = load_feature_rasters(features_dir)?;
    validate_grid_alignment(&rasters)?;
    let (rows, cols) = rasters[0].shape();
    let gt = *rasters[0].transform();
    let crs_epsg = rasters[0].crs().and_then(|c| c.epsg());
    println!("Loaded {} rasters ({}x{} grid, pixel {:.3}x{:.3})",
        rasters.len(), cols, rows, gt.pixel_width, gt.pixel_height);

    // --- 2. Read vector + build patch specs ---
    let half = size / 2;
    let stride = stride.unwrap_or(size).max(1);
    let mut specs: Vec<PatchSpec> = Vec::new();

    if let Some(points_path) = points {
        let fc = surtgis_core::vector::read_vector(points_path)
            .context("Failed to read points")?;
        println!("Points file has {} features", fc.len());
        for (idx, feat) in fc.iter().enumerate() {
            let Some(geo::Geometry::Point(p)) = feat.geometry.as_ref() else { continue };
            let label = match extract_label(feat, label_col) {
                Some(l) => l,
                None => continue,
            };
            let (col_f, row_f) = rasters[0].geo_to_pixel(p.x(), p.y());
            let col = col_f.floor() as isize;
            let row = row_f.floor() as isize;
            if row < half as isize || col < half as isize { continue; }
            if (row as usize + (size - half)) > rows || (col as usize + (size - half)) > cols {
                continue;
            }
            specs.push(PatchSpec {
                center_row: row as usize,
                center_col: col as usize,
                label_raw: label,
                source_idx: idx,
            });
        }
    } else if let Some(polygons_path) = polygons {
        let fc = surtgis_core::vector::read_vector(polygons_path)
            .context("Failed to read polygons")?;
        println!("Polygons file has {} features, grid stride = {}px", fc.len(), stride);
        for (idx, feat) in fc.iter().enumerate() {
            let label = match extract_label(feat, label_col) {
                Some(l) => l,
                None => continue,
            };
            let Some(geom) = feat.geometry.as_ref() else { continue };
            // Flatten MultiPolygon and Polygon into a list of Polygon refs
            let polys: Vec<geo::Polygon<f64>> = match geom {
                geo::Geometry::Polygon(p) => vec![p.clone()],
                geo::Geometry::MultiPolygon(mp) => mp.0.clone(),
                _ => continue,
            };
            for poly in &polys {
                let Some(bb) = poly.bounding_rect() else { continue };
                // Bbox in pixel space
                let (cx0, ry0) = rasters[0].geo_to_pixel(bb.min().x, bb.max().y);
                let (cx1, ry1) = rasters[0].geo_to_pixel(bb.max().x, bb.min().y);
                let row_min = (ry0.floor() as isize).max(half as isize) as usize;
                let row_max = (ry1.ceil() as isize).min((rows - (size - half)) as isize).max(0) as usize;
                let col_min = (cx0.floor() as isize).max(half as isize) as usize;
                let col_max = (cx1.ceil() as isize).min((cols - (size - half)) as isize).max(0) as usize;
                let mut r = row_min;
                while r <= row_max {
                    let mut c = col_min;
                    while c <= col_max {
                        // Convert (c, r) pixel origin to geo, then add a half-pixel offset
                        // to land at the cell centre for the point-in-polygon test.
                        let (x0, y0) = rasters[0].pixel_to_geo(c, r);
                        let x = x0 + 0.5 * gt.pixel_width;
                        let y = y0 + 0.5 * gt.pixel_height;
                        let pt = geo::Point::new(x, y);
                        if poly.contains(&pt) {
                            specs.push(PatchSpec {
                                center_row: r,
                                center_col: c,
                                label_raw: label,
                                source_idx: idx,
                            });
                        }
                        c += stride;
                    }
                    r += stride;
                }
            }
        }
    }

    let total_candidates = specs.len();
    if total_candidates == 0 {
        anyhow::bail!("No patch candidates produced — check that the vector has the expected geometry type and that the label column exists");
    }
    println!("Candidate patches before NaN/subsample: {}", total_candidates);

    // Optional subsample (deterministic).
    if let Some(cap) = max_patches {
        specs = subsample_deterministic(specs, cap, seed);
        println!("After --max-patches={} subsample: {}", cap, specs.len());
    }

    // --- 3. Decide label dtype ---
    let labels: Vec<LabelValue> = specs.iter().map(|s| s.label_raw).collect();
    let label_kind = decide_label_kind(&labels);
    let (label_dtype, label_bytes_each) = match label_kind {
        LabelKind::Int => ("<i8", 8usize),
        LabelKind::Float => ("<f4", 4usize),
    };

    // --- 4. Create output dir ---
    std::fs::create_dir_all(output)
        .with_context(|| format!("Failed to create output dir: {}", output.display()))?;

    // --- 5. First pass: check NaN threshold per candidate, count valid ---
    //     We need N up front to write the NPY header. Do the extraction + NaN
    //     check now and stash valid patches' data as Vec<f32>. Each kept patch
    //     holds bands * size * size * 4 bytes — for size 256, 10 bands, 10K
    //     patches = 26 GB. For v1 we warn and rely on --max-patches.
    let n_bands = rasters.len();
    let patch_pixels = size * size;
    let patch_bytes = n_bands * patch_pixels * 4; // f32
    let est_total_bytes = specs.len() * patch_bytes;
    eprintln!("Patch tensor estimate: {} patches × {} bands × {}² × 4 bytes = {:.2} GB",
        specs.len(), n_bands, size, est_total_bytes as f64 / 1e9);

    let mut kept: Vec<(PatchSpec, Vec<f32>)> = Vec::new();
    let mut nan_skipped = 0usize;

    for spec in &specs {
        let r0 = spec.center_row - half;
        let c0 = spec.center_col - half;
        let mut buf = vec![0f32; n_bands * patch_pixels];
        let mut nan_count = 0usize;
        for (bi, raster) in rasters.iter().enumerate() {
            let band_offset = bi * patch_pixels;
            for dr in 0..size {
                let row_offset = band_offset + dr * size;
                for dc in 0..size {
                    let v = raster.get(r0 + dr, c0 + dc).unwrap_or(f64::NAN);
                    if v.is_finite() {
                        buf[row_offset + dc] = v as f32;
                    } else {
                        buf[row_offset + dc] = f32::NAN;
                        nan_count += 1;
                    }
                }
            }
        }
        let nan_frac = nan_count as f64 / (n_bands * patch_pixels) as f64;
        if nan_frac > skip_nan_threshold {
            nan_skipped += 1;
            continue;
        }
        kept.push((spec.clone(), buf));
    }

    let n = kept.len();
    println!("After NaN threshold ({:.0}%): {} kept, {} skipped",
        skip_nan_threshold * 100.0, n, nan_skipped);
    if n == 0 {
        anyhow::bail!("All candidate patches were filtered by NaN threshold");
    }

    // --- 6. Write patches.npy ---
    let patches_path = output.join("patches.npy");
    let mut f_patches = OpenOptions::new().create(true).write(true).truncate(true)
        .open(&patches_path)
        .with_context(|| format!("Failed to open {}", patches_path.display()))?;
    write_npy_header(&mut f_patches, &[n, n_bands, size, size], "<f4")?;
    for (_, buf) in &kept {
        let bytes: &[u8] = bytemuck_cast_f32_to_bytes(buf);
        f_patches.write_all(bytes).context("Failed to write patch bytes")?;
    }
    f_patches.flush().ok();

    // --- 7. Write labels.npy ---
    let labels_path = output.join("labels.npy");
    let mut f_labels = OpenOptions::new().create(true).write(true).truncate(true)
        .open(&labels_path)
        .with_context(|| format!("Failed to open {}", labels_path.display()))?;
    write_npy_header(&mut f_labels, &[n], label_dtype)?;
    match label_kind {
        LabelKind::Int => {
            for (spec, _) in &kept {
                let v = match spec.label_raw {
                    LabelValue::Int(x) => x,
                    LabelValue::Float(x) => x as i64, // not reachable if kind is Int
                };
                f_labels.write_all(&v.to_le_bytes())?;
            }
        }
        LabelKind::Float => {
            for (spec, _) in &kept {
                let v = match spec.label_raw {
                    LabelValue::Int(x) => x as f32,
                    LabelValue::Float(x) => x as f32,
                };
                f_labels.write_all(&v.to_le_bytes())?;
            }
        }
    }
    f_labels.flush().ok();
    let _ = label_bytes_each; // kept for potential pre-alloc sanity

    // --- 8. Write manifest.csv ---
    let manifest_path = output.join("manifest.csv");
    let mut csv_w = csv::Writer::from_path(&manifest_path)
        .with_context(|| format!("Failed to create {}", manifest_path.display()))?;
    csv_w.write_record(&["idx", "label", "center_row", "center_col", "center_x", "center_y", "source_idx"])?;
    for (i, (spec, _)) in kept.iter().enumerate() {
        let (x0, y0) = rasters[0].pixel_to_geo(spec.center_col, spec.center_row);
        let x = x0 + 0.5 * gt.pixel_width;
        let y = y0 + 0.5 * gt.pixel_height;
        let label_str = match spec.label_raw {
            LabelValue::Int(v) => v.to_string(),
            LabelValue::Float(v) => format!("{}", v),
        };
        csv_w.write_record(&[
            i.to_string(), label_str,
            spec.center_row.to_string(), spec.center_col.to_string(),
            format!("{}", x), format!("{}", y),
            spec.source_idx.to_string(),
        ])?;
    }
    csv_w.flush().ok();

    // --- 9. Write meta.json ---
    let meta = serde_json::json!({
        "bands": feature_names,
        "patch_size": size,
        "n_patches": n,
        "label_dtype": label_dtype,
        "label_kind": match label_kind { LabelKind::Int => "int", LabelKind::Float => "float" },
        "crs_epsg": crs_epsg,
        "pixel_width": gt.pixel_width,
        "pixel_height": gt.pixel_height,
        "grid_rows": rows,
        "grid_cols": cols,
        "candidates_before_filter": total_candidates,
        "nan_skipped": nan_skipped,
        "nan_threshold": skip_nan_threshold,
        "seed": seed,
        "max_patches": max_patches,
        "source_mode": if points.is_some() { "points" } else { "polygons" },
    });
    std::fs::write(output.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;

    // --- 10. Summary ---
    println!();
    println!("=========================================");
    println!("PATCH EXTRACTION COMPLETE");
    println!("=========================================");
    println!("  Patches:   {}", n);
    println!("  Bands:     {} ({})", n_bands, feature_names.join(", "));
    println!("  Shape:     [{}, {}, {}, {}] ({})", n, n_bands, size, size, "<f4");
    println!("  Labels:    {} ({})", n, label_dtype);
    println!("  Output:    {}/", output.display());
    println!("  Time:      {:.1}s", start.elapsed().as_secs_f64());
    println!();
    println!("Load in Python:");
    println!("  import numpy as np");
    println!("  X = np.load('{}/patches.npy')  # [N, bands, H, W] f32", output.display());
    println!("  y = np.load('{}/labels.npy')   # [N] {}", output.display(),
        if label_kind == LabelKind::Int { "i64" } else { "f32" });

    Ok(())
}

/// Reinterpret a &[f32] as &[u8] without copying. Equivalent to bytemuck::cast_slice
/// but written inline to avoid the extra dep. Safety: f32 has no niche values and
/// any bit pattern is a valid f32; reading its bytes is always well-defined.
fn bytemuck_cast_f32_to_bytes(s: &[f32]) -> &[u8] {
    // SAFETY: f32 is `Pod`-equivalent; 4-byte aligned reads as u8 are fine.
    unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s))
    }
}

#[allow(dead_code)]
fn _hashmap_stub() -> HashMap<String, i64> { HashMap::new() } // avoid unused-import lint for HashMap
