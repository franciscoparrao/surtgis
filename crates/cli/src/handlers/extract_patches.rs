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

use super::gfm_profiles::{apply_band_norm_block, GfmProfile};
use super::stac_writer::{write_stac_output, ChipInfo, CollectionInfo};
use super::zarr_writer::{init_zarr_v2_array, write_chunk};
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

/// Non-recursive: only .tif/.tiff files at the top level of `dir`.
fn list_tifs_shallow(dir: &Path) -> Vec<PathBuf> {
    let mut tifs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() { continue; }
            if let Some(ext) = path.extension() {
                if ext.eq_ignore_ascii_case("tif") || ext.eq_ignore_ascii_case("tiff") {
                    tifs.push(path);
                }
            }
        }
    }
    tifs.sort();
    tifs
}

/// List immediate subdirectories of `dir`, sorted lexicographically.
fn list_subdirs(dir: &Path) -> Vec<PathBuf> {
    let mut subs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() { subs.push(path); }
        }
    }
    subs.sort();
    subs
}

/// Result of loading rasters from the features directory.
///
/// The outer index is the timestamp (length T ≥ 1); the inner index is
/// the band (length C, identical across timestamps). `timestamps[t]` is
/// the human-readable name of the t-th timestamp (e.g. `"t0"` for the
/// non-temporal case, or `"2024-01-15"` for explicit dates).
struct RasterSet {
    timestamps: Vec<String>,
    band_names: Vec<String>,
    rasters: Vec<Vec<surtgis_core::Raster<f64>>>,
}

impl RasterSet {
    fn n_timestamps(&self) -> usize { self.timestamps.len() }
    fn n_bands(&self) -> usize { self.band_names.len() }
}

/// Load one timestamp worth of feature rasters from `dir`. Honours an
/// optional `features.json` for explicit naming/order, then auto-discovers
/// any unregistered top-level .tif files. Subdirectories are ignored here —
/// callers that operate in multi-timestamp mode iterate them explicitly.
fn load_single_timestamp(dir: &Path) -> Result<(Vec<String>, Vec<surtgis_core::Raster<f64>>)> {
    let mut feature_names: Vec<String> = Vec::new();
    let mut rasters: Vec<surtgis_core::Raster<f64>> = Vec::new();
    let mut loaded_paths: HashSet<PathBuf> = HashSet::new();

    let features_json_path = dir.join("features.json");
    if features_json_path.exists() {
        let s = std::fs::read_to_string(&features_json_path)
            .with_context(|| format!("Failed to read {}", features_json_path.display()))?;
        let meta: serde_json::Value = serde_json::from_str(&s)
            .context("Failed to parse features.json")?;
        if let Some(entries) = meta["features"].as_array() {
            for entry in entries {
                let name = entry["name"].as_str().context("Feature entry missing 'name'")?;
                let file = entry["file"].as_str().context("Feature entry missing 'file'")?;
                let p = dir.join(file);
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

    for tif in list_tifs_shallow(dir) {
        let canonical = tif.canonicalize().unwrap_or_else(|_| tif.clone());
        if loaded_paths.contains(&canonical) { continue; }
        let name = tif.file_stem().map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string());
        match surtgis_core::io::read_geotiff::<f64, _>(&tif, None) {
            Ok(r) => {
                feature_names.push(name);
                rasters.push(r);
                loaded_paths.insert(canonical);
            }
            Err(e) => eprintln!("  WARNING: skipping {}: {}", tif.display(), e),
        }
    }

    Ok((feature_names, rasters))
}

/// Discover whether `features_dir` is single- or multi-timestamp and load
/// the corresponding raster set.
///
/// Detection: if `features_dir` has any top-level .tif files, treat as
/// single-timestamp. Else, if it has subdirectories each containing .tifs,
/// treat as multi-timestamp with one timestamp per subdir (sorted lex).
/// Mixed mode (both top-level .tifs AND subdirs with .tifs) errors.
fn load_raster_set(features_dir: &Path) -> Result<RasterSet> {
    let top_tifs = list_tifs_shallow(features_dir);
    let subdirs = list_subdirs(features_dir);
    let subdirs_with_tifs: Vec<PathBuf> = subdirs.into_iter()
        .filter(|d| !list_tifs_shallow(d).is_empty())
        .collect();

    if !top_tifs.is_empty() && !subdirs_with_tifs.is_empty() {
        anyhow::bail!(
            "Mixed mode in {}: found both top-level .tif files and subdirectories \
             containing .tifs. For multi-timestamp input, move all top-level .tifs \
             into a subdirectory.", features_dir.display(),
        );
    }

    if !top_tifs.is_empty() {
        let (names, rs) = load_single_timestamp(features_dir)?;
        if rs.is_empty() {
            anyhow::bail!("No feature rasters found in {}", features_dir.display());
        }
        return Ok(RasterSet {
            timestamps: vec!["t0".to_string()],
            band_names: names,
            rasters: vec![rs],
        });
    }

    if subdirs_with_tifs.is_empty() {
        anyhow::bail!("No feature rasters found in {}", features_dir.display());
    }

    // Multi-timestamp mode. Use the first subdir as the canonical band order.
    let mut timestamps: Vec<String> = Vec::with_capacity(subdirs_with_tifs.len());
    let mut all_rasters: Vec<Vec<surtgis_core::Raster<f64>>> = Vec::with_capacity(subdirs_with_tifs.len());
    let mut canonical_names: Vec<String> = Vec::new();

    for (ti, sub) in subdirs_with_tifs.iter().enumerate() {
        let ts_name = sub.file_name().map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("t{}", ti));
        let (names, rs) = load_single_timestamp(sub)
            .with_context(|| format!("Loading timestamp '{}'", ts_name))?;
        if rs.is_empty() {
            anyhow::bail!("Timestamp '{}' contains no rasters", ts_name);
        }
        if ti == 0 {
            canonical_names = names.clone();
        } else if names != canonical_names {
            anyhow::bail!(
                "Band-name mismatch at timestamp '{}'. Expected {:?}, got {:?}. \
                 All timestamps must declare the same bands in the same order.",
                ts_name, canonical_names, names,
            );
        }
        timestamps.push(ts_name);
        all_rasters.push(rs);
    }

    Ok(RasterSet {
        timestamps,
        band_names: canonical_names,
        rasters: all_rasters,
    })
}

/// Verify all rasters across all timestamps share the same grid (rows × cols
/// + transform). Returns an error pointing at the first mismatch so the user
/// can fix alignment.
fn validate_raster_set_alignment(set: &RasterSet) -> Result<()> {
    if set.rasters.is_empty() || set.rasters[0].is_empty() { return Ok(()); }
    let (r0, c0) = set.rasters[0][0].shape();
    let gt0 = *set.rasters[0][0].transform();
    let tol = 1e-6;
    for (ti, ts_rasters) in set.rasters.iter().enumerate() {
        for (bi, r) in ts_rasters.iter().enumerate() {
            let (ri, ci) = r.shape();
            let gti = r.transform();
            if ri != r0 || ci != c0 {
                anyhow::bail!(
                    "Raster shape mismatch at timestamp[{}] band[{}]: expected {}x{}, got {}x{}",
                    ti, bi, c0, r0, ci, ri,
                );
            }
            if (gti.origin_x - gt0.origin_x).abs() > tol
                || (gti.origin_y - gt0.origin_y).abs() > tol
                || (gti.pixel_width - gt0.pixel_width).abs() > tol
                || (gti.pixel_height - gt0.pixel_height).abs() > tol
            {
                anyhow::bail!(
                    "Raster transform mismatch at timestamp[{}] band[{}]. \
                     All rasters across all timestamps must share the same grid.",
                    ti, bi,
                );
            }
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

/// Output tensor format for the patches volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PatchOutputFormat {
    Npy,
    Zarr,
}

impl PatchOutputFormat {
    fn from_name(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "npy" => Ok(Self::Npy),
            "zarr" | "zarr2" | "zarr-v2" => Ok(Self::Zarr),
            other => anyhow::bail!("Unknown --output-format '{}'. Supported: npy, zarr.", other),
        }
    }
    fn label(&self) -> &'static str {
        match self { Self::Npy => "npy", Self::Zarr => "zarr" }
    }
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
    profile: Option<&str>,
    output_format: &str,
    emit_stac: bool,
    output: &Path,
) -> Result<()> {
    let start = Instant::now();

    if points.is_none() && polygons.is_none() {
        anyhow::bail!("Either --points or --polygons must be provided");
    }
    if size == 0 { anyhow::bail!("--size must be > 0"); }

    let out_fmt = PatchOutputFormat::from_name(output_format)?;

    // Resolve GFM profile up-front so we can echo it in the banner and use
    // its tile_size as the default when --size was left at its CLI default.
    let profile_spec = match profile {
        Some(name) => Some(GfmProfile::from_name(name)?.spec()),
        None => None,
    };

    println!("SurtGIS Extract Patches");
    println!("=========================================");
    println!("  Features dir:  {}", features_dir.display());
    if let Some(p) = points { println!("  Points:        {}", p.display()); }
    if let Some(p) = polygons { println!("  Polygons:      {}", p.display()); }
    println!("  Label column:  {}", label_col);
    println!("  Patch size:    {}x{}", size, size);
    if let Some(spec) = &profile_spec {
        println!("  GFM profile:   {} → {}", spec.name, spec.model_target);
        println!("                 expects {} bands, tile {}x{}, unit {}",
            spec.bands_order.len(), spec.tile_size, spec.tile_size, spec.expected_unit);
        if size != spec.tile_size {
            eprintln!("  WARNING: --size {} does not match profile tile {} — model inputs will need resizing",
                size, spec.tile_size);
        }
    }
    println!("  Output dir:    {}", output.display());
    println!("  Tensor format: {}", out_fmt.label());
    if emit_stac { println!("  STAC output:   on (ml-aoi + mlm)"); }
    println!();

    // --- 1. Load + validate rasters ---
    let raster_set = load_raster_set(features_dir)?;
    validate_raster_set_alignment(&raster_set)?;
    let n_timestamps = raster_set.n_timestamps();
    let n_bands = raster_set.n_bands();
    let feature_names = raster_set.band_names.clone();
    let timestamps_order = raster_set.timestamps.clone();
    let (rows, cols) = raster_set.rasters[0][0].shape();
    let gt = *raster_set.rasters[0][0].transform();
    let crs_epsg = raster_set.rasters[0][0].crs().and_then(|c| c.epsg());
    if n_timestamps > 1 {
        println!("Loaded {} timestamps × {} bands ({}x{} grid, pixel {:.3}x{:.3})",
            n_timestamps, n_bands, cols, rows, gt.pixel_width, gt.pixel_height);
        println!("  Timestamps: {}", timestamps_order.join(", "));
    } else {
        println!("Loaded {} rasters ({}x{} grid, pixel {:.3}x{:.3})",
            n_bands, cols, rows, gt.pixel_width, gt.pixel_height);
    }

    // If a GFM profile is set, validate that the band count matches what
    // the model expects. Band names are checked too but only as a soft
    // warning — users may have legitimate reasons to reorder.
    if let Some(spec) = &profile_spec {
        if n_bands != spec.bands_order.len() {
            anyhow::bail!(
                "Profile '{}' expects {} bands ({}), but {} feature rasters were loaded. \
                 Curate the features directory to contain exactly these bands in this order.",
                spec.name, spec.bands_order.len(), spec.bands_order.join(", "), n_bands,
            );
        }
        let mismatch: Vec<(usize, &str, &str)> = feature_names.iter().enumerate()
            .zip(spec.bands_order.iter())
            .filter_map(|((i, got), want)| {
                if got.eq_ignore_ascii_case(want) { None } else { Some((i, got.as_str(), *want)) }
            })
            .collect();
        if !mismatch.is_empty() {
            eprintln!("  WARNING: band names do not match profile order; assuming user-curated.");
            for (i, got, want) in mismatch {
                eprintln!("    band[{}]: got '{}', profile expects '{}'", i, got, want);
            }
        }
    }

    // --- 2. Read vector + build patch specs ---
    let half = size / 2;
    let stride = stride.unwrap_or(size).max(1);
    let mut specs: Vec<PatchSpec> = Vec::new();

    // Use the first raster of the first timestamp as the canonical grid
    // for geo↔pixel conversion. All other rasters are aligned to it.
    let canonical = &raster_set.rasters[0][0];

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
            let (col_f, row_f) = canonical.geo_to_pixel(p.x(), p.y());
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
                let (cx0, ry0) = canonical.geo_to_pixel(bb.min().x, bb.max().y);
                let (cx1, ry1) = canonical.geo_to_pixel(bb.max().x, bb.min().y);
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
                        let (x0, y0) = canonical.pixel_to_geo(c, r);
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
    //     For multi-timestamp inputs the output tensor is [N, C, T, H, W]; the
    //     internal buffer is laid out as a flat [C, T, H, W] block per patch.
    //     NaN threshold is checked over the full (C × T × H × W) volume so a
    //     single bad timestamp can knock the patch out.
    let patch_pixels = size * size;
    let voxels_per_patch = n_bands * n_timestamps * patch_pixels;
    let patch_bytes = voxels_per_patch * 4; // f32
    let est_total_bytes = specs.len() * patch_bytes;
    eprintln!(
        "Patch tensor estimate: {} patches × {} bands × {} timestamps × {}² × 4 bytes = {:.2} GB",
        specs.len(), n_bands, n_timestamps, size, est_total_bytes as f64 / 1e9,
    );

    let mut kept: Vec<(PatchSpec, Vec<f32>)> = Vec::new();
    let mut nan_skipped = 0usize;

    for spec in &specs {
        let r0 = spec.center_row - half;
        let c0 = spec.center_col - half;
        // Buffer laid out [C, T, H, W] flat: index = bi*T*HW + ti*HW + dr*W + dc.
        // We chose C-outer so the per-band z-score normalization in
        // `apply_band_norm_temporal` can stride across all timestamps of one
        // band contiguously — same convention Prithvi expects.
        let mut buf = vec![0f32; voxels_per_patch];
        let mut nan_count = 0usize;
        let ts_pixels = n_timestamps * patch_pixels;
        for bi in 0..n_bands {
            let band_offset = bi * ts_pixels;
            for ti in 0..n_timestamps {
                let raster = &raster_set.rasters[ti][bi];
                let ts_offset = band_offset + ti * patch_pixels;
                for dr in 0..size {
                    let row_offset = ts_offset + dr * size;
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
        }
        let nan_frac = nan_count as f64 / voxels_per_patch as f64;
        if nan_frac > skip_nan_threshold {
            nan_skipped += 1;
            continue;
        }
        // GFM profile: apply per-band z-score normalization in place across
        // all timestamps of that band. NaN pixels preserved as-is.
        if let Some(spec) = &profile_spec {
            if spec.band_norm.len() == n_bands {
                apply_band_norm_block(&mut buf, &spec.band_norm, n_timestamps * patch_pixels);
            }
        }
        kept.push((spec.clone(), buf));
    }

    let n = kept.len();
    println!("After NaN threshold ({:.0}%): {} kept, {} skipped",
        skip_nan_threshold * 100.0, n, nan_skipped);
    if n == 0 {
        anyhow::bail!("All candidate patches were filtered by NaN threshold");
    }

    // --- 6. Write patches tensor ---
    // Shape: [N, C, H, W] when T==1 (backward-compat), [N, C, T, H, W] when T>1.
    let tensor_shape: Vec<usize> = if n_timestamps > 1 {
        vec![n, n_bands, n_timestamps, size, size]
    } else {
        vec![n, n_bands, size, size]
    };

    match out_fmt {
        PatchOutputFormat::Npy => {
            let patches_path = output.join("patches.npy");
            let mut f_patches = OpenOptions::new().create(true).write(true).truncate(true)
                .open(&patches_path)
                .with_context(|| format!("Failed to open {}", patches_path.display()))?;
            write_npy_header(&mut f_patches, &tensor_shape, "<f4")?;
            for (_, buf) in &kept {
                let bytes: &[u8] = bytemuck_cast_f32_to_bytes(buf);
                f_patches.write_all(bytes).context("Failed to write patch bytes")?;
            }
            f_patches.flush().ok();
        }
        PatchOutputFormat::Zarr => {
            // One chunk per chip: chunk shape mirrors tensor_shape except the
            // leading N axis collapses to 1. So a load of chunk `i.0.0.0[.0]`
            // returns chip i with the full [C, (T,) H, W] payload.
            let chunk_shape: Vec<usize> = std::iter::once(1usize)
                .chain(tensor_shape.iter().skip(1).copied())
                .collect();
            let zarr_dir = output.join("patches.zarr");
            // Attributes go on the .zattrs alongside .zarray. We mirror the
            // top-level keys of meta.json so a Zarr-only consumer has the same
            // context as an NPY consumer reading meta.json.
            let zarr_attrs = serde_json::json!({
                "bands": feature_names,
                "patch_size": size,
                "n_patches": n,
                "n_timestamps": n_timestamps,
                "timestamps": timestamps_order,
                "tensor_layout": if n_timestamps > 1 { "[N, C, T, H, W]" } else { "[N, C, H, W]" },
                "crs_epsg": crs_epsg,
                "pixel_width": gt.pixel_width,
                "pixel_height": gt.pixel_height,
                "gfm_profile_name": profile_spec.as_ref().map(|s| s.name),
                "gfm_model_target": profile_spec.as_ref().map(|s| s.model_target),
            });
            init_zarr_v2_array(
                &zarr_dir, &tensor_shape, &chunk_shape, "<f4",
                serde_json::Value::String("NaN".to_string()),
                &zarr_attrs,
            )?;
            // Each chunk holds exactly one chip; coord on axis 0 = chip index.
            let mut chunk_coord = vec![0usize; tensor_shape.len()];
            for (chip_idx, (_, buf)) in kept.iter().enumerate() {
                chunk_coord[0] = chip_idx;
                let bytes: &[u8] = bytemuck_cast_f32_to_bytes(buf);
                write_chunk(&zarr_dir, &chunk_coord, bytes)?;
            }
        }
    }

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
        let (x0, y0) = canonical.pixel_to_geo(spec.center_col, spec.center_row);
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
    let profile_meta = profile_spec.as_ref().map(|spec| serde_json::json!({
        "name": spec.name,
        "model_target": spec.model_target,
        "bands_order": spec.bands_order,
        "tile_size": spec.tile_size,
        "band_norm_mean": spec.band_norm.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
        "band_norm_std": spec.band_norm.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
        "expected_unit": spec.expected_unit,
        "source_url": spec.source_url,
        "normalization_applied": spec.band_norm.len() == n_bands,
    }));

    let shape_label = if n_timestamps > 1 {
        format!("[{}, {}, {}, {}, {}]", n, n_bands, n_timestamps, size, size)
    } else {
        format!("[{}, {}, {}, {}]", n, n_bands, size, size)
    };

    let meta = serde_json::json!({
        "bands": feature_names,
        "patch_size": size,
        "n_patches": n,
        "n_timestamps": n_timestamps,
        "timestamps": timestamps_order,
        "tensor_shape": shape_label,
        "tensor_layout": if n_timestamps > 1 { "[N, C, T, H, W]" } else { "[N, C, H, W]" },
        "tensor_format": out_fmt.label(),
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
        "gfm_profile": profile_meta,
    });
    std::fs::write(output.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;

    // --- 9b. Optional STAC output ---
    if emit_stac {
        let asset_path = match out_fmt {
            PatchOutputFormat::Npy => "patches.npy",
            PatchOutputFormat::Zarr => "patches.zarr",
        };
        let asset_role = match out_fmt {
            PatchOutputFormat::Npy => "data",
            PatchOutputFormat::Zarr => "data-chunk",
        };
        let chips: Vec<ChipInfo> = kept.iter().enumerate().map(|(i, (spec, _))| ChipInfo {
            index: i,
            center_row: spec.center_row,
            center_col: spec.center_col,
            label_int: match spec.label_raw { LabelValue::Int(v) => Some(v), _ => None },
            label_float: match spec.label_raw { LabelValue::Float(v) => Some(v), _ => None },
            asset_path,
            asset_role,
        }).collect();

        let collection_id = output.file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "surtgis-extract-patches".to_string());
        let description = format!(
            "Training chips extracted by SurtGIS extract-patches v{} from {}. \
             {} bands × {} timestamps, tile {}x{}, source mode {}.",
            env!("CARGO_PKG_VERSION"), features_dir.display(),
            n_bands, n_timestamps, size, size,
            if points.is_some() { "points" } else { "polygons" },
        );
        let coll = CollectionInfo {
            id: &collection_id,
            description: &description,
            license: "proprietary",
            source_mode: if points.is_some() { "points" } else { "polygons" },
            patch_size: size,
            n_patches: n,
            n_bands,
            n_timestamps,
            band_names: &feature_names,
            timestamps: &timestamps_order,
            crs_epsg,
            gt: &gt,
            grid_rows: rows,
            grid_cols: cols,
            profile_spec: profile_spec.as_ref(),
        };
        write_stac_output(output, &coll, &chips)
            .context("Failed to write STAC output")?;
    }

    // --- 10. Summary ---
    println!();
    println!("=========================================");
    println!("PATCH EXTRACTION COMPLETE");
    println!("=========================================");
    println!("  Patches:    {}", n);
    println!("  Bands:      {} ({})", n_bands, feature_names.join(", "));
    if n_timestamps > 1 {
        println!("  Timestamps: {} ({})", n_timestamps, timestamps_order.join(", "));
    }
    println!("  Shape:      {} (<f4)", shape_label);
    println!("  Labels:     {} ({})", n, label_dtype);
    println!("  Output:     {}/", output.display());
    println!("  Time:       {:.1}s", start.elapsed().as_secs_f64());
    println!();
    println!("Load in Python:");
    let layout = if n_timestamps > 1 { "[N, C, T, H, W]" } else { "[N, C, H, W]" };
    match out_fmt {
        PatchOutputFormat::Npy => {
            println!("  import numpy as np");
            println!("  X = np.load('{}/patches.npy')  # {} f32", output.display(), layout);
        }
        PatchOutputFormat::Zarr => {
            println!("  import zarr");
            println!("  X = zarr.open('{}/patches.zarr', mode='r')  # {} f32", output.display(), layout);
            println!("  X_np = X[:]   # fully materialise, or X[i:j] for lazy access");
        }
    }
    println!("  import numpy as np");
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
