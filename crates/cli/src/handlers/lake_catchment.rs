//! `surtgis hydrology lake-catchment`: lake-aware catchments for many lakes.
//!
//! Per lake: a square UTM window around it; every DEM source warped onto
//! that window (STAC items read by range, or local tiles such as FABDEM);
//! the lake rasterized from its polygon (or extracted from a water mask
//! for point lakes); [`lake_catchment`] with the lake as the outlet; the
//! window doubled while the catchment or the lake touches its edge. One
//! output row per lake and DEM; with two or more DEMs, `dem_spread` says
//! how much they disagree.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use geo::{BoundingRect, Centroid, Geometry, MapCoords};
use rayon::prelude::*;
use surtgis_algorithms::hydrology::{
    LakeCatchmentParams, WaterBodyParams, lake_catchment, water_body_at,
};
use surtgis_cloud::blocking::{CogReaderBlocking, StacClientBlocking};
use surtgis_cloud::{BBox, CogReaderOptions, StacCatalog, StacClientOptions, StacSearchParams};
use surtgis_core::io::window::{geotiff_info, read_geotiff_window_bands};
use surtgis_core::vector::{AttributeValue, FeatureCollection, rasterize_polygons, read_vector};
use surtgis_core::warp::{self, GridSpec, Resampling, Transformer};
use surtgis_core::{CRS, GeoTransform, Raster};

/// Options of `hydrology lake-catchment`, as parsed from the command line.
pub struct Args {
    pub lakes: PathBuf,
    pub id_field: Option<String>,
    pub dems: Vec<String>,
    pub water_mask: Option<String>,
    pub water_value: f64,
    pub window_km: f64,
    pub max_window_km: f64,
    pub pixel_size: f64,
    pub jobs: usize,
    pub retries: u32,
    pub output: PathBuf,
    pub masks_dir: Option<PathBuf>,
    pub sink_depth: Option<f64>,
}

/// A raster source that can be warped onto any window grid.
enum Source {
    /// STAC collection; each item's `asset` is a COG read by range.
    Stac {
        catalog: String,
        collection: String,
        asset: String,
        /// One client per source, shared by every lake and window, so the
        /// SAS-token cache is reused (Planetary Computer rate-limits signing).
        client: std::sync::OnceLock<StacClientBlocking>,
    },
    /// Local GeoTIFF/COG tiles (a file, or every `.tif` under a directory).
    Tiles(Vec<Tile>),
}

struct Tile {
    path: PathBuf,
    epsg: u32,
    /// Bounds in the tile's own CRS.
    bounds: (f64, f64, f64, f64),
}

struct NamedSource {
    label: String,
    source: Source,
}

/// One input lake: id, geometry in EPSG:4326 and a representative point.
struct Lake {
    id: String,
    geom: Geometry<f64>,
    lon: f64,
    lat: f64,
    is_point: bool,
}

/// One output row.
struct Row {
    lake_id: String,
    dem: String,
    lon: f64,
    lat: f64,
    utm_epsg: u32,
    window_km: f64,
    lake_area_km2: f64,
    lake_perimeter_km: f64,
    lake_elev_m: f64,
    catchment_area_km2: f64,
    local_catchment_area_km2: f64,
    catchment_mean_elev_m: f64,
    water_trimmed: Option<f64>,
    edge_reached: bool,
    status: String,
    message: String,
}

pub fn run(a: Args) -> Result<()> {
    if a.dems.is_empty() {
        bail!(
            "give at least one --dem (stac:<catalog>:<collection>[:asset] or a local file/directory)"
        );
    }
    let dems: Vec<NamedSource> = a
        .dems
        .iter()
        .map(|s| parse_source(s))
        .collect::<Result<_>>()?;
    let water = a.water_mask.as_deref().map(parse_source).transpose()?;
    let lakes = read_lakes(&a.lakes, a.id_field.as_deref())?;
    if lakes.iter().any(|l| l.is_point) && water.is_none() {
        bail!(
            "{} has point lakes: give --water-mask (e.g. stac:pc:esa-worldcover:map) to extract \
             each lake, or pass lake polygons",
            a.lakes.display()
        );
    }
    if let Some(dir) = &a.masks_dir {
        std::fs::create_dir_all(dir)?;
    }
    eprintln!(
        "lake-catchment: {} lakes x {} DEM(s), window {}–{} km, {} jobs",
        lakes.len(),
        dems.len(),
        a.window_km,
        a.max_window_km,
        a.jobs
    );

    let done = Mutex::new(0usize);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a.jobs.max(1))
        .build()
        .context("thread pool")?;
    let mut rows: Vec<Row> = pool.install(|| {
        lakes
            .par_iter()
            .flat_map_iter(|lake| {
                let out: Vec<Row> = dems
                    .iter()
                    .map(|d| one(lake, d, water.as_ref(), &a))
                    .collect();
                let mut n = done.lock().unwrap_or_else(|e| e.into_inner());
                *n += 1;
                let summary: Vec<String> = out
                    .iter()
                    .map(|r| {
                        if r.status == "ok" {
                            format!("{} {:.2} km²", r.dem, r.catchment_area_km2)
                        } else {
                            format!("{} {}", r.dem, r.status)
                        }
                    })
                    .collect();
                eprintln!(
                    "  [{}/{}] {}: {}",
                    *n,
                    lakes.len(),
                    lake.id,
                    summary.join(", ")
                );
                out
            })
            .collect()
    });

    // Spread across DEMs (max / min catchment area among successful runs).
    let mut spread = std::collections::HashMap::<String, f64>::new();
    for id in lakes.iter().map(|l| &l.id) {
        let v: Vec<f64> = rows
            .iter()
            .filter(|r| &r.lake_id == id && r.status == "ok" && r.catchment_area_km2 > 0.0)
            .map(|r| r.catchment_area_km2)
            .collect();
        if v.len() >= 2 {
            let (mn, mx) = v
                .iter()
                .fold((f64::INFINITY, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
            spread.insert(id.clone(), mx / mn);
        }
    }
    rows.sort_by(|x, y| x.lake_id.cmp(&y.lake_id).then(x.dem.cmp(&y.dem)));
    write_csv(&a.output, &rows, &spread)?;
    let ok = rows.iter().filter(|r| r.status == "ok").count();
    println!(
        "Lake catchments: {ok}/{} lake x DEM runs ok -> {}",
        rows.len(),
        a.output.display()
    );
    Ok(())
}

/// Delineate one lake with one DEM, enlarging the window as needed.
fn one(lake: &Lake, dem: &NamedSource, water: Option<&NamedSource>, a: &Args) -> Row {
    let zone = (((lake.lon + 180.0) / 6.0).floor() as i64).clamp(0, 59) as u32 + 1;
    let epsg = if lake.lat >= 0.0 {
        32600 + zone
    } else {
        32700 + zone
    };
    let mut row = Row {
        lake_id: lake.id.clone(),
        dem: dem.label.clone(),
        lon: lake.lon,
        lat: lake.lat,
        utm_epsg: epsg,
        window_km: 0.0,
        lake_area_km2: f64::NAN,
        lake_perimeter_km: f64::NAN,
        lake_elev_m: f64::NAN,
        catchment_area_km2: f64::NAN,
        local_catchment_area_km2: f64::NAN,
        catchment_mean_elev_m: f64::NAN,
        water_trimmed: None,
        edge_reached: false,
        status: "error".into(),
        message: String::new(),
    };
    let result = (|| -> Result<()> {
        let to_utm = Transformer::new(4326, epsg)?;
        let (cx, cy) = to_utm
            .forward(lake.lon, lake.lat)
            .context("lake centre cannot be projected to UTM")?;
        // Start from the lake's own size when it has a polygon.
        let lake_utm: Option<Geometry<f64>> = if lake.is_point {
            None
        } else {
            Some(project_geometry(&lake.geom, &to_utm)?)
        };
        let lake_extent_km = lake_utm
            .as_ref()
            .and_then(|g| g.bounding_rect())
            .map(|r| r.width().max(r.height()) / 1000.0)
            .unwrap_or(0.0);
        let mut win_km = a.window_km.max(lake_extent_km * 3.0);
        loop {
            row.window_km = win_km;
            let half = win_km * 500.0;
            let n = ((2.0 * half) / a.pixel_size).ceil() as usize;
            let grid = GridSpec {
                transform: GeoTransform::new(cx - half, cy + half, a.pixel_size, -a.pixel_size),
                rows: n,
                cols: n,
                epsg,
            };
            let z = build_on_grid(&dem.source, &grid, Resampling::Nearest, a.retries)
                .with_context(|| format!("DEM {}", dem.label))?;
            let lake_mask = match &lake_utm {
                Some(g) => rasterize_lake(g, &grid, cx, cy)?,
                None => {
                    let w = water.expect("checked in run()");
                    let wm = build_on_grid(&w.source, &grid, Resampling::Nearest, a.retries)
                        .with_context(|| format!("water mask {}", w.label))?;
                    let mut bin = Raster::<u8>::new(n, n);
                    bin.set_transform(grid.transform);
                    for (o, v) in bin.data_mut().iter_mut().zip(wm.data().iter()) {
                        *o = u8::from((*v - a.water_value).abs() < 0.5);
                    }
                    let (c, r) = grid_cell(&grid, cx, cy);
                    let wb = water_body_at(&bin, r, c, WaterBodyParams::default())
                        .context("no water within 10 cells of the point")?;
                    row.water_trimmed =
                        Some(wb.raw_component_cells as f64 / wb.cells.max(1) as f64);
                    wb.mask
                }
            };
            let lc = lake_catchment(&z, &lake_mask, LakeCatchmentParams::default())?;
            let edge = lc.touches_edge || lc.lake_touches_edge;
            if !edge || win_km * 2.0 > a.max_window_km {
                row.lake_area_km2 = lc.lake_area / 1e6;
                row.lake_perimeter_km = lc.lake_perimeter / 1e3;
                row.lake_elev_m = lc.lake_elevation;
                row.catchment_area_km2 = lc.catchment_area / 1e6;
                row.catchment_mean_elev_m = lc.catchment_mean_elevation;
                if let Some(d) = a.sink_depth {
                    let mut p = LakeCatchmentParams::default();
                    p.sink_depth = Some(d);
                    let local = lake_catchment(&z, &lake_mask, p)?;
                    row.local_catchment_area_km2 = local.catchment_area / 1e6;
                }
                row.edge_reached = edge;
                row.status = if edge { "edge".into() } else { "ok".into() };
                if edge {
                    row.message = format!(
                        "catchment still reaches the edge of a {win_km} km window \
                         (--max-window-km {}); areas are lower bounds",
                        a.max_window_km
                    );
                }
                if let Some(dir) = &a.masks_dir {
                    let name = format!("{}_{}.tif", sanitize(&lake.id), sanitize(&dem.label));
                    let mut m = lc.mask.clone();
                    m.set_crs(Some(CRS::from_epsg(epsg)));
                    surtgis_core::io::write_geotiff(&m, dir.join(name), None)?;
                }
                return Ok(());
            }
            win_km *= 2.0;
        }
    })();
    if let Err(e) = result {
        row.status = "error".into();
        row.message = format!("{e:#}").replace(['\n', ','], " ");
    }
    row
}

/// Cell (col, row) of map coordinate (x, y) on `grid`, clamped.
fn grid_cell(grid: &GridSpec, x: f64, y: f64) -> (usize, usize) {
    let t = &grid.transform;
    let c = ((x - t.origin_x) / t.pixel_width).floor().max(0.0) as usize;
    let r = ((y - t.origin_y) / t.pixel_height).floor().max(0.0) as usize;
    (c.min(grid.cols - 1), r.min(grid.rows - 1))
}

/// Burn a (UTM) lake polygon onto the grid; a lake smaller than a cell
/// keeps at least the cell under its centre.
fn rasterize_lake(g: &Geometry<f64>, grid: &GridSpec, cx: f64, cy: f64) -> Result<Raster<u8>> {
    let mut fc = FeatureCollection::with_crs(Some(CRS::from_epsg(grid.epsg)));
    fc.push(surtgis_core::vector::Feature::new(g.clone()));
    let crs = CRS::from_epsg(grid.epsg);
    let burnt = rasterize_polygons(&fc, &grid.transform, grid.rows, grid.cols, None, Some(&crs))?;
    let mut m = Raster::<u8>::new(grid.rows, grid.cols);
    m.set_transform(grid.transform);
    let mut any = false;
    for (o, v) in m.data_mut().iter_mut().zip(burnt.data().iter()) {
        *o = u8::from(v.is_finite());
        any |= v.is_finite();
    }
    if !any {
        let (c, r) = grid_cell(grid, cx, cy);
        m.data_mut()[(r, c)] = 1;
    }
    Ok(m)
}

/// Warp every piece of `source` that overlaps `grid` onto it; cells no
/// source covers stay NaN. First valid value wins where pieces overlap.
fn build_on_grid(
    source: &Source,
    grid: &GridSpec,
    method: Resampling,
    retries: u32,
) -> Result<Raster<f64>> {
    let mut out = Raster::<f64>::new(grid.rows, grid.cols);
    out.set_transform(grid.transform);
    out.set_crs(Some(CRS::from_epsg(grid.epsg)));
    out.data_mut().fill(f64::NAN);
    let mut merge = |piece: &Raster<f64>, src_epsg: u32| -> Result<()> {
        let tf = Transformer::new(src_epsg, grid.epsg)?;
        let warped = warp::warp(
            std::slice::from_ref(piece),
            piece.transform(),
            &tf,
            grid,
            method,
        )?;
        let nd = piece.nodata();
        for (o, v) in out.data_mut().iter_mut().zip(warped[0].data().iter()) {
            if o.is_nan() && v.is_finite() && nd.is_none_or(|n| *v != n) {
                *o = *v;
            }
        }
        Ok(())
    };
    match source {
        Source::Tiles(tiles) => {
            for t in tiles {
                let tf = Transformer::new(t.epsg, grid.epsg)?;
                let Ok(b) = warp::source_window(grid, &tf, 2) else {
                    continue;
                };
                let (x0, y0, x1, y1) = t.bounds;
                if b.max_x < x0 || b.min_x > x1 || b.max_y < y0 || b.min_y > y1 {
                    continue;
                }
                let info = geotiff_info(&t.path)
                    .with_context(|| format!("reading {}", t.path.display()))?;
                let Some(pw) = info.window_for_bounds(0, b.min_x, b.min_y, b.max_x, b.max_y) else {
                    continue;
                };
                let piece = read_geotiff_window_bands::<f64, _>(&t.path, &info, 0, &pw)?
                    .into_iter()
                    .next()
                    .context("tile has no band")?;
                merge(&piece, t.epsg)?;
            }
        }
        Source::Stac {
            catalog,
            collection,
            asset,
            client,
        } => {
            let to_ll = Transformer::new(grid.epsg, 4326)?;
            let gb = grid.bounds();
            let corners = [
                (gb.min_x, gb.min_y),
                (gb.min_x, gb.max_y),
                (gb.max_x, gb.min_y),
                (gb.max_x, gb.max_y),
            ];
            let ll: Vec<(f64, f64)> = corners
                .iter()
                .filter_map(|&(x, y)| to_ll.forward(x, y))
                .collect();
            if ll.is_empty() {
                bail!("window cannot be projected to lon/lat");
            }
            let (w, e) = ll
                .iter()
                .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.0), b.max(p.0)));
            let (s, n) = ll
                .iter()
                .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.1), b.max(p.1)));
            if client.get().is_none() {
                let c = StacClientBlocking::new(
                    StacCatalog::from_str_or_url(catalog),
                    StacClientOptions::default(),
                )?;
                let _ = client.set(c);
            }
            let client = client.get().expect("set above");
            let params = StacSearchParams::new()
                .bbox(w, s, e, n)
                .collections(&[collection.as_str()])
                .limit(50);
            let items = client.search_all(&params)?;
            if items.is_empty() {
                bail!("no {collection} item covers the window");
            }
            for item in &items {
                let Some(a) = item.asset(asset) else { continue };
                let href = client
                    .sign_asset_href(&a.href, item.collection.as_deref().unwrap_or(collection))?;
                let mut last = None;
                for attempt in 0..retries.max(1) {
                    let res = (|| -> Result<()> {
                        let mut reader =
                            CogReaderBlocking::open(&href, CogReaderOptions::default())?;
                        let src_epsg = item
                            .epsg()
                            .or_else(|| reader.metadata().crs.as_ref().and_then(|c| c.epsg()))
                            .context("item has no EPSG")?;
                        let tf = Transformer::new(src_epsg, grid.epsg)?;
                        let b = warp::source_window(grid, &tf, 2)?;
                        let bb = BBox {
                            min_x: b.min_x,
                            min_y: b.min_y,
                            max_x: b.max_x,
                            max_y: b.max_y,
                        };
                        match reader.read_bbox::<f64>(&bb, None) {
                            Ok(piece) => merge(&piece, src_epsg),
                            // the item's footprint does not reach the window
                            Err(e) if e.to_string().contains("does not intersect") => Ok(()),
                            Err(e) => Err(e.into()),
                        }
                    })();
                    match res {
                        Ok(()) => {
                            last = None;
                            break;
                        }
                        Err(e) => {
                            last = Some(e);
                            std::thread::sleep(std::time::Duration::from_millis(500 << attempt));
                        }
                    }
                }
                if let Some(e) = last {
                    return Err(e.context(format!("item {}", item.id)));
                }
                let search_url = client.catalog().search_url();
                let root = search_url.strip_suffix("/search").unwrap_or(&search_url);
                surtgis_core::provenance::observe_stac_item(
                    surtgis_core::provenance::StacItemRecord::new(
                        root,
                        item.collection.as_deref().unwrap_or(collection),
                        item.id.as_str(),
                    )
                    .with_assets([asset.clone()]),
                );
            }
        }
    }
    if out.data().iter().all(|v| v.is_nan()) {
        bail!("no data in the window");
    }
    Ok(out)
}

/// `stac:<catalog>:<collection>[:asset]`, or a local file or directory.
fn parse_source(spec: &str) -> Result<NamedSource> {
    if let Some(rest) = spec.strip_prefix("stac:") {
        let parts: Vec<&str> = rest.split(':').collect();
        // catalog may be a URL with ':' (https://…); collection and asset are last
        let (catalog, collection, asset) = match parts.len() {
            2 => (
                parts[0].to_string(),
                parts[1].to_string(),
                "data".to_string(),
            ),
            n if n >= 3 && !parts[0].starts_with("http") => (
                parts[0].to_string(),
                parts[1].to_string(),
                parts[2..].join(":"),
            ),
            n if n >= 4 => (
                parts[..n - 2].join(":"),
                parts[n - 2].to_string(),
                parts[n - 1].to_string(),
            ),
            _ => bail!("'{spec}': expected stac:<catalog>:<collection>[:asset]"),
        };
        return Ok(NamedSource {
            label: collection.clone(),
            source: Source::Stac {
                catalog,
                collection,
                asset,
                client: std::sync::OnceLock::new(),
            },
        });
    }
    let path = Path::new(spec);
    let files: Vec<PathBuf> = if path.is_dir() {
        let mut v = Vec::new();
        collect_tifs(path, &mut v)?;
        v.sort();
        v
    } else if path.is_file() {
        vec![path.to_path_buf()]
    } else {
        bail!("'{spec}' is neither stac:<catalog>:<collection> nor an existing file or directory");
    };
    if files.is_empty() {
        bail!("no .tif under {}", path.display());
    }
    let tiles = files
        .into_iter()
        .map(|p| {
            let info = geotiff_info(&p).with_context(|| format!("reading {}", p.display()))?;
            let epsg = info
                .crs
                .as_ref()
                .and_then(|c| c.epsg())
                .with_context(|| format!("{} has no EPSG code", p.display()))?;
            let t = info.transform;
            let (x0, y0, x1, y1) = t.bounds(info.width as usize, info.height as usize);
            Ok(Tile {
                path: p,
                epsg,
                bounds: (x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let label = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "dem".into());
    Ok(NamedSource {
        label,
        source: Source::Tiles(tiles),
    })
}

fn collect_tifs(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            collect_tifs(&p, out)?;
        } else if p
            .extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("tif") || x.eq_ignore_ascii_case("tiff"))
        {
            out.push(p);
        }
    }
    Ok(())
}

/// Read lakes (points or polygons) and bring them to EPSG:4326.
fn read_lakes(path: &Path, id_field: Option<&str>) -> Result<Vec<Lake>> {
    let fc = read_vector(path).with_context(|| format!("reading {}", path.display()))?;
    let src_epsg = match fc.crs().and_then(|c| c.epsg()) {
        Some(e) => e,
        None => {
            eprintln!("Warning: {} has no CRS; assuming EPSG:4326", path.display());
            4326
        }
    };
    let to_ll = if src_epsg == 4326 {
        None
    } else {
        Some(Transformer::new(src_epsg, 4326)?)
    };
    let mut lakes = Vec::new();
    for (i, f) in fc.iter().enumerate() {
        let Some(g) = &f.geometry else { continue };
        let g = match &to_ll {
            Some(tf) => project_geometry(g, tf)?,
            None => g.clone(),
        };
        let id = match id_field.and_then(|k| f.properties.get(k)) {
            Some(AttributeValue::String(s)) => s.clone(),
            Some(AttributeValue::Int(v)) => v.to_string(),
            Some(AttributeValue::Float(v)) => v.to_string(),
            Some(_) | None => f.id.clone().unwrap_or_else(|| (i + 1).to_string()),
        };
        let (is_point, c) = match &g {
            Geometry::Point(p) => (true, *p),
            Geometry::Polygon(_) | Geometry::MultiPolygon(_) => {
                (false, g.centroid().context("lake polygon has no centroid")?)
            }
            other => bail!(
                "feature {id}: lakes must be points or polygons, got {:?}",
                std::mem::discriminant(other)
            ),
        };
        lakes.push(Lake {
            id,
            geom: g,
            lon: c.x(),
            lat: c.y(),
            is_point,
        });
    }
    if lakes.is_empty() {
        bail!("{} has no lake features", path.display());
    }
    Ok(lakes)
}

fn project_geometry(g: &Geometry<f64>, tf: &Transformer) -> Result<Geometry<f64>> {
    let failed = std::cell::Cell::new(false);
    let out = g.map_coords(|c| match tf.forward(c.x, c.y) {
        Some((x, y)) => geo::coord! { x: x, y: y },
        None => {
            failed.set(true);
            c
        }
    });
    if failed.get() {
        bail!("geometry cannot be projected");
    }
    Ok(out)
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn write_csv(
    path: &Path,
    rows: &[Row],
    spread: &std::collections::HashMap<String, f64>,
) -> Result<()> {
    let mut w =
        csv::Writer::from_path(path).with_context(|| format!("writing {}", path.display()))?;
    w.write_record([
        "lake_id",
        "dem",
        "lon",
        "lat",
        "utm_epsg",
        "window_km",
        "lake_area_km2",
        "lake_perimeter_km",
        "lake_elev_m",
        "catchment_area_km2",
        "catchment_lake_ratio",
        "local_catchment_area_km2",
        "catchment_mean_elev_m",
        "dem_spread",
        "water_trimmed",
        "edge_reached",
        "status",
        "message",
    ])?;
    let f = |v: f64, d: usize| {
        if v.is_finite() {
            format!("{v:.d$}")
        } else {
            String::new()
        }
    };
    for r in rows {
        let ratio = r.catchment_area_km2 / r.lake_area_km2;
        w.write_record([
            r.lake_id.clone(),
            r.dem.clone(),
            f(r.lon, 6),
            f(r.lat, 6),
            r.utm_epsg.to_string(),
            f(r.window_km, 0),
            f(r.lake_area_km2, 4),
            f(r.lake_perimeter_km, 3),
            f(r.lake_elev_m, 2),
            f(r.catchment_area_km2, 4),
            f(ratio, 3),
            f(r.local_catchment_area_km2, 4),
            f(r.catchment_mean_elev_m, 2),
            spread.get(&r.lake_id).map(|v| f(*v, 3)).unwrap_or_default(),
            r.water_trimmed.map(|v| f(v, 2)).unwrap_or_default(),
            r.edge_reached.to_string(),
            r.status.clone(),
            r.message.clone(),
        ])?;
    }
    w.flush()?;
    Ok(())
}
