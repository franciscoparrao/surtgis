//! Handler for STAC catalog subcommands.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use surtgis_algorithms::imagery::{
    CloudMaskStrategy, HlsFmask, LandsatQaMask, NoCloudMask, S2SclMask,
};
use surtgis_cloud::blocking::{CogReaderBlocking, StacClientBlocking};
use surtgis_cloud::composite::{cog_cache_key, overview_for_target_resolution};
use surtgis_cloud::{BBox, CogReaderOptions, StacCatalog, StacClientOptions, StacSearchParams};

use crate::commands::StacCommands;
use crate::helpers::{done, parse_bbox, spinner, write_result};
use crate::stac_introspect::{CloudMaskType, StacCollectionSchema};
use crate::streaming::resolve_asset_key;

/// Collection-specific configuration for cloud masking and asset keys
#[derive(Clone)]
pub enum CollectionProfile {
    /// Sentinel-2 L2A: SCL categorical cloud masking
    Sentinel2L2A {
        cloud_mask_strategy: Arc<dyn CloudMaskStrategy>,
    },
    /// Landsat Collection 2 L2: QA_PIXEL bitmask cloud masking
    LandsatC2L2 {
        cloud_mask_strategy: Arc<dyn CloudMaskStrategy>,
    },
    /// Sentinel-1 RTC: No cloud masking (SAR penetrates clouds)
    Sentinel1RTC,
}

impl std::fmt::Debug for CollectionProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sentinel2L2A { .. } => f.debug_struct("Sentinel2L2A").finish(),
            Self::LandsatC2L2 { .. } => f.debug_struct("LandsatC2L2").finish(),
            Self::Sentinel1RTC => f.debug_struct("Sentinel1RTC").finish(),
        }
    }
}

impl CollectionProfile {
    /// Create profile from collection name
    pub fn from_collection_name(name: &str) -> Result<Self> {
        match name {
            "sentinel-2-l2a" => Ok(Self::Sentinel2L2A {
                cloud_mask_strategy: Arc::new(S2SclMask::new()),
            }),
            "landsat-c2-l2" => Ok(Self::LandsatC2L2 {
                cloud_mask_strategy: Arc::new(LandsatQaMask::new()),
            }),
            "sentinel-1-rtc" => Ok(Self::Sentinel1RTC),
            // Any other collection: treat as no cloud masking needed
            // (derived products like WorldCover, DEMs, etc.)
            _ => {
                eprintln!(
                    "  ℹ Collection '{}' not recognized — proceeding without cloud masking",
                    name
                );
                Ok(Self::Sentinel1RTC)
            }
        }
    }

    /// Get the SCL/QA asset name for this collection (None for SAR)
    pub fn mask_asset_name(&self) -> Option<&str> {
        match self {
            Self::Sentinel2L2A { .. } => Some("scl"),
            Self::LandsatC2L2 { .. } => Some("QA_PIXEL"),
            Self::Sentinel1RTC => None,
        }
    }

    /// Get a description for logging
    pub fn description(&self) -> &str {
        match self {
            Self::Sentinel2L2A { .. } => "Sentinel-2 L2A",
            Self::LandsatC2L2 { .. } => "Landsat C2 L2",
            Self::Sentinel1RTC => "Sentinel-1 RTC",
        }
    }
}

/// Factory: create CloudMaskStrategy from auto-detected CloudMaskType
///
/// Supports:
/// - Categorical (e.g., Sentinel-2 SCL) → S2SclMask
/// - Bitmask with asset name "Fmask" (HLS S30/L30) → HlsFmask
/// - Bitmask otherwise (Landsat C2 QA_PIXEL) → LandsatQaMask
/// - None (e.g., SAR) → NoCloudMask
///
/// HLS Fmask and Landsat QA_PIXEL both ride the Bitmask variant but use
/// different bit assignments (HLS cloud is bit 1; Landsat cloud is bit 3),
/// so we route on the asset name. The HLS preprocessing convention is
/// required by Prithvi-EO-2.0 downstream consumers.
pub fn create_cloud_mask_strategy(mask_type: &CloudMaskType) -> Arc<dyn CloudMaskStrategy> {
    match mask_type {
        CloudMaskType::Categorical {
            asset: _,
            num_classes: _,
        } => {
            // For now, assume all categorical masks are S2-like (SCL)
            // Can be extended to support other categorical formats
            Arc::new(S2SclMask::new())
        }
        CloudMaskType::Bitmask { asset, bits: _ } => {
            if asset.eq_ignore_ascii_case("Fmask") || asset.to_lowercase().contains("fmask") {
                Arc::new(HlsFmask::new())
            } else {
                Arc::new(LandsatQaMask::new())
            }
        }
        CloudMaskType::None => Arc::new(NoCloudMask),
    }
}

/// Known STAC catalogs (curated + indexed)
#[derive(Clone, Debug)]
pub struct StacCatalogInfo {
    pub shorthand: &'static str,
    pub name: &'static str,
    pub url: &'static str,
    pub description: &'static str,
}

pub fn get_known_catalogs() -> Vec<StacCatalogInfo> {
    vec![
        StacCatalogInfo {
            shorthand: "pc",
            name: "Planetary Computer",
            url: "https://planetarycomputer.microsoft.com/api/stac/v1",
            description: "Microsoft's STAC: S2, Landsat, Sentinel-1, Copernicus DEM, GEBCO, etc.",
        },
        StacCatalogInfo {
            shorthand: "es",
            name: "Earth Search (AWS)",
            url: "https://earth-search.aws.element84.com/v1",
            description: "Element 84 on AWS: S2, Landsat, Sentinel-1, DEM collections",
        },
        StacCatalogInfo {
            shorthand: "cdse",
            name: "Copernicus Data Space (CDSE)",
            url: "https://catalogue.dataspace.copernicus.eu/odata/v1",
            description: "EU's Copernicus: S1, S2, S3, S5P, DEM (free, Europe-focused)",
        },
        StacCatalogInfo {
            shorthand: "usgs",
            name: "USGS 3DEP / OpenTopography",
            url: "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL30/SRTM_GL30_srtm",
            description: "USGS elevation data: SRTM, ASTER GDEM, HydroSHEDS",
        },
        StacCatalogInfo {
            shorthand: "osgeo",
            name: "OGC STAC Index API",
            url: "https://stacindex.org/api/v1",
            description: "STAC Index: Registry of all public STAC catalogs worldwide",
        },
    ]
}

/// Collections available in each STAC catalog
pub fn get_catalog_collections(catalog: &str) -> Vec<(&'static str, &'static str)> {
    match catalog {
        "pc" => vec![
            (
                "sentinel-2-l2a",
                "Sentinel-2 Level 2A (optical, 10-60m, 2016-present)",
            ),
            (
                "landsat-c2-l2",
                "Landsat Collection 2 Level 2 (optical, 30m, 1980-present)",
            ),
            ("sentinel-1-rtc", "Sentinel-1 RTC (SAR, 10m, 2015-present)"),
            ("cop-dem-glo-30", "Copernicus DEM 30m (elevation, global)"),
            ("nasadem", "NASADEM (elevation, 30m, global)"),
            ("gebco", "GEBCO bathymetry (ocean, 15 arc-seconds)"),
        ],
        "es" => vec![
            ("sentinel-2-l2a", "Sentinel-2 Level 2A (optical, 10-60m)"),
            (
                "landsat-c2-l2",
                "Landsat Collection 2 Level 2 (optical, 30m)",
            ),
            ("sentinel-1-rtc", "Sentinel-1 RTC (SAR, 10m)"),
        ],
        "cdse" => vec![
            (
                "sentinel-1-grd",
                "Sentinel-1 GRD (SAR, 10m, ground range detected)",
            ),
            (
                "sentinel-1-slc",
                "Sentinel-1 SLC (SAR, 10m, single look complex)",
            ),
            (
                "sentinel-2-l1c",
                "Sentinel-2 L1C (optical, 10-60m, L1 processing)",
            ),
            (
                "sentinel-2-l2a",
                "Sentinel-2 L2A (optical, 10-60m, atmospherically corrected)",
            ),
            (
                "sentinel-3-olci",
                "Sentinel-3 OLCI (optical, 300-1000m, ocean/land)",
            ),
            (
                "sentinel-5p",
                "Sentinel-5P (atmospheric, daily global coverage)",
            ),
        ],
        "usgs" => vec![
            (
                "srtm-30m",
                "SRTM 30m DEM (90m for polar regions, 2000 data)",
            ),
            (
                "aster-gdem",
                "ASTER GDEM (30m elevation, global, 2011 release)",
            ),
            (
                "nasadem",
                "NASADEM (30m, merged SRTM+ASTER, improved voids)",
            ),
            (
                "hydrosheds",
                "HydroSHEDS (hydrological datasets, 15 arc-seconds)",
            ),
        ],
        "osgeo" => vec![
            (
                "(registry)",
                "STAC Index: Search all public STAC catalogs worldwide",
            ),
            ("(api)", "Use API to discover 1000+ STAC catalogs globally"),
        ],
        _ => vec![(
            "(unknown)",
            "Use 'surtgis stac search --catalog <url> ...' to discover collections",
        )],
    }
}

/// Catalog info from STAC Index API
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StacIndexCatalog {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub url: Option<String>,
}

/// Fetch catalogs from OGC STAC Index API (<https://stacindex.org/api/v1>)
///
/// Returns a list of catalog summaries with ID, title, description, and URL.
/// This is optional - if it fails, users can still use curated catalogs.
fn fetch_stac_index_catalogs() -> Result<Vec<StacIndexCatalog>> {
    // Try cache first
    let cache_path = get_stac_index_cache_path();
    let cache_ttl_secs = 24 * 60 * 60; // 24 hours

    if let Ok(cached_catalogs) = load_from_cache(&cache_path, cache_ttl_secs) {
        eprintln!(
            "  📦 Using cached catalog list ({} catalogs)",
            cached_catalogs.len()
        );
        return Ok(cached_catalogs);
    }

    // Try real HTTP fetch
    eprintln!("  🌐 Discovering catalogs from STAC Index API...");
    match fetch_stac_index_http() {
        Ok(catalogs) => {
            // Save to cache for next time
            let _ = save_to_cache(&cache_path, &catalogs);
            eprintln!("  ✅ Found {} catalogs (cached)", catalogs.len());
            Ok(catalogs)
        }
        Err(e) => {
            eprintln!("  ℹ️  STAC Index API unavailable: {}", e);
            eprintln!("     Using curated catalogs instead");
            // Return empty - caller will show curated list
            Ok(Vec::new())
        }
    }
}

/// Safely truncate a UTF-8 string to a maximum length
fn truncate_utf8(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }

    // Truncate at character boundary
    let mut truncated = String::new();
    for c in s.chars() {
        if truncated.len() + c.len_utf8() <= max_len {
            truncated.push(c);
        } else {
            break;
        }
    }
    truncated.push_str("...");
    truncated
}

/// Get cache file path for STAC Index catalogs
fn get_stac_index_cache_path() -> std::path::PathBuf {
    #[cfg(unix)]
    {
        let cache_dir = std::env::var("XDG_CACHE_HOME").unwrap_or_else(|_| {
            format!(
                "{}/.cache",
                std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
            )
        });
        std::path::PathBuf::from(cache_dir)
            .join("surtgis")
            .join("stac_index_catalogs.json")
    }
    #[cfg(windows)]
    {
        let cache_dir = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string());
        std::path::PathBuf::from(cache_dir)
            .join("surtgis-cache")
            .join("stac_index_catalogs.json")
    }
    #[cfg(not(any(unix, windows)))]
    {
        std::path::PathBuf::from(".cache/stac_index_catalogs.json")
    }
}

/// Load catalogs from cache if valid
fn load_from_cache(path: &std::path::PathBuf, ttl_secs: u64) -> Result<Vec<StacIndexCatalog>> {
    use std::fs;
    use std::time::SystemTime;

    // Check if file exists and is fresh
    let metadata = fs::metadata(path).context("Cache file not found")?;
    let modified = metadata
        .modified()
        .context("Could not get modification time")?;
    let elapsed = SystemTime::now()
        .duration_since(modified)
        .context("Could not calculate cache age")?;

    if elapsed.as_secs() > ttl_secs {
        anyhow::bail!("Cache expired");
    }

    // Read and parse cache
    let content = fs::read_to_string(path).context("Could not read cache file")?;
    let catalogs: Vec<StacIndexCatalog> =
        serde_json::from_str(&content).context("Could not parse cache file")?;

    Ok(catalogs)
}

/// Save catalogs to cache
fn save_to_cache(path: &std::path::PathBuf, catalogs: &[StacIndexCatalog]) -> Result<()> {
    use std::fs;

    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }

    // Serialize and write
    let json = serde_json::to_string_pretty(catalogs).context("Could not serialize catalogs")?;
    fs::write(path, json).context("Could not write cache file")?;

    Ok(())
}

/// Fetch catalogs from STAC Index API via HTTP
/// Uses tokio runtime to handle async reqwest calls from sync context
fn fetch_stac_index_http() -> Result<Vec<StacIndexCatalog>> {
    // Use tokio to run async code from sync context
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to create async runtime")?;

    rt.block_on(async { fetch_stac_index_async().await })
}

/// Async function to fetch from STAC Index API
async fn fetch_stac_index_async() -> Result<Vec<StacIndexCatalog>> {
    use std::time::Duration;

    let url = "https://stacindex.org/api/catalogs";

    // Create HTTP client
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .context("Failed to create HTTP client")?;

    // Fetch catalogs
    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to fetch from STAC Index API")?;

    if !response.status().is_success() {
        anyhow::bail!("HTTP {}: {}", response.status(), url);
    }

    // Parse JSON response
    let json: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse STAC Index response")?;

    // Extract catalogs from response
    let catalogs = parse_stac_index_response(&json).context("Failed to parse catalog data")?;

    Ok(catalogs)
}

/// Parse STAC Index API JSON response and extract catalogs
fn parse_stac_index_response(json: &serde_json::Value) -> Result<Vec<StacIndexCatalog>> {
    let mut catalogs = Vec::new();

    // STAC Index API returns catalogs as root-level array
    let catalog_list = if let Some(arr) = json.as_array() {
        arr.clone()
    } else {
        anyhow::bail!("Expected array response from STAC Index API");
    };

    // Parse each catalog entry
    for item in catalog_list {
        // Extract fields from catalog object
        // STAC Index API schema:
        // id (numeric), slug (string), title, url, summary, isApi (bool), etc.

        let slug = item.get("slug").and_then(|v| v.as_str()).unwrap_or("");

        if slug.is_empty() {
            continue; // Skip entries without slug
        }

        let title = item
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or(slug)
            .to_string();

        // Use "summary" field from STAC Index if available, otherwise omit
        let description = item.get("summary").and_then(|v| v.as_str()).map(|s| {
            // Truncate at first newline or make more concise
            s.lines().next().unwrap_or(s).to_string()
        });

        let url = item
            .get("url")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        catalogs.push(StacIndexCatalog {
            id: slug.to_string(),
            title,
            description,
            url,
        });
    }

    // Sort by title for consistent display
    catalogs.sort_by(|a, b| a.title.cmp(&b.title));

    Ok(catalogs)
}

pub fn handle(action: StacCommands, compress: bool) -> Result<()> {
    match action {
        StacCommands::Search {
            catalog,
            bbox,
            datetime,
            collections,
            limit,
        } => {
            let cat = StacCatalog::from_str_or_url(&catalog);
            let pb = spinner("Searching STAC catalog...");
            let client = StacClientBlocking::new(cat, StacClientOptions::default())
                .context("Failed to create STAC client")?;

            let mut params = StacSearchParams::new().limit(limit);
            if let Some(ref b) = bbox {
                let bb = parse_bbox(b)?;
                params = params.bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y);
            }
            if let Some(ref dt) = datetime {
                params = params.datetime(dt);
            }
            if let Some(ref cols) = collections {
                let c: Vec<&str> = cols.split(',').map(|s| s.trim()).collect();
                params = params.collections(&c);
            }

            let results = client.search(&params).context("STAC search failed")?;
            pb.finish_and_clear();

            println!(
                "Found {} items (matched: {})",
                results.len(),
                results
                    .number_matched
                    .map_or("?".to_string(), |n| n.to_string())
            );
            println!();

            for item in &results.features {
                let dt = item.properties.datetime.as_deref().unwrap_or("-");
                let cc = item
                    .properties
                    .eo_cloud_cover
                    .map(|c| format!("{:.1}%", c))
                    .unwrap_or_else(|| "-".to_string());
                let col = item.collection.as_deref().unwrap_or("-");
                let asset_keys: Vec<&str> = item.assets.keys().map(|k| k.as_str()).collect();

                println!("  {} [{}]", item.id, col);
                println!("    datetime: {}  cloud: {}", dt, cc);
                println!("    assets: {}", asset_keys.join(", "));
            }

            if results.has_next() {
                println!("\n  (more results available -- increase --limit to fetch more)");
            }
        }

        StacCommands::Fetch {
            catalog,
            bbox,
            collection,
            asset,
            datetime,
            variable,
            time_step,
            output,
        } => {
            let cat = StacCatalog::from_str_or_url(&catalog);
            let bb = parse_bbox(&bbox)?;

            let mut params = StacSearchParams::new()
                .bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y)
                .collections(&[collection.as_str()])
                .limit(1);
            if let Some(ref dt) = datetime {
                params = params.datetime(dt);
            }

            let pb = spinner("Searching STAC catalog...");
            let client = StacClientBlocking::new(cat, StacClientOptions::default())
                .context("Failed to create STAC client")?;
            let results = client.search(&params).context("STAC search failed")?;

            let item = results
                .features
                .first()
                .ok_or_else(|| anyhow::anyhow!("No items found matching the search criteria"))?;
            pb.finish_and_clear();

            println!(
                "Item: {} [{}]",
                item.id,
                item.collection.as_deref().unwrap_or("-")
            );

            // Determine asset key
            let asset_key = if let Some(ref k) = asset {
                k.clone()
            } else {
                // Try COG first, then Zarr
                if let Some((k, _)) = item.first_cog_asset() {
                    println!("Auto-detected COG asset: {}", k);
                    k.clone()
                } else if let Some((k, _)) = item.first_zarr_asset() {
                    println!("Auto-detected Zarr asset: {}", k);
                    k.clone()
                } else {
                    anyhow::bail!(
                        "No COG or Zarr asset found. Specify --asset explicitly. Available: {}",
                        item.assets.keys().cloned().collect::<Vec<_>>().join(", ")
                    );
                }
            };

            let stac_asset = item.asset(&asset_key).ok_or_else(|| {
                anyhow::anyhow!(
                    "Asset '{}' not found. Available: {}",
                    asset_key,
                    item.assets.keys().cloned().collect::<Vec<_>>().join(", ")
                )
            })?;

            // Detect format
            let format = item.asset_format(&asset_key);

            // Sign the href if needed
            let href = client
                .sign_asset_href(&stac_asset.href, item.collection.as_deref().unwrap_or(""))
                .context("Failed to sign asset URL")?;

            match format {
                #[cfg(feature = "zarr")]
                surtgis_cloud::AssetFormat::Zarr => {
                    use surtgis_cloud::blocking::ZarrReaderBlocking;
                    use surtgis_cloud::{TimeReduction, TimeSelector, ZarrReaderOptions};

                    let var = variable.as_deref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "--variable is required for Zarr assets. \
                            Hint: the store may contain variables like 'precipitation_amount_1hour_Accumulation', 'ppt', etc."
                        )
                    })?;

                    // Extract SAS token for Azure Blob stores
                    let sas_token = href.split_once('?').map(|(_, q)| q.to_string());
                    let base_url = href.split('?').next().unwrap_or(&href);

                    let opts = ZarrReaderOptions { sas_token };

                    // Parse time step
                    let time = parse_time_step(&time_step)?;

                    let pb = spinner("Opening Zarr store...");
                    let start = Instant::now();
                    let reader = ZarrReaderBlocking::open(base_url, var, opts)
                        .context("Failed to open Zarr store")?;

                    let meta = reader.metadata();
                    println!(
                        "Zarr: {} — shape {:?}, dims {:?}",
                        meta.variable, meta.shape, meta.dimension_names
                    );
                    if let Some((t0, t1)) = &meta.time_range {
                        println!(
                            "Time range: {} to {}",
                            t0.format("%Y-%m-%d"),
                            t1.format("%Y-%m-%d")
                        );
                    }

                    pb.finish_and_clear();
                    let pb = spinner("Reading Zarr subset...");
                    let raster = reader
                        .read_bbox(&bb, &time)
                        .context("Failed to read Zarr subset")?;
                    pb.finish_and_clear();
                    let elapsed = start.elapsed();

                    let (rows, cols) = raster.shape();
                    println!("Fetched: {} x {} ({} cells)", cols, rows, raster.len());
                    write_result(&raster, &output, compress)?;
                    done("STAC Zarr fetch", &output, elapsed);
                }

                _ => {
                    // COG path (existing logic)
                    let pb = spinner("Fetching COG tiles...");
                    let start = Instant::now();
                    let opts = CogReaderOptions::default();
                    let mut reader = CogReaderBlocking::open(&href, opts)
                        .context("Failed to open remote COG")?;

                    let read_bb = {
                        use surtgis_cloud::reproject;
                        let epsg = item
                            .epsg()
                            .or_else(|| reader.metadata().crs.as_ref().and_then(|c| c.epsg()));
                        if let Some(epsg) = epsg {
                            if !reproject::is_wgs84(epsg) {
                                let reprojected = reproject::reproject_bbox_to_cog(&bb, epsg);
                                println!("Reprojected bbox to EPSG:{}", epsg);
                                reprojected
                            } else {
                                bb
                            }
                        } else {
                            bb
                        }
                    };

                    // `stac fetch` has no separate output-grid resolution (it
                    // just reads the COG at its own native resolution), so
                    // there's no ratio to compute here — full res (`None`) is
                    // the correct and only choice. (Overview selection only
                    // helps when reading is targeting a coarser output grid,
                    // e.g. `--align-to`, see `overview_for_target_resolution`.)
                    let raster: surtgis_core::Raster<f64> = reader
                        .read_bbox(&read_bb, None)
                        .context("Failed to read bounding box from COG")?;
                    pb.finish_and_clear();
                    let elapsed = start.elapsed();

                    let (rows, cols) = raster.shape();
                    println!("Fetched: {} x {} ({} cells)", cols, rows, raster.len());
                    write_result(&raster, &output, compress)?;
                    done("STAC fetch", &output, elapsed);
                }
            }
        }

        StacCommands::FetchMosaic {
            catalog,
            bbox,
            collection,
            asset,
            datetime,
            max_items,
            output,
        } => {
            let cat = StacCatalog::from_str_or_url(&catalog);
            let bb = parse_bbox(&bbox)?;

            let mut params = StacSearchParams::new()
                .bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y)
                .collections(&[collection.as_str()])
                .limit(max_items);
            if let Some(ref dt) = datetime {
                params = params.datetime(dt);
            }

            let pb = spinner("Searching STAC catalog...");
            let client_opts = StacClientOptions {
                max_items: max_items as usize,
                ..StacClientOptions::default()
            };
            let client = StacClientBlocking::new(cat, client_opts)
                .context("Failed to create STAC client")?;
            let items = client.search_all(&params).context("STAC search failed")?;
            pb.finish_and_clear();

            if items.is_empty() {
                anyhow::bail!("No items found matching the search criteria");
            }
            println!("Found {} items, fetching and mosaicking...", items.len());
            if items.len() as u32 >= max_items {
                eprintln!(
                    "  Note: hit the --max-scenes/--max-items cap of {}. More scenes may match \
                     this query; narrow --datetime/--bbox or raise --max-scenes. Each scene is \
                     held in RAM during mosaicking, so a large cap over a wide window can use \
                     several GB.",
                    max_items
                );
            }

            // Determine asset key from first item
            let asset_key = if let Some(ref k) = asset {
                k.clone()
            } else {
                let (k, _) = items[0].first_cog_asset().ok_or_else(|| {
                    anyhow::anyhow!(
                        "No COG asset found. Specify --asset explicitly. Available: {}",
                        items[0]
                            .assets
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
                println!("Auto-detected asset: {}", k);
                k.clone()
            };

            let start = Instant::now();
            let mut rasters: Vec<surtgis_core::Raster<f64>> = Vec::new();

            for (i, item) in items.iter().enumerate() {
                // Plain progress line so pipelines / non-TTY logs see progress
                // (the spinner below auto-hides when stderr is not a terminal).
                eprintln!("  [{}/{}] fetching {}", i + 1, items.len(), item.id);
                let pb = spinner(&format!(
                    "Fetching tile {} of {} [{}]...",
                    i + 1,
                    items.len(),
                    item.id
                ));

                let stac_asset = match item.asset(&asset_key) {
                    Some(a) => a,
                    None => {
                        pb.finish_and_clear();
                        eprintln!(
                            "  Warning: item {} missing asset '{}', skipping",
                            item.id, asset_key
                        );
                        continue;
                    }
                };

                let href = client
                    .sign_asset_href(&stac_asset.href, item.collection.as_deref().unwrap_or(""))
                    .context("Failed to sign asset URL")?;

                let opts = CogReaderOptions::default();
                let mut reader = match CogReaderBlocking::open(&href, opts) {
                    Ok(r) => r,
                    Err(e) => {
                        pb.finish_and_clear();
                        eprintln!(
                            "  Warning: failed to open COG for {}: {}, skipping",
                            item.id, e
                        );
                        continue;
                    }
                };

                // Auto-reproject bbox if COG is in a projected CRS
                let read_bb = {
                    use surtgis_cloud::reproject;
                    let epsg = item
                        .epsg()
                        .or_else(|| reader.metadata().crs.as_ref().and_then(|c| c.epsg()));
                    if let Some(epsg) = epsg {
                        if !reproject::is_wgs84(epsg) {
                            reproject::reproject_bbox_to_cog(&bb, epsg)
                        } else {
                            bb
                        }
                    } else {
                        bb
                    }
                };

                // Same as `stac fetch`: `fetch-mosaic` mosaics scenes at their
                // own native resolution (no separate output grid), so there's
                // no ratio to compute here — always full res.
                match reader.read_bbox::<f64>(&read_bb, None) {
                    Ok(raster) => {
                        pb.finish_and_clear();
                        let (rows, cols) = raster.shape();
                        println!(
                            "  [{}/{}] {} -- {} x {}",
                            i + 1,
                            items.len(),
                            item.id,
                            cols,
                            rows
                        );
                        rasters.push(raster);
                    }
                    Err(e) => {
                        pb.finish_and_clear();
                        eprintln!(
                            "  Warning: failed to read tile {}: {}, skipping",
                            item.id, e
                        );
                    }
                }
            }

            if rasters.is_empty() {
                anyhow::bail!("No tiles were successfully fetched");
            }

            let pb = spinner("Mosaicking tiles...");
            let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
            let result = surtgis_core::mosaic(&refs, None).context("Failed to mosaic tiles")?;
            pb.finish_and_clear();

            let elapsed = start.elapsed();
            let (rows, cols) = result.shape();
            println!(
                "Mosaic: {} tiles -> {} x {} ({} cells)",
                rasters.len(),
                cols,
                rows,
                result.len()
            );
            write_result(&result, &output, compress)?;
            done("STAC fetch-mosaic", &output, elapsed);
        }

        StacCommands::Composite {
            catalog,
            bbox,
            collection,
            asset,
            datetime,
            max_scenes,
            scl_asset: _scl_asset, // DEPRECATED: ignored (profile determines masking)
            scl_keep: _scl_keep,   // DEPRECATED: ignored (profile determines masking)
            align_to,
            naming,
            cache,
            strip_rows: cli_strip_rows,
            band_chunk_size,
            max_tile_failures,
            output,
        } => {
            // Both single- and multi-band composites run through the same
            // CompositeEngine (R8/R9): shared STAC search + mask, byte-budgeted
            // decode, and streaming output. `--asset "red,nir,..."` splits into
            // bands; a single `--asset red` is just `n_bands == 1`. This
            // retired the separate single-band path, which had no memory budget
            // (fixed strip height, one unbounded thread per tile).
            let assets: Vec<&str> = asset.split(',').map(|s| s.trim()).collect();
            return handle_multiband_composite(
                &catalog,
                &bbox,
                &collection,
                &assets,
                &datetime,
                max_scenes,
                align_to.as_ref(),
                &output,
                &naming,
                cache,
                cli_strip_rows,
                band_chunk_size,
                compress,
                max_tile_failures,
            );
        }

        StacCommands::ListCatalogs { search } => {
            println!("📚 Available STAC Catalogs\n");

            // Show search query if provided
            if let Some(ref query) = search {
                println!("🔍 Searching for: '{}'\n", query);
            }

            println!("═══════════════════════════════════════════════════════════════");
            println!("CURATED CATALOGS (reliable, actively maintained):\n");

            let mut curated_matches = 0;
            for catalog in get_known_catalogs() {
                // Skip osgeo if we're showing it separately
                if catalog.shorthand == "osgeo" {
                    continue;
                }

                // Check if matches search query
                if let Some(ref q) = search {
                    let query_lower = q.to_lowercase();
                    let matches = catalog.name.to_lowercase().contains(&query_lower)
                        || catalog.description.to_lowercase().contains(&query_lower)
                        || catalog.shorthand.to_lowercase().contains(&query_lower);
                    if !matches {
                        continue;
                    }
                }

                curated_matches += 1;
                println!("{:<6} {}", catalog.shorthand, catalog.name);
                println!("       {}", catalog.description);
                println!("       URL: {}\n", catalog.url);
            }

            // Try to fetch from STAC Index API
            println!("═══════════════════════════════════════════════════════════════");
            println!("STAC INDEX (dynamically discovered, 1000+ catalogs):\n");

            match fetch_stac_index_catalogs() {
                Ok(mut all_catalogs) => {
                    // Filter by search query if provided
                    if let Some(ref q) = search {
                        let query_lower = q.to_lowercase();
                        all_catalogs.retain(|cat| {
                            cat.title.to_lowercase().contains(&query_lower)
                                || cat.id.to_lowercase().contains(&query_lower)
                                || cat
                                    .description
                                    .as_ref()
                                    .map(|d| d.to_lowercase().contains(&query_lower))
                                    .unwrap_or(false)
                        });
                    }

                    if all_catalogs.is_empty() {
                        if search.is_some() {
                            println!("  ⚠️ No STAC Index catalogs match your search");
                        } else {
                            println!("  ⚠️ No catalogs found in STAC Index");
                        }
                    } else {
                        let total = all_catalogs.len();
                        let display_count = all_catalogs.len().min(15);
                        println!(
                            "  Showing {} of {} available catalogs:\n",
                            display_count, total
                        );
                        for (i, catalog_info) in all_catalogs.iter().take(display_count).enumerate()
                        {
                            let idx = i + 1;
                            println!("  [{}] {} ({})", idx, catalog_info.title, catalog_info.id);
                            if let Some(desc) = &catalog_info.description {
                                let preview = truncate_utf8(desc, 70);
                                println!("      {}", preview);
                            }
                            if let Some(url) = &catalog_info.url {
                                println!("      🔗 {}", url);
                            }
                            println!();
                        }
                        if total > 15 {
                            println!("  ... and {} more catalogs in STAC Index", total - 15);
                        }
                    }
                }
                Err(e) => {
                    if search.is_some() {
                        eprintln!("  ⚠️ Could not search STAC Index: {}", e);
                    } else {
                        eprintln!("  ⚠️ Could not fetch STAC Index: {}", e);
                    }
                    eprintln!("     (This is optional - you can still use curated catalogs above)");
                }
            }

            println!("\n═══════════════════════════════════════════════════════════════");
            println!("💡 You can also use any custom STAC API URL:");
            println!("   surtgis stac search --catalog https://your-stac-api.com/v1 ...");
            println!("\n💡 Search for specific data types:");
            println!("   surtgis stac list-catalogs --search sentinel-2");
            println!("   surtgis stac list-catalogs --search dem");
            println!("   surtgis stac list-catalogs --search thermal\n");
        }

        StacCommands::ListCollections { catalog } => {
            println!("📍 Catalog: {}\n", catalog);

            let collections = get_catalog_collections(&catalog);
            println!("📊 Available Collections:\n");

            let is_unknown = collections.len() == 1 && collections[0].0 == "(unknown)";
            if is_unknown {
                println!("  {}", collections[0].1);
            } else {
                for (id, desc) in &collections {
                    println!("  • {} - {}", id, desc);
                }
                println!("\n✨ Total: {} collections", collections.len());
            }

            println!(
                "\n💡 Usage: surtgis stac composite --catalog {} --collection <id> --asset <band> ...",
                catalog
            );
        }

        StacCommands::TimeSeries {
            catalog,
            bbox,
            collection,
            asset,
            datetime,
            interval,
            scl_asset,
            max_scenes,
            align_to,
            output,
        } => {
            // Multi-band support: --asset "red,nir,swir16,blue"
            let assets: Vec<&str> = asset.split(',').map(|s| s.trim()).collect();
            if assets.len() > 1 {
                println!("Multi-band time series: {} bands", assets.len());
                let total_start = Instant::now();
                for (band_idx, band) in assets.iter().enumerate() {
                    let band_outdir = output.join(band);
                    println!(
                        "\n[{}/{}] Band: {} → {}",
                        band_idx + 1,
                        assets.len(),
                        band,
                        band_outdir.display()
                    );
                    handle_time_series(
                        &catalog,
                        &bbox,
                        &collection,
                        band,
                        &datetime,
                        &interval,
                        &scl_asset,
                        max_scenes,
                        align_to.as_ref(),
                        &band_outdir,
                        compress,
                    )?;
                }
                println!(
                    "\nAll {} bands complete in {:.1?}",
                    assets.len(),
                    total_start.elapsed()
                );
            } else {
                handle_time_series(
                    &catalog,
                    &bbox,
                    &collection,
                    &asset,
                    &datetime,
                    &interval,
                    &scl_asset,
                    max_scenes,
                    align_to.as_ref(),
                    &output,
                    compress,
                )?;
            }
        }

        #[cfg(feature = "zarr")]
        StacCommands::DownloadClimate {
            catalog,
            bbox,
            collection,
            variable,
            datetime,
            aggregate,
            output,
        } => {
            use surtgis_cloud::blocking::ZarrReaderBlocking;
            use surtgis_cloud::zarr_auth::abfs_to_https_with_account;
            use surtgis_cloud::{AggMethod, TimeReduction, ZarrReaderOptions};

            let cat = StacCatalog::from_str_or_url(&catalog);
            let bb = parse_bbox(&bbox)?;

            // Parse aggregation
            let (interval_type, agg_method) = parse_aggregate(&aggregate)?;

            // Parse datetime range
            let (dt_start, dt_end) = parse_datetime_range(&datetime)?;

            // Search STAC for the collection item
            let pb = spinner("Searching STAC catalog...");
            let client = StacClientBlocking::new(cat, StacClientOptions::default())
                .context("Failed to create STAC client")?;

            let params = StacSearchParams::new()
                .bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y)
                .collections(&[collection.as_str()])
                .datetime(&datetime)
                .limit(1);

            let results = client.search(&params).context("STAC search failed")?;
            let item = results.features.first().ok_or_else(|| {
                anyhow::anyhow!(
                    "No items found for collection '{}' in range '{}'",
                    collection,
                    datetime
                )
            })?;
            pb.finish_and_clear();

            println!(
                "Item: {} [{}]",
                item.id,
                item.collection.as_deref().unwrap_or("-")
            );

            // Find the asset matching the variable name
            let stac_asset = item.asset(&variable).ok_or_else(|| {
                anyhow::anyhow!(
                    "Asset '{}' not found. Available: {}",
                    variable,
                    item.assets.keys().cloned().collect::<Vec<_>>().join(", ")
                )
            })?;

            // Get collection-level auth for Zarr
            let auth = client
                .get_collection_zarr_auth(&collection)
                .context("Failed to get collection auth")?;

            let (sas_token, store_url) = if let Some((token, account, _container)) = auth {
                let url = abfs_to_https_with_account(&stac_asset.href, Some(&account));
                (Some(token), url)
            } else {
                (None, stac_asset.href.clone())
            };

            let opts = ZarrReaderOptions { sas_token };

            // Open the Zarr store
            let pb = spinner("Opening Zarr store...");
            let reader = ZarrReaderBlocking::open(&store_url, &variable, opts)
                .context("Failed to open Zarr store")?;

            let meta = reader.metadata();
            println!(
                "Variable: {} — shape {:?}, dims {:?}",
                meta.variable, meta.shape, meta.dimension_names
            );
            if let Some((t0, t1)) = &meta.time_range {
                println!(
                    "Time range: {} to {}",
                    t0.format("%Y-%m-%d"),
                    t1.format("%Y-%m-%d")
                );
            }
            pb.finish_and_clear();

            // Generate time intervals
            let intervals = generate_intervals(dt_start, dt_end, interval_type);
            println!(
                "Downloading {} intervals ({}) for {}...",
                intervals.len(),
                aggregate,
                variable
            );

            // Create output directory
            std::fs::create_dir_all(&output).context("Failed to create output directory")?;

            let total_start = Instant::now();
            for (i, (int_start, int_end, label)) in intervals.iter().enumerate() {
                let time = if agg_method.is_some() {
                    TimeReduction::Aggregate {
                        start: *int_start,
                        end: *int_end,
                        method: agg_method.unwrap(),
                    }
                } else {
                    TimeReduction::Single(surtgis_cloud::TimeSelector::Nearest(*int_start))
                };

                let pb = spinner(&format!("[{}/{}] {}", i + 1, intervals.len(), label));
                match reader.read_bbox(&bb, &time) {
                    Ok(raster) => {
                        let suffix = agg_method
                            .map(|m| match m {
                                AggMethod::Mean => "mean",
                                AggMethod::Sum => "sum",
                                AggMethod::Min => "min",
                                AggMethod::Max => "max",
                            })
                            .unwrap_or("value");
                        let filename = format!("{}_{}.tif", label, suffix);
                        let out_path = output.join(&filename);
                        write_result(&raster, &out_path, compress)?;
                        let (rows, cols) = raster.shape();
                        pb.finish_and_clear();
                        println!("  {} — {}x{}", filename, cols, rows);
                    }
                    Err(e) => {
                        pb.finish_and_clear();
                        eprintln!("  Warning: {} — {}", label, e);
                    }
                }
            }

            let elapsed = total_start.elapsed();
            println!(
                "\nDone: {} intervals written to {} in {:.1?}",
                intervals.len(),
                output.display(),
                elapsed
            );
        }

        #[cfg(not(feature = "zarr"))]
        StacCommands::DownloadClimate { .. } => {
            anyhow::bail!("Zarr support not enabled. Recompile with --features zarr");
        }
    }

    Ok(())
}

/// Fetch and composite a single band from ANY STAC collection.
///
/// Features:
/// - **Catalog-agnostic**: Works with any STAC API (not just Planetary Computer)
/// - **Auto-detects**: Available bands, cloud masking strategy, CRS, resolution
/// - **Cloud masking**: Automatically selected (SCL, QA_PIXEL, or none for SAR)
/// - **Band matching**: Fuzzy matching on band names (B04 ≈ red ≈ banda_roja)
/// - **Multi-scene compositing**: Median stack across scenes
/// - **Grid alignment**: Optional resampling to reference DEM
/// - **No disk writes**: Returns raster in memory
pub fn fetch_stac_band(
    catalog: &str,
    bbox: &str,
    collection: &str,
    asset: &str,
    datetime: &str,
    max_scenes: usize,
    align_to: Option<&surtgis_core::Raster<f64>>,
) -> Result<surtgis_core::Raster<f64>> {
    use surtgis_core::mosaic;

    // Target output resolution, if we're aligning to a reference grid: lets
    // reads use a COG overview instead of full resolution when the reference
    // grid is much coarser than the source COG (see
    // `overview_for_target_resolution` / `select_overview` fix).
    let out_pixel_size: Option<f64> = align_to.map(|r| r.transform().pixel_width.abs());

    let cat = StacCatalog::from_str_or_url(catalog);
    let bb = parse_bbox(bbox)?;

    // Estimate spatial tile count from bbox size (same logic as composite)
    let bbox_w_km = (bb.max_x - bb.min_x).abs() * 111.0;
    let bbox_h_km = (bb.max_y - bb.min_y).abs() * 111.0;
    let tiles_est = (((bbox_w_km / 60.0).ceil() as usize).max(1))
        * (((bbox_h_km / 60.0).ceil() as usize).max(1));
    let search_limit = ((tiles_est * max_scenes * 5).max(1000)).min(10000) as u32;

    // Page size capped per catalog (PC=1000, ES=250, custom=250)
    let params = StacSearchParams::new()
        .bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y)
        .collections(&[collection])
        .datetime(datetime)
        .limit(search_limit.min(cat.max_page_size()));

    let pb = spinner(&format!(
        "Searching for {} scenes (limit={})...",
        collection, search_limit
    ));
    let client_opts = StacClientOptions {
        max_items: search_limit as usize,
        ..StacClientOptions::default()
    };
    let client =
        StacClientBlocking::new(cat, client_opts).context("Failed to create STAC client")?;
    let items = client.search_all(&params).context("STAC search failed")?;
    pb.finish_and_clear();

    if items.is_empty() {
        anyhow::bail!("No items found for {} in bbox {}", collection, bbox);
    }

    // Introspect first item to auto-detect collection schema
    let pb = spinner("Introspecting collection schema...");
    let schema = StacCollectionSchema::from_stac_item(collection, &items[0])
        .context("Failed to introspect STAC collection")?;
    pb.finish_and_clear();

    eprintln!("📊 Collection: {}", schema.collection_name);
    eprintln!("   Available bands: {}", schema.format_bands());
    eprintln!(
        "   Cloud masking: {}",
        match &schema.cloud_mask_type {
            CloudMaskType::Categorical { asset, num_classes } =>
                format!("Categorical {} ({} classes)", asset, num_classes),
            CloudMaskType::Bitmask { asset, bits } =>
                format!("Bitmask {} ({} bits)", asset, bits.len()),
            CloudMaskType::None => "None (SAR)".to_string(),
        }
    );

    // Find best matching band
    let band_info = schema.find_band_by_name(asset).context(format!(
        "Band '{}' not found. Available: {}",
        asset,
        schema.format_bands()
    ))?;
    eprintln!("   Band matched: {} → {}", asset, band_info.asset_key);

    // Create cloud masking strategy based on auto-detected type
    let cloud_mask_strategy = create_cloud_mask_strategy(&schema.cloud_mask_type);

    // Display info about found items
    for (idx, item) in items.iter().take(max_scenes).enumerate() {
        let date = item.properties.datetime.as_deref().unwrap_or("-");
        let cloud_cover = item.properties.eo_cloud_cover.unwrap_or(0.0);
        if idx < 3 {
            eprintln!(
                "  Scene {}: {} [{}% cloud]",
                idx + 1,
                date,
                cloud_cover as u32
            );
        }
    }
    if items.len() > 3 {
        eprintln!("  ... and {} more scenes", items.len() - 3);
    }

    // Download and composite scenes
    let mut rasters: Vec<surtgis_core::Raster<f64>> = Vec::new();
    let mut successful = 0;

    for (i, item) in items.iter().take(max_scenes).enumerate() {
        let date = item.properties.datetime.as_deref().unwrap_or("-");
        let cloud = item.properties.eo_cloud_cover.unwrap_or(0.0);
        let pb = spinner(&format!(
            "Downloading {}/{}: {} [{}% cloud, {}]...",
            i + 1,
            max_scenes,
            item.id,
            cloud as u32,
            date
        ));

        // Resolve data asset
        let data_asset = resolve_asset_key(item, &band_info.asset_key).and_then(|(_, a)| {
            client
                .sign_asset_href(&a.href, item.collection.as_deref().unwrap_or(""))
                .ok()
                .map(|h| (h, item.epsg()))
        });

        // Resolve cloud mask asset (if applicable)
        let mask_href = schema.cloud_mask_asset.as_ref().and_then(|mask_name| {
            resolve_asset_key(item, mask_name).and_then(|(_, a)| {
                client
                    .sign_asset_href(&a.href, item.collection.as_deref().unwrap_or(""))
                    .ok()
            })
        });

        let Some((data_href, epsg)) = data_asset else {
            pb.finish_and_clear();
            eprintln!(
                "  ⚠️ Skipping {}: asset {} not found",
                item.id, band_info.asset_key
            );
            continue;
        };

        // Read data
        let opts = CogReaderOptions::default();
        let mut reader = match CogReaderBlocking::open(&data_href, opts) {
            Ok(r) => r,
            Err(e) => {
                pb.finish_and_clear();
                eprintln!("  ⚠️ Skipping {}: failed to open COG: {}", item.id, e);
                continue;
            }
        };

        // Reproject bbox if needed
        let read_bb = {
            use surtgis_cloud::reproject;
            if let Some(epsg) = epsg {
                if !reproject::is_wgs84(epsg) {
                    reproject::reproject_bbox_to_cog(&bb, epsg)
                } else {
                    bb
                }
            } else {
                bb
            }
        };

        let data_overview = out_pixel_size.and_then(|out_px| {
            let meta = reader.metadata();
            overview_for_target_resolution(
                meta.width,
                meta.height,
                meta.geo_transform.pixel_width.abs(),
                &reader.overviews(),
                out_px,
            )
        });
        let mut raster = match reader.read_bbox::<f64>(&read_bb, data_overview) {
            Ok(r) => r,
            Err(e) => {
                pb.finish_and_clear();
                eprintln!("  ⚠️ Skipping {}: failed to read bbox: {}", item.id, e);
                continue;
            }
        };

        let (rrows, rcols) = raster.shape();
        pb.println(format!("  → Read {} × {} pixels", rcols, rrows));

        // Apply cloud masking using auto-detected strategy
        if let Some(mask_href) = &mask_href {
            let mask_desc = match &schema.cloud_mask_type {
                CloudMaskType::Categorical { asset, .. } => asset.clone(),
                CloudMaskType::Bitmask { asset, .. } => asset.clone(),
                CloudMaskType::None => "None".to_string(),
            };
            pb.println(format!("  → Applying {} cloud mask...", mask_desc));

            // Read mask asset as f64
            let mask_opts = CogReaderOptions::default();
            let mut mask_reader = match CogReaderBlocking::open(mask_href, mask_opts) {
                Ok(r) => r,
                Err(_) => {
                    // Mask is optional, continue without masking
                    pb.finish_and_clear();
                    rasters.push(raster);
                    successful += 1;
                    continue;
                }
            };

            // Read mask as f64 and apply strategy
            let mask_overview = out_pixel_size.and_then(|out_px| {
                let meta = mask_reader.metadata();
                overview_for_target_resolution(
                    meta.width,
                    meta.height,
                    meta.geo_transform.pixel_width.abs(),
                    &mask_reader.overviews(),
                    out_px,
                )
            });
            if let Ok(mask_raster) = mask_reader.read_bbox::<f64>(&read_bb, mask_overview) {
                raster = cloud_mask_strategy
                    .mask(&raster, &mask_raster)
                    .unwrap_or(raster); // If masking fails, use original
            }
        }

        pb.finish_and_clear();
        successful += 1;
        rasters.push(raster);

        if rasters.len() >= max_scenes {
            break;
        }
    }

    if rasters.is_empty() {
        anyhow::bail!(
            "Failed to download any {} {} scenes (0/{} successful)",
            collection,
            asset,
            items.len()
        );
    }

    eprintln!("  ✅ Successfully loaded {} scenes", successful);

    // Composite all scenes
    let pb = spinner(&format!(
        "Compositing {} scenes (median stack)...",
        rasters.len()
    ));
    let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
    let result = mosaic(&refs, None).context("Failed to mosaic scenes")?;
    let (comp_rows, comp_cols) = result.shape();
    pb.finish_and_clear();
    eprintln!("  ✓ Composite: {} × {} pixels", comp_cols, comp_rows);

    // Align to reference grid if provided
    let final_result = if let Some(reference) = align_to {
        let pb = spinner("Aligning to DEM grid (bilinear resample)...");
        let aligned = surtgis_core::resample_to_grid(
            &result,
            reference,
            surtgis_core::ResampleMethod::Bilinear,
        )
        .context("Failed to resample to DEM grid")?;
        let (out_rows, out_cols) = aligned.shape();
        pb.finish_and_clear();
        eprintln!("  ✓ Aligned: {} × {} pixels", out_cols, out_rows);
        aligned
    } else {
        result
    };

    eprintln!("  ✓ {} band ready for indices", asset);
    Ok(final_result)
}

// ---------------------------------------------------------------------------
// COG tile cache: skip HTTP downloads for previously-fetched tiles
// ---------------------------------------------------------------------------

/// Compute a cache path for a COG tile, based on the base URL (without SAS query params) and bbox.
fn cog_cache_path(href: &str, bb: &BBox) -> std::path::PathBuf {
    let hex = cog_cache_key(href, bb);

    let cache_dir = std::env::var("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
        .join("surtgis")
        .join("cog");

    cache_dir
        .join(&hex[..2])
        .join(&hex[2..4])
        .join(format!("{}.tif", &hex[4..]))
}

// ---------------------------------------------------------------------------
// Multi-band composite: single-pass, shared STAC search + shared mask
// ---------------------------------------------------------------------------

/// Read current process RSS in MB by parsing /proc/self/status. Returns 0 on
/// non-Linux or if the file isn't readable. Used for diagnostic logging at
/// strip / phase / band-chunk transitions so we can localise any RAM growth
/// to a specific phase of the pipeline.
fn read_rss_mb() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<usize>().ok())
                .map(|kb| kb / 1024)
        })
        .unwrap_or(0)
}

/// Background watchdog that samples /proc/self/status every `sample_interval`
/// and tracks the maximum RSS seen since the last reset. Lets us report
/// intra-chunk peaks that the boundary-only [ram] log lines miss.
///
/// Postdoc observed +815 MB undersample on Maule v0.7.0 (logged 12.3 GB vs
/// real-time 13.1 GB during the same chunk) — see
/// `BUG_RAM_V070_BUDGET_VS_REAL_MAULE_PC.md`. This addresses that gap.
struct RssPeakTracker {
    inner: Arc<RssPeakInner>,
}

struct RssPeakInner {
    peak_mb: AtomicUsize,
    stop: AtomicBool,
}

impl RssPeakTracker {
    fn start(sample_interval: Duration) -> Self {
        let inner = Arc::new(RssPeakInner {
            peak_mb: AtomicUsize::new(read_rss_mb()),
            stop: AtomicBool::new(false),
        });
        let clone = inner.clone();
        std::thread::Builder::new()
            .name("surtgis-rss-watchdog".into())
            .spawn(move || {
                while !clone.stop.load(Ordering::Relaxed) {
                    let cur = read_rss_mb();
                    clone.peak_mb.fetch_max(cur, Ordering::Relaxed);
                    std::thread::sleep(sample_interval);
                }
            })
            .ok();
        Self { inner }
    }

    /// Atomically swap the tracked peak with the current RSS and return the
    /// previous peak. Call at log boundaries so each report covers the window
    /// since the previous call.
    fn take_peak(&self) -> usize {
        let cur = read_rss_mb();
        self.inner.peak_mb.swap(cur, Ordering::Relaxed)
    }
}

impl Drop for RssPeakTracker {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::Relaxed);
    }
}

/// Resolves band/mask keys via the CLI's band-alias table.
///
/// Shared by every engine-routed composite (`handle_multiband_composite` and
/// `handle_time_series`), so it lives at module scope.
struct CliResolver;
impl surtgis_cloud::composite::AssetResolver for CliResolver {
    fn resolve(&self, item: &surtgis_cloud::stac_models::StacItem, key: &str) -> Option<String> {
        resolve_asset_key(item, key).map(|(_, a)| a.href.clone())
    }
}

/// Applies the collection's cloud-mask strategy.
struct CliMask(Arc<dyn CloudMaskStrategy>);
impl surtgis_cloud::composite::MaskApplier for CliMask {
    fn apply(
        &self,
        data: &surtgis_core::Raster<f64>,
        mask: &surtgis_core::Raster<f64>,
    ) -> surtgis_core::Raster<f64> {
        self.0.mask(data, mask).unwrap_or_else(|_| data.clone())
    }
}

/// Run one composite through the RAM-budgeted [`CompositeEngine`], streaming
/// each requested band straight to its `band_paths[i]` GeoTIFF.
///
/// This is the shared core behind both `stac composite` (multi-band, single
/// pass) and `stac time-series` (one call per temporal interval). The engine
/// enforces the RAM budget (`SURTGIS_RAM_BUDGET_GB`, default 16 GB) and, when
/// `align_to` is given, produces its output *directly* on that reference grid —
/// so callers must NOT resample post-hoc.
///
/// Masking is driven by [`CollectionProfile`]: Sentinel-2 (SCL), Landsat
/// (QA_PIXEL) and Sentinel-1 (none). Collections outside those three get no
/// mask (fallback).
#[allow(clippy::too_many_arguments)]
fn run_engine_composite(
    catalog: &str,
    collection: &str,
    bbox: &str,
    band_names: &[&str],
    datetime: &str,
    max_scenes: usize,
    align_to: Option<&std::path::PathBuf>,
    band_paths: &[std::path::PathBuf],
    band_chunk_size: usize,
    strip_rows_cfg: usize,
    use_cache: bool,
    compress: bool,
    max_tile_failures: usize,
    progress: &mut dyn surtgis_cloud::composite::CompositeProgress,
) -> Result<surtgis_cloud::composite::CompositeReport> {
    use surtgis_cloud::composite::{CompositeEngine, CompositeSpec, OutputGrid};

    let profile = CollectionProfile::from_collection_name(collection)?;

    // --- Build the spec ---
    let bb = parse_bbox(bbox)?;
    let align_grid = match align_to {
        Some(path) => {
            let reference: surtgis_core::Raster<f64> =
                surtgis_core::io::read_geotiff(path, None)
                    .context("Failed to read alignment reference raster")?;
            eprintln!(
                "  Using reference grid from --align-to: {}x{}",
                reference.shape().1,
                reference.shape().0
            );
            Some(OutputGrid::from_reference(&reference))
        }
        None => None,
    };
    let budget_gb = std::env::var("SURTGIS_RAM_BUDGET_GB")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|gb| *gb > 0.5)
        .unwrap_or(16.0);
    let spec = CompositeSpec {
        catalog: catalog.to_string(),
        collection: collection.to_string(),
        bbox_wgs84: bb,
        band_keys: band_names.iter().map(|s| s.to_string()).collect(),
        mask_key: profile.mask_asset_name().map(|s| s.to_string()),
        datetime: datetime.to_string(),
        max_scenes,
        align_grid,
        strip_rows: strip_rows_cfg,
        band_chunk_size,
        budget_gb,
        max_tile_failures,
        use_cache,
    };

    // --- Injected collaborators (dependency inversion) ---
    let cloud_mask_strategy: Arc<dyn CloudMaskStrategy> = match &profile {
        CollectionProfile::Sentinel2L2A {
            cloud_mask_strategy,
        }
        | CollectionProfile::LandsatC2L2 {
            cloud_mask_strategy,
        } => cloud_mask_strategy.clone(),
        CollectionProfile::Sentinel1RTC => Arc::new(NoCloudMask),
    };
    let mask_applier = CliMask(cloud_mask_strategy);

    // Stream each band strip straight to disk (no persistent
    // `n_bands × rows × cols × 8` RAM buffers).
    let mut sink = crate::composite_sink::StreamingTiffSink::new(band_paths.to_vec(), compress);

    // --- Run the engine, then assemble the streamed scratch into GeoTIFFs ---
    let mut engine = CompositeEngine::new(spec).context("Failed to create composite engine")?;
    let report = engine
        .run(&CliResolver, &mask_applier, &mut sink, progress)
        .context("Composite run failed")?;
    sink.finish()
        .context("Failed to assemble composite output files")?;

    Ok(report)
}

/// Handle multi-band `stac composite` in a single pass.
///
/// Shares the STAC search and SCL mask download across all bands, avoiding
/// redundant HTTP requests. For N bands, this is ~N× faster than running
/// N independent single-band composites.
fn handle_multiband_composite(
    catalog: &str,
    bbox_str: &str,
    collection: &str,
    band_names: &[&str],
    datetime: &str,
    max_scenes: usize,
    align_to: Option<&std::path::PathBuf>,
    output: &std::path::Path,
    naming: &str,
    use_cache: bool,
    strip_rows_cfg: usize,
    band_chunk_size: usize,
    compress: bool,
    max_tile_failures: usize,
) -> Result<()> {
    use surtgis_cloud::composite::{CompositeProgress, OutputGrid, StripPlan};

    let n_bands = band_names.len();
    println!(
        "Multi-band composite: {} bands [{}]",
        n_bands,
        band_names.join(", ")
    );

    let profile = CollectionProfile::from_collection_name(collection)?;
    eprintln!(
        "📷 Collection: {} (mask: {:?})",
        profile.description(),
        profile.mask_asset_name()
    );
    if use_cache {
        let sample = cog_cache_path("sample", &BBox::new(0.0, 0.0, 1.0, 1.0));
        if let Some(root) = sample.ancestors().nth(3) {
            eprintln!("📦 COG cache enabled: {}", root.display());
        }
    }

    // Resolve the per-band output paths up front (the engine streams each band
    // straight to disk via `run_engine_composite`).
    let stem = output.file_stem().unwrap_or_default().to_string_lossy();
    let use_asset_naming = naming.eq_ignore_ascii_case("asset");
    let band_paths: Vec<std::path::PathBuf> = if n_bands == 1 {
        // Single band: write exactly the requested `output` path, matching the
        // historical `stac composite --asset red -o out.tif` behaviour (the
        // per-band naming below is only meaningful for multi-band runs).
        vec![output.to_path_buf()]
    } else {
        band_names
            .iter()
            .map(|band_name| {
                if use_asset_naming {
                    output.with_file_name(format!("{}.tif", band_name))
                } else {
                    output.with_file_name(format!("{}_{}.tif", stem, band_name))
                }
            })
            .collect()
    };
    /// Bridges engine progress hooks to the CLI's existing stdout/stderr
    /// reporting and RAM diagnostics (RSS logging, intra-strip peak tracking,
    /// and mimalloc page reclamation at strip boundaries — allocator control
    /// belongs to the binary, not the library).
    struct CliProgress {
        num_strips: usize,
        n_bands: usize,
        band_names: Vec<String>,
        rss_peak: RssPeakTracker,
    }
    impl CompositeProgress for CliProgress {
        fn search_done(&mut self, items: usize, dates_total: usize, dates_used: usize) {
            println!(
                "Found {} items across {} dates (using {} dates)",
                items, dates_total, dates_used
            );
        }
        fn scenes_resolved(&mut self, n_scenes: usize, n_bands: usize) {
            eprintln!("  Resolved {} scenes with {} bands each", n_scenes, n_bands);
        }
        fn grid_ready(&mut self, grid: &OutputGrid, n_scenes: usize) {
            println!(
                "Output grid: {} x {} ({:.1}M cells), {} dates, {} bands",
                grid.cols,
                grid.rows,
                (grid.cols * grid.rows) as f64 / 1e6,
                n_scenes,
                self.n_bands
            );
        }
        fn plan_ready(&mut self, plan: &StripPlan, budget_gb: f64) {
            self.num_strips = plan.num_strips;
            let bd = &plan.breakdown;
            if plan.capped {
                eprintln!(
                    "⚠ strip_rows capped → {}: fitting the held set within {:.1} GB budget",
                    plan.strip_rows, budget_gb
                );
            }
            eprintln!(
                "  RAM budget ({:.1} GB, band_chunk={}) — output {:.1} | held {:.1} | decode {:.1} GB (strip_rows={}) → ~{:.1} GB planned; decode bounded at runtime by a {:.0} MB byte budget",
                budget_gb,
                plan.band_chunk,
                bd.output_gb,
                bd.held_gb,
                bd.decode_gb,
                plan.strip_rows,
                bd.estimated_total_gb,
                plan.decode_budget_bytes as f64 / 1e6,
            );
            eprintln!("  (override budget with SURTGIS_RAM_BUDGET_GB=<N>)");
        }
        fn strip_started(&mut self, idx: usize, num: usize, row_start: usize, row_end: usize) {
            print!(
                "\r  Strip {}/{} (rows {}-{}, {} bands)...",
                idx + 1,
                num,
                row_start,
                row_end,
                self.n_bands
            );
        }
        fn strip_finished(&mut self, idx: usize, num: usize) {
            // Force mimalloc to return idle segments to the OS at each strip
            // boundary (BUG_RAM_V070 item #6: strip-pair peaks from retained
            // free segments). No-op when mimalloc isn't the global allocator.
            #[cfg(all(not(target_arch = "wasm32"), feature = "mimalloc"))]
            {
                // SAFETY: mi_collect is the public mimalloc API, safe from any
                // thread when mimalloc is the global allocator.
                unsafe { libmimalloc_sys::mi_collect(true) };
            }
            eprintln!(
                "[ram] strip {}/{} finished: RSS={} MB",
                idx + 1,
                num,
                read_rss_mb()
            );
        }
        fn band_contribution(&mut self, band_idx: usize, contributed: usize, total: usize) {
            if band_idx == 0 || band_idx == self.n_bands - 1 {
                let name = self
                    .band_names
                    .get(band_idx)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                eprintln!(
                    "  band '{}': {} / {} scenes contributed data",
                    name, contributed, total
                );
            }
        }
        fn ram_checkpoint(&mut self, label: &str) {
            let peak = self.rss_peak.take_peak();
            eprintln!(
                "[ram] {}: RSS={} MB, peak_intra={} MB",
                label,
                read_rss_mb(),
                peak
            );
        }
        fn tile_failed(&mut self, msg: &str) {
            eprintln!("    [cog] tile FAILED after retries: {}", msg);
        }
    }
    let mut progress = CliProgress {
        num_strips: 0,
        n_bands,
        band_names: band_names.iter().map(|s| s.to_string()).collect(),
        rss_peak: RssPeakTracker::start(Duration::from_secs(2)),
    };

    // --- Run the engine (spec build + resolver/mask + streamed sink live in
    // the shared `run_engine_composite` helper, which also assembles the
    // per-band GeoTIFFs before returning) ---
    let start = Instant::now();
    eprintln!("[ram] baseline before composite: RSS={} MB", read_rss_mb());
    let report = run_engine_composite(
        catalog,
        collection,
        bbox_str,
        band_names,
        datetime,
        max_scenes,
        align_to,
        &band_paths,
        band_chunk_size,
        strip_rows_cfg,
        use_cache,
        compress,
        max_tile_failures,
        &mut progress,
    )?;
    println!(); // newline after the \r strip progress

    if report.failed_tiles > 0 {
        eprintln!(
            "⚠ {} tiles failed after retries ({} scenes affected); those regions were \
             gap-filled from neighbouring pixels/scenes. Last error: {}",
            report.failed_tiles,
            report.failed_dates,
            report.last_error.as_deref().unwrap_or("(none)")
        );
    }

    // The helper already assembled the N output GeoTIFFs; report their sizes.
    let grid = &report.grid;
    for (bi, band_name) in band_names.iter().enumerate() {
        let size = std::fs::metadata(&band_paths[bi])
            .map(|m| m.len())
            .unwrap_or(0);
        println!(
            "  ✓ {} → {} ({:.1} MB)",
            band_name,
            band_paths[bi].display(),
            size as f64 / 1e6
        );
    }

    println!(
        "Multi-band composite: {} dates × {} bands ({} tiles) → {} × {} ({:.1}M cells) in {:.1?}",
        report.scenes_used,
        n_bands,
        report.total_tiles,
        grid.cols,
        grid.rows,
        (grid.cols * grid.rows) as f64 / 1e6,
        start.elapsed()
    );

    Ok(())
}

/// Handle `surtgis stac time-series`: download one composite per temporal interval.
fn handle_time_series(
    catalog: &str,
    bbox: &str,
    collection: &str,
    asset: &str,
    datetime: &str,
    interval: &str,
    _scl_asset: &str,
    max_scenes: usize,
    align_to: Option<&std::path::PathBuf>,
    outdir: &std::path::PathBuf,
    compress: bool,
) -> Result<()> {
    // Parse datetime range "YYYY-MM-DD/YYYY-MM-DD"
    let parts: Vec<&str> = datetime.split('/').collect();
    if parts.len() != 2 {
        anyhow::bail!("datetime must be a range: YYYY-MM-DD/YYYY-MM-DD");
    }
    let start_date = parse_date(parts[0])?;
    let end_date = parse_date(parts[1])?;

    // Generate interval windows
    let intervals = split_date_range(&start_date, &end_date, interval)?;
    println!(
        "Time series: {} intervals ({}) from {} to {}",
        intervals.len(),
        interval,
        parts[0],
        parts[1]
    );

    std::fs::create_dir_all(outdir)?;
    let start = Instant::now();

    // Masking-semantics note (engine routing, R-timeseries): each interval now
    // runs through the RAM-budgeted `CompositeEngine` (the same core as
    // `stac composite`), instead of the legacy unbounded `fetch_stac_band`.
    // The engine masks via `CollectionProfile` — Sentinel-2 (SCL), Landsat
    // (QA_PIXEL) and Sentinel-1 (none) — which covers the realistic
    // time-series case (e.g. NDVI over Sentinel-2). For masked collections
    // *outside* those three the engine falls back to NO masking, whereas the
    // old `fetch_stac_band` auto-introspected the STAC schema. Intervals run
    // SEQUENTIALLY (one engine at a time): concurrent engines would each hold
    // their own RAM budget, defeating the point of routing through the engine.
    let total = intervals.len();
    let mut success = 0usize;
    let mut metadata: Vec<serde_json::Value> = Vec::with_capacity(total);

    for (i, (win_start, win_end)) in intervals.iter().enumerate() {
        let win_dt = format!("{}/{}", format_date(win_start), format_date(win_end));
        let label = format_date(win_start);
        println!(
            "[{}/{}] {} → {}",
            i + 1,
            total,
            format_date(win_start),
            format_date(win_end)
        );

        let filename = format!("{}_{}.tif", asset, label);
        let path = outdir.join(&filename);
        let band_paths = vec![path.clone()];

        // Headless progress: per-interval strip logging would spam. The
        // interval summary line below is the user-facing progress here.
        let mut progress = surtgis_cloud::composite::NoProgress;
        let run = run_engine_composite(
            catalog,
            collection,
            bbox,
            &[asset],
            &win_dt,
            max_scenes,
            align_to,
            &band_paths,
            1,   // band_chunk_size: single band per interval → minimum RAM
            512, // strip_rows: engine default (memory model may cap it)
            false,
            compress,
            0, // max_tile_failures: never abort, just gap-fill (as before)
            &mut progress,
        );

        match run {
            Ok(_report) => {
                // The engine doesn't report valid% directly, so read the written
                // GeoTIFF back to compute it (and grab the final grid dims). This
                // also preserves the historical metadata/console format exactly.
                let raster: surtgis_core::Raster<f64> =
                    match surtgis_core::io::read_geotiff(&path, None) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!(
                                "  ⚠️ [{}/{}] Wrote {} but could not read it back: {}",
                                i + 1,
                                total,
                                filename,
                                e
                            );
                            continue;
                        }
                    };
                let (rows, cols) = raster.shape();
                let valid = raster.data().iter().filter(|v| v.is_finite()).count();
                let cells = rows * cols;
                let pct = if cells > 0 {
                    valid as f64 / cells as f64 * 100.0
                } else {
                    0.0
                };

                println!(
                    "  [{}/{}] → {} ({}x{}, {:.1}% valid)",
                    i + 1,
                    total,
                    filename,
                    cols,
                    rows,
                    pct
                );

                metadata.push(serde_json::json!({
                    "index": i,
                    "date_start": format_date(win_start),
                    "date_end": format_date(win_end),
                    "file": filename,
                    "rows": rows,
                    "cols": cols,
                    "valid_pct": (pct * 10.0).round() / 10.0,
                }));
                success += 1;
            }
            Err(e) => {
                eprintln!("  ⚠️ [{}/{}] No data for {}: {}", i + 1, total, label, e);
            }
        }
    }

    // Write metadata JSON
    let meta_path = outdir.join("time_series.json");
    let meta_json = serde_json::json!({
        "catalog": catalog,
        "collection": collection,
        "asset": asset,
        "bbox": bbox,
        "datetime": datetime,
        "interval": interval,
        "total_intervals": intervals.len(),
        "successful": success,
        "rasters": metadata,
    });
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta_json)?)?;
    println!("\nMetadata → {}", meta_path.display());

    done(
        &format!("Time series ({}/{})", success, intervals.len()),
        outdir,
        start.elapsed(),
    );
    Ok(())
}

/// Simple date struct for interval splitting.
#[derive(Clone, Copy)]
pub(crate) struct SimpleDate {
    pub(crate) year: i32,
    pub(crate) month: u32,
    pub(crate) day: u32,
}

pub(crate) fn parse_date(s: &str) -> Result<SimpleDate> {
    // Strip time component if present: "2023-01-01T00:00:00Z" → "2023-01-01"
    let date_str = s.trim().split('T').next().unwrap_or(s.trim());
    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() != 3 {
        anyhow::bail!("invalid date format: '{}' (expected YYYY-MM-DD)", s);
    }
    Ok(SimpleDate {
        year: parts[0].parse().context("invalid year")?,
        month: parts[1].parse().context("invalid month")?,
        day: parts[2].parse().context("invalid day")?,
    })
}

pub(crate) fn format_date(d: &SimpleDate) -> String {
    format!("{:04}-{:02}-{:02}", d.year, d.month, d.day)
}

pub(crate) fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

pub(crate) fn advance_days(d: &SimpleDate, n: u32) -> SimpleDate {
    let mut y = d.year;
    let mut m = d.month;
    let mut day = d.day + n;
    loop {
        let dim = days_in_month(y, m);
        if day <= dim {
            break;
        }
        day -= dim;
        m += 1;
        if m > 12 {
            m = 1;
            y += 1;
        }
    }
    SimpleDate {
        year: y,
        month: m,
        day,
    }
}

pub(crate) fn advance_months(d: &SimpleDate, n: u32) -> SimpleDate {
    let mut m = d.month + n;
    let mut y = d.year;
    while m > 12 {
        m -= 12;
        y += 1;
    }
    let day = d.day.min(days_in_month(y, m));
    SimpleDate {
        year: y,
        month: m,
        day,
    }
}

pub(crate) fn date_le(a: &SimpleDate, b: &SimpleDate) -> bool {
    (a.year, a.month, a.day) <= (b.year, b.month, b.day)
}

pub(crate) fn split_date_range(
    start: &SimpleDate,
    end: &SimpleDate,
    interval: &str,
) -> Result<Vec<(SimpleDate, SimpleDate)>> {
    let mut windows = Vec::new();
    let mut cursor = *start;

    while date_le(&cursor, end) {
        let next = match interval.to_lowercase().as_str() {
            "monthly" | "month" => advance_months(&cursor, 1),
            "biweekly" | "2weeks" => advance_days(&cursor, 14),
            "weekly" | "week" => advance_days(&cursor, 7),
            "quarterly" | "quarter" => advance_months(&cursor, 3),
            "yearly" | "year" | "annual" => advance_months(&cursor, 12),
            custom => {
                let days: u32 = custom.parse()
                    .with_context(|| format!("invalid interval: '{}'. Use monthly, biweekly, weekly, quarterly, yearly, or a number of days", custom))?;
                advance_days(&cursor, days)
            }
        };
        // End of this window is day before next window (or end_date)
        let win_end = if date_le(&next, end) {
            advance_days(&next, 0) // next itself is start of next window
        } else {
            *end
        };
        // The datetime for STAC search uses the window bounds
        let actual_end = if date_le(&win_end, end) {
            win_end
        } else {
            *end
        };
        windows.push((cursor, actual_end));
        cursor = next;
    }

    if windows.is_empty() {
        anyhow::bail!("date range too short for interval '{}'", interval);
    }
    Ok(windows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_profile_sentinel2() {
        let profile = CollectionProfile::from_collection_name("sentinel-2-l2a").unwrap();
        assert_eq!(profile.mask_asset_name(), Some("scl"));
        assert_eq!(profile.description(), "Sentinel-2 L2A");
    }

    #[test]
    fn test_collection_profile_landsat() {
        let profile = CollectionProfile::from_collection_name("landsat-c2-l2").unwrap();
        assert_eq!(profile.mask_asset_name(), Some("QA_PIXEL"));
        assert_eq!(profile.description(), "Landsat C2 L2");
    }

    #[test]
    fn test_collection_profile_sentinel1() {
        let profile = CollectionProfile::from_collection_name("sentinel-1-rtc").unwrap();
        assert_eq!(profile.mask_asset_name(), None);
        assert_eq!(profile.description(), "Sentinel-1 RTC");
    }

    #[test]
    fn test_collection_profile_unknown_falls_back() {
        let profile = CollectionProfile::from_collection_name("esa-worldcover").unwrap();
        assert_eq!(profile.mask_asset_name(), None); // no cloud masking
    }

    #[test]
    fn test_collection_profile_debug() {
        let s2_profile = CollectionProfile::from_collection_name("sentinel-2-l2a").unwrap();
        let debug_str = format!("{:?}", s2_profile);
        assert!(debug_str.contains("Sentinel2L2A"));
    }
}

// ─── Zarr time step parsing ─────────────────────────────────────────

#[cfg(feature = "zarr")]
fn parse_time_step(s: &str) -> Result<surtgis_cloud::TimeReduction> {
    use surtgis_cloud::{TimeReduction, TimeSelector};

    match s.to_lowercase().as_str() {
        "first" => Ok(TimeReduction::Single(TimeSelector::First)),
        "last" => Ok(TimeReduction::Single(TimeSelector::Last)),
        other => {
            // Try parsing as ISO datetime
            if let Ok(dt) = chrono::NaiveDate::parse_from_str(other, "%Y-%m-%d") {
                let dt_utc = dt.and_hms_opt(0, 0, 0).unwrap().and_utc();
                Ok(TimeReduction::Single(TimeSelector::Nearest(dt_utc)))
            } else if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(other, "%Y-%m-%dT%H:%M:%S")
            {
                Ok(TimeReduction::Single(TimeSelector::Nearest(dt.and_utc())))
            } else {
                // Try as index
                if let Ok(idx) = other.parse::<usize>() {
                    Ok(TimeReduction::Single(TimeSelector::Index(idx)))
                } else {
                    anyhow::bail!(
                        "Invalid --time-step: '{}'. Use 'first', 'last', an ISO date (2020-06-15), or an index (0)",
                        other
                    );
                }
            }
        }
    }
}

// ─── Climate download helpers ───────────────────────────────────────

#[cfg(feature = "zarr")]
enum IntervalType {
    Daily,
    Monthly,
    Yearly,
    None,
}

#[cfg(feature = "zarr")]
fn parse_aggregate(s: &str) -> Result<(IntervalType, Option<surtgis_cloud::AggMethod>)> {
    use surtgis_cloud::AggMethod;
    match s.to_lowercase().replace('_', "-").as_str() {
        "none" => Ok((IntervalType::None, None)),
        "daily-sum" => Ok((IntervalType::Daily, Some(AggMethod::Sum))),
        "daily-mean" => Ok((IntervalType::Daily, Some(AggMethod::Mean))),
        "monthly-mean" => Ok((IntervalType::Monthly, Some(AggMethod::Mean))),
        "monthly-sum" => Ok((IntervalType::Monthly, Some(AggMethod::Sum))),
        "yearly-mean" => Ok((IntervalType::Yearly, Some(AggMethod::Mean))),
        "yearly-sum" => Ok((IntervalType::Yearly, Some(AggMethod::Sum))),
        other => anyhow::bail!(
            "Invalid --aggregate: '{}'. Options: none, daily-sum, daily-mean, monthly-mean, monthly-sum, yearly-mean, yearly-sum",
            other
        ),
    }
}

#[cfg(feature = "zarr")]
fn parse_datetime_range(
    s: &str,
) -> Result<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)> {
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 2 {
        anyhow::bail!("--datetime must be a range: 'YYYY-MM-DD/YYYY-MM-DD'");
    }
    let start = chrono::NaiveDate::parse_from_str(parts[0].trim(), "%Y-%m-%d")
        .context("Invalid start date")?
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc();
    let end = chrono::NaiveDate::parse_from_str(parts[1].trim(), "%Y-%m-%d")
        .context("Invalid end date")?
        .and_hms_opt(23, 59, 59)
        .unwrap()
        .and_utc();
    Ok((start, end))
}

#[cfg(feature = "zarr")]
fn generate_intervals(
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
    interval: IntervalType,
) -> Vec<(
    chrono::DateTime<chrono::Utc>,
    chrono::DateTime<chrono::Utc>,
    String,
)> {
    use chrono::{Datelike, NaiveDate, TimeZone, Utc};

    let mut intervals = Vec::new();

    match interval {
        IntervalType::None => {
            // Single interval covering the entire range
            let label = format!("{}_to_{}", start.format("%Y-%m-%d"), end.format("%Y-%m-%d"));
            intervals.push((start, end, label));
        }
        IntervalType::Daily => {
            let mut day = start.date_naive();
            let end_day = end.date_naive();
            while day <= end_day {
                let day_start = Utc.from_utc_datetime(&day.and_hms_opt(0, 0, 0).unwrap());
                let day_end = Utc.from_utc_datetime(&day.and_hms_opt(23, 59, 59).unwrap());
                let label = day.format("%Y-%m-%d").to_string();
                intervals.push((day_start, day_end, label));
                day = day.succ_opt().unwrap_or(day);
            }
        }
        IntervalType::Monthly => {
            let mut year = start.year();
            let mut month = start.month();
            while Utc.from_utc_datetime(
                &NaiveDate::from_ymd_opt(year, month, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
            ) <= end
            {
                let month_start = Utc.from_utc_datetime(
                    &NaiveDate::from_ymd_opt(year, month, 1)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap(),
                );
                let next_month = if month == 12 {
                    (year + 1, 1)
                } else {
                    (year, month + 1)
                };
                let month_end = Utc.from_utc_datetime(
                    &NaiveDate::from_ymd_opt(next_month.0, next_month.1, 1)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap(),
                ) - chrono::TimeDelta::seconds(1);
                let label = format!("{:04}-{:02}", year, month);
                intervals.push((month_start, month_end, label));
                month += 1;
                if month > 12 {
                    month = 1;
                    year += 1;
                }
            }
        }
        IntervalType::Yearly => {
            let mut year = start.year();
            while year <= end.year() {
                let year_start = Utc.from_utc_datetime(
                    &NaiveDate::from_ymd_opt(year, 1, 1)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap(),
                );
                let year_end = Utc.from_utc_datetime(
                    &NaiveDate::from_ymd_opt(year, 12, 31)
                        .unwrap()
                        .and_hms_opt(23, 59, 59)
                        .unwrap(),
                );
                let label = format!("{:04}", year);
                intervals.push((year_start, year_end, label));
                year += 1;
            }
        }
    }

    intervals
}
