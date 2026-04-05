//! Handler for STAC catalog subcommands.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use surtgis_algorithms::imagery::{CloudMaskStrategy, S2SclMask, LandsatQaMask, NoCloudMask};
use surtgis_cloud::blocking::{CogReaderBlocking, StacClientBlocking};
use surtgis_cloud::{BBox, CogReaderOptions, StacCatalog, StacClientOptions, StacItem, StacSearchParams};

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
            _ => anyhow::bail!("Unknown collection: {}", name),
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
/// - Bitmask (e.g., Landsat QA_PIXEL) → LandsatQaMask
/// - None (e.g., SAR) → NoCloudMask
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
        CloudMaskType::Bitmask { asset: _, bits: _ } => {
            // For now, assume all bitmask masks are Landsat-like (QA_PIXEL)
            // Can be extended to support other bitmask formats
            Arc::new(LandsatQaMask::new())
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

pub fn resolve_catalog_url(catalog: &str) -> String {
    for known in get_known_catalogs() {
        if catalog == known.shorthand {
            return known.url.to_string();
        }
    }
    // If not a shorthand, treat as full URL
    catalog.to_string()
}

/// Collections available in each STAC catalog
pub fn get_catalog_collections(catalog: &str) -> Vec<(&'static str, &'static str)> {
    match catalog {
        "pc" => vec![
            ("sentinel-2-l2a", "Sentinel-2 Level 2A (optical, 10-60m, 2016-present)"),
            ("landsat-c2-l2", "Landsat Collection 2 Level 2 (optical, 30m, 1980-present)"),
            ("sentinel-1-rtc", "Sentinel-1 RTC (SAR, 10m, 2015-present)"),
            ("cop-dem-glo-30", "Copernicus DEM 30m (elevation, global)"),
            ("nasadem", "NASADEM (elevation, 30m, global)"),
            ("gebco", "GEBCO bathymetry (ocean, 15 arc-seconds)"),
        ],
        "es" => vec![
            ("sentinel-2-l2a", "Sentinel-2 Level 2A (optical, 10-60m)"),
            ("landsat-c2-l2", "Landsat Collection 2 Level 2 (optical, 30m)"),
            ("sentinel-1-rtc", "Sentinel-1 RTC (SAR, 10m)"),
        ],
        "cdse" => vec![
            ("sentinel-1-grd", "Sentinel-1 GRD (SAR, 10m, ground range detected)"),
            ("sentinel-1-slc", "Sentinel-1 SLC (SAR, 10m, single look complex)"),
            ("sentinel-2-l1c", "Sentinel-2 L1C (optical, 10-60m, L1 processing)"),
            ("sentinel-2-l2a", "Sentinel-2 L2A (optical, 10-60m, atmospherically corrected)"),
            ("sentinel-3-olci", "Sentinel-3 OLCI (optical, 300-1000m, ocean/land)"),
            ("sentinel-5p", "Sentinel-5P (atmospheric, daily global coverage)"),
        ],
        "usgs" => vec![
            ("srtm-30m", "SRTM 30m DEM (90m for polar regions, 2000 data)"),
            ("aster-gdem", "ASTER GDEM (30m elevation, global, 2011 release)"),
            ("nasadem", "NASADEM (30m, merged SRTM+ASTER, improved voids)"),
            ("hydrosheds", "HydroSHEDS (hydrological datasets, 15 arc-seconds)"),
        ],
        "osgeo" => vec![
            ("(registry)", "STAC Index: Search all public STAC catalogs worldwide"),
            ("(api)", "Use API to discover 1000+ STAC catalogs globally"),
        ],
        _ => vec![
            ("(unknown)", "Use 'surtgis stac search --catalog <url> ...' to discover collections"),
        ],
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

/// Fetch catalogs from OGC STAC Index API (https://stacindex.org/api/v1)
///
/// Returns a list of catalog summaries with ID, title, description, and URL.
/// This is optional - if it fails, users can still use curated catalogs.
fn fetch_stac_index_catalogs() -> Result<Vec<StacIndexCatalog>> {
    // Try cache first
    let cache_path = get_stac_index_cache_path();
    let cache_ttl_secs = 24 * 60 * 60; // 24 hours

    if let Ok(cached_catalogs) = load_from_cache(&cache_path, cache_ttl_secs) {
        eprintln!("  📦 Using cached catalog list ({} catalogs)", cached_catalogs.len());
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
        let cache_dir = std::env::var("XDG_CACHE_HOME")
            .unwrap_or_else(|_| format!("{}/.cache", std::env::var("HOME").unwrap_or_else(|_| ".".to_string())));
        std::path::PathBuf::from(cache_dir).join("surtgis").join("stac_index_catalogs.json")
    }
    #[cfg(windows)]
    {
        let cache_dir = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string());
        std::path::PathBuf::from(cache_dir).join("surtgis-cache").join("stac_index_catalogs.json")
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
    let modified = metadata.modified().context("Could not get modification time")?;
    let elapsed = SystemTime::now()
        .duration_since(modified)
        .context("Could not calculate cache age")?;

    if elapsed.as_secs() > ttl_secs {
        anyhow::bail!("Cache expired");
    }

    // Read and parse cache
    let content = fs::read_to_string(path).context("Could not read cache file")?;
    let catalogs: Vec<StacIndexCatalog> = serde_json::from_str(&content)
        .context("Could not parse cache file")?;

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
    let json = serde_json::to_string_pretty(catalogs)
        .context("Could not serialize catalogs")?;
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

    rt.block_on(async {
        fetch_stac_index_async().await
    })
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
    let catalogs = parse_stac_index_response(&json)
        .context("Failed to parse catalog data")?;

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

        let slug = item
            .get("slug")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if slug.is_empty() {
            continue; // Skip entries without slug
        }

        let title = item
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or(slug)
            .to_string();

        // Use "summary" field from STAC Index if available, otherwise omit
        let description = item
            .get("summary")
            .and_then(|v| v.as_str())
            .map(|s| {
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
            let client =
                StacClientBlocking::new(cat, StacClientOptions::default())
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

            let results =
                client.search(&params).context("STAC search failed")?;
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
                let dt = item
                    .properties
                    .datetime
                    .as_deref()
                    .unwrap_or("-");
                let cc = item
                    .properties
                    .eo_cloud_cover
                    .map(|c| format!("{:.1}%", c))
                    .unwrap_or_else(|| "-".to_string());
                let col = item.collection.as_deref().unwrap_or("-");
                let asset_keys: Vec<&str> =
                    item.assets.keys().map(|k| k.as_str()).collect();

                println!("  {} [{}]", item.id, col);
                println!("    datetime: {}  cloud: {}", dt, cc);
                println!("    assets: {}", asset_keys.join(", "));
            }

            if results.has_next() {
                println!(
                    "\n  (more results available -- increase --limit to fetch more)"
                );
            }
        }

        StacCommands::Fetch {
            catalog,
            bbox,
            collection,
            asset,
            datetime,
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
            let client =
                StacClientBlocking::new(cat, StacClientOptions::default())
                    .context("Failed to create STAC client")?;
            let results =
                client.search(&params).context("STAC search failed")?;

            let item = results.features.first().ok_or_else(|| {
                anyhow::anyhow!("No items found matching the search criteria")
            })?;
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
                let (k, _) = item.first_cog_asset().ok_or_else(|| {
                    anyhow::anyhow!(
                        "No COG asset found. Specify --asset explicitly. Available: {}",
                        item.assets
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
                println!("Auto-detected asset: {}", k);
                k.clone()
            };

            let stac_asset = item.asset(&asset_key).ok_or_else(|| {
                anyhow::anyhow!(
                    "Asset '{}' not found. Available: {}",
                    asset_key,
                    item.assets
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;

            // Sign the href if needed
            let href = client
                .sign_asset_href(
                    &stac_asset.href,
                    item.collection.as_deref().unwrap_or(""),
                )
                .context("Failed to sign asset URL")?;

            // Read via CogReader
            let pb = spinner("Fetching COG tiles...");
            let start = Instant::now();
            let opts = CogReaderOptions::default();
            let mut reader = CogReaderBlocking::open(&href, opts)
                .context("Failed to open remote COG")?;

            // Auto-reproject bbox if COG is in a projected CRS (e.g. UTM)
            let read_bb = {
                use surtgis_cloud::reproject;
                // Prefer proj:epsg from STAC item (no extra HTTP request)
                let epsg = item.epsg().or_else(|| {
                    reader
                        .metadata()
                        .crs
                        .as_ref()
                        .and_then(|c| c.epsg())
                });
                if let Some(epsg) = epsg {
                    if !reproject::is_wgs84(epsg) {
                        let reprojected =
                            reproject::reproject_bbox_to_cog(&bb, epsg);
                        println!("Reprojected bbox to EPSG:{}", epsg);
                        reprojected
                    } else {
                        bb
                    }
                } else {
                    bb
                }
            };

            let raster: surtgis_core::Raster<f64> = reader
                .read_bbox(&read_bb, None)
                .context("Failed to read bounding box from COG")?;
            pb.finish_and_clear();
            let elapsed = start.elapsed();

            let (rows, cols) = raster.shape();
            println!(
                "Fetched: {} x {} ({} cells)",
                cols,
                rows,
                raster.len()
            );
            write_result(&raster, &output, compress)?;
            done("STAC fetch", &output, elapsed);
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
                    .sign_asset_href(
                        &stac_asset.href,
                        item.collection.as_deref().unwrap_or(""),
                    )
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
                    let epsg = item.epsg().or_else(|| {
                        reader
                            .metadata()
                            .crs
                            .as_ref()
                            .and_then(|c| c.epsg())
                    });
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
            let result = surtgis_core::mosaic(&refs, None)
                .context("Failed to mosaic tiles")?;
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
            scl_asset: _scl_asset,  // DEPRECATED: ignored (profile determines masking)
            scl_keep: _scl_keep,    // DEPRECATED: ignored (profile determines masking)
            align_to,
            output,
        } => {
            // Multi-band support: --asset "red,nir,swir16,blue"
            let assets: Vec<&str> = asset.split(',').map(|s| s.trim()).collect();
            if assets.len() > 1 {
                println!("Multi-band composite: {} bands", assets.len());
                let total_start = Instant::now();
                for (band_idx, band) in assets.iter().enumerate() {
                    let band_output = output.with_file_name(format!(
                        "{}_{}.tif",
                        output.file_stem().unwrap_or_default().to_string_lossy(),
                        band
                    ));
                    println!("\n[{}/{}] Band: {} → {}", band_idx + 1, assets.len(), band, band_output.display());

                    // Re-run composite for each band (SAS cache reuses tokens)
                    let band_cmd = StacCommands::Composite {
                        catalog: catalog.clone(),
                        bbox: bbox.clone(),
                        collection: collection.clone(),
                        asset: band.to_string(),
                        datetime: datetime.clone(),
                        max_scenes,
                        scl_asset: _scl_asset.clone(),
                        scl_keep: _scl_keep.clone(),
                        align_to: align_to.clone(),
                        output: band_output,
                    };
                    // Recursive call for single-band
                    handle(band_cmd, compress)?;
                }
                println!("\nAll {} bands complete in {:.1?}", assets.len(), total_start.elapsed());
                return Ok(());
            }

            // Get collection profile (determines cloud masking strategy)
            let profile = CollectionProfile::from_collection_name(&collection)?;
            let mask_asset_name = profile.mask_asset_name();

            eprintln!("📷 Collection: {}", profile.description());

            let cat = StacCatalog::from_str_or_url(&catalog);
            let bb = parse_bbox(&bbox)?;

            // Search with high limit to find ALL tiles across dates.
            // A large basin can have 6-10 MGRS tiles × 50+ dates = 500+ items.
            // We need to fetch all items to ensure every date has all its tiles.
            let search_limit = (max_scenes * 20).max(500) as u32;
            let params = StacSearchParams::new()
                .bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y)
                .collections(&[collection.as_str()])
                .datetime(&datetime)
                .limit(search_limit);

            let pb = spinner("Searching STAC catalog...");
            let client_opts = StacClientOptions {
                max_items: (max_scenes * 20).max(500) as usize,
                ..StacClientOptions::default()
            };
            let client = StacClientBlocking::new(cat, client_opts)
                .context("Failed to create STAC client")?;
            let items = client.search_all(&params).context("STAC search failed")?;
            pb.finish_and_clear();

            if items.is_empty() {
                anyhow::bail!("No items found matching the search criteria");
            }

            // Group items by acquisition date (YYYY-MM-DD)
            let mut by_date: BTreeMap<String, Vec<&StacItem>> = BTreeMap::new();
            for item in &items {
                let date = item
                    .properties
                    .datetime
                    .as_deref()
                    .unwrap_or("")
                    .get(..10)
                    .unwrap_or("unknown")
                    .to_string();
                by_date.entry(date).or_default().push(item);
            }

            let dates: Vec<String> =
                by_date.keys().take(max_scenes).cloned().collect();

            // Log tiles per date to verify all tiles are included
            let tiles_per_date: Vec<usize> = dates.iter()
                .map(|d| by_date.get(d).map(|g| g.len()).unwrap_or(0))
                .collect();
            let min_tiles = tiles_per_date.iter().min().copied().unwrap_or(0);
            let max_tiles = tiles_per_date.iter().max().copied().unwrap_or(0);

            println!(
                "Found {} items across {} dates (using {} dates, {}-{} tiles/date)",
                items.len(),
                by_date.len(),
                dates.len(),
                min_tiles, max_tiles,
            );

            let start = Instant::now();

            // -- Phase 1: Resolve asset URLs for all items (no data download) --
            // For each date, collect (data_href, scl_href, epsg) tuples
            #[allow(dead_code)]
            /// Per-tile info including its native EPSG for bbox reprojection
            struct TileRef {
                data_href: String,
                scl_href: String,
                epsg: Option<u32>,
            }

            struct SceneInfo {
                date: String,
                tiles: Vec<TileRef>,
                epsg: Option<u32>, // EPSG of the first tile (for output grid)
            }

            let mut scenes: Vec<SceneInfo> = Vec::new();

            for date in &dates {
                let group = &by_date[date];
                let mut tiles = Vec::new();
                let mut scene_epsg = None;

                let mut sign_failures = 0usize;
                let mut asset_missing = 0usize;

                for item in group {
                    let tile_epsg = item.epsg();
                    let _item_id = item.id.as_str();
                    // Log EPSG detection for first date
                    if scenes.is_empty() && tiles.is_empty() {
                    }

                    let data_result = match resolve_asset_key(item, &asset) {
                        Some((_, a)) => {
                            match client.sign_asset_href(&a.href, item.collection.as_deref().unwrap_or("")) {
                                Ok(h) => Some(h),
                                Err(_) => { sign_failures += 1; None }
                            }
                        }
                        None => { asset_missing += 1; None }
                    };

                    let scl_result = mask_asset_name.and_then(|mask_name| {
                        match resolve_asset_key(item, mask_name) {
                            Some((_, a)) => {
                                match client.sign_asset_href(&a.href, item.collection.as_deref().unwrap_or("")) {
                                    Ok(h) => Some(h),
                                    Err(_) => { sign_failures += 1; None }
                                }
                            }
                            None => None // Mask asset not found — proceed without masking
                        }
                    });

                    // Data asset is required; mask asset is optional.
                    // If mask is unavailable, proceed without cloud masking.
                    if let Some(dh) = data_result {
                        if scene_epsg.is_none() {
                            scene_epsg = tile_epsg;
                        }
                        tiles.push(TileRef {
                            data_href: dh,
                            scl_href: scl_result.unwrap_or_default(),
                            epsg: tile_epsg,
                        });
                    } else {
                        asset_missing += 1;
                    }
                }

                if tiles.is_empty() {
                    eprintln!("  ⚠ {}: 0 tiles resolved (items={}, sign_fail={}, asset_missing={})",
                        date, group.len(), sign_failures, asset_missing);
                }

                if !tiles.is_empty() {
                    scenes.push(SceneInfo {
                        date: date.clone(),
                        tiles,
                        epsg: scene_epsg,
                    });
                }
            }

            if scenes.is_empty() {
                anyhow::bail!("No valid scenes found");
            }

            // -- Phase 2: Determine output grid from first scene's metadata --
            let first_href = &scenes[0].tiles[0].data_href;
            let opts = CogReaderOptions::default();
            let probe_reader = CogReaderBlocking::open(first_href, opts)
                .context("Failed to probe first COG")?;
            let probe_meta = probe_reader.metadata();

            // Determine output grid:
            // If --align-to is present, use the DEM reference grid directly
            // (avoids the unreliable post-composite resample step).
            // Otherwise, use the COG native grid.
            let (out_cols, out_rows, out_transform, out_crs, cog_bb);

            eprintln!("  align_to = {:?}", align_to.as_ref().map(|p| p.display().to_string()));
            if let Some(ref align_path) = align_to {
                // Use reference DEM grid
                let reference: surtgis_core::Raster<f64> =
                    surtgis_core::io::read_geotiff(align_path, None)
                        .context("Failed to read alignment reference raster")?;
                let rgt = reference.transform();
                let (rr, rc) = reference.shape();

                out_rows = rr;
                out_cols = rc;
                out_transform = *rgt;
                out_crs = reference.crs().cloned();

                // Compute bbox from reference grid (in its CRS)
                let min_x = rgt.origin_x;
                let max_y = rgt.origin_y;
                let max_x = min_x + rc as f64 * rgt.pixel_width;
                let min_y = max_y + rr as f64 * rgt.pixel_height; // pixel_height is negative
                cog_bb = BBox::new(min_x, min_y, max_x, max_y);

                eprintln!("  Using reference grid from --align-to: {}x{} px=({:.1},{:.1})",
                    rc, rr, rgt.pixel_width, rgt.pixel_height);
            } else {
                // Use COG native grid
                let cog_bb_computed = {
                    use surtgis_cloud::reproject;
                    if let Some(epsg) = scenes[0].epsg {
                        if !reproject::is_wgs84(epsg) {
                            reproject::reproject_bbox_to_cog(&bb, epsg)
                        } else { bb }
                    } else { bb }
                };
                cog_bb = cog_bb_computed;

                let pixel_width = probe_meta.geo_transform.pixel_width.abs();
                let pixel_height = probe_meta.geo_transform.pixel_height.abs();
                out_cols = ((cog_bb.max_x - cog_bb.min_x) / pixel_width).round() as usize;
                out_rows = ((cog_bb.max_y - cog_bb.min_y) / pixel_height).round() as usize;

                out_transform = surtgis_core::GeoTransform::new(
                    cog_bb.min_x, cog_bb.max_y, pixel_width, -pixel_height,
                );
                out_crs = scenes[0].epsg.map(surtgis_core::CRS::from_epsg);
            }

            println!(
                "Output grid: {} x {} ({:.1}M cells), {} dates",
                out_cols, out_rows,
                (out_cols * out_rows) as f64 / 1e6,
                scenes.len()
            );
            eprintln!("  Grid config: cols={} rows={} px=({:.1},{:.1})",
                out_cols, out_rows, out_transform.pixel_width, out_transform.pixel_height);
            eprintln!("  Grid bbox: x=[{:.1}, {:.1}] y=[{:.1}, {:.1}]",
                cog_bb.min_x, cog_bb.max_x, cog_bb.min_y, cog_bb.max_y);

            // -- Phase 3: Strip-by-strip processing --
            let strip_rows = 512usize; // rows per output strip
            let num_strips = (out_rows + strip_rows - 1) / strip_rows;
            let n_scenes = scenes.len();

            let config = surtgis_core::io::StripWriterConfig {
                rows: out_rows,
                cols: out_cols,
                transform: out_transform,
                crs: out_crs,
                compress,
                rows_per_strip: strip_rows as u32,
            };

            let scenes_ref = &scenes;
            // Get cloud masking strategy from collection profile
            let cloud_mask_strategy = match &profile {
                CollectionProfile::Sentinel2L2A {
                    cloud_mask_strategy,
                } => cloud_mask_strategy.clone(),
                CollectionProfile::LandsatC2L2 {
                    cloud_mask_strategy,
                } => cloud_mask_strategy.clone(),
                CollectionProfile::Sentinel1RTC => Arc::new(NoCloudMask),
            };
            let mask_ref = &cloud_mask_strategy;
            let mut strip_idx_counter = 0usize;

            surtgis_core::io::write_geotiff_streaming(
                &output,
                &config,
                |_strip_idx, strip_out_rows| {
                    let current_strip = strip_idx_counter;
                    strip_idx_counter += 1;

                    let row_start = current_strip * strip_rows;
                    let row_end = (row_start + strip_out_rows).min(out_rows);
                    let actual_rows = row_end - row_start;

                    // Geographic bbox for this strip
                    let ph = out_transform.pixel_height.abs();
                    let strip_min_y = cog_bb.max_y - (row_end as f64 * ph);
                    let strip_max_y = cog_bb.max_y - (row_start as f64 * ph);
                    let strip_bb = BBox::new(
                        cog_bb.min_x, strip_min_y, cog_bb.max_x, strip_max_y,
                    );

                    eprintln!("  Strip {}/{}: bbox y=[{:.0}, {:.0}]",
                        current_strip + 1, num_strips, strip_min_y, strip_max_y);
                    print!(
                        "\r  Strip {}/{} (rows {}-{})...",
                        current_strip + 1, num_strips, row_start, row_end
                    );

                    // Collect masked strips from each scene.
                    // Strategy: read COG tiles at NATIVE resolution, mosaic+mask at native,
                    // then resample the clean result to the output grid resolution.
                    // This avoids tile alignment artifacts when COG res != output res.
                    let mut scene_strips: Vec<ndarray::Array2<f64>> = Vec::with_capacity(n_scenes);

                    // Reference raster for resampling scene results to output grid
                    let strip_ref = {
                        let mut r = surtgis_core::Raster::<f64>::new(actual_rows, out_cols);
                        r.set_transform(surtgis_core::GeoTransform::new(
                            strip_bb.min_x, strip_bb.max_y,
                            out_transform.pixel_width, out_transform.pixel_height,
                        ));
                        r
                    };

                    for scene in scenes_ref {
                        let is_first_strip = current_strip == 0;

                        // Expand strip_bb slightly to ensure it covers COG tiles
                        // that may be offset from the DEM grid by a few pixels.
                        let pad = 100.0; // 100m padding (covers 10 pixels at 10m)
                        let tile_bb = BBox::new(
                            strip_bb.min_x - pad,
                            strip_bb.min_y - pad,
                            strip_bb.max_x + pad,
                            strip_bb.max_y + pad,
                        );

                        // Download all tiles in parallel using std threads.
                        // Each tile involves 1-2 HTTP range requests (data + mask).
                        let tile_results: Vec<(
                            Option<surtgis_core::Raster<f64>>,
                            Option<surtgis_core::Raster<f64>>,
                        )> = {
                            let mut handles = Vec::with_capacity(scene.tiles.len());
                            for (tile_idx, tile) in scene.tiles.iter().enumerate() {
                                let data_href = tile.data_href.clone();
                                let scl_href = tile.scl_href.clone();
                                let bb = tile_bb;
                                let first = is_first_strip && tile_idx == 0;
                                handles.push(std::thread::spawn(move || {
                                    let data = read_cog_tile(&data_href, &bb, first);
                                    let mask = if scl_href.is_empty() {
                                        None
                                    } else {
                                        read_cog_tile_raw(&scl_href, &bb)
                                    };
                                    (data, mask)
                                }));
                            }
                            handles.into_iter().map(|h| h.join().unwrap_or((None, None))).collect()
                        };

                        let mut data_tiles: Vec<surtgis_core::Raster<f64>> = Vec::new();
                        let mut scl_tiles: Vec<surtgis_core::Raster<f64>> = Vec::new();
                        let mut data_ok = 0usize;
                        let mut data_fail = 0usize;
                        let mut scl_ok = 0usize;

                        for (data_opt, mask_opt) in tile_results {
                            if let Some(r) = data_opt {
                                data_tiles.push(r);
                                data_ok += 1;
                            } else {
                                data_fail += 1;
                            }
                            if let Some(r) = mask_opt {
                                scl_tiles.push(r);
                                scl_ok += 1;
                            }
                        }

                        if is_first_strip {
                            eprintln!("  ℹ {}: {} tiles (parallel), data={}/{} mask={}",
                                scene.date, scene.tiles.len(),
                                data_ok, data_ok + data_fail, scl_ok);
                        }

                        if data_tiles.is_empty() {
                            if is_first_strip {
                                eprintln!("  ⚠ {}: SKIPPED (no data tiles)", scene.date);
                            }
                            continue;
                        }

                        // Unify CRS: reproject tiles to the CRS of the first tile
                        // This handles multi-UTM-zone regions (e.g., EPSG:32719 + EPSG:32720)
                        fn unify_crs(tiles: &mut Vec<surtgis_core::Raster<f64>>) {
                            if tiles.len() <= 1 { return; }
                            let target_epsg = tiles[0].crs().and_then(|c| c.epsg());
                            if let Some(target) = target_epsg {
                                for i in 1..tiles.len() {
                                    if let Some(src_epsg) = tiles[i].crs().and_then(|c| c.epsg()) {
                                        if src_epsg != target {
                                            if let Some(reprojected) = surtgis_cloud::reproject::reproject_raster_utm(&tiles[i], src_epsg, target) {
                                                tiles[i] = reprojected;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        unify_crs(&mut data_tiles);
                        if !scl_tiles.is_empty() {
                            unify_crs(&mut scl_tiles);
                        }

                        // Mosaic spatial tiles for this date's strip
                        if is_first_strip {
                            eprintln!("  ℹ {}: mosaic input: data={} mask={} tiles",
                                scene.date, data_tiles.len(), scl_tiles.len());
                        }
                        let data_m = if data_tiles.len() == 1 {
                            data_tiles.into_iter().next().unwrap()
                        } else {
                            let refs: Vec<&surtgis_core::Raster<f64>> = data_tiles.iter().collect();
                            match surtgis_core::mosaic(&refs, None) {
                                Ok(m) => m,
                                Err(e) => {
                                    eprintln!("  ⚠ Mosaic failed for date (data): {}", e);
                                    continue;
                                }
                            }
                        };

                        // Mosaic mask tiles (optional — may be empty for collections without cloud mask)
                        let scl_m = if scl_tiles.is_empty() {
                            None
                        } else if scl_tiles.len() == 1 {
                            Some(scl_tiles.into_iter().next().unwrap())
                        } else {
                            let refs: Vec<&surtgis_core::Raster<f64>> = scl_tiles.iter().collect();
                            match surtgis_core::mosaic(&refs, None) {
                                Ok(m) => Some(m),
                                Err(e) => {
                                    eprintln!("  ⚠ Mosaic failed for date (mask): {}", e);
                                    None
                                }
                            }
                        };

                        // Diagnose between each step
                        if is_first_strip {
                            let dgt = data_m.transform();
                            let data_valid = data_m.data().iter().filter(|v| v.is_finite() && **v > 0.0).count();
                            let data_total = data_m.data().len();
                            eprintln!("    [mosaic] data_m: {}x{} px=({:.1},{:.1}) valid={}/{} ({:.0}%)",
                                data_m.shape().1, data_m.shape().0,
                                dgt.pixel_width, dgt.pixel_height,
                                data_valid, data_total,
                                if data_total > 0 { data_valid as f64 / data_total as f64 * 100.0 } else { 0.0 });

                            if let Some(ref scl) = scl_m {
                                let scl_valid = scl.data().iter().filter(|v| v.is_finite()).count();
                                let scl_total = scl.data().len();
                                eprintln!("    [mosaic] mask: {}x{} px=({:.1},{:.1}) valid={}/{} ({:.0}%)",
                                    scl.shape().1, scl.shape().0,
                                    scl.transform().pixel_width, scl.transform().pixel_height,
                                    scl_valid, scl_total,
                                    if scl_total > 0 { scl_valid as f64 / scl_total as f64 * 100.0 } else { 0.0 });
                            } else {
                                eprintln!("    [mosaic] no mask — proceeding without cloud masking");
                            }
                        }

                        // Debug: check mask value distribution before cloud mask
                        if is_first_strip {
                            if let Some(ref scl) = scl_m {
                                let mut scl_hist = [0usize; 16];
                                for v in scl.data().iter() {
                                    let vi = *v as usize;
                                    if vi < 16 { scl_hist[vi] += 1; }
                                }
                                let scl_nonzero: Vec<String> = scl_hist.iter().enumerate()
                                    .filter(|(_, c)| **c > 0)
                                    .map(|(i, c)| format!("{}:{}", i, c))
                                    .collect();
                                eprintln!("    [mask] histogram: {}", scl_nonzero.join(" "));
                            }
                        }

                        // Cloud mask using collection-specific strategy (if mask available)
                        let clean_result = if let Some(ref scl) = scl_m {
                            mask_ref.mask(&data_m, scl).ok()
                        } else {
                            Some(data_m) // No mask → use data as-is
                        };

                        match clean_result {
                            Some(clean) => {
                                // Resample from native resolution to output grid
                                let (clean_r, clean_c) = clean.shape();

                                if current_strip == 0 {
                                    let clean_valid = clean.data().iter().filter(|v| v.is_finite() && **v > 0.0).count();
                                    let clean_total = clean.data().len();
                                    let cgt = clean.transform();
                                    let rgt = strip_ref.transform();
                                    eprintln!("    [cloud] clean: {}x{} px=({:.1},{:.1}) valid={}/{} ({:.0}%)",
                                        clean_c, clean_r, cgt.pixel_width, cgt.pixel_height,
                                        clean_valid, clean_total,
                                        if clean_total > 0 { clean_valid as f64 / clean_total as f64 * 100.0 } else { 0.0 });
                                    eprintln!("    Resample: src origin=({:.1},{:.1}) px=({:.1},{:.1}) {}x{}",
                                        cgt.origin_x, cgt.origin_y, cgt.pixel_width, cgt.pixel_height, clean_c, clean_r);
                                    eprintln!("    Resample: dst origin=({:.1},{:.1}) px=({:.1},{:.1}) {}x{}",
                                        rgt.origin_x, rgt.origin_y, rgt.pixel_width, rgt.pixel_height,
                                        strip_ref.shape().1, strip_ref.shape().0);
                                }

                                let resampled = surtgis_core::resample_to_grid(
                                    &clean, &strip_ref,
                                    surtgis_core::ResampleMethod::Bilinear,
                                ).unwrap_or(clean);

                                let valid = resampled.data().iter().filter(|v| v.is_finite()).count();
                                let total = resampled.data().len();
                                if current_strip == 0 {
                                    let pct = if total > 0 { valid as f64 / total as f64 * 100.0 } else { 0.0 };
                                    eprintln!(", mosaic OK, resample {}x{}→{}x{}, {:.0}% clear ✓",
                                        clean_c, clean_r,
                                        resampled.shape().1, resampled.shape().0, pct);
                                }
                                if valid > 0 {
                                    scene_strips.push(resampled.data().to_owned());
                                } else if current_strip == 0 {
                                    eprintln!("  ⚠ {}: 100% cloudy, skipped", scene.date);
                                }
                            }
                            None => {
                                if current_strip == 0 {
                                    eprintln!("  ⚠ {}: cloud mask failed, skipping", scene.date);
                                }
                            }
                        }
                    }

                    // Compute per-pixel median across scenes for this strip,
                    // then fill remaining NaN with temporal nearest (most recent valid value).
                    let mut output = ndarray::Array2::<f64>::from_elem(
                        (actual_rows, out_cols), f64::NAN,
                    );

                    if !scene_strips.is_empty() {
                        let n = scene_strips.len();

                        // Diagnostic: check value ranges of individual scene strips
                        if current_strip == 0 {
                            for (si, strip) in scene_strips.iter().enumerate() {
                                let vals: Vec<f64> = strip.iter().filter(|v| v.is_finite()).copied().collect();
                                if !vals.is_empty() {
                                    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                                    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                    let over = vals.iter().filter(|&&v| v > 10000.0).count();
                                    if si < 3 || over > 0 {
                                        eprintln!("    scene_strip[{}]: n={} range=[{:.0},{:.0}] >10k={}",
                                            si, vals.len(), min, max, over);
                                    }
                                }
                            }
                        }

                        // Sort scene_strips by coverage (most pixels first) for greedy coverage
                        let mut coverage: Vec<(usize, usize)> = scene_strips.iter().enumerate()
                            .map(|(i, s)| {
                                let valid = s.iter().filter(|v| v.is_finite()).count();
                                (i, valid)
                            })
                            .collect();
                        coverage.sort_by(|a, b| b.1.cmp(&a.1));

                        // Phase 1: Median composite (all scenes)
                        for r in 0..actual_rows {
                            for c in 0..out_cols {
                                let mut values: Vec<f64> = Vec::with_capacity(n);
                                for strip in &scene_strips {
                                    if r < strip.nrows() && c < strip.ncols() {
                                        let v = strip[[r, c]];
                                        if v.is_finite() {
                                            values.push(v);
                                        }
                                    }
                                }
                                if !values.is_empty() {
                                    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                                    let mid = values.len() / 2;
                                    output[[r, c]] = if values.len() % 2 == 0 {
                                        (values[mid - 1] + values[mid]) / 2.0
                                    } else {
                                        values[mid]
                                    };
                                }
                            }
                        }

                        // Phase 2: Fill remaining NaN with temporal nearest
                        // (use the scene with highest coverage first, greedy)
                        let nan_before = output.iter().filter(|v| !v.is_finite()).count();
                        if nan_before > 0 {
                            for &(scene_idx, _) in &coverage {
                                let strip = &scene_strips[scene_idx];
                                let mut filled = 0usize;
                                for r in 0..actual_rows {
                                    for c in 0..out_cols {
                                        if !output[[r, c]].is_finite()
                                            && r < strip.nrows() && c < strip.ncols()
                                        {
                                            let v = strip[[r, c]];
                                            if v.is_finite() {
                                                output[[r, c]] = v;
                                                filled += 1;
                                            }
                                        }
                                    }
                                }
                                if filled == 0 { continue; }
                                // Check if all filled
                                let nan_remaining = output.iter().filter(|v| !v.is_finite()).count();
                                if nan_remaining == 0 { break; }
                            }
                        }

                        // Phase 3: Spatial fill for any remaining NaN
                        // (simple 3x3 mean of valid neighbors, iterate until stable)
                        let mut nan_remaining = output.iter().filter(|v| !v.is_finite()).count();
                        if nan_remaining > 0 && nan_remaining < output.len() {
                            let max_passes = 20;
                            for _pass in 0..max_passes {
                                let prev = output.clone();
                                let mut filled_this_pass = 0usize;
                                for r in 0..actual_rows {
                                    for c in 0..out_cols {
                                        if prev[[r, c]].is_finite() { continue; }
                                        // Collect valid neighbors in 3x3 window
                                        let mut sum = 0.0;
                                        let mut cnt = 0u32;
                                        for dr in -1i32..=1 {
                                            for dc in -1i32..=1 {
                                                let nr = r as i32 + dr;
                                                let nc = c as i32 + dc;
                                                if nr >= 0 && nr < actual_rows as i32
                                                    && nc >= 0 && nc < out_cols as i32
                                                {
                                                    let v = prev[[nr as usize, nc as usize]];
                                                    if v.is_finite() {
                                                        sum += v;
                                                        cnt += 1;
                                                    }
                                                }
                                            }
                                        }
                                        if cnt >= 2 {
                                            output[[r, c]] = sum / cnt as f64;
                                            filled_this_pass += 1;
                                        }
                                    }
                                }
                                if filled_this_pass == 0 { break; }
                                nan_remaining = output.iter().filter(|v| !v.is_finite()).count();
                                if nan_remaining == 0 { break; }
                            }
                        }
                    }

                    let valid_count = output.iter().filter(|v| v.is_finite()).count();
                    let total = output.len();
                    let pct = if total > 0 { valid_count as f64 / total as f64 * 100.0 } else { 0.0 };
                    if current_strip == 0 || current_strip == num_strips - 1 || scene_strips.is_empty() {
                        // Value range diagnostic
                        let valid_vals: Vec<f64> = output.iter().filter(|v| v.is_finite()).copied().collect();
                        let (vmin, vmax, vmean) = if !valid_vals.is_empty() {
                            let min = valid_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                            let max = valid_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let mean = valid_vals.iter().sum::<f64>() / valid_vals.len() as f64;
                            (min, max, mean)
                        } else { (0.0, 0.0, 0.0) };
                        let over_10k = valid_vals.iter().filter(|&&v| v > 10000.0).count();
                        eprintln!("  Strip {}/{}: scenes={}, {:.0}% valid, range=[{:.0},{:.0}] mean={:.0} >10k={}",
                            current_strip + 1, num_strips,
                            scene_strips.len(), pct, vmin, vmax, vmean, over_10k);
                    }
                    Ok(output)
                },
            ).context("Failed to write streaming composite")?;

            // Verify written file
            let file_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
            let expected_size = (out_rows * out_cols * 4) as u64; // f32
            eprintln!("  File written: {} bytes ({:.1} MB), expected ~{:.1} MB",
                file_size, file_size as f64 / 1e6, expected_size as f64 / 1e6);

            println!(); // newline after \r progress

            // If --align-to is specified AND we didn't already use the reference grid,
            // resample the output to match. (When align_to is present, we now write
            // directly to the reference grid, so this step is skipped.)
            if false && align_to.is_some() {
                let align_path = align_to.as_ref().unwrap();
                let composite: surtgis_core::Raster<f64> =
                    surtgis_core::io::read_geotiff(&output, None)
                        .context("Failed to re-read composite for alignment")?;
                let reference: surtgis_core::Raster<f64> =
                    surtgis_core::io::read_geotiff(align_path, None)
                        .context("Failed to read alignment reference raster")?;

                // Debug: log extents to diagnose alignment
                let cgt = composite.transform();
                let (cr, cc) = composite.shape();
                let rgt = reference.transform();
                let (rr, rc) = reference.shape();
                eprintln!("  Composite transform: origin=({:.1},{:.1}) px=({:.1},{:.1}) shape={}x{}",
                    cgt.origin_x, cgt.origin_y, cgt.pixel_width, cgt.pixel_height, cr, cc);
                eprintln!("  Reference transform: origin=({:.1},{:.1}) px=({:.1},{:.1}) shape={}x{}",
                    rgt.origin_x, rgt.origin_y, rgt.pixel_width, rgt.pixel_height, rr, rc);
                // Check sample pixel mapping
                let sample_x = rgt.origin_x + 0.5 * rgt.pixel_width;
                let sample_y = rgt.origin_y + 0.5 * rgt.pixel_height;
                let src_col_f = (sample_x - cgt.origin_x) / cgt.pixel_width - 0.5;
                let src_row_f = (sample_y - cgt.origin_y) / cgt.pixel_height - 0.5;
                eprintln!("  Pixel(0,0) ref→comp: geo=({:.1},{:.1}) → src_pix=({:.1},{:.1}) bounds=(0..{},0..{})",
                    sample_x, sample_y, src_col_f, src_row_f, cc, cr);

                let pb = spinner("Aligning to reference grid...");
                let aligned = surtgis_core::resample_to_grid(
                    &composite,
                    &reference,
                    surtgis_core::ResampleMethod::Bilinear,
                )
                .context("Failed to resample to reference grid")?;
                pb.finish_and_clear();

                let (ar, ac) = aligned.shape();
                let valid_aligned = aligned.data().iter().filter(|v| v.is_finite()).count();
                let total_aligned = ar * ac;
                let pct_aligned = if total_aligned > 0 { valid_aligned as f64 / total_aligned as f64 * 100.0 } else { 0.0 };
                eprintln!("  Aligned result: {}x{}, {}/{} valid ({:.1}%)",
                    ac, ar, valid_aligned, total_aligned, pct_aligned);
                if valid_aligned == 0 {
                    eprintln!("  ⚠ WARNING: Aligned raster has 0% valid pixels!");
                    eprintln!("    Composite CRS may differ from reference DEM CRS.");
                    eprintln!("    Composite origin: ({:.1}, {:.1}), Reference origin: ({:.1}, {:.1})",
                        cgt.origin_x, cgt.origin_y, rgt.origin_x, rgt.origin_y);
                }

                // Post-align spatial gap-fill (3x3 iterative mean)
                // The resample may introduce NaN at edges where bilinear hits NaN neighbors
                let nan_post = aligned.data().iter().filter(|v| !v.is_finite()).count();
                let mut aligned = aligned;
                if nan_post > 0 && nan_post < total_aligned {
                    let (ar, ac) = aligned.shape();
                    let max_passes = 30;
                    for _pass in 0..max_passes {
                        let prev = aligned.data().clone();
                        let mut filled = 0usize;
                        for r in 0..ar {
                            for c in 0..ac {
                                if prev[[r, c]].is_finite() { continue; }
                                let mut sum = 0.0;
                                let mut cnt = 0u32;
                                for dr in -1i32..=1 {
                                    for dc in -1i32..=1 {
                                        let nr = r as i32 + dr;
                                        let nc = c as i32 + dc;
                                        if nr >= 0 && nr < ar as i32 && nc >= 0 && nc < ac as i32 {
                                            let v = prev[[nr as usize, nc as usize]];
                                            if v.is_finite() { sum += v; cnt += 1; }
                                        }
                                    }
                                }
                                if cnt >= 2 {
                                    aligned.set(r, c, sum / cnt as f64);
                                    filled += 1;
                                }
                            }
                        }
                        if filled == 0 { break; }
                        let remaining = aligned.data().iter().filter(|v| !v.is_finite()).count();
                        if remaining == 0 { break; }
                    }
                    let final_valid = aligned.data().iter().filter(|v| v.is_finite()).count();
                    eprintln!("  Post-align gap-fill: {:.1}% → {:.1}%",
                        pct_aligned, final_valid as f64 / total_aligned as f64 * 100.0);
                }

                write_result(&aligned, &output, compress)?;
                println!(
                    "Aligned to reference: {} x {} -> {} x {}",
                    out_cols, out_rows, ac, ar
                );
            }

            let elapsed = start.elapsed();
            let total_items: usize = scenes.iter().map(|s| s.tiles.len()).sum();
            println!(
                "Composite: {} dates ({} tiles) -> {} x {} ({:.1}M cells)",
                scenes.len(), total_items, out_cols, out_rows,
                (out_cols * out_rows) as f64 / 1e6,
            );
            eprintln!(
                "  ℹ If fewer dates than expected were used, check ⚠ warnings above.\n    Common causes: CRS mismatch across UTM zones, cloud mask failures."
            );
            done("STAC composite", &output, elapsed);
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
                                || cat.description
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
                        println!("  Showing {} of {} available catalogs:\n", display_count, total);
                        for (i, catalog_info) in all_catalogs.iter().take(display_count).enumerate() {
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

            println!("\n💡 Usage: surtgis stac composite --catalog {} --collection <id> --asset <band> ...", catalog);
        }

        StacCommands::TimeSeries {
            catalog, bbox, collection, asset, datetime, interval,
            scl_asset, max_scenes, align_to, output,
        } => {
            // Multi-band support: --asset "red,nir,swir16,blue"
            let assets: Vec<&str> = asset.split(',').map(|s| s.trim()).collect();
            if assets.len() > 1 {
                println!("Multi-band time series: {} bands", assets.len());
                let total_start = Instant::now();
                for (band_idx, band) in assets.iter().enumerate() {
                    let band_outdir = output.join(band);
                    println!("\n[{}/{}] Band: {} → {}", band_idx + 1, assets.len(), band, band_outdir.display());
                    handle_time_series(
                        &catalog, &bbox, &collection, band, &datetime,
                        &interval, &scl_asset, max_scenes, align_to.as_ref(),
                        &band_outdir, compress,
                    )?;
                }
                println!("\nAll {} bands complete in {:.1?}", assets.len(), total_start.elapsed());
            } else {
                handle_time_series(
                    &catalog, &bbox, &collection, &asset, &datetime,
                    &interval, &scl_asset, max_scenes, align_to.as_ref(),
                    &output, compress,
                )?;
            }
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

    let cat = StacCatalog::from_str_or_url(catalog);
    let bb = parse_bbox(bbox)?;

    // Search for items
    let search_limit = (max_scenes * 4) as u32;
    let params = StacSearchParams::new()
        .bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y)
        .collections(&[collection])
        .datetime(datetime)
        .limit(search_limit);

    let pb = spinner(&format!(
        "Searching for {} scenes...",
        collection
    ));
    let client_opts = StacClientOptions {
        max_items: (max_scenes * 4) as usize,
        ..StacClientOptions::default()
    };
    let client = StacClientBlocking::new(cat, client_opts)
        .context("Failed to create STAC client")?;
    let items = client.search_all(&params).context("STAC search failed")?;
    pb.finish_and_clear();

    if items.is_empty() {
        anyhow::bail!(
            "No items found for {} in bbox {}",
            collection,
            bbox
        );
    }

    // Introspect first item to auto-detect collection schema
    let pb = spinner("Introspecting collection schema...");
    let schema = StacCollectionSchema::from_stac_item(collection, &items[0])
        .context("Failed to introspect STAC collection")?;
    pb.finish_and_clear();

    eprintln!("📊 Collection: {}", schema.collection_name);
    eprintln!("   Available bands: {}", schema.format_bands());
    eprintln!("   Cloud masking: {}", match &schema.cloud_mask_type {
        CloudMaskType::Categorical { asset, num_classes } => format!("Categorical {} ({} classes)", asset, num_classes),
        CloudMaskType::Bitmask { asset, bits } => format!("Bitmask {} ({} bits)", asset, bits.len()),
        CloudMaskType::None => "None (SAR)".to_string(),
    });

    // Find best matching band
    let band_info = schema.find_band_by_name(asset)
        .context(format!(
            "Band '{}' not found. Available: {}",
            asset, schema.format_bands()
        ))?;
    eprintln!("   Band matched: {} → {}", asset, band_info.asset_key);

    // Create cloud masking strategy based on auto-detected type
    let cloud_mask_strategy = create_cloud_mask_strategy(&schema.cloud_mask_type);

    // Display info about found items
    for (idx, item) in items.iter().take(max_scenes).enumerate() {
        let date = item.properties.datetime.as_deref().unwrap_or("-");
        let cloud_cover = item.properties.eo_cloud_cover.unwrap_or(0.0);
        if idx < 3 {
            eprintln!("  Scene {}: {} [{}% cloud]", idx + 1, date, cloud_cover as u32);
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
            i + 1, max_scenes, item.id, cloud as u32, date
        ));

        // Resolve data asset
        let data_asset = resolve_asset_key(item, &band_info.asset_key)
            .and_then(|(_, a)| {
                client.sign_asset_href(&a.href, item.collection.as_deref().unwrap_or(""))
                    .ok()
                    .map(|h| (h, item.epsg()))
            });

        // Resolve cloud mask asset (if applicable)
        let mask_href = schema.cloud_mask_asset.as_ref().and_then(|mask_name| {
            resolve_asset_key(item, mask_name)
                .and_then(|(_, a)| {
                    client.sign_asset_href(&a.href, item.collection.as_deref().unwrap_or(""))
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

        let mut raster = match reader.read_bbox::<f64>(&read_bb, None) {
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
            if let Ok(mask_raster) = mask_reader.read_bbox::<f64>(&read_bb, None) {
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
            collection, asset, items.len()
        );
    }

    eprintln!("  ✅ Successfully loaded {} scenes", successful);

    // Composite all scenes
    let pb = spinner(&format!(
        "Compositing {} scenes (median stack)...",
        rasters.len()
    ));
    let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
    let result = mosaic(&refs, None)
        .context("Failed to mosaic scenes")?;
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

/// Read a COG tile (data band) at native resolution into a raster.
/// Applies nodata filtering (DN=0 → NaN). Used by parallel tile download.
fn read_cog_tile(href: &str, bb: &BBox, log_meta: bool) -> Option<surtgis_core::Raster<f64>> {
    let mut dr = CogReaderBlocking::open(href, CogReaderOptions::default()).ok()?;
    let tile_meta = dr.metadata();
    if log_meta {
        eprintln!("    COG meta: {}x{} bps={} sf={} compression={} px={:.0}m",
            tile_meta.width, tile_meta.height,
            tile_meta.bits_per_sample, tile_meta.sample_format,
            tile_meta.compression,
            tile_meta.geo_transform.pixel_width.abs());
    }
    let mut r: surtgis_core::Raster<f64> = dr.read_bbox(bb, None).ok()?;
    let nodata_val = tile_meta.nodata.unwrap_or(0.0);
    for val in r.data_mut().iter_mut() {
        if *val == nodata_val || *val == 0.0 {
            *val = f64::NAN;
        }
    }
    Some(r)
}

/// Read a COG tile (mask/SCL band) at native resolution without nodata filtering.
fn read_cog_tile_raw(href: &str, bb: &BBox) -> Option<surtgis_core::Raster<f64>> {
    let mut sr = CogReaderBlocking::open(href, CogReaderOptions::default()).ok()?;
    sr.read_bbox(bb, None).ok()
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
    println!("Time series: {} intervals ({}) from {} to {}",
        intervals.len(), interval, parts[0], parts[1]);

    // Optionally load align-to reference
    let reference = match align_to {
        Some(path) => {
            let r: surtgis_core::Raster<f64> = surtgis_core::io::read_geotiff(path, None)
                .context("Failed to read align-to reference")?;
            Some(r)
        }
        None => None,
    };

    std::fs::create_dir_all(outdir)?;
    let start = Instant::now();

    let mut success = 0;
    let mut metadata: Vec<serde_json::Value> = Vec::new();

    for (i, (win_start, win_end)) in intervals.iter().enumerate() {
        let win_dt = format!("{}/{}", format_date(win_start), format_date(win_end));
        let label = format_date(win_start);

        println!("[{}/{}] {} → {}", i + 1, intervals.len(), format_date(win_start), format_date(win_end));

        match fetch_stac_band(
            catalog, bbox, collection, asset, &win_dt,
            max_scenes, reference.as_ref(),
        ) {
            Ok(raster) => {
                let (rows, cols) = raster.shape();
                let valid = raster.data().iter().filter(|v| v.is_finite()).count();
                let total = rows * cols;
                let pct = if total > 0 { valid as f64 / total as f64 * 100.0 } else { 0.0 };

                let filename = format!("{}_{}.tif", asset, label);
                let path = outdir.join(&filename);
                write_result(&raster, &path, compress)?;

                println!("  → {} ({}x{}, {:.1}% valid)", filename, cols, rows, pct);

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
                eprintln!("  ⚠️ No data for this interval: {}", e);
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

    done(&format!("Time series ({}/{})", success, intervals.len()), outdir, start.elapsed());
    Ok(())
}

/// Simple date struct for interval splitting.
#[derive(Clone, Copy)]
struct SimpleDate {
    year: i32,
    month: u32,
    day: u32,
}

fn parse_date(s: &str) -> Result<SimpleDate> {
    let parts: Vec<&str> = s.trim().split('-').collect();
    if parts.len() != 3 {
        anyhow::bail!("invalid date format: '{}' (expected YYYY-MM-DD)", s);
    }
    Ok(SimpleDate {
        year: parts[0].parse().context("invalid year")?,
        month: parts[1].parse().context("invalid month")?,
        day: parts[2].parse().context("invalid day")?,
    })
}

fn format_date(d: &SimpleDate) -> String {
    format!("{:04}-{:02}-{:02}", d.year, d.month, d.day)
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 29 } else { 28 },
        _ => 30,
    }
}

fn advance_days(d: &SimpleDate, n: u32) -> SimpleDate {
    let mut y = d.year;
    let mut m = d.month;
    let mut day = d.day + n;
    loop {
        let dim = days_in_month(y, m);
        if day <= dim { break; }
        day -= dim;
        m += 1;
        if m > 12 { m = 1; y += 1; }
    }
    SimpleDate { year: y, month: m, day }
}

fn advance_months(d: &SimpleDate, n: u32) -> SimpleDate {
    let mut m = d.month + n;
    let mut y = d.year;
    while m > 12 { m -= 12; y += 1; }
    let day = d.day.min(days_in_month(y, m));
    SimpleDate { year: y, month: m, day }
}

fn date_le(a: &SimpleDate, b: &SimpleDate) -> bool {
    (a.year, a.month, a.day) <= (b.year, b.month, b.day)
}

fn split_date_range(start: &SimpleDate, end: &SimpleDate, interval: &str) -> Result<Vec<(SimpleDate, SimpleDate)>> {
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
        let actual_end = if date_le(&win_end, end) { win_end } else { *end };
        windows.push((cursor, actual_end));
        cursor = next;
    }

    if windows.is_empty() {
        anyhow::bail!("date range too short for interval '{}'", interval);
    }
    Ok(windows)
}

/// DEPRECATED: Use fetch_stac_band() instead
/// Backward-compatible wrapper for Sentinel-2 specific fetching
#[deprecated(since = "0.3.0", note = "use fetch_stac_band() instead")]
pub fn fetch_s2_band_from_stac(
    catalog: &str,
    bbox: &str,
    collection: &str,
    asset: &str,
    datetime: &str,
    max_scenes: usize,
    _scl_asset: &str,
    _scl_keep: &str,
    align_to: Option<&surtgis_core::Raster<f64>>,
) -> Result<surtgis_core::Raster<f64>> {
    // Delegate to new multi-collection function
    // (scl_asset and scl_keep are ignored, profile determines masking strategy)
    fetch_stac_band(catalog, bbox, collection, asset, datetime, max_scenes, align_to)
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
    fn test_collection_profile_unknown() {
        assert!(CollectionProfile::from_collection_name("unknown-collection").is_err());
    }

    #[test]
    fn test_collection_profile_debug() {
        let s2_profile = CollectionProfile::from_collection_name("sentinel-2-l2a").unwrap();
        let debug_str = format!("{:?}", s2_profile);
        assert!(debug_str.contains("Sentinel2L2A"));
    }
}
