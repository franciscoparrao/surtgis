//! Handler for STAC catalog subcommands.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::time::Instant;

use surtgis_algorithms::imagery::cloud_mask_scl;
use surtgis_cloud::blocking::{CogReaderBlocking, StacClientBlocking};
use surtgis_cloud::{BBox, CogReaderOptions, StacCatalog, StacClientOptions, StacItem, StacSearchParams};

use crate::commands::StacCommands;
use crate::helpers::{done, parse_bbox, parse_scl_classes, spinner, write_result};
use crate::streaming::resolve_asset_key;

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
            scl_asset,
            scl_keep,
            output,
        } => {
            let cat = StacCatalog::from_str_or_url(&catalog);
            let bb = parse_bbox(&bbox)?;
            let keep_classes = parse_scl_classes(&scl_keep)?;

            // Search with high limit to find enough items across dates
            let search_limit = (max_scenes * 4) as u32;
            let params = StacSearchParams::new()
                .bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y)
                .collections(&[collection.as_str()])
                .datetime(&datetime)
                .limit(search_limit);

            let pb = spinner("Searching STAC catalog...");
            let client_opts = StacClientOptions {
                max_items: (max_scenes * 4) as usize,
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

            println!(
                "Found {} items across {} dates (using {} dates)",
                items.len(),
                by_date.len(),
                dates.len()
            );

            let start = Instant::now();

            // -- Phase 1: Resolve asset URLs for all items (no data download) --
            // For each date, collect (data_href, scl_href, epsg) tuples
            #[allow(dead_code)]
            struct SceneInfo {
                date: String,
                data_hrefs: Vec<String>,
                scl_hrefs: Vec<String>,
                epsg: Option<u32>,
            }

            let mut scenes: Vec<SceneInfo> = Vec::new();

            for date in &dates {
                let group = &by_date[date];
                let mut data_hrefs = Vec::new();
                let mut scl_hrefs = Vec::new();
                let mut scene_epsg = None;

                for item in group {
                    // Resolve and sign data asset
                    let data_result = resolve_asset_key(item, &asset)
                        .and_then(|(_, a)| {
                            client.sign_asset_href(
                                &a.href,
                                item.collection.as_deref().unwrap_or(""),
                            ).ok().map(|h| (h, item.epsg()))
                        });
                    // Resolve and sign SCL asset
                    let scl_result = resolve_asset_key(item, &scl_asset)
                        .and_then(|(_, a)| {
                            client.sign_asset_href(
                                &a.href,
                                item.collection.as_deref().unwrap_or(""),
                            ).ok()
                        });

                    if let (Some((dh, epsg)), Some(sh)) = (data_result, scl_result) {
                        data_hrefs.push(dh);
                        scl_hrefs.push(sh);
                        if scene_epsg.is_none() {
                            scene_epsg = epsg;
                        }
                    }
                }

                if !data_hrefs.is_empty() {
                    scenes.push(SceneInfo {
                        date: date.clone(),
                        data_hrefs,
                        scl_hrefs,
                        epsg: scene_epsg,
                    });
                }
            }

            if scenes.is_empty() {
                anyhow::bail!("No valid scenes found");
            }

            // -- Phase 2: Determine output grid from first scene's metadata --
            let first_href = &scenes[0].data_hrefs[0];
            let opts = CogReaderOptions::default();
            let probe_reader = CogReaderBlocking::open(first_href, opts)
                .context("Failed to probe first COG")?;
            let probe_meta = probe_reader.metadata();

            // Reproject user bbox to COG CRS if needed
            let cog_bb = {
                use surtgis_cloud::reproject;
                if let Some(epsg) = scenes[0].epsg {
                    if !reproject::is_wgs84(epsg) {
                        reproject::reproject_bbox_to_cog(&bb, epsg)
                    } else { bb }
                } else { bb }
            };

            let pixel_width = probe_meta.geo_transform.pixel_width.abs();
            let pixel_height = probe_meta.geo_transform.pixel_height.abs();
            let out_cols = ((cog_bb.max_x - cog_bb.min_x) / pixel_width).round() as usize;
            let out_rows = ((cog_bb.max_y - cog_bb.min_y) / pixel_height).round() as usize;

            let out_transform = surtgis_core::GeoTransform::new(
                cog_bb.min_x, cog_bb.max_y, pixel_width, -pixel_height,
            );
            let out_crs = scenes[0].epsg.map(surtgis_core::CRS::from_epsg);

            println!(
                "Output grid: {} x {} ({:.1}M cells), {} dates",
                out_cols, out_rows,
                (out_cols * out_rows) as f64 / 1e6,
                scenes.len()
            );

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
            let keep_ref = &keep_classes;
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
                    let strip_min_y = cog_bb.max_y - (row_end as f64 * pixel_height);
                    let strip_max_y = cog_bb.max_y - (row_start as f64 * pixel_height);
                    let strip_bb = BBox::new(
                        cog_bb.min_x, strip_min_y, cog_bb.max_x, strip_max_y,
                    );

                    print!(
                        "\r  Strip {}/{} (rows {}-{})...",
                        current_strip + 1, num_strips, row_start, row_end
                    );

                    // Collect masked strips from each scene
                    let mut scene_strips: Vec<ndarray::Array2<f64>> = Vec::with_capacity(n_scenes);

                    for scene in scenes_ref {
                        // Read data tiles for this strip bbox
                        let mut data_tiles: Vec<surtgis_core::Raster<f64>> = Vec::new();
                        let mut scl_tiles: Vec<surtgis_core::Raster<f64>> = Vec::new();

                        for (dh, sh) in scene.data_hrefs.iter().zip(scene.scl_hrefs.iter()) {
                            let opts = CogReaderOptions::default();
                            // Read data window
                            if let Ok(mut dr) = CogReaderBlocking::open(dh, opts) {
                                if let Ok(r) = dr.read_bbox::<f64>(&strip_bb, None) {
                                    data_tiles.push(r);
                                }
                            }
                            let opts2 = CogReaderOptions::default();
                            // Read SCL window
                            if let Ok(mut sr) = CogReaderBlocking::open(sh, opts2) {
                                if let Ok(r) = sr.read_bbox::<f64>(&strip_bb, None) {
                                    scl_tiles.push(r);
                                }
                            }
                        }

                        if data_tiles.is_empty() || scl_tiles.is_empty() {
                            continue;
                        }

                        // Mosaic spatial tiles for this date's strip
                        let data_m = if data_tiles.len() == 1 {
                            data_tiles.into_iter().next().unwrap()
                        } else {
                            let refs: Vec<&surtgis_core::Raster<f64>> = data_tiles.iter().collect();
                            match surtgis_core::mosaic(&refs, None) {
                                Ok(m) => m,
                                Err(_) => continue,
                            }
                        };

                        let scl_m = if scl_tiles.len() == 1 {
                            scl_tiles.into_iter().next().unwrap()
                        } else {
                            let refs: Vec<&surtgis_core::Raster<f64>> = scl_tiles.iter().collect();
                            match surtgis_core::mosaic(&refs, None) {
                                Ok(m) => m,
                                Err(_) => continue,
                            }
                        };

                        // Cloud mask
                        if let Ok(clean) = cloud_mask_scl(&data_m, &scl_m, keep_ref) {
                            scene_strips.push(clean.data().to_owned());
                        }
                    }

                    // Compute per-pixel median across scenes for this strip
                    let mut output = ndarray::Array2::<f64>::from_elem(
                        (actual_rows, out_cols), f64::NAN,
                    );

                    if !scene_strips.is_empty() {
                        // All strips should cover roughly the same area but may differ
                        // in exact dimensions. Use the output grid as reference.
                        let n = scene_strips.len();
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
                    }

                    Ok(output)
                },
            ).context("Failed to write streaming composite")?;

            println!(); // newline after \r progress
            let elapsed = start.elapsed();
            println!(
                "Composite: {} scenes -> {} x {} ({:.1}M cells)",
                scenes.len(), out_cols, out_rows,
                (out_cols * out_rows) as f64 / 1e6,
            );
            done("STAC composite", &output, elapsed);
        }
    }

    Ok(())
}
