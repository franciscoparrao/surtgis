//! CLI handlers for vector geoprocessing commands.

use std::time::Instant;

use anyhow::{Context, Result};

use surtgis_algorithms::vector::overlay;
use surtgis_algorithms::vector::{buffer_points, BufferParams};
use surtgis_core::vector::{read_vector, FeatureCollection};

use crate::commands::VectorCommands;
use crate::helpers::spinner;

pub fn handle(action: VectorCommands) -> Result<()> {
    match action {
        VectorCommands::Intersection { input_a, input_b, output } => {
            let start = Instant::now();
            let a = read_vector(&input_a).context("Failed to read layer A")?;
            let b = read_vector(&input_b).context("Failed to read layer B")?;
            println!("A: {} features, B: {} features", a.features.len(), b.features.len());

            let pb = spinner("Computing intersection...");
            let result = overlay::intersection(&a, &b);
            pb.finish_and_clear();

            println!("Result: {} features in {:.1?}", result.features.len(), start.elapsed());
            write_vector(&result, &output)?;
        }

        VectorCommands::Union { input_a, input_b, output } => {
            let start = Instant::now();
            let a = read_vector(&input_a)?;
            let b = read_vector(&input_b)?;

            let pb = spinner("Computing union...");
            let result = overlay::union(&a, &b);
            pb.finish_and_clear();

            println!("Result: {} features in {:.1?}", result.features.len(), start.elapsed());
            write_vector(&result, &output)?;
        }

        VectorCommands::Difference { input_a, input_b, output } => {
            let start = Instant::now();
            let a = read_vector(&input_a)?;
            let b = read_vector(&input_b)?;

            let pb = spinner("Computing difference...");
            let result = overlay::difference(&a, &b);
            pb.finish_and_clear();

            println!("Result: {} features in {:.1?}", result.features.len(), start.elapsed());
            write_vector(&result, &output)?;
        }

        VectorCommands::SymDifference { input_a, input_b, output } => {
            let start = Instant::now();
            let a = read_vector(&input_a)?;
            let b = read_vector(&input_b)?;

            let pb = spinner("Computing symmetric difference...");
            let result = overlay::symmetric_difference(&a, &b);
            pb.finish_and_clear();

            println!("Result: {} features in {:.1?}", result.features.len(), start.elapsed());
            write_vector(&result, &output)?;
        }

        VectorCommands::Dissolve { input, output } => {
            let start = Instant::now();
            let fc = read_vector(&input)?;
            println!("Input: {} features", fc.features.len());

            let pb = spinner("Dissolving...");
            let result = overlay::dissolve(&fc);
            pb.finish_and_clear();

            println!("Dissolved to {} features in {:.1?}", result.features.len(), start.elapsed());
            write_vector(&result, &output)?;
        }

        VectorCommands::Buffer { input, distance, segments, output } => {
            let start = Instant::now();
            let fc = read_vector(&input)?;
            println!("Input: {} features, distance: {}", fc.features.len(), distance);

            let pb = spinner("Buffering...");
            let params = BufferParams {
                distance,
                segments,
            };

            let mut result = FeatureCollection::new();
            for feature in &fc.features {
                if let Some(geom) = &feature.geometry {
                    if let geo::Geometry::Point(pt) = geom {
                        let buffered = buffer_points(pt, &params);
                        result.push(surtgis_core::vector::Feature {
                            geometry: Some(geo::Geometry::Polygon(buffered)),
                            properties: feature.properties.clone(),
                            id: feature.id.clone(),
                        });
                    }
                }
            }
            pb.finish_and_clear();

            println!("Buffered {} features in {:.1?}", result.features.len(), start.elapsed());
            write_vector(&result, &output)?;
        }
    }
    Ok(())
}

fn write_vector(fc: &FeatureCollection, path: &std::path::Path) -> Result<()> {
    use geo::algorithm::coords_iter::CoordsIter;

    let features: Vec<serde_json::Value> = fc.features.iter().filter_map(|f| {
        let geom = f.geometry.as_ref()?;
        let coords: Vec<Vec<Vec<f64>>> = match geom {
            geo::Geometry::Polygon(p) => {
                let mut rings = vec![];
                rings.push(p.exterior().0.iter().map(|c| vec![c.x, c.y]).collect());
                for interior in p.interiors() {
                    rings.push(interior.0.iter().map(|c| vec![c.x, c.y]).collect());
                }
                rings
            }
            _ => return None,
        };
        Some(serde_json::json!({
            "type": "Feature",
            "geometry": { "type": "Polygon", "coordinates": coords },
            "properties": {}
        }))
    }).collect();

    let geojson = serde_json::json!({
        "type": "FeatureCollection",
        "features": features
    });

    std::fs::write(path, serde_json::to_string_pretty(&geojson)?)
        .context("Failed to write output")?;
    println!("Written: {}", path.display());
    Ok(())
}
