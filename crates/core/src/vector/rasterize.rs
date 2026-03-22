//! Polygon rasterization and raster clipping.

use geo::BoundingRect;
use geo::Contains;
use geo_types::{Geometry, Point, Polygon};

use crate::error::{Error, Result};
use crate::raster::{GeoTransform, Raster};

use super::{AttributeValue, FeatureCollection};

/// Rasterize polygons: for each cell, set value based on which polygon contains it.
///
/// The cell value comes from:
/// - If `attribute` is `Some(name)`: the feature property with that name (must be numeric).
/// - Otherwise: the feature's 1-based sequential index (1.0, 2.0, ...).
///
/// Cells not inside any polygon remain NaN.
pub fn rasterize_polygons(
    features: &FeatureCollection,
    transform: &GeoTransform,
    rows: usize,
    cols: usize,
    attribute: Option<&str>,
) -> Result<Raster<f64>> {
    let mut output = Raster::filled(rows, cols, f64::NAN);
    output.set_transform(*transform);

    for (idx, feature) in features.iter().enumerate() {
        let value = match attribute {
            Some(attr_name) => match feature.get_property(attr_name) {
                Some(AttributeValue::Int(v)) => *v as f64,
                Some(AttributeValue::Float(v)) => *v,
                Some(_) => {
                    return Err(Error::Other(format!(
                        "Property '{attr_name}' is not numeric for feature {idx}"
                    )));
                }
                None => {
                    return Err(Error::Other(format!(
                        "Property '{attr_name}' not found in feature {idx}"
                    )));
                }
            },
            None => (idx + 1) as f64,
        };

        let geom = match &feature.geometry {
            Some(g) => g,
            None => continue,
        };

        match geom {
            Geometry::Polygon(poly) => {
                rasterize_single_polygon(poly, transform, rows, cols, value, &mut output);
            }
            Geometry::MultiPolygon(mp) => {
                for poly in &mp.0 {
                    rasterize_single_polygon(poly, transform, rows, cols, value, &mut output);
                }
            }
            _ => {
                // Skip non-polygon geometries silently
            }
        }
    }

    Ok(output)
}

/// Rasterize a single polygon into the output raster, using bounding-box optimization.
fn rasterize_single_polygon(
    polygon: &Polygon<f64>,
    transform: &GeoTransform,
    rows: usize,
    cols: usize,
    value: f64,
    output: &mut Raster<f64>,
) {
    // Get polygon bounding box to limit iteration
    let bbox = match polygon.bounding_rect() {
        Some(r) => r,
        None => return,
    };

    // Convert bbox corners to pixel coordinates
    let (min_col_f, min_row_f) = transform.geo_to_pixel(bbox.min().x, bbox.max().y);
    let (max_col_f, max_row_f) = transform.geo_to_pixel(bbox.max().x, bbox.min().y);

    // Clamp to raster bounds
    let r_start = (min_row_f.floor().max(0.0) as usize).min(rows);
    let r_end = ((max_row_f.ceil() as usize) + 1).min(rows);
    let c_start = (min_col_f.floor().max(0.0) as usize).min(cols);
    let c_end = ((max_col_f.ceil() as usize) + 1).min(cols);

    for row in r_start..r_end {
        for col in c_start..c_end {
            let (x, y) = transform.pixel_to_geo(col, row);
            let pt = Point::new(x, y);
            if polygon.contains(&pt) {
                // Last-writer-wins for overlapping polygons
                let _ = output.set(row, col, value);
            }
        }
    }
}

/// Clip a raster by a polygon: cells outside the polygon are set to NaN.
pub fn clip_raster_by_polygon(raster: &Raster<f64>, polygon: &Polygon<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = raster.shape();
    let transform = raster.transform();

    // Clone the raster, then mask out exterior cells
    let mut output = raster.clone();

    // Get polygon bounding box for optimization
    let bbox = polygon.bounding_rect();

    for row in 0..rows {
        for col in 0..cols {
            let (x, y) = transform.pixel_to_geo(col, row);
            let pt = Point::new(x, y);

            // Quick bbox rejection
            if let Some(ref rect) = bbox {
                if x < rect.min().x || x > rect.max().x || y < rect.min().y || y > rect.max().y {
                    let _ = output.set(row, col, f64::NAN);
                    continue;
                }
            }

            if !polygon.contains(&pt) {
                let _ = output.set(row, col, f64::NAN);
            }
        }
    }

    Ok(output)
}

/// Clip a raster by all polygons in a FeatureCollection (union of polygons).
///
/// A cell is kept if it falls inside *any* polygon in the collection.
pub fn clip_raster(raster: &Raster<f64>, features: &FeatureCollection) -> Result<Raster<f64>> {
    if features.is_empty() {
        return Err(Error::Other(
            "FeatureCollection is empty, nothing to clip by".into(),
        ));
    }

    // Collect all polygons from all features
    let mut polygons: Vec<&Polygon<f64>> = Vec::new();
    for feature in features.iter() {
        match &feature.geometry {
            Some(Geometry::Polygon(p)) => polygons.push(p),
            Some(Geometry::MultiPolygon(mp)) => {
                for p in &mp.0 {
                    polygons.push(p);
                }
            }
            _ => {}
        }
    }

    if polygons.is_empty() {
        return Err(Error::Other(
            "No polygon geometries found in FeatureCollection".into(),
        ));
    }

    // Build combined bounding box from all polygons
    let mut combined_min_x = f64::INFINITY;
    let mut combined_min_y = f64::INFINITY;
    let mut combined_max_x = f64::NEG_INFINITY;
    let mut combined_max_y = f64::NEG_INFINITY;

    for poly in &polygons {
        if let Some(rect) = poly.bounding_rect() {
            combined_min_x = combined_min_x.min(rect.min().x);
            combined_min_y = combined_min_y.min(rect.min().y);
            combined_max_x = combined_max_x.max(rect.max().x);
            combined_max_y = combined_max_y.max(rect.max().y);
        }
    }

    let (rows, cols) = raster.shape();
    let transform = raster.transform();
    let mut output = raster.clone();

    for row in 0..rows {
        for col in 0..cols {
            let (x, y) = transform.pixel_to_geo(col, row);

            // Quick combined bbox rejection
            if x < combined_min_x || x > combined_max_x || y < combined_min_y || y > combined_max_y
            {
                let _ = output.set(row, col, f64::NAN);
                continue;
            }

            let pt = Point::new(x, y);
            let inside = polygons.iter().any(|poly| poly.contains(&pt));
            if !inside {
                let _ = output.set(row, col, f64::NAN);
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Feature;
    use geo_types::Coord;

    /// Create a simple rectangular polygon.
    fn rect_polygon(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Polygon<f64> {
        Polygon::new(
            geo_types::LineString::new(vec![
                Coord {
                    x: x_min,
                    y: y_min,
                },
                Coord {
                    x: x_max,
                    y: y_min,
                },
                Coord {
                    x: x_max,
                    y: y_max,
                },
                Coord {
                    x: x_min,
                    y: y_max,
                },
                Coord {
                    x: x_min,
                    y: y_min,
                },
            ]),
            vec![],
        )
    }

    #[test]
    fn test_clip_raster_by_polygon() {
        // 10x10 raster, each cell is 1.0 unit, origin at (0, 10), pixel_height = -1.0
        let mut raster = Raster::filled(10, 10, 1.0_f64);
        raster.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        // Clip polygon covers cells (2..8, 3..7) in geographic coords
        let poly = rect_polygon(2.0, 3.0, 8.0, 7.0);
        let clipped = clip_raster_by_polygon(&raster, &poly).unwrap();

        let (rows, cols) = clipped.shape();
        let mut valid_count = 0;
        for r in 0..rows {
            for c in 0..cols {
                let v = clipped.get(r, c).unwrap();
                if v.is_finite() {
                    valid_count += 1;
                }
            }
        }
        // The polygon covers a 6x4 area in geo coords; with 1-unit cells, that's
        // approximately 6*4 = 24 cell-centers inside (exact count depends on center vs edge).
        // Cell centers at col=2..7 (x=2.5..7.5) are inside [2,8], row=3..6 (y=3.5..6.5) are inside [3,7].
        // That gives 6 cols * 4 rows = 24 cells.
        assert!(valid_count > 0, "Some cells must survive clipping");
        assert!(valid_count < 100, "Not all cells should survive");
        assert_eq!(valid_count, 24);
    }

    #[test]
    fn test_rasterize_single_polygon() {
        let transform = GeoTransform::new(0.0, 10.0, 1.0, -1.0);
        let poly = rect_polygon(2.0, 3.0, 8.0, 7.0);

        let mut fc = FeatureCollection::new();
        fc.push(Feature::new(Geometry::Polygon(poly)));

        let result = rasterize_polygons(&fc, &transform, 10, 10, None).unwrap();

        let mut count_ones = 0;
        let mut count_nan = 0;
        for r in 0..10 {
            for c in 0..10 {
                let v = result.get(r, c).unwrap();
                if v == 1.0 {
                    count_ones += 1;
                } else if v.is_nan() {
                    count_nan += 1;
                }
            }
        }
        assert_eq!(count_ones, 24);
        assert_eq!(count_nan, 76);
    }

    #[test]
    fn test_rasterize_with_attribute() {
        let transform = GeoTransform::new(0.0, 5.0, 1.0, -1.0);
        let poly = rect_polygon(1.0, 1.0, 4.0, 4.0);

        let mut feat = Feature::new(Geometry::Polygon(poly));
        feat.set_property("class", AttributeValue::Float(42.0));

        let mut fc = FeatureCollection::new();
        fc.push(feat);

        let result = rasterize_polygons(&fc, &transform, 5, 5, Some("class")).unwrap();

        // Check that rasterized cells have value 42.0
        let mut found_42 = false;
        for r in 0..5 {
            for c in 0..5 {
                let v = result.get(r, c).unwrap();
                if v == 42.0 {
                    found_42 = true;
                }
            }
        }
        assert!(found_42);
    }

    #[test]
    fn test_clip_raster_union() {
        let mut raster = Raster::filled(10, 10, 5.0_f64);
        raster.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        // Two non-overlapping polygons
        let poly1 = rect_polygon(0.0, 8.0, 3.0, 10.0); // top-left area
        let poly2 = rect_polygon(7.0, 0.0, 10.0, 3.0); // bottom-right area

        let mut fc = FeatureCollection::new();
        fc.push(Feature::new(Geometry::Polygon(poly1)));
        fc.push(Feature::new(Geometry::Polygon(poly2)));

        let clipped = clip_raster(&raster, &fc).unwrap();

        let mut valid = 0;
        for r in 0..10 {
            for c in 0..10 {
                if clipped.get(r, c).unwrap().is_finite() {
                    valid += 1;
                }
            }
        }
        // Two small rectangles, each covering some cells
        assert!(valid > 0);
        assert!(valid < 100);
    }

    #[test]
    fn test_empty_feature_collection_errors() {
        let raster = Raster::filled(5, 5, 1.0_f64);
        let fc = FeatureCollection::new();
        assert!(clip_raster(&raster, &fc).is_err());
    }
}
