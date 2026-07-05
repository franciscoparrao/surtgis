//! Polygon rasterization and raster clipping.

use geo::BoundingRect;
use geo::Contains;
use geo_types::{Geometry, LineString, Point, Polygon};

use crate::crs::CRS;
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
///
/// # CRS validation
///
/// `raster_crs` should be the CRS of the reference raster the output grid is
/// built from (`rows`/`cols`/`transform`) — pass `reference.crs()`. When
/// both `features.crs()` and `raster_crs` are known and disagree (per
/// [`CRS::is_equivalent`]), this returns [`Error::Other`] instead of
/// silently rasterizing in the wrong place (e.g. a WGS84 vector over a UTM
/// raster, which previously produced an empty or garbage result with no
/// error). If either CRS is unknown (`None`), no validation is performed —
/// callers with untrustworthy CRS metadata should reproject explicitly
/// rather than rely on this check.
pub fn rasterize_polygons(
    features: &FeatureCollection,
    transform: &GeoTransform,
    rows: usize,
    cols: usize,
    attribute: Option<&str>,
    raster_crs: Option<&CRS>,
) -> Result<Raster<f64>> {
    if let (Some(vector_crs), Some(raster_crs)) = (features.crs(), raster_crs) {
        if !vector_crs.is_equivalent(raster_crs) {
            return Err(Error::Other(format!(
                "vector CRS ({}) does not match raster CRS ({}); reproject the vector data before rasterizing",
                vector_crs.identifier(),
                raster_crs.identifier()
            )));
        }
    }

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

/// Rasterize a single polygon into the output raster.
///
/// For north-up transforms this uses an even-odd **scanline fill**: per raster
/// row it intersects the pixel-centre latitude with every ring edge (exterior
/// and holes), then fills the columns between intersection pairs. That is the
/// same centre-based rule GDAL/`rasterio` use, and it is O(edges·rows + filled
/// cells) instead of a point-in-polygon test per bounding-box cell — the latter
/// made large geographic grids ~1000× slower. Rotated transforms fall back to
/// the exact per-cell test, which does not assume horizontal scanlines.
fn rasterize_single_polygon(
    polygon: &Polygon<f64>,
    transform: &GeoTransform,
    rows: usize,
    cols: usize,
    value: f64,
    output: &mut Raster<f64>,
) {
    if !transform.is_north_up() {
        rasterize_single_polygon_pointwise(polygon, transform, rows, cols, value, output);
        return;
    }

    let bbox = match polygon.bounding_rect() {
        Some(r) => r,
        None => return,
    };

    // Rows spanned by the polygon bbox (pixel space); pixel_height is negative
    // for north-up, so max-y maps to the smaller row index.
    let (_, min_row_f) = transform.geo_to_pixel(bbox.min().x, bbox.max().y);
    let (_, max_row_f) = transform.geo_to_pixel(bbox.max().x, bbox.min().y);
    let r_start = (min_row_f.floor().max(0.0) as usize).min(rows);
    let r_end = ((max_row_f.ceil() as usize) + 1).min(rows);

    // Exterior ring first, then every hole; all contribute to the even-odd count.
    let rings: Vec<&LineString<f64>> = std::iter::once(polygon.exterior())
        .chain(polygon.interiors())
        .collect();

    let mut xs: Vec<f64> = Vec::new();
    for row in r_start..r_end {
        // Pixel-centre latitude of this row (north-up ⇒ depends only on row).
        let (_, y) = transform.pixel_to_geo(0, row);

        xs.clear();
        for ring in &rings {
            let pts = &ring.0;
            let n = pts.len();
            if n < 2 {
                continue;
            }
            for i in 0..n {
                let a = pts[i];
                let b = pts[(i + 1) % n];
                // Half-open crossing rule: count the edge iff `y` lies in
                // [min(y0,y1), max(y0,y1)) — avoids double-counting shared
                // vertices and skips horizontal edges (y0 == y1).
                if (a.y <= y && y < b.y) || (b.y <= y && y < a.y) {
                    let t = (y - a.y) / (b.y - a.y);
                    xs.push(a.x + t * (b.x - a.x));
                }
            }
        }
        if xs.len() < 2 {
            continue;
        }
        xs.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));

        // Fill the columns whose centre falls in each [x_start, x_end) span.
        for pair in xs.chunks_exact(2) {
            let (col_a, _) = transform.geo_to_pixel(pair[0], y);
            let (col_b, _) = transform.geo_to_pixel(pair[1], y);
            // A column `c` has centre at fractional col `c + 0.5`; it is inside
            // the span when `c + 0.5 ∈ [col_a, col_b)`.
            let c_start = ((col_a - 0.5).ceil().max(0.0) as usize).min(cols);
            let c_end = ((col_b - 0.5).ceil().max(0.0) as usize).min(cols);
            for col in c_start..c_end {
                // Last-writer-wins for overlapping polygons.
                let _ = output.set(row, col, value);
            }
        }
    }
}

/// Exact per-cell fallback: point-in-polygon test over the bounding box.
///
/// Used for rotated transforms, where raster rows are not horizontal in
/// geographic space and the scanline assumption does not hold.
fn rasterize_single_polygon_pointwise(
    polygon: &Polygon<f64>,
    transform: &GeoTransform,
    rows: usize,
    cols: usize,
    value: f64,
    output: &mut Raster<f64>,
) {
    let bbox = match polygon.bounding_rect() {
        Some(r) => r,
        None => return,
    };

    let (min_col_f, min_row_f) = transform.geo_to_pixel(bbox.min().x, bbox.max().y);
    let (max_col_f, max_row_f) = transform.geo_to_pixel(bbox.max().x, bbox.min().y);

    let r_start = (min_row_f.floor().max(0.0) as usize).min(rows);
    let r_end = ((max_row_f.ceil() as usize) + 1).min(rows);
    let c_start = (min_col_f.floor().max(0.0) as usize).min(cols);
    let c_end = ((max_col_f.ceil() as usize) + 1).min(cols);

    for row in r_start..r_end {
        for col in c_start..c_end {
            let (x, y) = transform.pixel_to_geo(col, row);
            let pt = Point::new(x, y);
            if polygon.contains(&pt) {
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
                Coord { x: x_min, y: y_min },
                Coord { x: x_max, y: y_min },
                Coord { x: x_max, y: y_max },
                Coord { x: x_min, y: y_max },
                Coord { x: x_min, y: y_min },
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

        let result = rasterize_polygons(&fc, &transform, 10, 10, None, None).unwrap();

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
    fn test_rasterize_matching_crs_ok() {
        // Vector and reference raster agree on CRS (EPSG:32719): rasterizing
        // must behave exactly as without CRS metadata at all.
        let transform = GeoTransform::new(0.0, 10.0, 1.0, -1.0);
        let poly = rect_polygon(2.0, 3.0, 8.0, 7.0);

        let mut fc = FeatureCollection::with_crs(Some(crate::crs::CRS::from_epsg(32719)));
        fc.push(Feature::new(Geometry::Polygon(poly)));

        let result = rasterize_polygons(
            &fc,
            &transform,
            10,
            10,
            None,
            Some(&crate::crs::CRS::from_epsg(32719)),
        )
        .unwrap();

        let mut count_ones = 0;
        for r in 0..10 {
            for c in 0..10 {
                if result.get(r, c).unwrap() == 1.0 {
                    count_ones += 1;
                }
            }
        }
        assert_eq!(count_ones, 24);
    }

    #[test]
    fn test_rasterize_mismatched_crs_errors() {
        // Closes A-13: a WGS84 vector rasterized against a UTM reference
        // raster must fail loudly instead of silently returning an
        // empty/garbage grid.
        let transform = GeoTransform::new(0.0, 10.0, 1.0, -1.0);
        let poly = rect_polygon(2.0, 3.0, 8.0, 7.0);

        let mut fc = FeatureCollection::with_crs(Some(crate::crs::CRS::wgs84()));
        fc.push(Feature::new(Geometry::Polygon(poly)));

        let raster_crs = crate::crs::CRS::from_epsg(32719);
        let err = rasterize_polygons(&fc, &transform, 10, 10, None, Some(&raster_crs))
            .expect_err("mismatched CRS must error, not silently rasterize");

        let msg = err.to_string();
        assert!(msg.contains("4326"), "message should name the vector CRS: {msg}");
        assert!(msg.contains("32719"), "message should name the raster CRS: {msg}");
    }

    #[test]
    fn test_rasterize_unknown_crs_skips_validation() {
        // Either side unknown (`None`) must not fail the check — same
        // lenient semantics as `raster::validate::check_same_crs`.
        let transform = GeoTransform::new(0.0, 10.0, 1.0, -1.0);
        let poly = rect_polygon(2.0, 3.0, 8.0, 7.0);

        let mut fc_no_crs = FeatureCollection::new();
        fc_no_crs.push(Feature::new(Geometry::Polygon(poly.clone())));
        let raster_crs = crate::crs::CRS::from_epsg(32719);
        assert!(
            rasterize_polygons(&fc_no_crs, &transform, 10, 10, None, Some(&raster_crs)).is_ok()
        );

        let mut fc_with_crs = FeatureCollection::with_crs(Some(crate::crs::CRS::wgs84()));
        fc_with_crs.push(Feature::new(Geometry::Polygon(poly)));
        assert!(rasterize_polygons(&fc_with_crs, &transform, 10, 10, None, None).is_ok());
    }

    #[test]
    fn test_rasterize_with_attribute() {
        let transform = GeoTransform::new(0.0, 5.0, 1.0, -1.0);
        let poly = rect_polygon(1.0, 1.0, 4.0, 4.0);

        let mut feat = Feature::new(Geometry::Polygon(poly));
        feat.set_property("class", AttributeValue::Float(42.0));

        let mut fc = FeatureCollection::new();
        fc.push(feat);

        let result = rasterize_polygons(&fc, &transform, 5, 5, Some("class"), None).unwrap();

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

    /// Rasterize `poly` with the scanline path (default) and with the exact
    /// per-cell point-in-polygon fallback, and assert both masks are identical.
    /// This guards the "0-px discrepancy vs rasterio" property while the fast
    /// scanline path is what actually runs.
    fn assert_scanline_matches_pointwise(poly: &Polygon<f64>, transform: &GeoTransform) {
        let (rows, cols) = (40usize, 40usize);

        let mut scan = Raster::filled(rows, cols, f64::NAN);
        scan.set_transform(*transform);
        rasterize_single_polygon(poly, transform, rows, cols, 1.0, &mut scan);

        let mut exact = Raster::filled(rows, cols, f64::NAN);
        exact.set_transform(*transform);
        rasterize_single_polygon_pointwise(poly, transform, rows, cols, 1.0, &mut exact);

        let mut diffs = 0;
        for r in 0..rows {
            for c in 0..cols {
                let a = scan.get(r, c).unwrap().is_finite();
                let b = exact.get(r, c).unwrap().is_finite();
                if a != b {
                    diffs += 1;
                }
            }
        }
        assert_eq!(diffs, 0, "scanline vs pointwise disagree on {diffs} cells");
    }

    #[test]
    fn test_scanline_parity_convex() {
        // North-up grid, 1-unit cells, origin (0, 40). Edges are offset from the
        // half-integer pixel centres so no centre lands exactly on the boundary
        // (that degenerate tie is where even-odd/GDAL and geo::Contains legitimately
        // differ; real geographic polygons never align to cell centres, which is
        // why the field study saw 0-px discrepancy vs rasterio).
        let t = GeoTransform::new(0.0, 40.0, 1.0, -1.0);
        assert_scanline_matches_pointwise(&rect_polygon(3.2, 4.3, 30.7, 22.8), &t);
    }

    #[test]
    fn test_scanline_parity_concave() {
        // An L-shaped (concave) polygon: the scanline must produce two fill
        // spans on the rows that cross the notch.
        let t = GeoTransform::new(0.0, 40.0, 1.0, -1.0);
        let l = Polygon::new(
            geo_types::LineString::new(vec![
                Coord { x: 5.0, y: 5.0 },
                Coord { x: 30.0, y: 5.0 },
                Coord { x: 30.0, y: 12.0 },
                Coord { x: 15.0, y: 12.0 },
                Coord { x: 15.0, y: 30.0 },
                Coord { x: 5.0, y: 30.0 },
                Coord { x: 5.0, y: 5.0 },
            ]),
            vec![],
        );
        assert_scanline_matches_pointwise(&l, &t);
    }

    #[test]
    fn test_scanline_parity_with_hole() {
        // Square with a square hole: interior-ring edges must re-toggle the
        // even-odd count so hole cells stay empty.
        let t = GeoTransform::new(0.0, 40.0, 1.0, -1.0);
        let with_hole = Polygon::new(
            geo_types::LineString::new(vec![
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 34.0, y: 4.0 },
                Coord { x: 34.0, y: 34.0 },
                Coord { x: 4.0, y: 34.0 },
                Coord { x: 4.0, y: 4.0 },
            ]),
            vec![geo_types::LineString::new(vec![
                Coord { x: 14.0, y: 14.0 },
                Coord { x: 24.0, y: 14.0 },
                Coord { x: 24.0, y: 24.0 },
                Coord { x: 14.0, y: 24.0 },
                Coord { x: 14.0, y: 14.0 },
            ])],
        );
        assert_scanline_matches_pointwise(&with_hole, &t);
    }
}
