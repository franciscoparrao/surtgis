//! Geometry simplification algorithms
//!
//! - Douglas-Peucker: preserves shape character, fast
//! - Visvalingam-Whyatt: area-based, better for cartographic display

use geo::{Geometry, LineString, MultiLineString, MultiPolygon, Polygon};
use geo::{Simplify, SimplifyVw};

/// Parameters for simplification
#[derive(Debug, Clone)]
pub struct SimplifyParams {
    /// Tolerance (epsilon for DP, area threshold for VW)
    pub tolerance: f64,
}

impl Default for SimplifyParams {
    fn default() -> Self {
        Self { tolerance: 1.0 }
    }
}

/// Simplify a LineString using Douglas-Peucker algorithm.
///
/// Removes vertices that deviate less than `tolerance` from the
/// simplified line.
///
/// # Arguments
/// * `line` - Input LineString
/// * `tolerance` - Maximum allowed deviation
pub fn simplify_dp(geom: &Geometry<f64>, tolerance: f64) -> Geometry<f64> {
    match geom {
        Geometry::LineString(ls) => Geometry::LineString(ls.simplify(&tolerance)),
        Geometry::Polygon(p) => Geometry::Polygon(simplify_polygon_dp(p, tolerance)),
        Geometry::MultiLineString(mls) => {
            let simplified: Vec<LineString<f64>> =
                mls.0.iter().map(|ls| ls.simplify(&tolerance)).collect();
            Geometry::MultiLineString(MultiLineString::new(simplified))
        }
        Geometry::MultiPolygon(mp) => {
            let simplified: Vec<Polygon<f64>> = mp
                .0
                .iter()
                .map(|p| simplify_polygon_dp(p, tolerance))
                .collect();
            Geometry::MultiPolygon(MultiPolygon::new(simplified))
        }
        other => other.clone(),
    }
}

fn simplify_polygon_dp(polygon: &Polygon<f64>, tolerance: f64) -> Polygon<f64> {
    let exterior = polygon.exterior().simplify(&tolerance);
    let interiors: Vec<LineString<f64>> = polygon
        .interiors()
        .iter()
        .map(|ring| ring.simplify(&tolerance))
        .filter(|ring| ring.0.len() >= 4) // Must remain valid ring
        .collect();
    Polygon::new(exterior, interiors)
}

/// Simplify a geometry using Visvalingam-Whyatt algorithm.
///
/// Removes vertices based on the effective area they contribute.
/// Better results for cartographic generalization.
///
/// # Arguments
/// * `geom` - Input geometry
/// * `tolerance` - Minimum effective area to retain a vertex
pub fn simplify_vw(geom: &Geometry<f64>, tolerance: f64) -> Geometry<f64> {
    match geom {
        Geometry::LineString(ls) => Geometry::LineString(ls.simplify_vw(&tolerance)),
        Geometry::Polygon(p) => Geometry::Polygon(simplify_polygon_vw(p, tolerance)),
        Geometry::MultiLineString(mls) => {
            let simplified: Vec<LineString<f64>> =
                mls.0.iter().map(|ls| ls.simplify_vw(&tolerance)).collect();
            Geometry::MultiLineString(MultiLineString::new(simplified))
        }
        Geometry::MultiPolygon(mp) => {
            let simplified: Vec<Polygon<f64>> = mp
                .0
                .iter()
                .map(|p| simplify_polygon_vw(p, tolerance))
                .collect();
            Geometry::MultiPolygon(MultiPolygon::new(simplified))
        }
        other => other.clone(),
    }
}

fn simplify_polygon_vw(polygon: &Polygon<f64>, tolerance: f64) -> Polygon<f64> {
    let exterior = polygon.exterior().simplify_vw(&tolerance);
    let interiors: Vec<LineString<f64>> = polygon
        .interiors()
        .iter()
        .map(|ring| ring.simplify_vw(&tolerance))
        .filter(|ring| ring.0.len() >= 4)
        .collect();
    Polygon::new(exterior, interiors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zigzag_line() -> LineString<f64> {
        // A zigzag line with small deviations
        LineString::from(vec![
            (0.0, 0.0),
            (1.0, 0.1),  // small deviation
            (2.0, 0.0),
            (3.0, -0.05), // tiny deviation
            (4.0, 0.0),
            (5.0, 0.2),  // larger deviation
            (6.0, 0.0),
            (7.0, 0.0),
            (8.0, 0.0),
            (9.0, 0.0),
            (10.0, 0.0),
        ])
    }

    fn complex_polygon() -> Polygon<f64> {
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (1.0, 0.1),
            (2.0, 0.0),
            (3.0, 0.05),
            (4.0, 0.0),
            (5.0, 0.0),
            (5.0, 5.0),
            (4.0, 4.9),
            (3.0, 5.0),
            (2.0, 5.1),
            (1.0, 5.0),
            (0.0, 5.0),
            (0.0, 0.0),
        ]);
        Polygon::new(exterior, vec![])
    }

    #[test]
    fn test_simplify_dp_reduces_vertices() {
        let line = zigzag_line();
        let original_count = line.0.len();

        let simplified = simplify_dp(&Geometry::LineString(line), 0.15);

        if let Geometry::LineString(ls) = simplified {
            assert!(
                ls.0.len() < original_count,
                "Should reduce vertices: {} -> {}",
                original_count,
                ls.0.len()
            );
            // Start and end should be preserved
            assert_eq!(ls.0.first().unwrap().x, 0.0);
            assert_eq!(ls.0.last().unwrap().x, 10.0);
        } else {
            panic!("Expected LineString");
        }
    }

    #[test]
    fn test_simplify_dp_high_tolerance() {
        let line = zigzag_line();
        let simplified = simplify_dp(&Geometry::LineString(line), 10.0);

        if let Geometry::LineString(ls) = simplified {
            // With very high tolerance, should reduce to just 2 points
            assert_eq!(ls.0.len(), 2, "High tolerance should leave only endpoints");
        }
    }

    #[test]
    fn test_simplify_dp_zero_tolerance() {
        let line = zigzag_line();
        let original_count = line.0.len();
        let simplified = simplify_dp(&Geometry::LineString(line), 0.0);

        if let Geometry::LineString(ls) = simplified {
            assert_eq!(ls.0.len(), original_count, "Zero tolerance should keep all vertices");
        }
    }

    #[test]
    fn test_simplify_polygon() {
        let poly = complex_polygon();
        let original_count = poly.exterior().0.len();

        let simplified = simplify_dp(&Geometry::Polygon(poly), 0.15);

        if let Geometry::Polygon(p) = simplified {
            assert!(
                p.exterior().0.len() < original_count,
                "Polygon should be simplified: {} -> {}",
                original_count,
                p.exterior().0.len()
            );
            // Should still be a closed ring
            assert_eq!(p.exterior().0.first(), p.exterior().0.last());
        }
    }

    #[test]
    fn test_simplify_vw() {
        let line = zigzag_line();
        let original_count = line.0.len();

        let simplified = simplify_vw(&Geometry::LineString(line), 0.5);

        if let Geometry::LineString(ls) = simplified {
            assert!(
                ls.0.len() < original_count,
                "VW should reduce vertices: {} -> {}",
                original_count,
                ls.0.len()
            );
        }
    }

    #[test]
    fn test_simplify_preserves_non_simplifiable() {
        // A Point cannot be simplified
        let point = Geometry::Point(geo::Point::new(1.0, 2.0));
        let result = simplify_dp(&point, 1.0);

        if let Geometry::Point(p) = result {
            assert_eq!(p.x(), 1.0);
            assert_eq!(p.y(), 2.0);
        }
    }
}
