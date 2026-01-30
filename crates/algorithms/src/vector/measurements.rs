//! Geometric measurements: area, length, perimeter

use geo::{Area as GeoArea, Euclidean, Geometry, Length};

/// Calculate the area of a geometry.
///
/// Returns unsigned area. For geographic CRS, results are in CRS units squared
/// (e.g., square degrees — project to a metric CRS for square meters).
pub fn area(geom: &Geometry<f64>) -> f64 {
    match geom {
        Geometry::Polygon(p) => p.unsigned_area(),
        Geometry::MultiPolygon(mp) => mp.unsigned_area(),
        Geometry::Rect(r) => r.unsigned_area(),
        _ => 0.0,
    }
}

/// Calculate the length of a linear geometry.
///
/// Returns Euclidean length in CRS units.
pub fn length(geom: &Geometry<f64>) -> f64 {
    match geom {
        Geometry::LineString(ls) => ls.length::<Euclidean>(),
        Geometry::MultiLineString(mls) => {
            mls.0.iter().map(|ls| ls.length::<Euclidean>()).sum()
        }
        Geometry::Line(l) => {
            let dx = l.end.x - l.start.x;
            let dy = l.end.y - l.start.y;
            (dx * dx + dy * dy).sqrt()
        }
        _ => 0.0,
    }
}

/// Calculate the perimeter of a polygon geometry.
///
/// Returns the total length of exterior and interior rings.
pub fn perimeter(geom: &Geometry<f64>) -> f64 {
    match geom {
        Geometry::Polygon(p) => {
            let ext = p.exterior().length::<Euclidean>();
            let int: f64 = p.interiors().iter().map(|r| r.length::<Euclidean>()).sum();
            ext + int
        }
        Geometry::MultiPolygon(mp) => {
            mp.0.iter()
                .map(|p| {
                    let ext = p.exterior().length::<Euclidean>();
                    let int: f64 = p.interiors().iter().map(|r| r.length::<Euclidean>()).sum();
                    ext + int
                })
                .sum()
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{Line, LineString, MultiLineString, Polygon, Coord};

    fn square() -> Polygon<f64> {
        Polygon::new(
            LineString::from(vec![
                (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0),
            ]),
            vec![],
        )
    }

    #[test]
    fn test_area_square() {
        let a = area(&Geometry::Polygon(square()));
        assert!((a - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_area_triangle() {
        let triangle = Polygon::new(
            LineString::from(vec![
                (0.0, 0.0), (10.0, 0.0), (5.0, 10.0), (0.0, 0.0),
            ]),
            vec![],
        );
        let a = area(&Geometry::Polygon(triangle));
        assert!((a - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_area_non_polygon() {
        let line = Geometry::LineString(LineString::from(vec![(0.0, 0.0), (10.0, 10.0)]));
        assert_eq!(area(&line), 0.0);
    }

    #[test]
    fn test_length_line() {
        let line = Geometry::LineString(LineString::from(vec![
            (0.0, 0.0), (3.0, 4.0),
        ]));
        let l = length(&line);
        assert!((l - 5.0).abs() < 1e-10); // 3-4-5 triangle
    }

    #[test]
    fn test_length_multiline() {
        let mls = Geometry::MultiLineString(MultiLineString::new(vec![
            LineString::from(vec![(0.0, 0.0), (10.0, 0.0)]),
            LineString::from(vec![(0.0, 0.0), (0.0, 5.0)]),
        ]));
        let l = length(&mls);
        assert!((l - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_length_segment() {
        let line = Geometry::Line(Line::new(
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 6.0, y: 8.0 },
        ));
        let l = length(&line);
        assert!((l - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_perimeter_square() {
        let p = perimeter(&Geometry::Polygon(square()));
        assert!((p - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_perimeter_with_hole() {
        let poly = Polygon::new(
            LineString::from(vec![
                (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0),
            ]),
            vec![LineString::from(vec![
                (2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0), (2.0, 2.0),
            ])],
        );
        let p = perimeter(&Geometry::Polygon(poly));
        // Exterior: 40, Interior: 24
        assert!((p - 64.0).abs() < 1e-10);
    }
}
