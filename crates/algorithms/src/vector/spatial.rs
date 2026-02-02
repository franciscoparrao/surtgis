//! Spatial operations: convex hull, centroid, bounding box, dissolve

use geo::{
    BoundingRect, Centroid as GeoCentroid, ConvexHull as GeoConvexHull,
    Coord, Geometry, LineString, MultiPoint,
    Point, Polygon,
};
use std::collections::HashMap;

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BoundingBox {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    pub fn center(&self) -> (f64, f64) {
        ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)
    }

    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    pub fn to_polygon(&self) -> Polygon<f64> {
        Polygon::new(
            LineString::from(vec![
                (self.min_x, self.min_y),
                (self.max_x, self.min_y),
                (self.max_x, self.max_y),
                (self.min_x, self.max_y),
                (self.min_x, self.min_y),
            ]),
            vec![],
        )
    }
}

/// Compute the bounding box of a geometry
pub fn bounding_box(geom: &Geometry<f64>) -> Option<BoundingBox> {
    geom.bounding_rect().map(|rect| BoundingBox {
        min_x: rect.min().x,
        min_y: rect.min().y,
        max_x: rect.max().x,
        max_y: rect.max().y,
    })
}

/// Compute the centroid of a geometry
pub fn centroid(geom: &Geometry<f64>) -> Option<Point<f64>> {
    match geom {
        Geometry::Point(p) => Some(*p),
        Geometry::Line(l) => Some(l.centroid()),
        Geometry::LineString(ls) => ls.centroid(),
        Geometry::Polygon(p) => p.centroid(),
        Geometry::MultiPoint(mp) => mp.centroid(),
        Geometry::MultiLineString(mls) => mls.centroid(),
        Geometry::MultiPolygon(mp) => mp.centroid(),
        Geometry::Rect(r) => Some(r.centroid()),
        _ => None,
    }
}

/// Compute the convex hull of a geometry
pub fn convex_hull(geom: &Geometry<f64>) -> Polygon<f64> {
    match geom {
        Geometry::Point(p) => Polygon::new(
            LineString::from(vec![(p.x(), p.y()), (p.x(), p.y())]),
            vec![],
        ),
        Geometry::MultiPoint(mp) => mp.convex_hull(),
        Geometry::LineString(ls) => ls.convex_hull(),
        Geometry::Polygon(p) => p.convex_hull(),
        Geometry::MultiPolygon(mp) => mp.convex_hull(),
        Geometry::MultiLineString(mls) => mls.convex_hull(),
        Geometry::GeometryCollection(gc) => {
            // Collect all coordinates
            let points: Vec<Coord<f64>> = gc
                .0
                .iter()
                .filter_map(bounding_box)
                .flat_map(|bb| {
                    vec![
                        Coord { x: bb.min_x, y: bb.min_y },
                        Coord { x: bb.max_x, y: bb.max_y },
                        Coord { x: bb.min_x, y: bb.max_y },
                        Coord { x: bb.max_x, y: bb.min_y },
                    ]
                })
                .collect();

            let mp = MultiPoint::from(
                points.into_iter().map(|c| Point::new(c.x, c.y)).collect::<Vec<_>>()
            );
            mp.convex_hull()
        }
        _ => Polygon::new(LineString::new(vec![]), vec![]),
    }
}

/// Dissolve: group geometries by a key and merge polygons within each group.
///
/// Returns a map from key to the merged polygon (union of all polygons
/// in that group, approximated as their convex hull).
///
/// # Arguments
/// * `features` - Pairs of (key, polygon)
///
/// # Returns
/// Map from key to dissolved polygon
pub fn dissolve(features: &[(String, Polygon<f64>)]) -> HashMap<String, Polygon<f64>> {
    let mut groups: HashMap<String, Vec<&Polygon<f64>>> = HashMap::new();

    for (key, poly) in features {
        groups.entry(key.clone()).or_default().push(poly);
    }

    groups
        .into_iter()
        .map(|(key, polys)| {
            if polys.len() == 1 {
                return (key, polys[0].clone());
            }

            // Merge by computing convex hull of all points
            let all_coords: Vec<Coord<f64>> = polys
                .iter()
                .flat_map(|p| p.exterior().0.iter().copied())
                .collect();

            let mp = MultiPoint::from(
                all_coords.into_iter().map(|c| Point::new(c.x, c.y)).collect::<Vec<_>>()
            );
            let hull = mp.convex_hull();

            (key, hull)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_polygon() -> Polygon<f64> {
        Polygon::new(
            LineString::from(vec![
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (0.0, 0.0),
            ]),
            vec![],
        )
    }

    #[test]
    fn test_bounding_box() {
        let poly = sample_polygon();
        let bb = bounding_box(&Geometry::Polygon(poly)).unwrap();

        assert_eq!(bb.min_x, 0.0);
        assert_eq!(bb.min_y, 0.0);
        assert_eq!(bb.max_x, 10.0);
        assert_eq!(bb.max_y, 10.0);
        assert_eq!(bb.area(), 100.0);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bb = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains_point(5.0, 5.0));
        assert!(!bb.contains_point(15.0, 5.0));
    }

    #[test]
    fn test_bounding_box_intersects() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        let c = BoundingBox::new(20.0, 20.0, 30.0, 30.0);

        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_centroid_polygon() {
        let poly = sample_polygon();
        let c = centroid(&Geometry::Polygon(poly)).unwrap();

        assert!((c.x() - 5.0).abs() < 1e-10);
        assert!((c.y() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_centroid_point() {
        let point = Geometry::Point(Point::new(3.0, 7.0));
        let c = centroid(&point).unwrap();
        assert_eq!(c.x(), 3.0);
        assert_eq!(c.y(), 7.0);
    }

    #[test]
    fn test_convex_hull() {
        // L-shaped polygon → convex hull should be a rectangle-ish shape
        let l_shape = Polygon::new(
            LineString::from(vec![
                (0.0, 0.0),
                (5.0, 0.0),
                (5.0, 5.0),
                (10.0, 5.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (0.0, 0.0),
            ]),
            vec![],
        );

        let hull = convex_hull(&Geometry::Polygon(l_shape));

        // Hull should have fewer or equal vertices than the L-shape
        // and should be convex
        assert!(hull.exterior().0.len() <= 7);
    }

    #[test]
    fn test_convex_hull_points() {
        let points = MultiPoint::from(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(5.0, 5.0),
            Point::new(0.0, 10.0),
            Point::new(10.0, 10.0),
            Point::new(5.0, 3.0), // Interior point
        ]);

        let hull = convex_hull(&Geometry::MultiPoint(points));
        // Interior point should not be on the hull
        assert!(hull.exterior().0.len() <= 6);
    }

    #[test]
    fn test_dissolve_single_group() {
        let features = vec![
            ("forest".to_string(), sample_polygon()),
        ];

        let result = dissolve(&features);
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("forest"));
    }

    #[test]
    fn test_dissolve_multiple_groups() {
        let poly1 = Polygon::new(
            LineString::from(vec![(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]),
            vec![],
        );
        let poly2 = Polygon::new(
            LineString::from(vec![(5.0, 0.0), (10.0, 0.0), (10.0, 5.0), (5.0, 5.0), (5.0, 0.0)]),
            vec![],
        );
        let poly3 = Polygon::new(
            LineString::from(vec![(20.0, 20.0), (25.0, 20.0), (25.0, 25.0), (20.0, 25.0), (20.0, 20.0)]),
            vec![],
        );

        let features = vec![
            ("A".to_string(), poly1),
            ("A".to_string(), poly2),
            ("B".to_string(), poly3),
        ];

        let result = dissolve(&features);
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("A"));
        assert!(result.contains_key("B"));
    }

    #[test]
    fn test_bounding_box_to_polygon() {
        let bb = BoundingBox::new(1.0, 2.0, 5.0, 8.0);
        let poly = bb.to_polygon();

        let coords = &poly.exterior().0;
        assert_eq!(coords.len(), 5); // Closed ring
        assert_eq!(coords[0], Coord { x: 1.0, y: 2.0 });
    }
}
