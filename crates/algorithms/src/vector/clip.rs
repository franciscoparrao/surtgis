//! Clipping operations
//!
//! Clip geometries by a rectangular extent using Cohen-Sutherland
//! for lines and Sutherland-Hodgman for polygons.

use geo::{
    Coord, Geometry, GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon,
    Polygon,
};

/// A clipping rectangle
#[derive(Debug, Clone, Copy)]
pub struct ClipRect {
    /// Minimum x (left edge).
    pub min_x: f64,
    /// Minimum y (bottom edge).
    pub min_y: f64,
    /// Maximum x (right edge).
    pub max_x: f64,
    /// Maximum y (top edge).
    pub max_y: f64,
}

impl ClipRect {
    /// Construct a clipping rectangle from its minimum and maximum coordinates.
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }
}

/// Edge of the clipping rectangle
#[derive(Debug, Clone, Copy)]
enum Edge {
    Left,
    Right,
    Bottom,
    Top,
}

impl Edge {
    fn is_inside(&self, p: &Coord<f64>, rect: &ClipRect) -> bool {
        match self {
            Edge::Left => p.x >= rect.min_x,
            Edge::Right => p.x <= rect.max_x,
            Edge::Bottom => p.y >= rect.min_y,
            Edge::Top => p.y <= rect.max_y,
        }
    }

    fn intersect(&self, p: &Coord<f64>, q: &Coord<f64>, rect: &ClipRect) -> Coord<f64> {
        let (px, py) = (p.x, p.y);
        let (qx, qy) = (q.x, q.y);
        let dx = qx - px;
        let dy = qy - py;

        match self {
            Edge::Left => {
                let t = (rect.min_x - px) / dx;
                Coord {
                    x: rect.min_x,
                    y: py + t * dy,
                }
            }
            Edge::Right => {
                let t = (rect.max_x - px) / dx;
                Coord {
                    x: rect.max_x,
                    y: py + t * dy,
                }
            }
            Edge::Bottom => {
                let t = (rect.min_y - py) / dy;
                Coord {
                    x: px + t * dx,
                    y: rect.min_y,
                }
            }
            Edge::Top => {
                let t = (rect.max_y - py) / dy;
                Coord {
                    x: px + t * dx,
                    y: rect.max_y,
                }
            }
        }
    }
}

/// Clip a polygon against one edge (Sutherland-Hodgman step)
fn clip_polygon_edge(vertices: &[Coord<f64>], edge: Edge, rect: &ClipRect) -> Vec<Coord<f64>> {
    if vertices.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::new();
    let n = vertices.len();

    for i in 0..n {
        let current = &vertices[i];
        let next = &vertices[(i + 1) % n];

        let current_inside = edge.is_inside(current, rect);
        let next_inside = edge.is_inside(next, rect);

        match (current_inside, next_inside) {
            (true, true) => {
                output.push(*next);
            }
            (true, false) => {
                output.push(edge.intersect(current, next, rect));
            }
            (false, true) => {
                output.push(edge.intersect(current, next, rect));
                output.push(*next);
            }
            (false, false) => {}
        }
    }

    output
}

/// Clip a geometry by a rectangular extent.
///
/// Uses Sutherland-Hodgman algorithm for polygons and Cohen-Sutherland
/// approach for line strings.
///
/// # Arguments
/// * `geom` - Input geometry
/// * `rect` - Clipping rectangle
///
/// # Returns
/// Clipped geometry, or None if completely outside
pub fn clip_by_rect(geom: &Geometry<f64>, rect: ClipRect) -> Option<Geometry<f64>> {
    match geom {
        Geometry::Point(p) => {
            if rect.contains(p.x(), p.y()) {
                Some(geom.clone())
            } else {
                None
            }
        }

        Geometry::Polygon(poly) => clip_polygon(poly, &rect).map(Geometry::Polygon),

        Geometry::LineString(ls) => {
            let pieces = clip_linestring_pieces(ls, &rect);
            match pieces.len() {
                0 => None,
                1 => Some(Geometry::LineString(LineString::new(
                    pieces.into_iter().next().unwrap(),
                ))),
                _ => Some(Geometry::MultiLineString(MultiLineString::new(
                    pieces.into_iter().map(LineString::new).collect(),
                ))),
            }
        }

        // CR-5: Multi* variants used to pass through unclipped (`other.clone()`).
        // The `shapefile` crate converts every ESRI Polygon into a geo::MultiPolygon,
        // so this was the dominant real-world case, not an edge case.
        Geometry::MultiPolygon(mp) => {
            let polys: Vec<Polygon<f64>> =
                mp.0.iter()
                    .filter_map(|poly| clip_polygon(poly, &rect))
                    .collect();

            match polys.len() {
                0 => None,
                1 => Some(Geometry::Polygon(polys.into_iter().next().unwrap())),
                _ => Some(Geometry::MultiPolygon(MultiPolygon::new(polys))),
            }
        }

        Geometry::MultiLineString(mls) => {
            let lines: Vec<LineString<f64>> = mls
                .0
                .iter()
                .flat_map(|ls| clip_linestring_pieces(ls, &rect))
                .map(LineString::new)
                .collect();

            match lines.len() {
                0 => None,
                1 => Some(Geometry::LineString(lines.into_iter().next().unwrap())),
                _ => Some(Geometry::MultiLineString(MultiLineString::new(lines))),
            }
        }

        Geometry::MultiPoint(mp) => {
            let points: Vec<_> =
                mp.0.iter()
                    .filter(|p| rect.contains(p.x(), p.y()))
                    .cloned()
                    .collect();

            match points.len() {
                0 => None,
                1 => Some(Geometry::Point(points.into_iter().next().unwrap())),
                _ => Some(Geometry::MultiPoint(MultiPoint::new(points))),
            }
        }

        Geometry::GeometryCollection(gc) => {
            let geoms: Vec<Geometry<f64>> =
                gc.0.iter().filter_map(|g| clip_by_rect(g, rect)).collect();

            if geoms.is_empty() {
                None
            } else {
                Some(Geometry::GeometryCollection(GeometryCollection::from(
                    geoms,
                )))
            }
        }

        // Line, Rect, Triangle: not covered by this audit, keep prior behaviour.
        other => Some(other.clone()),
    }
}

/// Clip a single polygon (exterior + interior rings) against `rect` using
/// Sutherland-Hodgman. Returns `None` if the exterior ring is fully clipped away.
///
/// CR-6: interior rings (holes) used to be dropped unconditionally, which
/// silently overestimated the area of any polygon with holes after clipping.
fn clip_polygon(poly: &Polygon<f64>, rect: &ClipRect) -> Option<Polygon<f64>> {
    let mut exterior: Vec<Coord<f64>> = poly.exterior().0.to_vec();
    if exterior.len() > 1 && exterior.first() == exterior.last() {
        exterior.pop();
    }

    for edge in [Edge::Left, Edge::Right, Edge::Bottom, Edge::Top] {
        exterior = clip_polygon_edge(&exterior, edge, rect);
        if exterior.is_empty() {
            return None;
        }
    }
    exterior.push(exterior[0]);

    let mut interiors = Vec::new();
    for interior in poly.interiors() {
        let mut ring: Vec<Coord<f64>> = interior.0.to_vec();
        if ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
        }

        let mut clipped = ring;
        for edge in [Edge::Left, Edge::Right, Edge::Bottom, Edge::Top] {
            clipped = clip_polygon_edge(&clipped, edge, rect);
            if clipped.is_empty() {
                break;
            }
        }

        // A degenerate ring (< 3 vertices) can't bound an area; drop it.
        if clipped.len() >= 3 {
            clipped.push(clipped[0]);
            interiors.push(LineString::new(clipped));
        }
    }

    Some(Polygon::new(LineString::new(exterior), interiors))
}

/// Clip a line string against `rect`, returning each contiguous clipped
/// segment as a separate vertex list.
///
/// CR-9: a line that exits and re-enters the clip window used to be
/// concatenated into a single `LineString`, creating a false bridge segment
/// straight across the interior of the window. Discontinuous pieces are now
/// kept separate so the caller can emit a `MultiLineString`.
fn clip_linestring_pieces(ls: &LineString<f64>, rect: &ClipRect) -> Vec<Vec<Coord<f64>>> {
    let mut pieces: Vec<Vec<Coord<f64>>> = Vec::new();
    let mut current: Vec<Coord<f64>> = Vec::new();

    for window in ls.0.windows(2) {
        let (p0, p1) = (window[0], window[1]);
        match clip_segment(p0, p1, rect) {
            Some((c0, c1)) => {
                match current.last() {
                    Some(last) if *last == c0 => {}
                    Some(_) => {
                        // Discontinuity: the new segment doesn't start where the
                        // previous one ended, so close off the current piece.
                        if current.len() >= 2 {
                            pieces.push(std::mem::take(&mut current));
                        } else {
                            current.clear();
                        }
                        current.push(c0);
                    }
                    None => current.push(c0),
                }
                current.push(c1);
            }
            None => {
                // Segment fully outside the window: breaks the current piece.
                if current.len() >= 2 {
                    pieces.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
    }

    if current.len() >= 2 {
        pieces.push(current);
    }

    pieces
}

/// Cohen-Sutherland region codes
const INSIDE: u8 = 0b0000;
const LEFT: u8 = 0b0001;
const RIGHT: u8 = 0b0010;
const BOTTOM: u8 = 0b0100;
const TOP: u8 = 0b1000;

fn outcode(p: Coord<f64>, rect: &ClipRect) -> u8 {
    let mut code = INSIDE;
    if p.x < rect.min_x {
        code |= LEFT;
    }
    if p.x > rect.max_x {
        code |= RIGHT;
    }
    if p.y < rect.min_y {
        code |= BOTTOM;
    }
    if p.y > rect.max_y {
        code |= TOP;
    }
    code
}

fn clip_segment(
    mut p0: Coord<f64>,
    mut p1: Coord<f64>,
    rect: &ClipRect,
) -> Option<(Coord<f64>, Coord<f64>)> {
    let mut code0 = outcode(p0, rect);
    let mut code1 = outcode(p1, rect);

    loop {
        if (code0 | code1) == 0 {
            return Some((p0, p1)); // Both inside
        }
        if (code0 & code1) != 0 {
            return None; // Both outside same region
        }

        let code_out = if code0 != 0 { code0 } else { code1 };
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;

        let new_point = if code_out & TOP != 0 {
            let t = (rect.max_y - p0.y) / dy;
            Coord {
                x: p0.x + t * dx,
                y: rect.max_y,
            }
        } else if code_out & BOTTOM != 0 {
            let t = (rect.min_y - p0.y) / dy;
            Coord {
                x: p0.x + t * dx,
                y: rect.min_y,
            }
        } else if code_out & RIGHT != 0 {
            let t = (rect.max_x - p0.x) / dx;
            Coord {
                x: rect.max_x,
                y: p0.y + t * dy,
            }
        } else {
            let t = (rect.min_x - p0.x) / dx;
            Coord {
                x: rect.min_x,
                y: p0.y + t * dy,
            }
        };

        if code_out == code0 {
            p0 = new_point;
            code0 = outcode(p0, rect);
        } else {
            p1 = new_point;
            code1 = outcode(p1, rect);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::Point;

    fn unit_rect() -> ClipRect {
        ClipRect::new(0.0, 0.0, 10.0, 10.0)
    }

    #[test]
    fn test_clip_point_inside() {
        let point = Geometry::Point(Point::new(5.0, 5.0));
        let result = clip_by_rect(&point, unit_rect());
        assert!(result.is_some());
    }

    #[test]
    fn test_clip_point_outside() {
        let point = Geometry::Point(Point::new(15.0, 5.0));
        let result = clip_by_rect(&point, unit_rect());
        assert!(result.is_none());
    }

    #[test]
    fn test_clip_polygon_fully_inside() {
        let poly = Geometry::Polygon(Polygon::new(
            LineString::from(vec![
                (2.0, 2.0),
                (8.0, 2.0),
                (8.0, 8.0),
                (2.0, 8.0),
                (2.0, 2.0),
            ]),
            vec![],
        ));

        let result = clip_by_rect(&poly, unit_rect());
        assert!(result.is_some());
    }

    #[test]
    fn test_clip_polygon_partial() {
        // Polygon extends beyond the clip rect
        let poly = Geometry::Polygon(Polygon::new(
            LineString::from(vec![
                (-5.0, -5.0),
                (5.0, -5.0),
                (5.0, 5.0),
                (-5.0, 5.0),
                (-5.0, -5.0),
            ]),
            vec![],
        ));

        let result = clip_by_rect(&poly, unit_rect());
        assert!(result.is_some());

        if let Some(Geometry::Polygon(clipped)) = result {
            // All clipped coordinates should be within the rect
            for coord in clipped.exterior().0.iter() {
                assert!(
                    coord.x >= -0.001
                        && coord.x <= 10.001
                        && coord.y >= -0.001
                        && coord.y <= 10.001,
                    "Clipped coord ({}, {}) outside rect",
                    coord.x,
                    coord.y
                );
            }
        }
    }

    #[test]
    fn test_clip_polygon_fully_outside() {
        let poly = Geometry::Polygon(Polygon::new(
            LineString::from(vec![
                (20.0, 20.0),
                (30.0, 20.0),
                (30.0, 30.0),
                (20.0, 30.0),
                (20.0, 20.0),
            ]),
            vec![],
        ));

        let result = clip_by_rect(&poly, unit_rect());
        assert!(result.is_none());
    }

    #[test]
    fn test_clip_line_partial() {
        let line = Geometry::LineString(LineString::from(vec![(-5.0, 5.0), (15.0, 5.0)]));

        let result = clip_by_rect(&line, unit_rect());
        assert!(result.is_some());

        if let Some(Geometry::LineString(clipped)) = result {
            assert!(clipped.0.len() >= 2);
            // Endpoints should be on the rect boundary
            assert!((clipped.0[0].x - 0.0).abs() < 1e-10);
            assert!((clipped.0.last().unwrap().x - 10.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_clip_line_fully_outside() {
        let line = Geometry::LineString(LineString::from(vec![(20.0, 20.0), (30.0, 30.0)]));

        let result = clip_by_rect(&line, unit_rect());
        assert!(result.is_none());
    }
}
