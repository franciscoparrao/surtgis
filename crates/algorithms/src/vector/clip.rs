//! Clipping operations
//!
//! Clip geometries by a rectangular extent using Cohen-Sutherland
//! for lines and Sutherland-Hodgman for polygons.

use geo::{Coord, Geometry, LineString, Polygon};

/// A clipping rectangle
#[derive(Debug, Clone, Copy)]
pub struct ClipRect {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl ClipRect {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
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
                Coord { x: rect.min_x, y: py + t * dy }
            }
            Edge::Right => {
                let t = (rect.max_x - px) / dx;
                Coord { x: rect.max_x, y: py + t * dy }
            }
            Edge::Bottom => {
                let t = (rect.min_y - py) / dy;
                Coord { x: px + t * dx, y: rect.min_y }
            }
            Edge::Top => {
                let t = (rect.max_y - py) / dy;
                Coord { x: px + t * dx, y: rect.max_y }
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

        Geometry::Polygon(poly) => {
            // Sutherland-Hodgman: clip against each edge
            let mut vertices: Vec<Coord<f64>> = poly
                .exterior()
                .0.to_vec();

            // Remove closing vertex for algorithm
            if vertices.len() > 1 && vertices.first() == vertices.last() {
                vertices.pop();
            }

            for edge in [Edge::Left, Edge::Right, Edge::Bottom, Edge::Top] {
                vertices = clip_polygon_edge(&vertices, edge, &rect);
                if vertices.is_empty() {
                    return None;
                }
            }

            // Close the ring
            if !vertices.is_empty() {
                vertices.push(vertices[0]);
            }

            Some(Geometry::Polygon(Polygon::new(
                LineString::new(vertices),
                vec![], // Interior rings not clipped for simplicity
            )))
        }

        Geometry::LineString(ls) => {
            // Clip each segment
            let mut clipped_coords: Vec<Coord<f64>> = Vec::new();

            for window in ls.0.windows(2) {
                let (p0, p1) = (window[0], window[1]);
                if let Some((c0, c1)) = clip_segment(p0, p1, &rect) {
                    if clipped_coords.last() != Some(&c0) {
                        clipped_coords.push(c0);
                    }
                    clipped_coords.push(c1);
                }
            }

            if clipped_coords.len() >= 2 {
                Some(Geometry::LineString(LineString::new(clipped_coords)))
            } else {
                None
            }
        }

        other => Some(other.clone()),
    }
}

/// Cohen-Sutherland region codes
const INSIDE: u8 = 0b0000;
const LEFT: u8 = 0b0001;
const RIGHT: u8 = 0b0010;
const BOTTOM: u8 = 0b0100;
const TOP: u8 = 0b1000;

fn outcode(p: Coord<f64>, rect: &ClipRect) -> u8 {
    let mut code = INSIDE;
    if p.x < rect.min_x { code |= LEFT; }
    if p.x > rect.max_x { code |= RIGHT; }
    if p.y < rect.min_y { code |= BOTTOM; }
    if p.y > rect.max_y { code |= TOP; }
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
            Coord { x: p0.x + t * dx, y: rect.max_y }
        } else if code_out & BOTTOM != 0 {
            let t = (rect.min_y - p0.y) / dy;
            Coord { x: p0.x + t * dx, y: rect.min_y }
        } else if code_out & RIGHT != 0 {
            let t = (rect.max_x - p0.x) / dx;
            Coord { x: rect.max_x, y: p0.y + t * dy }
        } else {
            let t = (rect.min_x - p0.x) / dx;
            Coord { x: rect.min_x, y: p0.y + t * dy }
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
                (2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0), (2.0, 2.0),
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
                (-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0), (-5.0, -5.0),
            ]),
            vec![],
        ));

        let result = clip_by_rect(&poly, unit_rect());
        assert!(result.is_some());

        if let Some(Geometry::Polygon(clipped)) = result {
            // All clipped coordinates should be within the rect
            for coord in clipped.exterior().0.iter() {
                assert!(
                    coord.x >= -0.001 && coord.x <= 10.001
                        && coord.y >= -0.001 && coord.y <= 10.001,
                    "Clipped coord ({}, {}) outside rect",
                    coord.x, coord.y
                );
            }
        }
    }

    #[test]
    fn test_clip_polygon_fully_outside() {
        let poly = Geometry::Polygon(Polygon::new(
            LineString::from(vec![
                (20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0), (20.0, 20.0),
            ]),
            vec![],
        ));

        let result = clip_by_rect(&poly, unit_rect());
        assert!(result.is_none());
    }

    #[test]
    fn test_clip_line_partial() {
        let line = Geometry::LineString(LineString::from(vec![
            (-5.0, 5.0), (15.0, 5.0),
        ]));

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
        let line = Geometry::LineString(LineString::from(vec![
            (20.0, 20.0), (30.0, 30.0),
        ]));

        let result = clip_by_rect(&line, unit_rect());
        assert!(result.is_none());
    }
}
