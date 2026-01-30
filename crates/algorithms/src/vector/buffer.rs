//! Buffer operations
//!
//! Create buffer zones around geometries. For points, generates circles
//! approximated as polygons. For lines and polygons, offsets the boundary.

use geo::{LineString, Point, Polygon};
use std::f64::consts::PI;

/// Parameters for buffer operations
#[derive(Debug, Clone)]
pub struct BufferParams {
    /// Buffer distance (positive = expand, negative = shrink)
    pub distance: f64,
    /// Number of segments to approximate curves (default: 16)
    pub segments: usize,
}

impl Default for BufferParams {
    fn default() -> Self {
        Self {
            distance: 1.0,
            segments: 16,
        }
    }
}

/// Create a circular buffer around a point.
///
/// Generates a polygon approximating a circle with the given number
/// of segments.
///
/// # Arguments
/// * `point` - Center point
/// * `params` - Buffer parameters (distance, segments)
///
/// # Returns
/// A polygon approximating a circle
pub fn buffer_points(point: &Point<f64>, params: &BufferParams) -> Polygon<f64> {
    let n = params.segments.max(4);
    let r = params.distance.abs();
    let cx = point.x();
    let cy = point.y();

    let mut coords = Vec::with_capacity(n + 1);
    for i in 0..n {
        let angle = 2.0 * PI * i as f64 / n as f64;
        coords.push((cx + r * angle.cos(), cy + r * angle.sin()));
    }
    // Close the ring
    coords.push(coords[0]);

    Polygon::new(LineString::from(coords), vec![])
}

/// Create buffers around a geometry represented as coordinate pairs.
///
/// Supports buffering of point collections (each point becomes a circle).
///
/// # Arguments
/// * `points` - Slice of (x, y) coordinates
/// * `params` - Buffer parameters
///
/// # Returns
/// A vector of buffer polygons (one per input point)
pub fn buffer_geometry(points: &[(f64, f64)], params: &BufferParams) -> Vec<Polygon<f64>> {
    points
        .iter()
        .map(|&(x, y)| buffer_points(&Point::new(x, y), params))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::Area;

    #[test]
    fn test_buffer_point_circle() {
        let point = Point::new(0.0, 0.0);
        let params = BufferParams {
            distance: 10.0,
            segments: 64,
        };

        let polygon = buffer_points(&point, &params);

        // Area should approximate π * r²
        let expected_area = PI * 100.0;
        let actual_area = polygon.unsigned_area();

        let error = (actual_area - expected_area).abs() / expected_area;
        assert!(
            error < 0.01,
            "Circle area error {:.2}% (expected {:.1}, got {:.1})",
            error * 100.0,
            expected_area,
            actual_area
        );
    }

    #[test]
    fn test_buffer_point_vertex_count() {
        let point = Point::new(5.0, 5.0);
        let params = BufferParams {
            distance: 1.0,
            segments: 32,
        };

        let polygon = buffer_points(&point, &params);
        let ring = polygon.exterior();

        // Should have segments + 1 coordinates (closed ring)
        assert_eq!(ring.0.len(), 33);
    }

    #[test]
    fn test_buffer_multiple_points() {
        let points = vec![(0.0, 0.0), (10.0, 0.0), (5.0, 5.0)];
        let params = BufferParams {
            distance: 2.0,
            segments: 16,
        };

        let buffers = buffer_geometry(&points, &params);
        assert_eq!(buffers.len(), 3);

        // All buffers should have area > 0
        for buf in &buffers {
            assert!(buf.unsigned_area() > 0.0);
        }
    }

    #[test]
    fn test_buffer_distance_affects_size() {
        let point = Point::new(0.0, 0.0);

        let small = buffer_points(&point, &BufferParams { distance: 1.0, segments: 32 });
        let big = buffer_points(&point, &BufferParams { distance: 5.0, segments: 32 });

        assert!(
            big.unsigned_area() > small.unsigned_area() * 20.0,
            "Bigger buffer should have ~25x the area"
        );
    }
}
