//! Vector overlay operations: intersection, union, difference, dissolve.
//!
//! Wraps `geo::BooleanOps` for polygon-polygon operations on feature collections.

use geo::algorithm::bool_ops::BooleanOps;
use geo_types::{Geometry, MultiPolygon, Polygon};

use surtgis_core::vector::{Feature, FeatureCollection};

/// Compute the geometric intersection of two feature collections.
///
/// Returns features from A clipped to the geometry of B.
pub fn intersection(a: &FeatureCollection, b: &FeatureCollection) -> FeatureCollection {
    overlay(a, b, OverlayOp::Intersection)
}

/// Compute the geometric union of two feature collections.
pub fn union(a: &FeatureCollection, b: &FeatureCollection) -> FeatureCollection {
    overlay(a, b, OverlayOp::Union)
}

/// Compute the geometric difference: A minus B.
pub fn difference(a: &FeatureCollection, b: &FeatureCollection) -> FeatureCollection {
    overlay(a, b, OverlayOp::Difference)
}

/// Compute the symmetric difference (XOR) of two collections.
pub fn symmetric_difference(a: &FeatureCollection, b: &FeatureCollection) -> FeatureCollection {
    overlay(a, b, OverlayOp::SymDifference)
}

/// Dissolve all features into a single geometry (union all).
pub fn dissolve(fc: &FeatureCollection) -> FeatureCollection {
    let polys = extract_polygons(fc);
    if polys.is_empty() {
        return FeatureCollection::new();
    }

    let mut result = MultiPolygon::new(vec![polys[0].clone()]);
    for poly in &polys[1..] {
        let mp = MultiPolygon::new(vec![poly.clone()]);
        result = result.union(&mp);
    }

    let mut out = FeatureCollection::new();
    for poly in result.0 {
        out.push(Feature {
            geometry: Some(Geometry::Polygon(poly)),
            properties: Default::default(),
            id: None,
        });
    }
    out
}

// ─── Internal ────────────────────────────────────────────────────────

enum OverlayOp {
    Intersection,
    Union,
    Difference,
    SymDifference,
}

fn overlay(a: &FeatureCollection, b: &FeatureCollection, op: OverlayOp) -> FeatureCollection {
    let a_polys = extract_polygons(a);
    let b_polys = extract_polygons(b);

    // CR-7: an empty operand used to make difference/union/symmetric_difference
    // silently return an empty collection, discarding the non-empty side.
    // A - ∅ = A, A ∪ ∅ = A, A Δ ∅ = A (and only intersection collapses to ∅).
    if b_polys.is_empty() {
        return match op {
            OverlayOp::Intersection => FeatureCollection::new(),
            OverlayOp::Difference | OverlayOp::Union | OverlayOp::SymDifference => a.clone(),
        };
    }
    // Symmetric case: ∅ ∩ B = ∅, ∅ - B = ∅, ∅ ∪ B = B, ∅ Δ B = B.
    if a_polys.is_empty() {
        return match op {
            OverlayOp::Intersection | OverlayOp::Difference => FeatureCollection::new(),
            OverlayOp::Union | OverlayOp::SymDifference => b.clone(),
        };
    }

    let b_merged = merge_polygons(&b_polys);
    let mut out = FeatureCollection::new();

    match op {
        // CR-8: union/xor used to loop per-A-polygon and union/xor against the
        // *whole* of B each time, so B ended up duplicated once per A polygon
        // (n times for n polygons in A), inflating the resulting area. The
        // correct semantics is a single union/xor of A merged against B merged.
        OverlayOp::Union | OverlayOp::SymDifference => {
            let a_merged = merge_polygons(&a_polys);
            let result = match op {
                OverlayOp::Union => a_merged.union(&b_merged),
                OverlayOp::SymDifference => a_merged.xor(&b_merged),
                _ => unreachable!(),
            };
            push_result_polygons(&mut out, result);
        }

        // Intersection/difference are directional (relative to B) and keep
        // per-feature attributes from A, so these still iterate per A-feature.
        OverlayOp::Intersection | OverlayOp::Difference => {
            for feature in &a.features {
                let Some(geom) = &feature.geometry else {
                    continue;
                };
                let feature_polys = geometry_to_polygons(geom);

                for a_poly in &feature_polys {
                    let a_mp = MultiPolygon::new(vec![a_poly.clone()]);
                    let result = match op {
                        OverlayOp::Intersection => a_mp.intersection(&b_merged),
                        OverlayOp::Difference => a_mp.difference(&b_merged),
                        _ => unreachable!(),
                    };

                    for poly in result.0 {
                        if poly.exterior().0.len() >= 4 {
                            out.push(Feature {
                                geometry: Some(Geometry::Polygon(poly)),
                                properties: feature.properties.clone(),
                                id: feature.id.clone(),
                            });
                        }
                    }
                }
            }
        }
    }

    out
}

/// Merge a non-empty slice of polygons into a single `MultiPolygon` by
/// repeated boolean union. Panics if `polys` is empty (callers must check).
fn merge_polygons(polys: &[Polygon<f64>]) -> MultiPolygon<f64> {
    let mut merged = MultiPolygon::new(vec![polys[0].clone()]);
    for p in &polys[1..] {
        merged = merged.union(&MultiPolygon::new(vec![p.clone()]));
    }
    merged
}

/// Push each ring of a boolean-op result as its own feature. Used for
/// union/xor where the result no longer maps to a single input feature, so
/// there is no unambiguous set of attributes to propagate.
fn push_result_polygons(out: &mut FeatureCollection, result: MultiPolygon<f64>) {
    for poly in result.0 {
        if poly.exterior().0.len() >= 4 {
            out.push(Feature {
                geometry: Some(Geometry::Polygon(poly)),
                properties: Default::default(),
                id: None,
            });
        }
    }
}

fn extract_polygons(fc: &FeatureCollection) -> Vec<Polygon<f64>> {
    fc.features
        .iter()
        .filter_map(|f| f.geometry.as_ref())
        .flat_map(geometry_to_polygons)
        .collect()
}

fn geometry_to_polygons(geom: &Geometry<f64>) -> Vec<Polygon<f64>> {
    match geom {
        Geometry::Polygon(p) => vec![p.clone()],
        Geometry::MultiPolygon(mp) => mp.0.clone(),
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::{ClipRect, clip_by_rect};
    use geo::{Area, LineString};

    /// Build an axis-aligned square polygon from (x0,y0) to (x1,y1).
    fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Polygon<f64> {
        Polygon::new(
            LineString::from(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]),
            vec![],
        )
    }

    fn fc_of(polys: Vec<Polygon<f64>>) -> FeatureCollection {
        let mut fc = FeatureCollection::new();
        for p in polys {
            fc.push(Feature::new(Geometry::Polygon(p)));
        }
        fc
    }

    /// Total unsigned area of all polygon geometries in a collection.
    fn total_area(fc: &FeatureCollection) -> f64 {
        fc.features
            .iter()
            .filter_map(|f| f.geometry.as_ref())
            .map(|g| match g {
                Geometry::Polygon(p) => p.unsigned_area(),
                Geometry::MultiPolygon(mp) => mp.unsigned_area(),
                _ => 0.0,
            })
            .sum()
    }

    // ─── intersection / difference / union / symmetric_difference ──────
    // A = [0,2]x[0,2] (area 4), B = [1,3]x[1,3] (area 4), overlap = [1,2]x[1,2] (area 1)

    #[test]
    fn test_intersection_partial_overlap() {
        let a = fc_of(vec![square(0.0, 0.0, 2.0, 2.0)]);
        let b = fc_of(vec![square(1.0, 1.0, 3.0, 3.0)]);
        let result = intersection(&a, &b);
        assert!((total_area(&result) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_difference_partial_overlap() {
        let a = fc_of(vec![square(0.0, 0.0, 2.0, 2.0)]);
        let b = fc_of(vec![square(1.0, 1.0, 3.0, 3.0)]);
        let result = difference(&a, &b);
        assert!((total_area(&result) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_union_partial_overlap() {
        let a = fc_of(vec![square(0.0, 0.0, 2.0, 2.0)]);
        let b = fc_of(vec![square(1.0, 1.0, 3.0, 3.0)]);
        let result = union(&a, &b);
        assert!((total_area(&result) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_symmetric_difference_partial_overlap() {
        let a = fc_of(vec![square(0.0, 0.0, 2.0, 2.0)]);
        let b = fc_of(vec![square(1.0, 1.0, 3.0, 3.0)]);
        let result = symmetric_difference(&a, &b);
        assert!((total_area(&result) - 6.0).abs() < 1e-9);
    }

    // ─── CR-7: empty operand must not silently drop the non-empty side ─

    #[test]
    fn test_difference_b_empty_returns_a() {
        let a = fc_of(vec![square(0.0, 0.0, 2.0, 2.0)]);
        let b = FeatureCollection::new();
        let result = difference(&a, &b);
        assert_eq!(result.features.len(), a.features.len());
        assert!((total_area(&result) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_union_b_empty_returns_a() {
        let a = fc_of(vec![square(0.0, 0.0, 2.0, 2.0)]);
        let b = FeatureCollection::new();
        let result = union(&a, &b);
        assert!((total_area(&result) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_symmetric_difference_b_empty_returns_a() {
        let a = fc_of(vec![square(0.0, 0.0, 2.0, 2.0)]);
        let b = FeatureCollection::new();
        let result = symmetric_difference(&a, &b);
        assert!((total_area(&result) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_intersection_b_empty_returns_empty() {
        let a = fc_of(vec![square(0.0, 0.0, 2.0, 2.0)]);
        let b = FeatureCollection::new();
        let result = intersection(&a, &b);
        assert!(result.features.is_empty());
    }

    #[test]
    fn test_union_a_empty_returns_b() {
        let a = FeatureCollection::new();
        let b = fc_of(vec![square(0.0, 0.0, 2.0, 2.0)]);
        let result = union(&a, &b);
        assert!((total_area(&result) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_difference_a_empty_returns_empty() {
        let a = FeatureCollection::new();
        let b = fc_of(vec![square(0.0, 0.0, 2.0, 2.0)]);
        let result = difference(&a, &b);
        assert!(result.features.is_empty());
    }

    // ─── CR-8: union must not duplicate B once per polygon in A ────────

    #[test]
    fn test_union_does_not_duplicate_b() {
        // A has 2 disjoint unit squares (area 1 each); B has 1 disjoint unit
        // square (area 1), none overlapping. Correct union area = 3.
        // The old per-A-polygon loop unioned each A polygon against the whole
        // of B, so B was emitted twice (once per A polygon) => buggy area = 4.
        let a = fc_of(vec![square(0.0, 0.0, 1.0, 1.0), square(5.0, 5.0, 6.0, 6.0)]);
        let b = fc_of(vec![square(10.0, 10.0, 11.0, 11.0)]);
        let result = union(&a, &b);
        assert!(
            (total_area(&result) - 3.0).abs() < 1e-9,
            "expected union area 3.0 (no duplication), got {}",
            total_area(&result)
        );
    }

    // ─── CR-5: clip must recurse into Multi* geometries ─────────────────

    #[test]
    fn test_clip_multipolygon_drops_outside_part_and_unwraps() {
        let rect = ClipRect::new(0.0, 0.0, 10.0, 10.0);
        let mp = Geometry::MultiPolygon(MultiPolygon::new(vec![
            square(2.0, 2.0, 8.0, 8.0),     // fully inside
            square(-5.0, -5.0, -1.0, -1.0), // fully outside
        ]));

        let result = clip_by_rect(&mp, rect).expect("should keep the inside part");
        match result {
            Geometry::Polygon(p) => {
                assert!((p.unsigned_area() - 36.0).abs() < 1e-9);
            }
            other => panic!("expected a single clipped Polygon, got {other:?}"),
        }
    }

    #[test]
    fn test_clip_multipolygon_keeps_multiple_parts() {
        let rect = ClipRect::new(0.0, 0.0, 10.0, 10.0);
        let mp = Geometry::MultiPolygon(MultiPolygon::new(vec![
            square(-5.0, 2.0, 5.0, 4.0), // straddles left edge
            square(5.0, 6.0, 15.0, 8.0), // straddles right edge
        ]));

        let result = clip_by_rect(&mp, rect).expect("both parts intersect the rect");
        match result {
            Geometry::MultiPolygon(clipped) => {
                assert_eq!(clipped.0.len(), 2);
                for poly in &clipped.0 {
                    for coord in poly.exterior().0.iter() {
                        assert!(coord.x >= -1e-9 && coord.x <= 10.0 + 1e-9);
                    }
                }
            }
            other => panic!("expected a MultiPolygon with 2 parts, got {other:?}"),
        }
    }

    // ─── CR-6: clip must preserve (and clip) interior rings/holes ──────

    #[test]
    fn test_clip_polygon_with_hole_preserves_hole() {
        // Exterior [0,10]x[0,10] with a hole [4,6]x[4,6] (area 4), clipped to
        // [1,9]x[1,9]. Expected area = clipped exterior (8*8=64) - hole (4).
        let rect = ClipRect::new(1.0, 1.0, 9.0, 9.0);
        let poly = Polygon::new(
            LineString::from(vec![
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (0.0, 0.0),
            ]),
            vec![LineString::from(vec![
                (4.0, 4.0),
                (6.0, 4.0),
                (6.0, 6.0),
                (4.0, 6.0),
                (4.0, 4.0),
            ])],
        );

        let result = clip_by_rect(&Geometry::Polygon(poly), rect).expect("polygon overlaps rect");
        match result {
            Geometry::Polygon(clipped) => {
                assert_eq!(clipped.interiors().len(), 1, "hole must be preserved");
                assert!((clipped.unsigned_area() - 60.0).abs() < 1e-9);
            }
            other => panic!("expected a Polygon, got {other:?}"),
        }
    }

    // ─── CR-9: clip must not bridge a reentrant line with a false segment ─

    #[test]
    fn test_clip_line_reentrant_produces_multilinestring() {
        let rect = ClipRect::new(0.0, 0.0, 10.0, 10.0);
        // Starts inside, exits north past the top edge, stays outside for a
        // segment entirely beyond the window, then re-enters from the top.
        let line = Geometry::LineString(LineString::from(vec![
            (2.0, 2.0),
            (2.0, 15.0),
            (8.0, 15.0),
            (8.0, 2.0),
        ]));

        let result = clip_by_rect(&line, rect).expect("line intersects the rect twice");
        match result {
            Geometry::MultiLineString(mls) => {
                assert_eq!(mls.0.len(), 2, "expected two disjoint clipped segments");
                // No fabricated bridge: the two pieces must not share endpoints.
                let piece0_end = *mls.0[0].0.last().unwrap();
                let piece1_start = mls.0[1].0[0];
                assert_ne!(
                    (piece0_end.x, piece0_end.y),
                    (piece1_start.x, piece1_start.y)
                );
            }
            other => panic!("expected a MultiLineString with 2 parts, got {other:?}"),
        }
    }
}
