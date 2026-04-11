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
    let b_polys = extract_polygons(b);
    if b_polys.is_empty() {
        return FeatureCollection::new();
    }

    // Merge B into a single MultiPolygon for overlay
    let b_merged = if b_polys.len() == 1 {
        MultiPolygon::new(vec![b_polys[0].clone()])
    } else {
        let mut merged = MultiPolygon::new(vec![b_polys[0].clone()]);
        for p in &b_polys[1..] {
            merged = merged.union(&MultiPolygon::new(vec![p.clone()]));
        }
        merged
    };

    let mut out = FeatureCollection::new();

    for feature in &a.features {
        let Some(geom) = &feature.geometry else { continue };
        let a_polys = geometry_to_polygons(geom);

        for a_poly in &a_polys {
            let a_mp = MultiPolygon::new(vec![a_poly.clone()]);
            let result = match op {
                OverlayOp::Intersection => a_mp.intersection(&b_merged),
                OverlayOp::Union => a_mp.union(&b_merged),
                OverlayOp::Difference => a_mp.difference(&b_merged),
                OverlayOp::SymDifference => a_mp.xor(&b_merged),
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

    out
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
