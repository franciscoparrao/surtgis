//! 2D k-d tree for spatial indexing
//!
//! Provides O(log n) nearest-neighbor and k-nearest-neighbor queries
//! for scattered point data. Replaces O(n·m) brute-force search in
//! interpolation algorithms (IDW, kriging, natural neighbor).
//!
//! Reference:
//! Bentley, J.L. (1975). Multidimensional binary search trees used
//! for associative searching. CACM, 18(9).

use super::SamplePoint;

/// A 2D k-d tree for efficient spatial queries on sample points.
#[derive(Debug)]
pub struct KdTree {
    nodes: Vec<KdNode>,
    /// Original points stored in tree-order
    points: Vec<SamplePoint>,
}

#[derive(Debug)]
struct KdNode {
    /// Index into `points`
    point_idx: usize,
    /// Split dimension: 0 = x, 1 = y
    split_dim: u8,
    /// Left child index (None = leaf)
    left: Option<usize>,
    /// Right child index (None = leaf)
    right: Option<usize>,
}

/// Result of a nearest-neighbor query
#[derive(Debug, Clone, Copy)]
pub struct NearestResult {
    pub point: SamplePoint,
    pub distance_sq: f64,
    pub index: usize,
}

impl KdTree {
    /// Build a k-d tree from sample points.
    ///
    /// Construction is O(n log n) using median-of-coordinate splitting.
    pub fn build(points: &[SamplePoint]) -> Self {
        if points.is_empty() {
            return Self {
                nodes: Vec::new(),
                points: Vec::new(),
            };
        }

        let mut indices: Vec<usize> = (0..points.len()).collect();
        let stored_points: Vec<SamplePoint> = points.to_vec();
        let mut nodes = Vec::with_capacity(points.len());

        build_recursive(&stored_points, &mut indices, 0, &mut nodes);

        Self {
            nodes,
            points: stored_points,
        }
    }

    /// Number of points in the tree.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Find the single nearest point to (qx, qy).
    ///
    /// Returns `None` if the tree is empty.
    /// Complexity: O(log n) average case.
    pub fn nearest(&self, qx: f64, qy: f64) -> Option<NearestResult> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut best_dist_sq = f64::MAX;
        let mut best_idx = 0;

        self.nearest_recursive(0, qx, qy, &mut best_dist_sq, &mut best_idx);

        Some(NearestResult {
            point: self.points[best_idx],
            distance_sq: best_dist_sq,
            index: best_idx,
        })
    }

    /// Find the k nearest points to (qx, qy).
    ///
    /// Returns up to k results sorted by ascending distance.
    /// Complexity: O(k log n) average case.
    pub fn k_nearest(&self, qx: f64, qy: f64, k: usize) -> Vec<NearestResult> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }

        // Use a max-heap (sorted descending by distance) of size k
        let mut heap: Vec<(f64, usize)> = Vec::with_capacity(k + 1);

        self.knn_recursive(0, qx, qy, k, &mut heap);

        // Sort ascending by distance
        heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        heap.iter()
            .map(|&(dist_sq, idx)| NearestResult {
                point: self.points[idx],
                distance_sq: dist_sq,
                index: idx,
            })
            .collect()
    }

    /// Find all points within a given radius of (qx, qy).
    ///
    /// Returns results in no particular order.
    pub fn within_radius(&self, qx: f64, qy: f64, radius: f64) -> Vec<NearestResult> {
        if self.nodes.is_empty() || radius <= 0.0 {
            return Vec::new();
        }

        let radius_sq = radius * radius;
        let mut results = Vec::new();

        self.radius_recursive(0, qx, qy, radius_sq, &mut results);

        results
    }

    fn nearest_recursive(
        &self,
        node_idx: usize,
        qx: f64,
        qy: f64,
        best_dist_sq: &mut f64,
        best_idx: &mut usize,
    ) {
        let node = &self.nodes[node_idx];
        let p = &self.points[node.point_idx];

        let dx = qx - p.x;
        let dy = qy - p.y;
        let dist_sq = dx * dx + dy * dy;

        if dist_sq < *best_dist_sq {
            *best_dist_sq = dist_sq;
            *best_idx = node.point_idx;
        }

        // Determine which side to search first
        let diff = if node.split_dim == 0 { dx } else { dy };
        let (first, second) = if diff < 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        // Search the closer side first
        if let Some(child) = first {
            self.nearest_recursive(child, qx, qy, best_dist_sq, best_idx);
        }

        // Check if we need to search the other side
        if diff * diff < *best_dist_sq {
            if let Some(child) = second {
                self.nearest_recursive(child, qx, qy, best_dist_sq, best_idx);
            }
        }
    }

    fn knn_recursive(
        &self,
        node_idx: usize,
        qx: f64,
        qy: f64,
        k: usize,
        heap: &mut Vec<(f64, usize)>,
    ) {
        let node = &self.nodes[node_idx];
        let p = &self.points[node.point_idx];

        let dx = qx - p.x;
        let dy = qy - p.y;
        let dist_sq = dx * dx + dy * dy;

        // Insert into heap if closer than the k-th best
        let max_dist_sq = if heap.len() >= k {
            heap[0].0 // max-heap: first element is farthest
        } else {
            f64::MAX
        };

        if dist_sq < max_dist_sq || heap.len() < k {
            if heap.len() >= k {
                // Remove the farthest point
                heap.remove(0);
            }
            // Insert maintaining descending order (max-heap via sorted vec)
            let pos = heap
                .binary_search_by(|probe| {
                    probe
                        .0
                        .partial_cmp(&dist_sq)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .reverse()
                })
                .unwrap_or_else(|e| e);
            heap.insert(pos, (dist_sq, node.point_idx));
        }

        let diff = if node.split_dim == 0 { dx } else { dy };
        let (first, second) = if diff < 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(child) = first {
            self.knn_recursive(child, qx, qy, k, heap);
        }

        let threshold = if heap.len() >= k {
            heap[0].0
        } else {
            f64::MAX
        };

        if diff * diff < threshold {
            if let Some(child) = second {
                self.knn_recursive(child, qx, qy, k, heap);
            }
        }
    }

    fn radius_recursive(
        &self,
        node_idx: usize,
        qx: f64,
        qy: f64,
        radius_sq: f64,
        results: &mut Vec<NearestResult>,
    ) {
        let node = &self.nodes[node_idx];
        let p = &self.points[node.point_idx];

        let dx = qx - p.x;
        let dy = qy - p.y;
        let dist_sq = dx * dx + dy * dy;

        if dist_sq <= radius_sq {
            results.push(NearestResult {
                point: *p,
                distance_sq: dist_sq,
                index: node.point_idx,
            });
        }

        let diff = if node.split_dim == 0 { dx } else { dy };

        // Always check both sides if the splitting plane is within radius
        if let Some(left) = node.left {
            if diff > 0.0 || diff * diff <= radius_sq {
                self.radius_recursive(left, qx, qy, radius_sq, results);
            }
        }

        if let Some(right) = node.right {
            if diff < 0.0 || diff * diff <= radius_sq {
                self.radius_recursive(right, qx, qy, radius_sq, results);
            }
        }
    }
}

/// Recursively build the k-d tree.
fn build_recursive(
    points: &[SamplePoint],
    indices: &mut [usize],
    depth: usize,
    nodes: &mut Vec<KdNode>,
) -> usize {
    let n = indices.len();
    let split_dim = (depth % 2) as u8;

    // Sort by split dimension
    indices.sort_by(|&a, &b| {
        let va = if split_dim == 0 {
            points[a].x
        } else {
            points[a].y
        };
        let vb = if split_dim == 0 {
            points[b].x
        } else {
            points[b].y
        };
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let median = n / 2;
    let point_idx = indices[median];

    let node_idx = nodes.len();
    nodes.push(KdNode {
        point_idx,
        split_dim,
        left: None,
        right: None,
    });

    if median > 0 {
        let mut left_indices = indices[..median].to_vec();
        let left_idx = build_recursive(points, &mut left_indices, depth + 1, nodes);
        nodes[node_idx].left = Some(left_idx);
    }

    if median + 1 < n {
        let mut right_indices = indices[median + 1..].to_vec();
        let right_idx = build_recursive(points, &mut right_indices, depth + 1, nodes);
        nodes[node_idx].right = Some(right_idx);
    }

    node_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_points() -> Vec<SamplePoint> {
        vec![
            SamplePoint::new(2.0, 3.0, 10.0),
            SamplePoint::new(5.0, 4.0, 20.0),
            SamplePoint::new(9.0, 6.0, 30.0),
            SamplePoint::new(4.0, 7.0, 40.0),
            SamplePoint::new(8.0, 1.0, 50.0),
            SamplePoint::new(7.0, 2.0, 60.0),
            SamplePoint::new(1.0, 8.0, 70.0),
            SamplePoint::new(6.0, 5.0, 80.0),
        ]
    }

    #[test]
    fn test_build_and_size() {
        let pts = sample_points();
        let tree = KdTree::build(&pts);
        assert_eq!(tree.len(), 8);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_empty_tree() {
        let tree = KdTree::build(&[]);
        assert!(tree.is_empty());
        assert!(tree.nearest(0.0, 0.0).is_none());
        assert!(tree.k_nearest(0.0, 0.0, 3).is_empty());
        assert!(tree.within_radius(0.0, 0.0, 10.0).is_empty());
    }

    #[test]
    fn test_nearest_exact() {
        let pts = sample_points();
        let tree = KdTree::build(&pts);

        // Query at exact point location
        let result = tree.nearest(5.0, 4.0).unwrap();
        assert!(result.distance_sq < 1e-10);
        assert!((result.point.value - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_correct() {
        let pts = sample_points();
        let tree = KdTree::build(&pts);

        // Query near (6, 5) — should find (6, 5) with value 80
        let result = tree.nearest(6.1, 5.1).unwrap();
        assert!((result.point.value - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_matches_brute_force() {
        let pts = sample_points();
        let tree = KdTree::build(&pts);

        // Test many query points
        for qx in 0..10 {
            for qy in 0..10 {
                let qx = qx as f64 + 0.5;
                let qy = qy as f64 + 0.5;

                let tree_result = tree.nearest(qx, qy).unwrap();

                // Brute force
                let bf_result = pts
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.dist_sq(qx, qy)
                            .partial_cmp(&b.dist_sq(qx, qy))
                            .unwrap()
                    })
                    .unwrap();

                assert!(
                    (tree_result.distance_sq - bf_result.1.dist_sq(qx, qy)).abs() < 1e-10,
                    "Mismatch at ({}, {}): tree={:.4}, bf={:.4}",
                    qx,
                    qy,
                    tree_result.distance_sq,
                    bf_result.1.dist_sq(qx, qy)
                );
            }
        }
    }

    #[test]
    fn test_k_nearest() {
        let pts = sample_points();
        let tree = KdTree::build(&pts);

        let results = tree.k_nearest(5.0, 5.0, 3);
        assert_eq!(results.len(), 3);

        // Should be sorted ascending
        for i in 1..results.len() {
            assert!(results[i].distance_sq >= results[i - 1].distance_sq);
        }

        // Verify against brute force
        let mut dists: Vec<(f64, usize)> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| (p.dist_sq(5.0, 5.0), i))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (i, r) in results.iter().enumerate() {
            assert!(
                (r.distance_sq - dists[i].0).abs() < 1e-10,
                "k={}: tree={:.4}, bf={:.4}",
                i,
                r.distance_sq,
                dists[i].0
            );
        }
    }

    #[test]
    fn test_k_nearest_more_than_points() {
        let pts = sample_points();
        let tree = KdTree::build(&pts);

        let results = tree.k_nearest(5.0, 5.0, 100);
        assert_eq!(results.len(), pts.len());
    }

    #[test]
    fn test_within_radius() {
        let pts = sample_points();
        let tree = KdTree::build(&pts);

        let results = tree.within_radius(5.0, 5.0, 2.0);

        // Verify all results are within radius
        for r in &results {
            assert!(r.distance_sq <= 4.0 + 1e-10);
        }

        // Verify against brute force
        let bf_count = pts.iter().filter(|p| p.dist_sq(5.0, 5.0) <= 4.0).count();
        assert_eq!(results.len(), bf_count);
    }

    #[test]
    fn test_within_radius_zero() {
        let pts = sample_points();
        let tree = KdTree::build(&pts);
        let results = tree.within_radius(5.0, 5.0, 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_point() {
        let pts = vec![SamplePoint::new(3.0, 4.0, 100.0)];
        let tree = KdTree::build(&pts);

        let result = tree.nearest(0.0, 0.0).unwrap();
        assert!((result.distance_sq - 25.0).abs() < 1e-10);

        let knn = tree.k_nearest(0.0, 0.0, 5);
        assert_eq!(knn.len(), 1);
    }

    #[test]
    fn test_collinear_points() {
        // All points on a line
        let pts: Vec<SamplePoint> = (0..10)
            .map(|i| SamplePoint::new(i as f64, 0.0, i as f64))
            .collect();
        let tree = KdTree::build(&pts);

        let result = tree.nearest(4.5, 0.0).unwrap();
        assert!(result.distance_sq <= 0.25 + 1e-10);

        let knn = tree.k_nearest(4.5, 0.0, 3);
        assert_eq!(knn.len(), 3);
    }

    #[test]
    fn test_large_dataset() {
        // 1000 random-ish points
        let pts: Vec<SamplePoint> = (0..1000)
            .map(|i| {
                let x = ((i * 7 + 13) % 100) as f64;
                let y = ((i * 11 + 37) % 100) as f64;
                SamplePoint::new(x, y, i as f64)
            })
            .collect();
        let tree = KdTree::build(&pts);
        assert_eq!(tree.len(), 1000);

        // Spot-check nearest matches brute force
        let result = tree.nearest(50.0, 50.0).unwrap();
        let bf = pts
            .iter()
            .map(|p| p.dist_sq(50.0, 50.0))
            .fold(f64::MAX, f64::min);
        assert!((result.distance_sq - bf).abs() < 1e-10);
    }
}
