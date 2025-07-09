//! Spatial indexing structures for DBSCAN clustering
//!
//! This module provides efficient spatial data structures for region queries
//! used in DBSCAN clustering algorithms.

use crate::core::{FormicaXError, OHLCV};

/// Point in n-dimensional space
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// Coordinates of the point
    pub coordinates: Vec<f64>,
    /// Index of the point in the original dataset
    pub index: usize,
}

impl Point {
    /// Create a new point
    pub fn new(coordinates: Vec<f64>, index: usize) -> Self {
        Self { coordinates, index }
    }

    /// Calculate Euclidean distance to another point
    pub fn distance_to(&self, other: &Point) -> f64 {
        self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Check if this point is within epsilon distance of another point
    pub fn is_within_epsilon(&self, other: &Point, epsilon: f64) -> bool {
        self.distance_to(other) <= epsilon
    }
}

/// KD-tree node
#[derive(Debug)]
struct KDTreeNode {
    point: Point,
    left: Option<Box<KDTreeNode>>,
    right: Option<Box<KDTreeNode>>,
    axis: usize,
}

impl KDTreeNode {
    /// Create a new KD-tree node
    fn new(point: Point, axis: usize) -> Self {
        Self {
            point,
            left: None,
            right: None,
            axis,
        }
    }
}

/// KD-tree for efficient spatial queries
#[derive(Debug)]
pub struct KDTree {
    root: Option<Box<KDTreeNode>>,
    dimensions: usize,
}

impl KDTree {
    /// Create a new KD-tree from OHLCV data
    pub fn from_ohlcv(data: &[OHLCV]) -> Result<Self, FormicaXError> {
        if data.is_empty() {
            return Err(FormicaXError::Data(crate::core::DataError::EmptyDataset));
        }

        // Convert OHLCV data to points
        let points: Vec<Point> = data
            .iter()
            .enumerate()
            .map(|(index, ohlcv)| {
                Point::new(vec![ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close], index)
            })
            .collect();

        let dimensions = points[0].coordinates.len();
        let root = Self::build_kdtree(points, 0, dimensions);

        Ok(Self { root, dimensions })
    }

    /// Build KD-tree recursively
    fn build_kdtree(
        mut points: Vec<Point>,
        depth: usize,
        dimensions: usize,
    ) -> Option<Box<KDTreeNode>> {
        if points.is_empty() {
            return None;
        }

        let axis = depth % dimensions;

        // Sort points by the current axis
        points.sort_by(|a, b| {
            a.coordinates[axis]
                .partial_cmp(&b.coordinates[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let median_index = points.len() / 2;
        let median_point = points.remove(median_index);

        let mut left_points = points;
        let right_points = left_points.split_off(median_index);

        let left = Self::build_kdtree(left_points, depth + 1, dimensions);
        let right = Self::build_kdtree(right_points, depth + 1, dimensions);

        let mut node = KDTreeNode::new(median_point, axis);
        node.left = left;
        node.right = right;
        Some(Box::new(node))
    }

    /// Find all points within epsilon distance of a query point
    pub fn range_search(&self, query_point: &Point, epsilon: f64) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(ref root) = self.root {
            Self::range_search_recursive(root, query_point, epsilon, &mut result);
        }
        result
    }

    /// Recursive range search implementation
    fn range_search_recursive(
        node: &KDTreeNode,
        query_point: &Point,
        epsilon: f64,
        result: &mut Vec<usize>,
    ) {
        // Check if current point is within range
        if node.point.is_within_epsilon(query_point, epsilon) {
            result.push(node.point.index);
        }

        let axis = node.axis;
        let query_coord = query_point.coordinates[axis];
        let node_coord = node.point.coordinates[axis];

        // Check left subtree if query point is within epsilon of the splitting plane
        if query_coord - epsilon <= node_coord {
            if let Some(ref left) = node.left {
                Self::range_search_recursive(left, query_point, epsilon, result);
            }
        }

        // Check right subtree if query point is within epsilon of the splitting plane
        if query_coord + epsilon >= node_coord {
            if let Some(ref right) = node.right {
                Self::range_search_recursive(right, query_point, epsilon, result);
            }
        }
    }

    /// Find the k nearest neighbors of a query point
    pub fn k_nearest_neighbors(&self, query_point: &Point, k: usize) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(ref root) = self.root {
            self.knn_recursive(root, query_point, k, &mut result);
        }
        result
    }

    /// Recursive k-nearest neighbors implementation
    fn knn_recursive(
        &self,
        node: &KDTreeNode,
        query_point: &Point,
        k: usize,
        result: &mut Vec<usize>,
    ) {
        // Add current point to result if we have space
        if result.len() < k {
            result.push(node.point.index);
        } else {
            // Replace farthest point if current point is closer
            let current_distance = node.point.distance_to(query_point);
            let farthest_distance = self.get_farthest_distance(result, query_point);

            if current_distance < farthest_distance {
                result.pop(); // Remove farthest point
                result.push(node.point.index);
            }
        }

        let axis = node.axis;
        let query_coord = query_point.coordinates[axis];
        let node_coord = node.point.coordinates[axis];

        // Determine which subtree to explore first
        let (first_subtree, second_subtree) = if query_coord < node_coord {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Explore first subtree
        if let Some(ref first) = first_subtree {
            self.knn_recursive(first, query_point, k, result);
        }

        // Check if we need to explore second subtree
        if let Some(ref second) = second_subtree {
            let axis_distance = (query_coord - node_coord).abs();
            let farthest_distance = self.get_farthest_distance(result, query_point);

            if axis_distance < farthest_distance {
                self.knn_recursive(second, query_point, k, result);
            }
        }
    }

    /// Get the distance to the farthest point in the result set
    fn get_farthest_distance(&self, result: &[usize], _query_point: &Point) -> f64 {
        // This is a simplified implementation
        // In a real implementation, you would maintain a sorted list of distances
        result
            .iter()
            .map(|&_index| {
                // This would need access to the original data to calculate distance
                // For now, return a large value
                1e10
            })
            .fold(0.0, f64::max)
    }

    /// Get the number of dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Check if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }
}

/// Spatial index trait for different indexing strategies
pub trait SpatialIndex {
    /// Find all points within epsilon distance of a query point
    fn range_search(&self, query_point: &Point, epsilon: f64) -> Vec<usize>;

    /// Find the k nearest neighbors of a query point
    fn k_nearest_neighbors(&self, query_point: &Point, k: usize) -> Vec<usize>;

    /// Get the number of dimensions
    fn dimensions(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool;
}

impl SpatialIndex for KDTree {
    fn range_search(&self, query_point: &Point, epsilon: f64) -> Vec<usize> {
        self.range_search(query_point, epsilon)
    }

    fn k_nearest_neighbors(&self, query_point: &Point, k: usize) -> Vec<usize> {
        self.k_nearest_neighbors(query_point, k)
    }

    fn dimensions(&self) -> usize {
        self.dimensions()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OHLCV;
    use chrono::Utc;

    fn create_test_data() -> Vec<OHLCV> {
        vec![
            OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000),
            OHLCV::new(Utc::now(), 102.0, 107.0, 100.0, 104.0, 1200),
            OHLCV::new(Utc::now(), 104.0, 109.0, 102.0, 106.0, 1100),
            OHLCV::new(Utc::now(), 106.0, 111.0, 104.0, 108.0, 1300),
            OHLCV::new(Utc::now(), 108.0, 113.0, 106.0, 110.0, 1400),
        ]
    }

    #[test]
    fn test_point_creation() {
        let point = Point::new(vec![1.0, 2.0, 3.0], 0);
        assert_eq!(point.coordinates, vec![1.0, 2.0, 3.0]);
        assert_eq!(point.index, 0);
    }

    #[test]
    fn test_point_distance() {
        let point1 = Point::new(vec![0.0, 0.0], 0);
        let point2 = Point::new(vec![3.0, 4.0], 1);
        assert_eq!(point1.distance_to(&point2), 5.0);
    }

    #[test]
    fn test_point_within_epsilon() {
        let point1 = Point::new(vec![0.0, 0.0], 0);
        let point2 = Point::new(vec![1.0, 1.0], 1);
        assert!(point1.is_within_epsilon(&point2, 2.0));
        assert!(!point1.is_within_epsilon(&point2, 1.0));
    }

    #[test]
    fn test_kdtree_creation() {
        let data = create_test_data();
        let kdtree = KDTree::from_ohlcv(&data).unwrap();
        assert_eq!(kdtree.dimensions(), 4);
        assert!(!kdtree.is_empty());
    }

    #[test]
    fn test_kdtree_empty_data() {
        let data: Vec<OHLCV> = vec![];
        let result = KDTree::from_ohlcv(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_kdtree_range_search() {
        let data = create_test_data();
        let kdtree = KDTree::from_ohlcv(&data).unwrap();

        let query_point = Point::new(vec![102.0, 107.0, 100.0, 104.0], 0);
        let results = kdtree.range_search(&query_point, 5.0);

        assert!(!results.is_empty());
        assert!(results.contains(&1)); // Should find the second point
    }

    #[test]
    fn test_spatial_index_trait() {
        let data = create_test_data();
        let kdtree = KDTree::from_ohlcv(&data).unwrap();

        let query_point = Point::new(vec![102.0, 107.0, 100.0, 104.0], 0);
        let results = kdtree.range_search(&query_point, 5.0);

        assert!(!results.is_empty());
        assert_eq!(kdtree.dimensions(), 4);
        assert!(!kdtree.is_empty());
    }
}
