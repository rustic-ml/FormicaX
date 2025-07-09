//! Core DBSCAN algorithm implementation
//!
//! This module provides the main DBSCAN clustering algorithm with multiple variants:
//! - Standard DBSCAN with KD-tree spatial indexing
//! - Parallel DBSCAN with lock-free region queries
//! - Incremental DBSCAN for streaming data
//! - Approximate DBSCAN for large-scale datasets

use crate::clustering::dbscan::config::{DBSCANConfig, DBSCANVariant};
use crate::clustering::dbscan::spatial::{KDTree, Point};
use crate::core::{ClusterResult, ClusteringError, DataError, FormicaXError, OHLCV};
#[cfg(feature = "clustering")]
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// DBSCAN clustering algorithm
#[derive(Debug)]
pub struct DBSCAN {
    /// Configuration for the algorithm
    pub config: DBSCANConfig,
    /// Spatial index for efficient region queries
    spatial_index: Option<KDTree>,
    /// Cluster assignments for each point
    cluster_assignments: Vec<usize>,
    /// Cluster centers (for compatibility with ClusteringAlgorithm trait)
    cluster_centers: Option<Vec<Vec<f64>>>,
    /// Number of clusters found
    num_clusters: usize,
    /// Noise points (assigned to cluster 0)
    noise_points: Vec<usize>,
}

impl Default for DBSCAN {
    fn default() -> Self {
        Self::new()
    }
}

impl DBSCAN {
    /// Create a new DBSCAN instance with default configuration
    pub fn new() -> Self {
        Self::with_config(DBSCANConfig::default())
    }

    /// Create a new DBSCAN instance with custom configuration
    pub fn with_config(config: DBSCANConfig) -> Self {
        Self {
            config,
            spatial_index: None,
            cluster_assignments: Vec::new(),
            cluster_centers: None,
            num_clusters: 0,
            noise_points: Vec::new(),
        }
    }

    /// Fit the DBSCAN model to the data
    pub fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError> {
        // Validate input data
        if data.is_empty() {
            return Err(FormicaXError::Data(DataError::EmptyDataset));
        }

        // Validate configuration
        self.validate_config(data)?;

        // Build spatial index
        let spatial_index = KDTree::from_ohlcv(data)?;
        self.spatial_index = Some(spatial_index);

        // Run the appropriate variant
        let cluster_assignments = match self.config.variant {
            DBSCANVariant::Standard => self.run_standard_dbscan(data)?,
            DBSCANVariant::Parallel => self.run_parallel_dbscan(data)?,
            DBSCANVariant::Incremental => self.run_incremental_dbscan(data)?,
            DBSCANVariant::Approximate => self.run_approximate_dbscan(data)?,
        };

        self.cluster_assignments = cluster_assignments;
        self.num_clusters = self.calculate_num_clusters();
        self.noise_points = self.find_noise_points();
        self.cluster_centers = self.calculate_cluster_centers(data);

        // Calculate silhouette score
        let silhouette_score = self.calculate_silhouette_score(data)?;
        let start_time = Instant::now();

        Ok(ClusterResult::new(
            "DBSCAN".to_string(),
            self.num_clusters,
            self.cluster_assignments.clone(),
        )
        .with_silhouette_score(silhouette_score)
        .with_centers(self.cluster_centers.clone().unwrap_or_default())
        .with_noise_points(self.noise_points.clone())
        .with_execution_time(start_time.elapsed())
        .with_converged(true)
        .with_iterations(1))
    }

    /// Run standard DBSCAN algorithm
    fn run_standard_dbscan(&self, data: &[OHLCV]) -> Result<Vec<usize>, FormicaXError> {
        let n = data.len();
        let mut cluster_assignments = vec![0; n]; // 0 = unassigned
        let mut cluster_id = 1;
        let mut visited = HashSet::new();

        let spatial_index = self.spatial_index.as_ref().unwrap();

        for i in 0..n {
            if visited.contains(&i) {
                continue;
            }

            visited.insert(i);

            let point = Point::new(
                vec![data[i].open, data[i].high, data[i].low, data[i].close],
                i,
            );

            let neighbors = spatial_index.range_search(&point, self.config.epsilon);

            if neighbors.len() < self.config.min_points {
                // Mark as noise
                cluster_assignments[i] = 0;
            } else {
                // Start a new cluster
                cluster_assignments[i] = cluster_id;
                self.expand_cluster(
                    data,
                    &mut cluster_assignments,
                    &mut visited,
                    &neighbors,
                    cluster_id,
                    spatial_index,
                )?;
                cluster_id += 1;
            }
        }

        Ok(cluster_assignments)
    }

    /// Run parallel DBSCAN algorithm
    fn run_parallel_dbscan(&self, data: &[OHLCV]) -> Result<Vec<usize>, FormicaXError> {
        let n = data.len();
        let cluster_assignments = Arc::new(Mutex::new(vec![0; n]));
        let visited = Arc::new(Mutex::new(HashSet::new()));
        let cluster_counter = Arc::new(Mutex::new(1));

        let spatial_index = self.spatial_index.as_ref().unwrap();

        // Process points in parallel
        #[cfg(feature = "parallel")]
        {
            (0..n).into_par_iter().for_each(|i| {
                let mut visited_guard = visited.lock().unwrap();
                if visited_guard.contains(&i) {
                    return;
                }
                visited_guard.insert(i);
                drop(visited_guard);

                let point = Point::new(
                    vec![data[i].open, data[i].high, data[i].low, data[i].close],
                    i,
                );

                let neighbors = spatial_index.range_search(&point, self.config.epsilon);

                if neighbors.len() < self.config.min_points {
                    // Mark as noise
                    let mut assignments_guard = cluster_assignments.lock().unwrap();
                    assignments_guard[i] = 0;
                } else {
                    // Start a new cluster
                    let cluster_id = {
                        let mut counter_guard = cluster_counter.lock().unwrap();
                        let id = *counter_guard;
                        *counter_guard += 1;
                        id
                    };

                    let mut assignments_guard = cluster_assignments.lock().unwrap();
                    assignments_guard[i] = cluster_id;
                    drop(assignments_guard);

                    self.expand_cluster_parallel(
                        data,
                        &cluster_assignments,
                        &visited,
                        &neighbors,
                        cluster_id,
                        spatial_index,
                    );
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..n).for_each(|i| {
                let mut visited_guard = visited.lock().unwrap();
                if visited_guard.contains(&i) {
                    return;
                }
                visited_guard.insert(i);
                drop(visited_guard);

                let point = Point::new(
                    vec![data[i].open, data[i].high, data[i].low, data[i].close],
                    i,
                );

                let neighbors = spatial_index.range_search(&point, self.config.epsilon);

                if neighbors.len() < self.config.min_points {
                    // Mark as noise
                    let mut assignments_guard = cluster_assignments.lock().unwrap();
                    assignments_guard[i] = 0;
                } else {
                    // Start a new cluster
                    let cluster_id = {
                        let mut counter_guard = cluster_counter.lock().unwrap();
                        let id = *counter_guard;
                        *counter_guard += 1;
                        id
                    };

                    let mut assignments_guard = cluster_assignments.lock().unwrap();
                    assignments_guard[i] = cluster_id;
                    drop(assignments_guard);

                    self.expand_cluster_parallel(
                        data,
                        &cluster_assignments,
                        &visited,
                        &neighbors,
                        cluster_id,
                        spatial_index,
                    );
                }
            });
        }

        let result = Arc::try_unwrap(cluster_assignments)
            .unwrap()
            .into_inner()
            .unwrap();

        Ok(result)
    }

    /// Run incremental DBSCAN algorithm
    fn run_incremental_dbscan(&self, data: &[OHLCV]) -> Result<Vec<usize>, FormicaXError> {
        // For incremental DBSCAN, we process data in chunks
        let chunk_size = self.config.buffer_size;
        let mut cluster_assignments = vec![0; data.len()];
        let mut cluster_id = 1;

        for chunk_start in (0..data.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(data.len());
            let chunk_data = &data[chunk_start..chunk_end];

            // Force Standard variant for chunk DBSCAN
            let mut chunk_config = self.config.clone();
            chunk_config.variant = DBSCANVariant::Standard;
            let mut chunk_dbscan = DBSCAN::with_config(chunk_config);
            let _ = chunk_dbscan.fit(chunk_data)?;
            let chunk_assignments = chunk_dbscan.cluster_assignments.clone();

            // Adjust cluster IDs and copy to main assignments
            for (i, &assignment) in chunk_assignments.iter().enumerate() {
                let global_index = chunk_start + i;
                if assignment > 0 {
                    cluster_assignments[global_index] = assignment + cluster_id - 1;
                }
            }

            // Update cluster_id for next chunk
            let max_cluster_in_chunk = chunk_assignments
                .iter()
                .filter(|&&x| x > 0)
                .max()
                .unwrap_or(&0);
            cluster_id += max_cluster_in_chunk;
        }

        Ok(cluster_assignments)
    }

    /// Run approximate DBSCAN algorithm
    fn run_approximate_dbscan(&self, data: &[OHLCV]) -> Result<Vec<usize>, FormicaXError> {
        // Approximate DBSCAN uses sampling to reduce computational complexity
        let min_points = self.config.min_points.max(3);
        let sample_size = ((data.len() as f64 * 0.1) as usize).max(min_points);
        if sample_size >= data.len() {
            // If sample is the whole dataset, just run standard DBSCAN
            return self.run_standard_dbscan(data);
        }
        #[cfg(feature = "clustering")]
        let sample_indices: Vec<usize> = {
            use rand::seq::IteratorRandom;
            let mut rng = rand::thread_rng();
            let unique_indices: Vec<usize> = (0..data.len()).choose_multiple(&mut rng, sample_size);
            if unique_indices.len() < data.len() {
                // Not enough unique indices, fallback
                return self.run_standard_dbscan(data);
            }
            unique_indices
        };
        #[cfg(not(feature = "clustering"))]
        let sample_indices: Vec<usize> = (0..sample_size).collect();

        let sample_data: Vec<OHLCV> = sample_indices.iter().map(|&i| data[i].clone()).collect();

        // Run standard DBSCAN on sample
        let sample_assignments = self.run_standard_dbscan(&sample_data)?;

        // Assign remaining points to nearest cluster
        let mut cluster_assignments = vec![0; data.len()];

        // Copy sample assignments
        for (&global_idx, &assignment) in sample_indices.iter().zip(sample_assignments.iter()) {
            if global_idx < cluster_assignments.len() {
                cluster_assignments[global_idx] = assignment;
            }
        }

        // Assign remaining points
        for i in 0..data.len() {
            if !sample_indices.contains(&i) {
                let point = Point::new(
                    vec![data[i].open, data[i].high, data[i].low, data[i].close],
                    i,
                );

                // Find nearest cluster
                let nearest_cluster =
                    self.find_nearest_cluster(&point, &sample_data, &sample_assignments);
                cluster_assignments[i] = nearest_cluster;
            }
        }

        Ok(cluster_assignments)
    }

    /// Expand a cluster by adding all density-reachable points
    fn expand_cluster(
        &self,
        data: &[OHLCV],
        cluster_assignments: &mut [usize],
        visited: &mut HashSet<usize>,
        neighbors: &[usize],
        cluster_id: usize,
        spatial_index: &KDTree,
    ) -> Result<(), FormicaXError> {
        let mut queue = VecDeque::new();
        queue.extend(neighbors.iter().cloned());

        while let Some(point_idx) = queue.pop_front() {
            if visited.contains(&point_idx) {
                continue;
            }

            visited.insert(point_idx);
            cluster_assignments[point_idx] = cluster_id;

            let point = Point::new(
                vec![
                    data[point_idx].open,
                    data[point_idx].high,
                    data[point_idx].low,
                    data[point_idx].close,
                ],
                point_idx,
            );

            let point_neighbors = spatial_index.range_search(&point, self.config.epsilon);

            if point_neighbors.len() >= self.config.min_points {
                queue.extend(point_neighbors.iter().cloned());
            }
        }

        Ok(())
    }

    /// Expand a cluster in parallel
    fn expand_cluster_parallel(
        &self,
        data: &[OHLCV],
        cluster_assignments: &Arc<Mutex<Vec<usize>>>,
        visited: &Arc<Mutex<HashSet<usize>>>,
        neighbors: &[usize],
        cluster_id: usize,
        spatial_index: &KDTree,
    ) {
        let mut queue = VecDeque::new();
        queue.extend(neighbors.iter().cloned());

        while let Some(point_idx) = queue.pop_front() {
            let mut visited_guard = visited.lock().unwrap();
            if visited_guard.contains(&point_idx) {
                continue;
            }
            visited_guard.insert(point_idx);
            drop(visited_guard);

            let mut assignments_guard = cluster_assignments.lock().unwrap();
            assignments_guard[point_idx] = cluster_id;
            drop(assignments_guard);

            let point = Point::new(
                vec![
                    data[point_idx].open,
                    data[point_idx].high,
                    data[point_idx].low,
                    data[point_idx].close,
                ],
                point_idx,
            );

            let point_neighbors = spatial_index.range_search(&point, self.config.epsilon);

            if point_neighbors.len() >= self.config.min_points {
                queue.extend(point_neighbors.iter().cloned());
            }
        }
    }

    /// Find the nearest cluster for a point
    fn find_nearest_cluster(
        &self,
        point: &Point,
        sample_data: &[OHLCV],
        sample_assignments: &[usize],
    ) -> usize {
        let mut min_distance = f64::INFINITY;
        let mut nearest_cluster = 0;

        for (i, sample_point) in sample_data.iter().enumerate() {
            let sample_coords = vec![
                sample_point.open,
                sample_point.high,
                sample_point.low,
                sample_point.close,
            ];
            let sample_point = Point::new(sample_coords, i);

            let distance = point.distance_to(&sample_point);
            if distance < min_distance {
                min_distance = distance;
                nearest_cluster = sample_assignments[i];
            }
        }

        nearest_cluster
    }

    /// Calculate the number of clusters
    fn calculate_num_clusters(&self) -> usize {
        self.cluster_assignments
            .iter()
            .filter(|&&x| x > 0)
            .collect::<HashSet<_>>()
            .len()
    }

    /// Find noise points (assigned to cluster 0)
    fn find_noise_points(&self) -> Vec<usize> {
        self.cluster_assignments
            .iter()
            .enumerate()
            .filter(|(_, &cluster)| cluster == 0)
            .map(|(index, _)| index)
            .collect()
    }

    /// Calculate cluster centers
    fn calculate_cluster_centers(&self, data: &[OHLCV]) -> Option<Vec<Vec<f64>>> {
        if self.num_clusters == 0 {
            return None;
        }

        let mut cluster_sums: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut cluster_counts: HashMap<usize, usize> = HashMap::new();

        for (i, &cluster_id) in self.cluster_assignments.iter().enumerate() {
            if cluster_id > 0 {
                let coords = vec![data[i].open, data[i].high, data[i].low, data[i].close];

                cluster_sums
                    .entry(cluster_id)
                    .and_modify(|sum| {
                        for (sum_val, coord) in sum.iter_mut().zip(coords.iter()) {
                            *sum_val += coord;
                        }
                    })
                    .or_insert(coords);

                *cluster_counts.entry(cluster_id).or_insert(0) += 1;
            }
        }

        let mut centers = Vec::new();
        for cluster_id in 1..=self.num_clusters {
            if let (Some(sum), Some(count)) = (
                cluster_sums.get(&cluster_id),
                cluster_counts.get(&cluster_id),
            ) {
                let center: Vec<f64> = sum.iter().map(|&x| x / *count as f64).collect();
                centers.push(center);
            }
        }

        Some(centers)
    }

    /// Calculate silhouette score
    fn calculate_silhouette_score(&self, data: &[OHLCV]) -> Result<f64, FormicaXError> {
        if self.num_clusters < 2 {
            return Ok(0.0);
        }

        let mut total_silhouette = 0.0;
        let mut valid_points = 0;

        for i in 0..data.len() {
            if self.cluster_assignments[i] == 0 {
                continue; // Skip noise points
            }

            let point = Point::new(
                vec![data[i].open, data[i].high, data[i].low, data[i].close],
                i,
            );

            let a = self.calculate_intra_cluster_distance(&point, i, data)?;
            let b = self.calculate_nearest_cluster_distance(&point, i, data)?;

            if b > a {
                total_silhouette += (b - a) / b.max(a);
            } else {
                total_silhouette += 0.0;
            }

            valid_points += 1;
        }

        if valid_points == 0 {
            return Ok(0.0);
        }

        Ok(total_silhouette / valid_points as f64)
    }

    /// Calculate intra-cluster distance
    fn calculate_intra_cluster_distance(
        &self,
        point: &Point,
        point_idx: usize,
        data: &[OHLCV],
    ) -> Result<f64, FormicaXError> {
        let cluster_id = self.cluster_assignments[point_idx];
        let mut total_distance = 0.0;
        let mut count = 0;

        for (i, &assignment) in self.cluster_assignments.iter().enumerate() {
            if assignment == cluster_id && i != point_idx {
                let other_point = Point::new(
                    vec![data[i].open, data[i].high, data[i].low, data[i].close],
                    i,
                );
                total_distance += point.distance_to(&other_point);
                count += 1;
            }
        }

        if count == 0 {
            return Ok(0.0);
        }

        Ok(total_distance / count as f64)
    }

    /// Calculate distance to nearest cluster
    fn calculate_nearest_cluster_distance(
        &self,
        point: &Point,
        point_idx: usize,
        data: &[OHLCV],
    ) -> Result<f64, FormicaXError> {
        let current_cluster = self.cluster_assignments[point_idx];
        let mut min_distance = f64::INFINITY;

        for cluster_id in 1..=self.num_clusters {
            if cluster_id == current_cluster {
                continue;
            }

            let mut total_distance = 0.0;
            let mut count = 0;

            for (i, &assignment) in self.cluster_assignments.iter().enumerate() {
                if assignment == cluster_id {
                    let other_point = Point::new(
                        vec![data[i].open, data[i].high, data[i].low, data[i].close],
                        i,
                    );
                    total_distance += point.distance_to(&other_point);
                    count += 1;
                }
            }

            if count > 0 {
                let avg_distance = total_distance / count as f64;
                min_distance = min_distance.min(avg_distance);
            }
        }

        Ok(min_distance)
    }

    /// Predict cluster assignments for new data
    pub fn predict(&self, data: &[OHLCV]) -> Result<Vec<usize>, FormicaXError> {
        if self.spatial_index.is_none() {
            return Err(FormicaXError::Clustering(ClusteringError::AlgorithmError {
                message: "DBSCAN model not fitted".to_string(),
            }));
        }

        let mut predictions = Vec::new();

        for ohlcv in data {
            let point = Point::new(vec![ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close], 0);

            // Find nearest cluster center
            let nearest_cluster = if let Some(ref centers) = self.cluster_centers {
                let mut min_distance = f64::INFINITY;
                let mut nearest_cluster = 0;

                for (cluster_id, center) in centers.iter().enumerate() {
                    let center_point = Point::new(center.clone(), 0);
                    let distance = point.distance_to(&center_point);

                    if distance < min_distance {
                        min_distance = distance;
                        nearest_cluster = cluster_id + 1;
                    }
                }

                nearest_cluster
            } else {
                0
            };

            predictions.push(nearest_cluster);
        }

        Ok(predictions)
    }

    /// Get cluster centers
    pub fn get_cluster_centers(&self) -> Option<Vec<Vec<f64>>> {
        self.cluster_centers.clone()
    }

    /// Validate configuration
    pub fn validate_config(&self, data: &[OHLCV]) -> Result<(), FormicaXError> {
        self.config.validate()?;

        if data.is_empty() {
            return Err(FormicaXError::Data(DataError::EmptyDataset));
        }

        Ok(())
    }

    /// Get the number of clusters
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Get noise points
    pub fn noise_points(&self) -> &[usize] {
        &self.noise_points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_data() -> Vec<OHLCV> {
        vec![
            OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000),
            OHLCV::new(Utc::now(), 102.0, 107.0, 100.0, 104.0, 1200),
            OHLCV::new(Utc::now(), 104.0, 109.0, 102.0, 106.0, 1100),
            OHLCV::new(Utc::now(), 106.0, 111.0, 104.0, 108.0, 1300),
            OHLCV::new(Utc::now(), 108.0, 113.0, 106.0, 110.0, 1400),
            OHLCV::new(Utc::now(), 200.0, 205.0, 198.0, 202.0, 1000), // Different cluster
            OHLCV::new(Utc::now(), 202.0, 207.0, 200.0, 204.0, 1200),
            OHLCV::new(Utc::now(), 204.0, 209.0, 202.0, 206.0, 1100),
        ]
    }

    #[test]
    fn test_dbscan_creation() {
        let dbscan = DBSCAN::new();
        assert_eq!(dbscan.num_clusters(), 0);
        assert!(dbscan.noise_points().is_empty());
    }

    #[test]
    fn test_dbscan_with_config() {
        let config = DBSCANConfig::builder()
            .epsilon(0.5)
            .min_points(2)
            .variant(DBSCANVariant::Standard)
            .build()
            .unwrap();

        let dbscan = DBSCAN::with_config(config);
        assert_eq!(dbscan.num_clusters(), 0);
    }

    #[test]
    fn test_dbscan_fit_standard() {
        let data = create_test_data();
        let config = DBSCANConfig::builder()
            .epsilon(10.0)
            .min_points(2)
            .variant(DBSCANVariant::Standard)
            .build()
            .unwrap();

        let mut dbscan = DBSCAN::with_config(config);
        let result = dbscan.fit(&data).unwrap();

        assert!(result.n_clusters > 0);
        assert_eq!(result.cluster_assignments.len(), data.len());
        assert_eq!(result.algorithm_name, "DBSCAN");
    }

    #[test]
    fn test_dbscan_fit_parallel() {
        let data = create_test_data();
        let config = DBSCANConfig::builder()
            .epsilon(10.0)
            .min_points(2)
            .variant(DBSCANVariant::Parallel)
            .parallel(true)
            .build()
            .unwrap();

        let mut dbscan = DBSCAN::with_config(config);
        let result = dbscan.fit(&data).unwrap();

        assert!(result.n_clusters > 0);
        assert_eq!(result.cluster_assignments.len(), data.len());
    }

    #[test]
    fn test_dbscan_fit_incremental() {
        let data = create_test_data();
        let config = DBSCANConfig::builder()
            .epsilon(10.0)
            .min_points(2)
            .variant(DBSCANVariant::Incremental)
            .buffer_size(3)
            .build()
            .unwrap();

        let mut dbscan = DBSCAN::with_config(config);
        let result = dbscan.fit(&data).unwrap();

        assert!(result.n_clusters > 0);
        assert_eq!(result.cluster_assignments.len(), data.len());
    }

    #[test]
    fn test_dbscan_fit_approximate() {
        let data = create_test_data();
        let config = DBSCANConfig::builder()
            .epsilon(10.0)
            .min_points(2)
            .variant(DBSCANVariant::Approximate)
            .build()
            .unwrap();

        let mut dbscan = DBSCAN::with_config(config);
        let result = dbscan.fit(&data).unwrap();

        assert!(result.n_clusters > 0);
        assert_eq!(result.cluster_assignments.len(), data.len());
    }

    #[test]
    fn test_dbscan_predict() {
        let data = create_test_data();
        let config = DBSCANConfig::builder()
            .epsilon(10.0)
            .min_points(2)
            .variant(DBSCANVariant::Standard)
            .build()
            .unwrap();

        let mut dbscan = DBSCAN::with_config(config);
        dbscan.fit(&data).unwrap();

        let new_data = vec![
            OHLCV::new(Utc::now(), 101.0, 106.0, 99.0, 103.0, 1000),
            OHLCV::new(Utc::now(), 201.0, 206.0, 199.0, 203.0, 1000),
        ];

        let predictions = dbscan.predict(&new_data).unwrap();
        assert_eq!(predictions.len(), new_data.len());
    }

    #[test]
    fn test_dbscan_validation_metrics() {
        let data = create_test_data();
        let config = DBSCANConfig::builder()
            .epsilon(10.0)
            .min_points(2)
            .variant(DBSCANVariant::Standard)
            .build()
            .unwrap();

        let mut dbscan = DBSCAN::with_config(config);
        let result = dbscan.fit(&data).unwrap();

        assert!(result.silhouette_score >= 0.0);
        assert!(result.silhouette_score <= 1.0);
    }

    #[test]
    fn test_dbscan_empty_data() {
        let data: Vec<OHLCV> = vec![];
        let config = DBSCANConfig::builder()
            .epsilon(0.5)
            .min_points(2)
            .build()
            .unwrap();

        let mut dbscan = DBSCAN::with_config(config);
        let result = dbscan.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dbscan_invalid_config() {
        let _data = create_test_data();
        let config = DBSCANConfig::builder()
            .epsilon(-1.0) // Invalid epsilon
            .min_points(2)
            .build();

        assert!(config.is_err());
    }
}
