//! Core KMeans algorithm implementation (Lloyd, Elkan, etc.)

use super::config::KMeansConfig;
use crate::core::{ClusterResult, FormicaXError, OHLCV};
use std::collections::HashMap;
use std::time::Instant;

/// KMeans clustering algorithm
pub struct KMeans {
    config: KMeansConfig,
    centroids: Option<Vec<Vec<f64>>>,
    assignments: Option<Vec<usize>>,
    converged: bool,
    iterations: usize,
}

impl KMeans {
    /// Create a new KMeans instance with the given config
    pub fn with_config(config: KMeansConfig) -> Self {
        Self {
            config,
            centroids: None,
            assignments: None,
            converged: false,
            iterations: 0,
        }
    }

    /// Fit the KMeans algorithm to the data
    pub fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError> {
        let start_time = Instant::now();

        // Validate input data
        if data.is_empty() {
            return Err(FormicaXError::Data(crate::core::DataError::EmptyDataset));
        }

        if data.len() < self.config.k {
            return Err(FormicaXError::Data(
                crate::core::DataError::InsufficientData {
                    min_points: self.config.k,
                    actual_points: data.len(),
                },
            ));
        }

        // Convert OHLCV data to feature vectors
        let features = self.ohlcv_to_features(data)?;

        // Initialize centroids
        let mut centroids = self.initialize_centroids(&features)?;

        // Main KMeans loop (Lloyd's algorithm)
        let mut assignments = vec![0; data.len()];
        let mut converged = false;
        let mut iterations = 0;

        while iterations < self.config.max_iterations && !converged {
            // Assignment step: assign each point to nearest centroid
            let new_assignments = self.assign_to_centroids(&features, &centroids)?;

            // Check for convergence
            converged = new_assignments == assignments;
            assignments = new_assignments;

            if !converged {
                // Update step: recalculate centroids
                centroids = self.update_centroids(&features, &assignments)?;
            }

            iterations += 1;
        }

        // Store algorithm state
        self.centroids = Some(centroids.clone());
        self.assignments = Some(assignments.clone());
        self.converged = converged;
        self.iterations = iterations;

        // Calculate quality metrics
        let silhouette_score = self.calculate_silhouette_score(&features, &assignments)?;
        let inertia = self.calculate_inertia(&features, &centroids, &assignments);

        let execution_time = start_time.elapsed();

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert(
            "variant".to_string(),
            serde_json::Value::String(format!("{:?}", self.config.variant)),
        );
        metadata.insert(
            "parallel".to_string(),
            serde_json::Value::Bool(self.config.parallel),
        );
        metadata.insert(
            "simd".to_string(),
            serde_json::Value::Bool(self.config.simd),
        );

        Ok(ClusterResult {
            algorithm_name: format!("KMeans-{:?}", self.config.variant),
            n_clusters: self.config.k,
            cluster_assignments: assignments,
            cluster_centers: Some(centroids),
            inertia: Some(inertia),
            silhouette_score,
            iterations,
            converged,
            execution_time,
            noise_points: vec![],  // KMeans doesn't produce noise points
            core_points: vec![],   // KMeans doesn't distinguish core points
            border_points: vec![], // KMeans doesn't distinguish border points
            metadata,
        })
    }

    /// Convert OHLCV data to feature vectors for clustering
    fn ohlcv_to_features(&self, data: &[OHLCV]) -> Result<Vec<Vec<f64>>, FormicaXError> {
        let mut features = Vec::with_capacity(data.len());

        for ohlcv in data {
            // Use OHLCV values as features: [open, high, low, close, volume]
            // Normalize volume to be in a similar range as prices
            let volume_normalized = ohlcv.volume as f64 / 1000.0; // Simple normalization

            features.push(vec![
                ohlcv.open,
                ohlcv.high,
                ohlcv.low,
                ohlcv.close,
                volume_normalized,
            ]);
        }

        Ok(features)
    }

    /// Initialize centroids using k-means++ or random initialization
    fn initialize_centroids(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FormicaXError> {
        let mut centroids = Vec::with_capacity(self.config.k);

        // For now, use simple random initialization
        // TODO: Implement k-means++ initialization
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..features.len()).collect();
        indices.shuffle(&mut rng);

        for i in 0..self.config.k {
            let idx = indices[i % indices.len()];
            centroids.push(features[idx].clone());
        }

        Ok(centroids)
    }

    /// Assign each data point to the nearest centroid
    fn assign_to_centroids(
        &self,
        features: &[Vec<f64>],
        centroids: &[Vec<f64>],
    ) -> Result<Vec<usize>, FormicaXError> {
        let mut assignments = Vec::with_capacity(features.len());

        for feature in features {
            let mut min_distance = f64::INFINITY;
            let mut best_cluster = 0;

            for (cluster_id, centroid) in centroids.iter().enumerate() {
                let distance = self.euclidean_distance(feature, centroid);
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster_id;
                }
            }

            assignments.push(best_cluster);
        }

        Ok(assignments)
    }

    /// Update centroids based on current assignments
    fn update_centroids(
        &self,
        features: &[Vec<f64>],
        assignments: &[usize],
    ) -> Result<Vec<Vec<f64>>, FormicaXError> {
        let n_features = features[0].len();
        let mut new_centroids = vec![vec![0.0; n_features]; self.config.k];
        let mut cluster_sizes = vec![0; self.config.k];

        // Sum up all points in each cluster
        for (feature, &assignment) in features.iter().zip(assignments) {
            cluster_sizes[assignment] += 1;
            for (j, &value) in feature.iter().enumerate() {
                new_centroids[assignment][j] += value;
            }
        }

        // Calculate means
        for (centroid, &size) in new_centroids.iter_mut().zip(&cluster_sizes) {
            if size > 0 {
                for value in centroid.iter_mut() {
                    *value /= size as f64;
                }
            }
        }

        Ok(new_centroids)
    }

    /// Calculate Euclidean distance between two feature vectors
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate silhouette score for clustering quality
    fn calculate_silhouette_score(
        &self,
        _features: &[Vec<f64>],
        _assignments: &[usize],
    ) -> Result<f64, FormicaXError> {
        // Simplified silhouette score calculation
        // TODO: Implement full silhouette score calculation
        Ok(0.5) // Placeholder
    }

    /// Calculate inertia (within-cluster sum of squares)
    fn calculate_inertia(
        &self,
        features: &[Vec<f64>],
        centroids: &[Vec<f64>],
        assignments: &[usize],
    ) -> f64 {
        let mut inertia = 0.0;

        for (feature, &assignment) in features.iter().zip(assignments) {
            let distance = self.euclidean_distance(feature, &centroids[assignment]);
            inertia += distance.powi(2);
        }

        inertia
    }

    /// Get the current centroids
    pub fn get_centroids(&self) -> Option<&Vec<Vec<f64>>> {
        self.centroids.as_ref()
    }

    /// Get the current assignments
    pub fn get_assignments(&self) -> Option<&Vec<usize>> {
        self.assignments.as_ref()
    }

    /// Check if the algorithm converged
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Get the number of iterations performed
    pub fn get_iterations(&self) -> usize {
        self.iterations
    }
}
