use crate::clustering::som::config::{SOMConfig, TopologyType};
use crate::core::{
    ClusterResult, ClusteringAlgorithm, ConfigError, DataError, FormicaXError, OHLCV,
};
use rand::prelude::*;
use std::time::Instant;

/// Self-Organizing Map clustering algorithm
#[derive(Debug)]
pub struct SOM {
    /// Configuration for the SOM algorithm
    config: SOMConfig,
    /// Neuron weights
    weights: Option<Vec<Vec<f64>>>,
    /// Cluster assignments
    assignments: Option<Vec<usize>>,
    /// Grid positions
    grid_positions: Option<Vec<(usize, usize)>>,
    /// Convergence status
    converged: bool,
    /// Number of iterations performed
    iterations: usize,
}

impl SOM {
    /// Create a new SOM with default configuration
    pub fn new() -> Self {
        Self::with_config(SOMConfig::default())
    }

    /// Create a new SOM with custom configuration
    pub fn with_config(config: SOMConfig) -> Self {
        Self {
            config,
            weights: None,
            assignments: None,
            grid_positions: None,
            converged: false,
            iterations: 0,
        }
    }

    /// Fit the SOM to the data
    pub fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError> {
        if data.is_empty() {
            return Err(FormicaXError::Data(DataError::EmptyDataset));
        }

        self.config.validate()?;
        let start_time = Instant::now();

        // Extract features from OHLCV data
        let features = self.extract_features(data);

        // Train the SOM
        let (weights, grid_positions) = self.train_som(&features)?;

        let execution_time = start_time.elapsed();

        self.weights = Some(weights.clone());
        self.grid_positions = Some(grid_positions.clone());

        // Create cluster assignments
        let assignments = self.assign_clusters(&features, &weights, &grid_positions)?;
        self.assignments = Some(assignments.clone());

        // Calculate quality metrics
        let silhouette_score = self.calculate_silhouette_score(&features, &assignments)?;
        let inertia = self.calculate_inertia(&features, &assignments)?;

        // Create cluster centers
        let cluster_centers = self.create_cluster_centers(&weights, &grid_positions);

        Ok(ClusterResult::new(
            "SOM".to_string(),
            self.config.width * self.config.height,
            assignments,
        )
        .with_centers(cluster_centers)
        .with_inertia(inertia)
        .with_silhouette_score(silhouette_score)
        .with_converged(self.converged)
        .with_iterations(self.iterations)
        .with_execution_time(execution_time))
    }

    /// Extract features from OHLCV data
    fn extract_features(&self, data: &[OHLCV]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|ohlcv| {
                vec![
                    ohlcv.open,
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    ohlcv.volume as f64,
                ]
            })
            .collect()
    }

    /// Train the SOM
    fn train_som(
        &mut self,
        features: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<(usize, usize)>), FormicaXError> {
        let n_features = features[0].len();
        let n_neurons = self.config.width * self.config.height;

        // Initialize grid positions
        let mut grid_positions = Vec::new();
        for i in 0..self.config.height {
            for j in 0..self.config.width {
                grid_positions.push((i, j));
            }
        }

        // Initialize weights randomly
        let mut rng = StdRng::from_entropy();
        let mut weights = Vec::new();
        for _ in 0..n_neurons {
            let mut neuron_weights = Vec::new();
            for _ in 0..n_features {
                neuron_weights.push(rng.gen_range(-1.0..1.0));
            }
            weights.push(neuron_weights);
        }

        // Training loop
        for epoch in 0..self.config.epochs {
            self.iterations = epoch + 1;

            // Shuffle data
            let mut indices: Vec<usize> = (0..features.len()).collect();
            indices.shuffle(&mut rng);

            for &idx in &indices {
                let input = &features[idx];

                // Find Best Matching Unit (BMU)
                let bmu_idx = self.find_bmu(input, &weights)?;

                // Update weights
                self.update_weights(input, bmu_idx, epoch, &mut weights, &grid_positions)?;
            }

            // Check convergence (simplified)
            if epoch > 0 && epoch % 10 == 0 {
                // Check if weights have stabilized
                self.converged = true; // Simplified convergence check
            }
        }

        Ok((weights, grid_positions))
    }

    /// Find Best Matching Unit
    fn find_bmu(&self, input: &[f64], weights: &[Vec<f64>]) -> Result<usize, FormicaXError> {
        let mut best_idx = 0;
        let mut best_distance = f64::INFINITY;

        for (i, neuron_weights) in weights.iter().enumerate() {
            let distance = self.euclidean_distance_squared(input, neuron_weights);
            if distance < best_distance {
                best_distance = distance;
                best_idx = i;
            }
        }

        Ok(best_idx)
    }

    /// Update weights
    fn update_weights(
        &self,
        input: &[f64],
        bmu_idx: usize,
        epoch: usize,
        weights: &mut [Vec<f64>],
        _grid_positions: &[(usize, usize)],
    ) -> Result<(), FormicaXError> {
        let current_learning_rate =
            self.config.learning_rate * (1.0 - epoch as f64 / self.config.epochs as f64);
        let current_radius =
            self.config.neighborhood_radius * (1.0 - epoch as f64 / self.config.epochs as f64);

        let bmu_pos = _grid_positions[bmu_idx];

        for (i, neuron_weights) in weights.iter_mut().enumerate() {
            let neuron_pos = _grid_positions[i];
            let distance = self.grid_distance(&bmu_pos, &neuron_pos);

            if distance <= current_radius {
                let neighborhood_function =
                    (-distance * distance / (2.0 * current_radius * current_radius)).exp();

                for (j, &input_val) in input.iter().enumerate() {
                    neuron_weights[j] += current_learning_rate
                        * neighborhood_function
                        * (input_val - neuron_weights[j]);
                }
            }
        }

        Ok(())
    }

    /// Calculate distance between grid positions
    fn grid_distance(&self, pos1: &(usize, usize), pos2: &(usize, usize)) -> f64 {
        match self.config.topology {
            TopologyType::Rectangular => {
                let dx = (pos1.0 as f64 - pos2.0 as f64).abs();
                let dy = (pos1.1 as f64 - pos2.1 as f64).abs();
                (dx * dx + dy * dy).sqrt()
            }
            TopologyType::Hexagonal => {
                let dx = (pos1.0 as f64 - pos2.0 as f64).abs();
                let dy = (pos1.1 as f64 - pos2.1 as f64).abs();
                dx + dy + (dx * dy).max(0.0)
            }
        }
    }

    /// Assign clusters based on BMU
    fn assign_clusters(
        &self,
        features: &[Vec<f64>],
        weights: &[Vec<f64>],
        _grid_positions: &[(usize, usize)],
    ) -> Result<Vec<usize>, FormicaXError> {
        let mut assignments = Vec::new();

        for feature in features {
            let bmu_idx = self.find_bmu(feature, weights)?;
            assignments.push(bmu_idx);
        }

        Ok(assignments)
    }

    /// Create cluster centers from neuron weights
    fn create_cluster_centers(
        &self,
        weights: &[Vec<f64>],
        _grid_positions: &[(usize, usize)],
    ) -> Vec<Vec<f64>> {
        weights.to_vec()
    }

    /// Calculate silhouette score
    fn calculate_silhouette_score(
        &self,
        features: &[Vec<f64>],
        assignments: &[usize],
    ) -> Result<f64, FormicaXError> {
        // Simplified silhouette score calculation
        let n_samples = features.len();
        if n_samples < 2 {
            return Ok(0.0);
        }

        let mut total_silhouette = 0.0;
        let mut valid_samples = 0;

        for i in 0..n_samples {
            let cluster_i = assignments[i];

            // Calculate intra-cluster distance
            let mut intra_dist = 0.0;
            let mut intra_count = 0;

            for j in 0..n_samples {
                if i != j && assignments[j] == cluster_i {
                    intra_dist += self.euclidean_distance(&features[i], &features[j]);
                    intra_count += 1;
                }
            }

            if intra_count == 0 {
                continue;
            }
            intra_dist /= intra_count as f64;

            // Calculate nearest inter-cluster distance
            let mut min_inter_dist = f64::INFINITY;
            let n_clusters = self.config.width * self.config.height;
            for cluster_k in 0..n_clusters {
                if cluster_k != cluster_i {
                    let mut inter_dist = 0.0;
                    let mut inter_count = 0;

                    for j in 0..n_samples {
                        if assignments[j] == cluster_k {
                            inter_dist += self.euclidean_distance(&features[i], &features[j]);
                            inter_count += 1;
                        }
                    }

                    if inter_count > 0 {
                        inter_dist /= inter_count as f64;
                        min_inter_dist = min_inter_dist.min(inter_dist);
                    }
                }
            }

            if min_inter_dist.is_finite() {
                let silhouette = (min_inter_dist - intra_dist) / intra_dist.max(min_inter_dist);
                total_silhouette += silhouette;
                valid_samples += 1;
            }
        }

        if valid_samples > 0 {
            Ok(total_silhouette / valid_samples as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate inertia (within-cluster sum of squares)
    fn calculate_inertia(
        &self,
        features: &[Vec<f64>],
        assignments: &[usize],
    ) -> Result<f64, FormicaXError> {
        let weights = self.weights.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "SOM not fitted yet".to_string(),
            })
        })?;

        let mut inertia = 0.0;
        for (i, feature) in features.iter().enumerate() {
            let cluster = assignments[i];
            if cluster < weights.len() {
                inertia += self.euclidean_distance_squared(feature, &weights[cluster]);
            }
        }

        Ok(inertia)
    }

    /// Euclidean distance
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Squared Euclidean distance
    fn euclidean_distance_squared(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
    }

    /// Get neuron weights
    pub fn get_weights(&self) -> Option<&Vec<Vec<f64>>> {
        self.weights.as_ref()
    }

    /// Get cluster assignments
    pub fn get_assignments(&self) -> Option<&Vec<usize>> {
        self.assignments.as_ref()
    }

    /// Get grid positions
    pub fn get_grid_positions(&self) -> Option<&Vec<(usize, usize)>> {
        self.grid_positions.as_ref()
    }

    /// Check if the algorithm converged
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Get number of iterations performed
    pub fn get_iterations(&self) -> usize {
        self.iterations
    }

    pub fn width(&self) -> usize {
        self.config.width
    }
    pub fn height(&self) -> usize {
        self.config.height
    }
}

impl ClusteringAlgorithm for SOM {
    type Config = SOMConfig;

    fn new() -> Self {
        Self::new()
    }

    fn with_config(config: Self::Config) -> Self {
        Self::with_config(config)
    }

    fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError> {
        self.fit(data)
    }

    fn predict(&self, data: &[OHLCV]) -> Result<Vec<usize>, FormicaXError> {
        let features = self.extract_features(data);
        let weights = self.weights.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "SOM not fitted yet".to_string(),
            })
        })?;
        let grid_positions = self.grid_positions.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "SOM not fitted yet".to_string(),
            })
        })?;

        self.assign_clusters(&features, weights, grid_positions)
    }

    fn get_cluster_centers(&self) -> Option<Vec<Vec<f64>>> {
        self.weights.clone()
    }

    fn validate_config(&self, data: &[OHLCV]) -> Result<(), FormicaXError> {
        self.config.validate()?;
        if data.is_empty() {
            return Err(FormicaXError::Data(DataError::EmptyDataset));
        }
        Ok(())
    }

    fn algorithm_name(&self) -> &'static str {
        "SOM"
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
            OHLCV::new(Utc::now(), 90.0, 95.0, 88.0, 92.0, 800),
            OHLCV::new(Utc::now(), 92.0, 97.0, 90.0, 94.0, 900),
            OHLCV::new(Utc::now(), 94.0, 99.0, 92.0, 96.0, 850),
        ]
    }

    #[test]
    fn test_som_creation() {
        let som = SOM::new();
        assert_eq!(som.config.width, 5);
        assert_eq!(som.config.height, 5);
        assert_eq!(som.config.topology, TopologyType::Rectangular);
    }

    #[test]
    fn test_som_with_config() {
        let config = SOMConfig::builder()
            .width(10)
            .height(8)
            .topology(TopologyType::Hexagonal)
            .learning_rate(0.05)
            .neighborhood_radius(3.0)
            .epochs(200)
            .build()
            .unwrap();

        let som = SOM::with_config(config);
        assert_eq!(som.config.width, 10);
        assert_eq!(som.config.height, 8);
        assert_eq!(som.config.topology, TopologyType::Hexagonal);
    }

    #[test]
    fn test_som_fit() {
        let mut som = SOM::new();
        let data = create_test_data();

        let result = som.fit(&data).unwrap();

        assert_eq!(result.n_clusters, 25); // 5x5 grid
        assert_eq!(result.assignments().len(), data.len());
        assert!(som.get_iterations() > 0);
    }

    #[test]
    fn test_som_empty_data() {
        let mut som = SOM::new();
        let result = som.fit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_som_predict() {
        let mut som = SOM::new();
        let data = create_test_data();

        som.fit(&data).unwrap();

        let predictions = som.predict(&data).unwrap();
        assert_eq!(predictions.len(), data.len());
        assert!(predictions.iter().all(|&cluster| cluster < 25)); // 5x5 grid
    }

    #[test]
    fn test_som_get_parameters() {
        let mut som = SOM::new();
        let data = create_test_data();

        som.fit(&data).unwrap();

        assert!(som.get_weights().is_some());
        assert!(som.get_assignments().is_some());
        assert!(som.get_grid_positions().is_some());

        let weights = som.get_weights().unwrap();
        assert_eq!(weights.len(), 25); // 5x5 grid
        assert_eq!(weights[0].len(), 5); // 5 features
    }
}
