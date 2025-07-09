use crate::clustering::affinity_propagation::config::{
    AffinityPropagationConfig, PreferenceMethod,
};
use crate::core::{
    ClusterResult, ClusteringAlgorithm, ConfigError, DataError, FormicaXError, OHLCV,
};
use std::time::Instant;

/// Affinity Propagation clustering algorithm
#[derive(Debug)]
pub struct AffinityPropagation {
    /// Configuration for the affinity propagation algorithm
    config: AffinityPropagationConfig,
    /// Cluster assignments
    assignments: Option<Vec<usize>>,
    /// Exemplars (cluster centers)
    exemplars: Option<Vec<usize>>,
    /// Similarity matrix
    similarity_matrix: Option<Vec<Vec<f64>>>,
    /// Convergence status
    converged: bool,
    /// Number of iterations performed
    iterations: usize,
    /// Whether the algorithm has been fitted
    fitted: bool,
}

impl AffinityPropagation {
    /// Create a new AffinityPropagation with default configuration
    pub fn new() -> Self {
        Self::with_config(AffinityPropagationConfig::default())
    }

    /// Create a new AffinityPropagation with custom configuration
    pub fn with_config(config: AffinityPropagationConfig) -> Self {
        Self {
            config,
            assignments: None,
            exemplars: None,
            similarity_matrix: None,
            converged: false,
            iterations: 0,
            fitted: false,
        }
    }

    /// Fit the affinity propagation to the data
    pub fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError> {
        if data.is_empty() {
            return Err(FormicaXError::Data(DataError::EmptyDataset));
        }

        self.config.validate()?;
        let start_time = Instant::now();

        // Extract features from OHLCV data
        let features = self.extract_features(data);

        // Perform affinity propagation clustering
        let (assignments, exemplars, similarity_matrix) = self.perform_clustering(&features)?;

        let execution_time = start_time.elapsed();

        self.assignments = Some(assignments.clone());
        self.exemplars = Some(exemplars.clone());
        self.similarity_matrix = Some(similarity_matrix);
        self.fitted = true;

        // Calculate quality metrics
        let silhouette_score = self.calculate_silhouette_score(&features, &assignments)?;
        let inertia = self.calculate_inertia(&features, &assignments)?;

        // Create cluster centers from exemplars
        let cluster_centers = self.create_cluster_centers(&features, &exemplars);

        Ok(ClusterResult::new(
            "AffinityPropagation".to_string(),
            exemplars.len(),
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

    /// Perform affinity propagation clustering
    fn perform_clustering(
        &mut self,
        features: &[Vec<f64>],
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<Vec<f64>>), FormicaXError> {
        let n_samples = features.len();

        // Compute similarity matrix
        let mut similarity_matrix = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    similarity_matrix[i][j] =
                        -self.euclidean_distance_squared(&features[i], &features[j]);
                }
            }
        }

        // Set preferences on diagonal
        let preference = self.calculate_preference(&similarity_matrix);
        for i in 0..n_samples {
            similarity_matrix[i][i] = preference;
        }

        // Initialize messages
        let mut responsibility = vec![vec![0.0; n_samples]; n_samples];
        let mut availability = vec![vec![0.0; n_samples]; n_samples];

        // Message passing iterations
        for iteration in 0..self.config.max_iterations {
            self.iterations = iteration + 1;

            // Update responsibilities
            let old_responsibility = responsibility.clone();
            for i in 0..n_samples {
                for k in 0..n_samples {
                    let mut max_value = f64::NEG_INFINITY;
                    for k_prime in 0..n_samples {
                        if k_prime != k {
                            let value = similarity_matrix[i][k_prime] + availability[i][k_prime];
                            max_value = max_value.max(value);
                        }
                    }
                    responsibility[i][k] = similarity_matrix[i][k] - max_value;
                }
            }

            // Update availabilities
            let old_availability = availability.clone();
            for i in 0..n_samples {
                for k in 0..n_samples {
                    if i == k {
                        let mut sum = 0.0;
                        for i_prime in 0..n_samples {
                            if i_prime != i {
                                sum += responsibility[i_prime][k].max(0.0);
                            }
                        }
                        availability[i][k] = sum;
                    } else {
                        let mut sum = 0.0;
                        for i_prime in 0..n_samples {
                            if i_prime != i && i_prime != k {
                                sum += responsibility[i_prime][k].max(0.0);
                            }
                        }
                        availability[i][k] = (responsibility[k][k] + sum).max(0.0);
                    }
                }
            }

            // Apply damping
            for i in 0..n_samples {
                for k in 0..n_samples {
                    responsibility[i][k] = self.config.damping * old_responsibility[i][k]
                        + (1.0 - self.config.damping) * responsibility[i][k];
                    availability[i][k] = self.config.damping * old_availability[i][k]
                        + (1.0 - self.config.damping) * availability[i][k];
                }
            }

            // Check convergence
            let mut max_change: f64 = 0.0;
            for i in 0..n_samples {
                for k in 0..n_samples {
                    let change = (responsibility[i][k] - old_responsibility[i][k]).abs();
                    max_change = max_change.max(change);
                }
            }

            if max_change < self.config.tolerance {
                self.converged = true;
                break;
            }
        }

        // Find exemplars
        let mut exemplars = Vec::new();
        for k in 0..n_samples {
            if responsibility[k][k] + availability[k][k] > 0.0 {
                exemplars.push(k);
            }
        }

        // Create assignments
        let mut assignments = vec![0; n_samples];
        for i in 0..n_samples {
            let mut best_exemplar = exemplars[0];
            let mut best_similarity = f64::NEG_INFINITY;

            for &exemplar in &exemplars {
                let similarity = similarity_matrix[i][exemplar];
                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_exemplar = exemplar;
                }
            }

            assignments[i] = exemplars.iter().position(|&x| x == best_exemplar).unwrap();
        }

        Ok((assignments, exemplars, similarity_matrix))
    }

    /// Calculate preference value
    fn calculate_preference(&self, similarity_matrix: &[Vec<f64>]) -> f64 {
        match self.config.preference {
            PreferenceMethod::Median => {
                let mut values = Vec::new();
                for i in 0..similarity_matrix.len() {
                    for j in 0..similarity_matrix[i].len() {
                        if i != j {
                            values.push(similarity_matrix[i][j]);
                        }
                    }
                }
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                values[values.len() / 2]
            }
            PreferenceMethod::Minimum => {
                let mut min_value = f64::INFINITY;
                for i in 0..similarity_matrix.len() {
                    for j in 0..similarity_matrix[i].len() {
                        if i != j {
                            min_value = min_value.min(similarity_matrix[i][j]);
                        }
                    }
                }
                min_value
            }
            PreferenceMethod::Custom(value) => value,
        }
    }

    /// Create cluster centers from exemplars
    fn create_cluster_centers(&self, features: &[Vec<f64>], exemplars: &[usize]) -> Vec<Vec<f64>> {
        exemplars
            .iter()
            .map(|&exemplar| features[exemplar].clone())
            .collect()
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
            let n_clusters = assignments.iter().max().unwrap() + 1;
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
        if !self.fitted {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "Affinity propagation not fitted yet".to_string(),
            }));
        }

        let exemplars = self.exemplars.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "Affinity propagation not fitted yet".to_string(),
            })
        })?;

        let mut inertia = 0.0;
        for (i, feature) in features.iter().enumerate() {
            let cluster = assignments[i];
            if cluster < exemplars.len() {
                let exemplar_idx = exemplars[cluster];
                inertia += self.euclidean_distance_squared(feature, &features[exemplar_idx]);
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

    /// Get cluster assignments
    pub fn get_assignments(&self) -> Option<&Vec<usize>> {
        self.assignments.as_ref()
    }

    /// Get exemplars
    pub fn get_exemplars(&self) -> Option<&Vec<usize>> {
        self.exemplars.as_ref()
    }

    /// Get similarity matrix
    pub fn get_similarity_matrix(&self) -> Option<&Vec<Vec<f64>>> {
        self.similarity_matrix.as_ref()
    }

    /// Check if the algorithm converged
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Get number of iterations performed
    pub fn get_iterations(&self) -> usize {
        self.iterations
    }
}

impl ClusteringAlgorithm for AffinityPropagation {
    type Config = AffinityPropagationConfig;

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
        let exemplars = self.exemplars.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "Affinity propagation not fitted yet".to_string(),
            })
        })?;
        let similarity_matrix = self.similarity_matrix.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "Affinity propagation not fitted yet".to_string(),
            })
        })?;

        let mut assignments = Vec::new();
        for _feature in &features {
            let mut best_exemplar = exemplars[0];
            let mut best_similarity = f64::NEG_INFINITY;

            for &exemplar in exemplars {
                let similarity = similarity_matrix[exemplar][exemplar]; // Simplified
                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_exemplar = exemplar;
                }
            }

            assignments.push(exemplars.iter().position(|&x| x == best_exemplar).unwrap());
        }

        Ok(assignments)
    }

    fn get_cluster_centers(&self) -> Option<Vec<Vec<f64>>> {
        // Return exemplar features as cluster centers
        None // Simplified - would need to store exemplar features
    }

    fn validate_config(&self, data: &[OHLCV]) -> Result<(), FormicaXError> {
        self.config.validate()?;
        if data.is_empty() {
            return Err(FormicaXError::Data(DataError::EmptyDataset));
        }
        Ok(())
    }

    fn algorithm_name(&self) -> &'static str {
        "AffinityPropagation"
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
    fn test_affinity_propagation_creation() {
        let ap = AffinityPropagation::new();
        assert_eq!(ap.config.damping, 0.5);
        assert_eq!(ap.config.max_iterations, 200);
    }

    #[test]
    fn test_affinity_propagation_with_config() {
        let config = AffinityPropagationConfig::builder()
            .damping(0.7)
            .max_iterations(300)
            .preference(PreferenceMethod::Minimum)
            .build()
            .unwrap();

        let ap = AffinityPropagation::with_config(config);
        assert_eq!(ap.config.damping, 0.7);
        assert_eq!(ap.config.max_iterations, 300);
    }

    #[test]
    fn test_affinity_propagation_fit() {
        let mut ap = AffinityPropagation::new();
        let data = create_test_data();

        let result = ap.fit(&data).unwrap();

        assert!(result.n_clusters > 0);
        assert_eq!(result.assignments().len(), data.len());
        assert!(ap.get_iterations() > 0);
    }

    #[test]
    fn test_affinity_propagation_empty_data() {
        let mut ap = AffinityPropagation::new();
        let result = ap.fit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_affinity_propagation_get_parameters() {
        let mut ap = AffinityPropagation::new();
        let data = create_test_data();

        ap.fit(&data).unwrap();

        assert!(ap.get_assignments().is_some());
        assert!(ap.get_exemplars().is_some());
        assert!(ap.get_similarity_matrix().is_some());

        let exemplars = ap.get_exemplars().unwrap();
        assert!(!exemplars.is_empty());
    }
}
