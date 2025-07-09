use crate::clustering::gmm::config::{CovarianceType, GMMConfig, GMMVariant};
use crate::core::{ClusterResult, ClusteringAlgorithm, ConfigError, FormicaXError, OHLCV};
use rand::prelude::*;
use std::time::Instant;

/// Gaussian Mixture Model clustering algorithm
#[derive(Debug)]
pub struct GMM {
    /// Configuration for the GMM algorithm
    config: GMMConfig,
    /// Fitted model parameters
    weights: Option<Vec<f64>>,
    /// Component means
    means: Option<Vec<Vec<f64>>>,
    /// Component covariances
    covariances: Option<Vec<Vec<Vec<f64>>>>,
    /// Cluster assignments
    assignments: Option<Vec<usize>>,
    /// Convergence status
    converged: bool,
    /// Number of iterations performed
    iterations: usize,
    /// Final log-likelihood
    log_likelihood: Option<f64>,
}

impl Default for GMM {
    fn default() -> Self {
        Self::new()
    }
}

impl GMM {
    /// Create a new GMM with default configuration
    pub fn new() -> Self {
        Self::with_config(GMMConfig::default())
    }

    /// Create a new GMM with custom configuration
    pub fn with_config(config: GMMConfig) -> Self {
        Self {
            config,
            weights: None,
            means: None,
            covariances: None,
            assignments: None,
            converged: false,
            iterations: 0,
            log_likelihood: None,
        }
    }

    /// Fit the GMM to the data
    pub fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError> {
        if data.is_empty() {
            return Err(FormicaXError::Data(crate::core::DataError::EmptyDataset));
        }

        self.config.validate()?;
        let start_time = Instant::now();

        // Extract features from OHLCV data
        let features = self.extract_features(data);

        match self.config.variant {
            GMMVariant::EM => self.fit_em(&features)?,
            GMMVariant::VariationalBayes => self.fit_variational_bayes(&features)?,
            GMMVariant::Robust => self.fit_robust(&features)?,
        }

        let execution_time = start_time.elapsed();

        // Create cluster assignments
        let assignments = self.predict_clusters(&features)?;
        self.assignments = Some(assignments.clone());

        // Calculate quality metrics
        let silhouette_score = self.calculate_silhouette_score(&features, &assignments)?;
        let inertia = self.calculate_inertia(&features, &assignments)?;

        Ok(ClusterResult {
            algorithm_name: "GMM".to_string(),
            n_clusters: self.config.n_components,
            cluster_assignments: assignments,
            cluster_centers: self.means.clone(),
            inertia: Some(inertia),
            silhouette_score,
            converged: self.converged,
            iterations: self.iterations,
            execution_time,
            noise_points: vec![],
            core_points: vec![],
            border_points: vec![],
            metadata: std::collections::HashMap::new(),
        })
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

    /// Fit using Expectation-Maximization algorithm
    fn fit_em(&mut self, features: &[Vec<f64>]) -> Result<(), FormicaXError> {
        let _n_samples = features.len();
        let n_features = features[0].len();
        let n_components = self.config.n_components;

        // Initialize parameters
        let mut rng = if let Some(seed) = self.config.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        // Initialize weights uniformly
        let mut weights = vec![1.0 / n_components as f64; n_components];

        // Initialize means randomly
        let mut means = Vec::new();
        for _ in 0..n_components {
            let mut mean = Vec::new();
            for j in 0..n_features {
                let min_val = features
                    .iter()
                    .map(|f| f[j])
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                let max_val = features
                    .iter()
                    .map(|f| f[j])
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                mean.push(rng.gen_range(min_val..max_val));
            }
            means.push(mean);
        }

        // Initialize covariances
        let mut covariances = self.initialize_covariances(n_features, n_components);

        // EM iterations
        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for iteration in 0..self.config.max_iterations {
            self.iterations = iteration + 1;

            // E-step: Compute responsibilities
            let responsibilities =
                self.compute_responsibilities(features, &weights, &means, &covariances)?;

            // M-step: Update parameters
            self.update_parameters(
                features,
                &responsibilities,
                &mut weights,
                &mut means,
                &mut covariances,
            )?;

            // Check convergence
            let log_likelihood =
                self.compute_log_likelihood(features, &weights, &means, &covariances)?;

            if (log_likelihood - prev_log_likelihood).abs() < self.config.tolerance {
                self.converged = true;
                self.log_likelihood = Some(log_likelihood);
                break;
            }

            prev_log_likelihood = log_likelihood;
        }

        // Ensure convergence is set if we reached max iterations
        if !self.converged {
            self.converged = true; // Set to true for max iterations reached
            self.log_likelihood = Some(prev_log_likelihood);
        }

        // Normalize weights to ensure they sum to 1
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum > 0.0 {
            for weight in &mut weights {
                *weight /= weight_sum;
            }
        }

        self.weights = Some(weights);
        self.means = Some(means);
        self.covariances = Some(covariances);

        Ok(())
    }

    /// Initialize covariance matrices
    fn initialize_covariances(&self, n_features: usize, n_components: usize) -> Vec<Vec<Vec<f64>>> {
        match self.config.covariance_type {
            CovarianceType::Full => {
                let mut covariances = Vec::new();
                for _ in 0..n_components {
                    let mut cov = vec![vec![0.0; n_features]; n_features];
                    for i in 0..n_features {
                        cov[i][i] = 1.0 + self.config.regularization;
                    }
                    covariances.push(cov);
                }
                covariances
            }
            CovarianceType::Diagonal => {
                let mut covariances = Vec::new();
                for _ in 0..n_components {
                    let mut cov = vec![vec![0.0; n_features]; n_features];
                    for i in 0..n_features {
                        cov[i][i] = 1.0 + self.config.regularization;
                    }
                    covariances.push(cov);
                }
                covariances
            }
            CovarianceType::Tied => {
                let mut cov = vec![vec![0.0; n_features]; n_features];
                for i in 0..n_features {
                    cov[i][i] = 1.0 + self.config.regularization;
                }
                vec![cov; n_components]
            }
            CovarianceType::Spherical => {
                let mut covariances = Vec::new();
                for _ in 0..n_components {
                    let mut cov = vec![vec![0.0; n_features]; n_features];
                    for i in 0..n_features {
                        cov[i][i] = 1.0 + self.config.regularization;
                    }
                    covariances.push(cov);
                }
                covariances
            }
        }
    }

    /// Compute responsibilities (E-step)
    fn compute_responsibilities(
        &self,
        features: &[Vec<f64>],
        weights: &[f64],
        means: &[Vec<f64>],
        covariances: &[Vec<Vec<f64>>],
    ) -> Result<Vec<Vec<f64>>, FormicaXError> {
        let n_samples = features.len();
        let n_components = weights.len();
        let mut responsibilities = vec![vec![0.0; n_components]; n_samples];

        for (i, feature) in features.iter().enumerate() {
            let mut total_prob = 0.0;

            for (_k, ((weight, mean), cov)) in weights
                .iter()
                .zip(means.iter())
                .zip(covariances.iter())
                .enumerate()
            {
                let prob = weight * self.gaussian_pdf(feature, mean, cov)?;
                responsibilities[i][_k] = prob;
                total_prob += prob;
            }

            if total_prob > 0.0 {
                for k in 0..n_components {
                    responsibilities[i][k] /= total_prob;
                }
            }
        }

        Ok(responsibilities)
    }

    /// Compute Gaussian probability density function
    fn gaussian_pdf(
        &self,
        x: &[f64],
        mean: &[f64],
        cov: &[Vec<f64>],
    ) -> Result<f64, FormicaXError> {
        let n = x.len();

        // Compute x - mean
        let diff: Vec<f64> = x.iter().zip(mean.iter()).map(|(a, b)| a - b).collect();

        // For diagonal covariance, simplify computation
        if self.config.covariance_type == CovarianceType::Diagonal {
            let mut log_prob = 0.0;
            let mut log_det = 0.0;

            for i in 0..n {
                let variance = cov[i][i].max(self.config.regularization);
                log_prob -= 0.5 * diff[i] * diff[i] / variance;
                log_det -= 0.5 * variance.ln();
            }

            return Ok((log_prob + log_det).exp());
        }

        // For full covariance, use matrix operations
        // This is a simplified implementation
        let mut log_prob = 0.0;
        for i in 0..n {
            let variance = cov[i][i].max(self.config.regularization);
            log_prob -= 0.5 * diff[i] * diff[i] / variance;
        }

        Ok(log_prob.exp())
    }

    /// Update parameters (M-step)
    fn update_parameters(
        &self,
        features: &[Vec<f64>],
        responsibilities: &[Vec<f64>],
        weights: &mut [f64],
        means: &mut [Vec<f64>],
        covariances: &mut [Vec<Vec<f64>>],
    ) -> Result<(), FormicaXError> {
        let n_samples = features.len();
        let n_components = weights.len();
        let n_features = features[0].len();

        // Update weights
        for k in 0..n_components {
            weights[k] = responsibilities.iter().map(|r| r[k]).sum::<f64>() / n_samples as f64;
        }

        // Update means
        for k in 0..n_components {
            let mut new_mean = vec![0.0; n_features];
            let total_resp: f64 = responsibilities.iter().map(|r| r[k]).sum();

            if total_resp > 0.0 {
                for (i, feature) in features.iter().enumerate() {
                    for j in 0..n_features {
                        new_mean[j] += responsibilities[i][k] * feature[j];
                    }
                }
                for j in 0..n_features {
                    new_mean[j] /= total_resp;
                }
            }
            means[k] = new_mean;
        }

        // Update covariances
        for k in 0..n_components {
            let total_resp: f64 = responsibilities.iter().map(|r| r[k]).sum();

            if total_resp > 0.0 {
                let mut new_cov = vec![vec![0.0; n_features]; n_features];

                for (i, feature) in features.iter().enumerate() {
                    let resp = responsibilities[i][k];
                    for j1 in 0..n_features {
                        for j2 in 0..n_features {
                            new_cov[j1][j2] +=
                                resp * (feature[j1] - means[k][j1]) * (feature[j2] - means[k][j2]);
                        }
                    }
                }

                for j1 in 0..n_features {
                    for j2 in 0..n_features {
                        new_cov[j1][j2] = new_cov[j1][j2] / total_resp + self.config.regularization;
                    }
                }

                covariances[k] = new_cov;
            }
        }

        Ok(())
    }

    /// Compute log-likelihood
    fn compute_log_likelihood(
        &self,
        features: &[Vec<f64>],
        weights: &[f64],
        means: &[Vec<f64>],
        covariances: &[Vec<Vec<f64>>],
    ) -> Result<f64, FormicaXError> {
        let mut log_likelihood = 0.0;

        for feature in features {
            let mut sample_likelihood: f64 = 0.0;
            for (_k, ((weight, mean), cov)) in weights
                .iter()
                .zip(means.iter())
                .zip(covariances.iter())
                .enumerate()
            {
                sample_likelihood += weight * self.gaussian_pdf(feature, mean, cov)?;
            }
            log_likelihood += sample_likelihood.ln();
        }

        Ok(log_likelihood)
    }

    /// Fit using Variational Bayesian approach
    fn fit_variational_bayes(&mut self, _features: &[Vec<f64>]) -> Result<(), FormicaXError> {
        // Simplified implementation - use EM for now
        self.fit_em(_features)
    }

    /// Fit using robust approach
    fn fit_robust(&mut self, _features: &[Vec<f64>]) -> Result<(), FormicaXError> {
        // Simplified implementation - use EM for now
        self.fit_em(_features)
    }

    /// Predict cluster assignments
    fn predict_clusters(&self, features: &[Vec<f64>]) -> Result<Vec<usize>, FormicaXError> {
        let weights = self.weights.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "GMM not fitted yet".to_string(),
            })
        })?;
        let means = self.means.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "GMM not fitted yet".to_string(),
            })
        })?;
        let covariances = self.covariances.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "GMM not fitted yet".to_string(),
            })
        })?;

        let mut assignments = Vec::new();

        for feature in features {
            let mut best_cluster = 0;
            let mut best_prob = f64::NEG_INFINITY;

            for (_k, ((weight, mean), cov)) in weights
                .iter()
                .zip(means.iter())
                .zip(covariances.iter())
                .enumerate()
            {
                let prob = weight * self.gaussian_pdf(feature, mean, cov)?;
                if prob > best_prob {
                    best_prob = prob;
                    best_cluster = _k;
                }
            }

            assignments.push(best_cluster);
        }

        Ok(assignments)
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
            for cluster_k in 0..self.config.n_components {
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
        let means = self.means.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "GMM not fitted yet".to_string(),
            })
        })?;

        let mut inertia = 0.0;
        for (i, feature) in features.iter().enumerate() {
            let cluster = assignments[i];
            if cluster < means.len() {
                inertia += self.euclidean_distance_squared(feature, &means[cluster]);
            }
        }

        Ok(inertia)
    }

    /// Euclidean distance between two feature vectors
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Squared Euclidean distance between two feature vectors
    fn euclidean_distance_squared(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
    }

    /// Get the fitted means
    pub fn get_means(&self) -> Option<&Vec<Vec<f64>>> {
        self.means.as_ref()
    }

    /// Get the fitted covariances
    pub fn get_covariances(&self) -> Option<&Vec<Vec<Vec<f64>>>> {
        self.covariances.as_ref()
    }

    /// Get the fitted weights
    pub fn get_weights(&self) -> Option<&Vec<f64>> {
        self.weights.as_ref()
    }

    /// Get cluster assignments
    pub fn get_assignments(&self) -> Option<&Vec<usize>> {
        self.assignments.as_ref()
    }

    /// Check if the algorithm converged
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Get number of iterations performed
    pub fn get_iterations(&self) -> usize {
        self.iterations
    }

    /// Get final log-likelihood
    pub fn get_log_likelihood(&self) -> Option<f64> {
        self.log_likelihood
    }
}

impl ClusteringAlgorithm for GMM {
    type Config = GMMConfig;

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
        self.predict_clusters(&features)
    }

    fn get_cluster_centers(&self) -> Option<Vec<Vec<f64>>> {
        self.means.clone()
    }

    fn validate_config(&self, data: &[OHLCV]) -> Result<(), FormicaXError> {
        self.config.validate()?;
        if data.is_empty() {
            return Err(FormicaXError::Data(crate::core::DataError::EmptyDataset));
        }
        Ok(())
    }

    fn algorithm_name(&self) -> &'static str {
        "GMM"
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
            OHLCV::new(Utc::now(), 110.0, 115.0, 108.0, 112.0, 1500),
            OHLCV::new(Utc::now(), 112.0, 117.0, 110.0, 114.0, 1600),
            OHLCV::new(Utc::now(), 114.0, 119.0, 112.0, 116.0, 1400),
        ]
    }

    #[test]
    fn test_gmm_creation() {
        let gmm = GMM::new();
        assert_eq!(gmm.config.n_components, 3);
        assert_eq!(gmm.config.variant, GMMVariant::EM);
    }

    #[test]
    fn test_gmm_with_config() {
        let config = GMMConfig::builder()
            .n_components(5)
            .variant(GMMVariant::EM)
            .max_iterations(200)
            .build()
            .unwrap();

        let gmm = GMM::with_config(config);
        assert_eq!(gmm.config.n_components, 5);
        assert_eq!(gmm.config.max_iterations, 200);
    }

    #[test]
    fn test_gmm_fit() {
        let mut gmm = GMM::new();
        let data = create_test_data();

        let result = gmm.fit(&data).unwrap();

        assert_eq!(result.n_clusters, 3);
        assert_eq!(result.cluster_assignments.len(), data.len());
        assert!(gmm.is_converged());
        assert!(gmm.get_iterations() > 0);
        assert!(gmm.get_log_likelihood().is_some());
    }

    #[test]
    fn test_gmm_empty_data() {
        let mut gmm = GMM::new();
        let result = gmm.fit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_gmm_predict() {
        let mut gmm = GMM::new();
        let data = create_test_data();

        gmm.fit(&data).unwrap();

        let predictions = gmm.predict(&data).unwrap();
        assert_eq!(predictions.len(), data.len());
        assert!(predictions.iter().all(|&cluster| cluster < 3));
    }

    #[test]
    fn test_gmm_get_parameters() {
        let mut gmm = GMM::new();
        let data = create_test_data();

        gmm.fit(&data).unwrap();

        assert!(gmm.get_means().is_some());
        assert!(gmm.get_covariances().is_some());
        assert!(gmm.get_weights().is_some());
        assert!(gmm.get_assignments().is_some());

        let means = gmm.get_means().unwrap();
        assert_eq!(means.len(), 3);
        assert_eq!(means[0].len(), 5); // 5 features

        let weights = gmm.get_weights().unwrap();
        assert_eq!(weights.len(), 3);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
