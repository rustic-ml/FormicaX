//! Core traits and interfaces for FormicaX
//!
//! This module defines the unified interfaces that all clustering algorithms
//! must implement, providing a consistent API across different algorithms.

use crate::core::{ClusterResult, FormicaXError, OHLCV};

/// Unified trait for all clustering algorithms
///
/// This trait provides a common interface for all clustering algorithms
/// in FormicaX, ensuring consistent behavior and API across different
/// implementations.
pub trait ClusteringAlgorithm {
    /// Configuration type for this algorithm
    type Config;

    /// Create a new instance with default configuration
    fn new() -> Self
    where
        Self: Default;

    /// Create a new instance with custom configuration
    fn with_config(config: Self::Config) -> Self;

    /// Fit the clustering algorithm to the data
    ///
    /// This method performs the clustering and returns comprehensive results
    /// including cluster assignments, quality metrics, and metadata.
    fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError>;

    /// Predict cluster assignments for new data
    ///
    /// This method assigns cluster labels to new data points based on
    /// the fitted model.
    fn predict(&self, data: &[OHLCV]) -> Result<Vec<usize>, FormicaXError>;

    /// Get cluster centers/centroids (if applicable)
    ///
    /// Returns the cluster centers as vectors of features.
    /// Returns None if the algorithm doesn't use centers.
    fn get_cluster_centers(&self) -> Option<Vec<Vec<f64>>>;

    /// Validate the algorithm configuration
    ///
    /// Checks if the current configuration is valid for the given data.
    fn validate_config(&self, data: &[OHLCV]) -> Result<(), FormicaXError>;

    /// Get algorithm name
    fn algorithm_name(&self) -> &'static str;

    /// Check if the algorithm supports incremental updates
    fn supports_incremental(&self) -> bool {
        false
    }

    /// Update the model with new data (if supported)
    ///
    /// This method allows incremental updates to the clustering model.
    /// Returns an error if incremental updates are not supported.
    fn update(&mut self, _data: &[OHLCV]) -> Result<ClusterResult, FormicaXError> {
        Err(FormicaXError::Clustering(
            crate::core::ClusteringError::AlgorithmError {
                message: "Incremental updates not supported for this algorithm".to_string(),
            },
        ))
    }
}

/// Trait for clustering algorithms that support parallel processing
pub trait ParallelClusteringAlgorithm: ClusteringAlgorithm {
    /// Enable or disable parallel processing
    fn set_parallel(&mut self, parallel: bool);

    /// Check if parallel processing is enabled
    fn is_parallel(&self) -> bool;

    /// Set the number of threads for parallel processing
    fn set_num_threads(&mut self, num_threads: usize);

    /// Get the number of threads used for parallel processing
    fn num_threads(&self) -> usize;
}

/// Trait for clustering algorithms that support SIMD optimization
pub trait SimdClusteringAlgorithm: ClusteringAlgorithm {
    /// Enable or disable SIMD optimization
    fn set_simd(&mut self, simd: bool);

    /// Check if SIMD optimization is enabled
    fn is_simd(&self) -> bool;

    /// Get the SIMD instruction set being used
    fn simd_instruction_set(&self) -> Option<&'static str>;
}

/// Trait for clustering algorithms that support streaming data
pub trait StreamingClusteringAlgorithm: ClusteringAlgorithm {
    /// Process data in streaming fashion
    ///
    /// This method processes data incrementally without loading
    /// the entire dataset into memory.
    fn fit_streaming<I>(&mut self, data_stream: I) -> Result<ClusterResult, FormicaXError>
    where
        I: Iterator<Item = OHLCV>;

    /// Get the current streaming state
    fn streaming_state(&self) -> Option<StreamingState>;
}

/// State information for streaming clustering algorithms
#[derive(Debug, Clone)]
pub struct StreamingState {
    /// Number of data points processed so far
    pub n_processed: usize,
    /// Current number of clusters
    pub n_clusters: usize,
    /// Whether the model is stable
    pub is_stable: bool,
    /// Last update timestamp
    pub last_update: std::time::Instant,
}

/// Trait for clustering algorithms that support ensemble methods
pub trait EnsembleClusteringAlgorithm: ClusteringAlgorithm {
    /// Create an ensemble of multiple clustering runs
    ///
    /// This method runs the clustering algorithm multiple times
    /// and combines the results for improved robustness.
    fn fit_ensemble(
        &mut self,
        data: &[OHLCV],
        n_runs: usize,
    ) -> Result<Vec<ClusterResult>, FormicaXError>;

    /// Combine multiple clustering results into a consensus
    fn consensus(
        results: &[ClusterResult],
        method: ConsensusMethod,
    ) -> Result<ClusterResult, FormicaXError>;
}

/// Methods for combining ensemble clustering results
#[derive(Debug, Clone, Copy)]
pub enum ConsensusMethod {
    /// Majority voting for cluster assignments
    MajorityVote,
    /// Weighted voting based on silhouette scores
    WeightedVote,
    /// Hierarchical clustering of cluster assignments
    Hierarchical,
    /// Graph-based consensus clustering
    GraphBased,
}

/// Trait for clustering algorithms that support validation metrics
pub trait ValidatedClusteringAlgorithm: ClusteringAlgorithm {
    /// Calculate validation metrics for the clustering result
    fn calculate_validation_metrics(
        &self,
        data: &[OHLCV],
        result: &ClusterResult,
    ) -> Result<ValidationMetrics, FormicaXError>;

    /// Validate clustering quality
    fn validate_quality(
        &self,
        result: &ClusterResult,
        threshold: f64,
    ) -> Result<bool, FormicaXError>;
}

/// Comprehensive validation metrics for clustering results
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// Silhouette score (-1 to 1, higher is better)
    pub silhouette_score: f64,
    /// Calinski-Harabasz index (higher is better)
    pub calinski_harabasz: f64,
    /// Davies-Bouldin index (lower is better)
    pub davies_bouldin: f64,
    /// Dunn index (higher is better)
    pub dunn_index: f64,
    /// Adjusted Rand index (for labeled data)
    pub adjusted_rand: Option<f64>,
    /// Normalized mutual information (for labeled data)
    pub normalized_mutual_info: Option<f64>,
    /// Homogeneity score (for labeled data)
    pub homogeneity: Option<f64>,
    /// Completeness score (for labeled data)
    pub completeness: Option<f64>,
    /// V-measure (for labeled data)
    pub v_measure: Option<f64>,
}

impl ValidationMetrics {
    /// Create a new validation metrics instance
    pub fn new(silhouette_score: f64) -> Self {
        Self {
            silhouette_score,
            calinski_harabasz: 0.0,
            davies_bouldin: 0.0,
            dunn_index: 0.0,
            adjusted_rand: None,
            normalized_mutual_info: None,
            homogeneity: None,
            completeness: None,
            v_measure: None,
        }
    }

    /// Calculate overall quality score
    pub fn overall_score(&self) -> f64 {
        // Normalize metrics to 0-1 range and combine
        let silhouette_norm = (self.silhouette_score + 1.0) / 2.0;
        let ch_norm = self.calinski_harabasz / (self.calinski_harabasz + 1.0);
        let db_norm = 1.0 / (1.0 + self.davies_bouldin);
        let dunn_norm = self.dunn_index / (self.dunn_index + 1.0);

        (silhouette_norm + ch_norm + db_norm + dunn_norm) / 4.0
    }

    /// Check if clustering quality is acceptable
    pub fn is_acceptable(&self, threshold: f64) -> bool {
        self.overall_score() >= threshold
    }
}

/// Trait for clustering algorithms that support parameter optimization
pub trait OptimizableClusteringAlgorithm: ClusteringAlgorithm {
    /// Parameter space for optimization
    type ParameterSpace;

    /// Optimize algorithm parameters using cross-validation
    fn optimize_parameters(
        &self,
        data: &[OHLCV],
        parameter_space: Self::ParameterSpace,
        cv_folds: usize,
    ) -> Result<Self::Config, FormicaXError>;

    /// Grid search over parameter space
    fn grid_search(
        &self,
        data: &[OHLCV],
        parameter_space: Self::ParameterSpace,
    ) -> Result<Vec<(Self::Config, f64)>, FormicaXError>;
}

/// Trait for clustering algorithms that support model persistence
pub trait PersistableClusteringAlgorithm: ClusteringAlgorithm {
    /// Save the model to a file
    fn save_model(&self, path: &str) -> Result<(), FormicaXError>;

    /// Load the model from a file
    fn load_model(path: &str) -> Result<Self, FormicaXError>
    where
        Self: Sized;

    /// Serialize the model to bytes
    fn to_bytes(&self) -> Result<Vec<u8>, FormicaXError>;

    /// Deserialize the model from bytes
    fn from_bytes(bytes: &[u8]) -> Result<Self, FormicaXError>
    where
        Self: Sized;
}

/// Trait for clustering algorithms that support real-time processing
pub trait RealTimeClusteringAlgorithm: ClusteringAlgorithm {
    /// Process data in real-time with minimal latency
    fn process_realtime(&mut self, data_point: &OHLCV) -> Result<Option<usize>, FormicaXError>;

    /// Get real-time performance metrics
    fn realtime_metrics(&self) -> RealTimeMetrics;

    /// Set real-time processing parameters
    fn set_realtime_params(&mut self, params: RealTimeParams);
}

/// Real-time processing parameters
#[derive(Debug, Clone)]
pub struct RealTimeParams {
    /// Maximum processing time per data point
    pub max_processing_time: std::time::Duration,
    /// Buffer size for batch processing
    pub buffer_size: usize,
    /// Whether to enable adaptive processing
    pub adaptive: bool,
}

/// Real-time performance metrics
#[derive(Debug, Clone)]
pub struct RealTimeMetrics {
    /// Average processing time per data point
    pub avg_processing_time: std::time::Duration,
    /// Maximum processing time observed
    pub max_processing_time: std::time::Duration,
    /// Number of data points processed
    pub n_processed: usize,
    /// Current buffer utilization
    pub buffer_utilization: f64,
}

impl Default for RealTimeParams {
    fn default() -> Self {
        Self {
            max_processing_time: std::time::Duration::from_millis(1),
            buffer_size: 1000,
            adaptive: true,
        }
    }
}

impl Default for RealTimeMetrics {
    fn default() -> Self {
        Self {
            avg_processing_time: std::time::Duration::ZERO,
            max_processing_time: std::time::Duration::ZERO,
            n_processed: 0,
            buffer_utilization: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_validation_metrics_creation() {
        let metrics = ValidationMetrics::new(0.75);
        assert_eq!(metrics.silhouette_score, 0.75);
        assert_eq!(metrics.calinski_harabasz, 0.0);
        assert_eq!(metrics.davies_bouldin, 0.0);
    }

    #[test]
    fn test_validation_metrics_overall_score() {
        let mut metrics = ValidationMetrics::new(0.5);
        metrics.calinski_harabasz = 100.0;
        metrics.davies_bouldin = 0.5;
        metrics.dunn_index = 2.0;

        let score = metrics.overall_score();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_validation_metrics_acceptability() {
        let mut metrics = ValidationMetrics::new(0.8);
        // Set other metrics to reasonable values to ensure overall score is good
        metrics.calinski_harabasz = 100.0;
        metrics.davies_bouldin = 0.5;
        metrics.dunn_index = 2.0;

        assert!(metrics.is_acceptable(0.7));
        assert!(!metrics.is_acceptable(0.9));
    }

    #[test]
    fn test_realtime_params_default() {
        let params = RealTimeParams::default();
        assert_eq!(
            params.max_processing_time,
            std::time::Duration::from_millis(1)
        );
        assert_eq!(params.buffer_size, 1000);
        assert!(params.adaptive);
    }

    #[test]
    fn test_realtime_metrics_default() {
        let metrics = RealTimeMetrics::default();
        assert_eq!(metrics.avg_processing_time, std::time::Duration::ZERO);
        assert_eq!(metrics.max_processing_time, std::time::Duration::ZERO);
        assert_eq!(metrics.n_processed, 0);
        assert_eq!(metrics.buffer_utilization, 0.0);
    }

    proptest! {
        #[test]
        fn test_validation_metrics_properties(
            silhouette in -1.0..1.0f64,
            calinski_harabasz in 0.0..1000.0f64,
            davies_bouldin in 0.0..10.0f64,
            dunn_index in 0.0..10.0f64
        ) {
            let mut metrics = ValidationMetrics::new(silhouette);
            metrics.calinski_harabasz = calinski_harabasz;
            metrics.davies_bouldin = davies_bouldin;
            metrics.dunn_index = dunn_index;

            let score = metrics.overall_score();
            assert!((0.0..=1.0).contains(&score));

            // Test acceptability with various thresholds
            assert!(metrics.is_acceptable(0.0));
            assert!(!metrics.is_acceptable(1.1));
        }
    }
}
