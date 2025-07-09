//! Core data models for FormicaX
//!
//! This module contains the fundamental data structures used throughout
//! the FormicaX library, including OHLCV data and clustering results.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// OHLCV (Open, High, Low, Close, Volume) data structure
///
/// Represents a single data point in financial time series data.
/// All price fields are stored as f64 for maximum precision,
/// and volume is stored as u64 for large volume values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OHLCV {
    /// Timestamp in UTC
    pub timestamp: DateTime<Utc>,
    /// Opening price
    pub open: f64,
    /// Highest price during the period
    pub high: f64,
    /// Lowest price during the period
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: u64,
}

impl OHLCV {
    /// Create a new OHLCV data point
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: u64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Validate OHLCV data for logical consistency
    pub fn validate(&self) -> Result<(), crate::core::DataError> {
        // Check for positive prices
        if self.open <= 0.0 || self.high <= 0.0 || self.low <= 0.0 || self.close <= 0.0 {
            return Err(crate::core::DataError::InvalidOHLCV {
                message: "All prices must be positive".to_string(),
            });
        }

        // Check logical consistency: high >= low
        if self.high < self.low {
            return Err(crate::core::DataError::InvalidOHLCV {
                message: "High price cannot be less than low price".to_string(),
            });
        }

        // Check logical consistency: high >= open and high >= close
        if self.high < self.open || self.high < self.close {
            return Err(crate::core::DataError::InvalidOHLCV {
                message: "High price must be >= open and close prices".to_string(),
            });
        }

        // Check logical consistency: low <= open and low <= close
        if self.low > self.open || self.low > self.close {
            return Err(crate::core::DataError::InvalidOHLCV {
                message: "Low price must be <= open and close prices".to_string(),
            });
        }

        Ok(())
    }

    /// Calculate the price range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the body size (|close - open|)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate the upper shadow (high - max(open, close))
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    /// Calculate the lower shadow (min(open, close) - low)
    pub fn lower_shadow(&self) -> f64 {
        self.open.min(self.close) - self.low
    }

    /// Calculate the typical price (high + low + close) / 3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the weighted close price (high + low + close + close) / 4
    pub fn weighted_close(&self) -> f64 {
        (self.high + self.low + self.close + self.close) / 4.0
    }

    /// Check if this is a bullish candle (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if this is a bearish candle (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Check if this is a doji candle (close ≈ open)
    pub fn is_doji(&self, tolerance: f64) -> bool {
        (self.close - self.open).abs() <= tolerance
    }

    /// Convert to feature vector for clustering
    pub fn to_features(&self) -> Vec<f64> {
        vec![
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume as f64,
            self.range(),
            self.body_size(),
            self.upper_shadow(),
            self.lower_shadow(),
            self.typical_price(),
            self.weighted_close(),
        ]
    }

    /// Calculate returns relative to a previous OHLCV
    pub fn returns(&self, previous: &OHLCV) -> f64 {
        (self.close - previous.close) / previous.close
    }

    /// Calculate log returns relative to a previous OHLCV
    pub fn log_returns(&self, previous: &OHLCV) -> f64 {
        (self.close / previous.close).ln()
    }
}

/// Comprehensive clustering result with metadata and quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterResult {
    /// Name of the clustering algorithm used
    pub algorithm_name: String,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Cluster assignments for each data point
    pub cluster_assignments: Vec<usize>,
    /// Cluster centers/centroids (if applicable)
    pub cluster_centers: Option<Vec<Vec<f64>>>,
    /// Inertia/within-cluster sum of squares (for K-Means)
    pub inertia: Option<f64>,
    /// Silhouette score for clustering quality
    pub silhouette_score: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Total execution time
    pub execution_time: Duration,
    /// Noise points (for DBSCAN)
    pub noise_points: Vec<usize>,
    /// Core points (for DBSCAN)
    pub core_points: Vec<usize>,
    /// Border points (for DBSCAN)
    pub border_points: Vec<usize>,
    /// Additional algorithm-specific metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl ClusterResult {
    /// Create a new clustering result
    pub fn new(algorithm_name: String, n_clusters: usize, cluster_assignments: Vec<usize>) -> Self {
        Self {
            algorithm_name,
            n_clusters,
            cluster_assignments,
            cluster_centers: None,
            inertia: None,
            silhouette_score: 0.0,
            iterations: 0,
            converged: false,
            execution_time: Duration::ZERO,
            noise_points: Vec::new(),
            core_points: Vec::new(),
            border_points: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set cluster centers
    pub fn with_centers(mut self, centers: Vec<Vec<f64>>) -> Self {
        self.cluster_centers = Some(centers);
        self
    }

    /// Set inertia
    pub fn with_inertia(mut self, inertia: f64) -> Self {
        self.inertia = Some(inertia);
        self
    }

    /// Set silhouette score
    pub fn with_silhouette_score(mut self, score: f64) -> Self {
        self.silhouette_score = score;
        self
    }

    /// Set iteration count
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set convergence status
    pub fn with_converged(mut self, converged: bool) -> Self {
        self.converged = converged;
        self
    }

    /// Set execution time
    pub fn with_execution_time(mut self, time: Duration) -> Self {
        self.execution_time = time;
        self
    }

    /// Set noise points
    pub fn with_noise_points(mut self, noise_points: Vec<usize>) -> Self {
        self.noise_points = noise_points;
        self
    }

    /// Set core points
    pub fn with_core_points(mut self, core_points: Vec<usize>) -> Self {
        self.core_points = core_points;
        self
    }

    /// Set border points
    pub fn with_border_points(mut self, border_points: Vec<usize>) -> Self {
        self.border_points = border_points;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get cluster assignments as a slice
    pub fn assignments(&self) -> &[usize] {
        &self.cluster_assignments
    }

    /// Get the number of data points
    pub fn n_points(&self) -> usize {
        self.cluster_assignments.len()
    }

    /// Get cluster sizes
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0; self.n_clusters];
        for &assignment in &self.cluster_assignments {
            if assignment < self.n_clusters {
                sizes[assignment] += 1;
            }
        }
        sizes
    }

    /// Check if clustering is valid
    pub fn is_valid(&self) -> bool {
        self.n_clusters > 0
            && !self.cluster_assignments.is_empty()
            && self
                .cluster_assignments
                .iter()
                .all(|&x| x < self.n_clusters)
    }

    /// Get cluster quality assessment
    pub fn quality_assessment(&self) -> ClusterQuality {
        ClusterQuality {
            silhouette_score: self.silhouette_score,
            n_clusters: self.n_clusters,
            n_points: self.n_points(),
            cluster_sizes: self.cluster_sizes(),
            converged: self.converged,
            execution_time: self.execution_time,
        }
    }
}

/// Cluster quality assessment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterQuality {
    /// Silhouette score (-1 to 1, higher is better)
    pub silhouette_score: f64,
    /// Number of clusters
    pub n_clusters: usize,
    /// Total number of data points
    pub n_points: usize,
    /// Size of each cluster
    pub cluster_sizes: Vec<usize>,
    /// Whether algorithm converged
    pub converged: bool,
    /// Execution time
    pub execution_time: Duration,
}

impl ClusterQuality {
    /// Assess overall clustering quality
    pub fn overall_quality(&self) -> QualityLevel {
        if !self.converged {
            return QualityLevel::Poor;
        }

        if self.silhouette_score >= 0.7 {
            QualityLevel::Excellent
        } else if self.silhouette_score >= 0.5 {
            QualityLevel::Good
        } else if self.silhouette_score >= 0.25 {
            QualityLevel::Fair
        } else {
            QualityLevel::Poor
        }
    }

    /// Check for balanced clusters
    pub fn is_balanced(&self) -> bool {
        if self.cluster_sizes.is_empty() {
            return false;
        }

        let min_size = self.cluster_sizes.iter().min().unwrap();
        let max_size = self.cluster_sizes.iter().max().unwrap();

        // Consider balanced if min/max ratio > 0.1
        (*min_size as f64 / *max_size as f64) > 0.1
    }

    /// Get cluster size statistics
    pub fn cluster_size_stats(&self) -> ClusterSizeStats {
        if self.cluster_sizes.is_empty() {
            return ClusterSizeStats {
                min: 0,
                max: 0,
                mean: 0.0,
                std_dev: 0.0,
            };
        }

        let min = *self.cluster_sizes.iter().min().unwrap();
        let max = *self.cluster_sizes.iter().max().unwrap();
        let mean =
            self.cluster_sizes.iter().sum::<usize>() as f64 / self.cluster_sizes.len() as f64;

        let variance = self
            .cluster_sizes
            .iter()
            .map(|&size| {
                let diff = size as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.cluster_sizes.len() as f64;

        let std_dev = variance.sqrt();

        ClusterSizeStats {
            min,
            max,
            mean,
            std_dev,
        }
    }
}

/// Quality level classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualityLevel {
    Poor,
    Fair,
    Good,
    Excellent,
}

/// Cluster size statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSizeStats {
    pub min: usize,
    pub max: usize,
    pub mean: f64,
    pub std_dev: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_ohlcv_creation() {
        let timestamp = Utc::now();
        let ohlcv = OHLCV::new(timestamp, 100.0, 105.0, 98.0, 102.0, 1000);

        assert_eq!(ohlcv.open, 100.0);
        assert_eq!(ohlcv.high, 105.0);
        assert_eq!(ohlcv.low, 98.0);
        assert_eq!(ohlcv.close, 102.0);
        assert_eq!(ohlcv.volume, 1000);
    }

    #[test]
    fn test_ohlcv_validation_valid() {
        let timestamp = Utc::now();
        let ohlcv = OHLCV::new(timestamp, 100.0, 105.0, 98.0, 102.0, 1000);
        assert!(ohlcv.validate().is_ok());
    }

    #[test]
    fn test_ohlcv_validation_invalid_high_low() {
        let timestamp = Utc::now();
        let ohlcv = OHLCV::new(timestamp, 100.0, 95.0, 98.0, 102.0, 1000);
        assert!(ohlcv.validate().is_err());
    }

    #[test]
    fn test_ohlcv_validation_negative_prices() {
        let timestamp = Utc::now();
        let ohlcv = OHLCV::new(timestamp, -100.0, 105.0, 98.0, 102.0, 1000);
        assert!(ohlcv.validate().is_err());
    }

    #[test]
    fn test_ohlcv_calculations() {
        let timestamp = Utc::now();
        let ohlcv = OHLCV::new(timestamp, 100.0, 105.0, 98.0, 102.0, 1000);

        assert_eq!(ohlcv.range(), 7.0);
        assert_eq!(ohlcv.body_size(), 2.0);
        assert_eq!(ohlcv.upper_shadow(), 3.0);
        assert_eq!(ohlcv.lower_shadow(), 2.0);
        assert_eq!(ohlcv.typical_price(), (105.0 + 98.0 + 102.0) / 3.0);
        assert_eq!(ohlcv.weighted_close(), (105.0 + 98.0 + 102.0 + 102.0) / 4.0);
    }

    #[test]
    fn test_ohlcv_candle_types() {
        let timestamp = Utc::now();
        let bullish = OHLCV::new(timestamp, 100.0, 105.0, 98.0, 102.0, 1000);
        let bearish = OHLCV::new(timestamp, 102.0, 105.0, 98.0, 100.0, 1000);
        let doji = OHLCV::new(timestamp, 100.0, 105.0, 98.0, 100.1, 1000);

        assert!(bullish.is_bullish());
        assert!(!bullish.is_bearish());
        assert!(bearish.is_bearish());
        assert!(!bearish.is_bullish());
        assert!(doji.is_doji(0.2));
    }

    #[test]
    fn test_ohlcv_features() {
        let timestamp = Utc::now();
        let ohlcv = OHLCV::new(timestamp, 100.0, 105.0, 98.0, 102.0, 1000);
        let features = ohlcv.to_features();

        assert_eq!(features.len(), 11);
        assert_eq!(features[0], 100.0); // open
        assert_eq!(features[1], 105.0); // high
        assert_eq!(features[2], 98.0); // low
        assert_eq!(features[3], 102.0); // close
        assert_eq!(features[4], 1000.0); // volume
    }

    #[test]
    fn test_cluster_result_creation() {
        let result = ClusterResult::new("K-Means".to_string(), 3, vec![0, 1, 2, 0, 1, 2]);

        assert_eq!(result.algorithm_name, "K-Means");
        assert_eq!(result.n_clusters, 3);
        assert_eq!(result.n_points(), 6);
        assert!(result.is_valid());
    }

    #[test]
    fn test_cluster_result_builder_pattern() {
        let result = ClusterResult::new("K-Means".to_string(), 3, vec![0, 1, 2])
            .with_centers(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]])
            .with_inertia(123.45)
            .with_silhouette_score(0.75)
            .with_iterations(50)
            .with_converged(true)
            .with_execution_time(Duration::from_millis(100));

        assert_eq!(result.inertia, Some(123.45));
        assert_eq!(result.silhouette_score, 0.75);
        assert_eq!(result.iterations, 50);
        assert!(result.converged);
        assert_eq!(result.execution_time, Duration::from_millis(100));
    }

    #[test]
    fn test_cluster_sizes() {
        let result = ClusterResult::new("K-Means".to_string(), 3, vec![0, 0, 1, 1, 1, 2]);

        let sizes = result.cluster_sizes();
        assert_eq!(sizes, vec![2, 3, 1]);
    }

    #[test]
    fn test_cluster_quality_assessment() {
        let result = ClusterResult::new("K-Means".to_string(), 3, vec![0, 0, 1, 1, 1, 2])
            .with_silhouette_score(0.8)
            .with_converged(true);

        let quality = result.quality_assessment();
        assert_eq!(quality.silhouette_score, 0.8);
        assert_eq!(quality.n_clusters, 3);
        assert_eq!(quality.n_points, 6);
        assert!(quality.converged);
        assert_eq!(quality.overall_quality(), QualityLevel::Excellent);
    }

    proptest! {
        #[test]
        fn test_ohlcv_validation_properties(
            open in 1.0..1000.0f64,
            high in 1.0..1000.0f64,
            low in 1.0..1000.0f64,
            close in 1.0..1000.0f64,
            volume in 1u64..1000000u64
        ) {
            let timestamp = Utc::now();
            let ohlcv = OHLCV::new(timestamp, open, high, low, close, volume);

            // If validation passes, check logical properties
            if ohlcv.validate().is_ok() {
                assert!(ohlcv.high >= ohlcv.low);
                assert!(ohlcv.high >= ohlcv.open);
                assert!(ohlcv.high >= ohlcv.close);
                assert!(ohlcv.low <= ohlcv.open);
                assert!(ohlcv.low <= ohlcv.close);
                assert!(ohlcv.open > 0.0);
                assert!(ohlcv.high > 0.0);
                assert!(ohlcv.low > 0.0);
                assert!(ohlcv.close > 0.0);
            }
        }
    }

    proptest! {
        #[test]
        fn test_cluster_result_properties(
            _n_clusters in 1usize..10usize,
            assignments in prop::collection::vec(0usize..10usize, 10..100)
        ) {
            let max_cluster = assignments.iter().max().copied().unwrap_or(0);
            let n_clusters = (max_cluster + 1).max(1);

            let result = ClusterResult::new(
                "Test".to_string(),
                n_clusters,
                assignments.clone(),
            );

            assert_eq!(result.n_points(), assignments.len());
            assert!(result.is_valid());

            let sizes = result.cluster_sizes();
            assert_eq!(sizes.len(), n_clusters);
            assert_eq!(sizes.iter().sum::<usize>(), assignments.len());
        }
    }
}
