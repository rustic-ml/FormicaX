//! DBSCAN (Density-Based Spatial Clustering of Applications with Noise) module
//!
//! This module provides high-performance DBSCAN clustering with multiple variants:
//! - Standard DBSCAN with KD-tree spatial indexing
//! - Parallel DBSCAN with lock-free region queries
//! - Incremental DBSCAN for streaming data
//! - Approximate DBSCAN for large-scale datasets

pub mod algorithm;
pub mod config;
pub mod spatial;

use crate::core::{ClusterResult, ClusteringAlgorithm, FormicaXError, OHLCV};

// Re-export main types for convenience
pub use algorithm::DBSCAN;
pub use config::{DBSCANConfig, DBSCANConfigBuilder, DBSCANVariant};

impl ClusteringAlgorithm for DBSCAN {
    type Config = DBSCANConfig;

    fn new() -> Self {
        Self::with_config(DBSCANConfig::default())
    }

    fn with_config(config: Self::Config) -> Self {
        Self::with_config(config)
    }

    fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError> {
        self.fit(data)
    }

    fn predict(&self, data: &[OHLCV]) -> Result<Vec<usize>, FormicaXError> {
        self.predict(data)
    }

    fn get_cluster_centers(&self) -> Option<Vec<Vec<f64>>> {
        self.get_cluster_centers()
    }

    fn validate_config(&self, data: &[OHLCV]) -> Result<(), FormicaXError> {
        self.validate_config(data)
    }

    fn algorithm_name(&self) -> &'static str {
        "DBSCAN"
    }

    fn supports_incremental(&self) -> bool {
        matches!(self.config.variant, DBSCANVariant::Incremental)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OHLCV;
    use chrono::Utc;
    use proptest::prelude::*;

    #[test]
    fn test_dbscan_creation() {
        let dbscan = DBSCAN::new();
        assert_eq!(dbscan.algorithm_name(), "DBSCAN");
    }

    #[test]
    fn test_dbscan_with_config() {
        let config = DBSCANConfig::builder()
            .epsilon(0.5)
            .min_points(5)
            .variant(DBSCANVariant::Standard)
            .build()
            .unwrap();

        let dbscan = DBSCAN::with_config(config);
        assert_eq!(dbscan.algorithm_name(), "DBSCAN");
    }

    #[test]
    fn test_dbscan_supports_incremental() {
        let config = DBSCANConfig::builder()
            .variant(DBSCANVariant::Incremental)
            .build()
            .unwrap();

        let dbscan = DBSCAN::with_config(config);
        assert!(dbscan.supports_incremental());
    }

    proptest! {
        #[test]
        fn test_dbscan_properties(
            epsilon in 0.1..10.0f64,
            min_points in 2..20usize,
            data_size in 10..100usize
        ) {
            // Generate test data
            let data: Vec<OHLCV> = (0..data_size)
                .map(|i| {
                    OHLCV::new(
                        Utc::now() + chrono::Duration::hours(i as i64),
                        100.0 + i as f64,
                        105.0 + i as f64,
                        98.0 + i as f64,
                        102.0 + i as f64,
                        1000 + i as u64,
                    )
                })
                .collect();

            let config = DBSCANConfig::builder()
                .epsilon(epsilon)
                .min_points(min_points)
                .variant(DBSCANVariant::Standard)
                .build()
                .unwrap();

            let dbscan = DBSCAN::with_config(config);

            // Test that configuration validation works
            let validation_result = dbscan.validate_config(&data);
            assert!(validation_result.is_ok());
        }
    }
}
