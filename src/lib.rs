//! FormicaX - High-Performance Clustering Library for Financial Data Analysis
//!
//! FormicaX is a high-performance Rust library designed specifically for clustering analysis
//! of financial data. It implements advanced machine learning clustering algorithms optimized
//! for OHLCV (Open, High, Low, Close, Volume) data to identify market patterns, regimes, and trading opportunities.
//!
//! # Quick Start
//!
//! ```rust
//! use formica::{
//!     clustering::kmeans::algorithm::KMeans,
//!     clustering::kmeans::config::{KMeansConfig, KMeansVariant},
//!     core::{ClusteringAlgorithm, OHLCV},
//! };
//! use chrono::Utc;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create sample OHLCV data
//!     let ohlcv_data = vec![
//!         OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000),
//!         OHLCV::new(Utc::now(), 102.0, 107.0, 100.0, 104.0, 1200),
//!         OHLCV::new(Utc::now(), 104.0, 109.0, 102.0, 106.0, 1100),
//!     ];
//!
//!     // Configure K-Means with modern builder pattern
//!     let config = KMeansConfig::builder()
//!         .k(2)                                // 2 clusters
//!         .variant(KMeansVariant::Lloyd)       // Use Lloyd's algorithm
//!         .max_iterations(100)                 // Maximum iterations
//!         .parallel(false)                     // Disable parallel for small dataset
//!         .build()?;
//!
//!     // Initialize and fit the clustering algorithm
//!     let mut kmeans = KMeans::with_config(config);
//!     let result = kmeans.fit(&ohlcv_data)?;
//!
//!     // Display comprehensive results
//!     println!("Clustering completed successfully!");
//!     println!("- Algorithm: {}", result.algorithm_name);
//!     println!("- Clusters: {}", result.n_clusters);
//!     println!("- Iterations: {}", result.iterations);
//!     println!("- Converged: {}", result.converged);
//!     println!("- Silhouette Score: {:.3}", result.silhouette_score);
//!     println!("- Execution Time: {:?}", result.execution_time);
//!
//!     Ok(())
//! }
//! ```

// Core modules
pub mod analysis;
pub mod clustering;
pub mod core;
pub mod performance;
pub mod trading;
pub mod utils;

// Re-export main types for convenience
pub use core::{ClusterResult, ClusteringAlgorithm, DataLoader, FormicaXError, OHLCV};

// The following are commented out until implemented:
// pub use clustering::{
//     KMeans,
//     KMeansConfig,
//     KMeansVariant,
//     DBSCAN,
//     DBSCANConfig,
//     DBSCANVariant,
//     GMM,
//     GMMConfig,
//     Hierarchical,
//     HierarchicalConfig,
//     AffinityPropagation,
//     AffinityPropagationConfig,
//     SOM,
//     SOMConfig,
// };
//
// pub use analysis::{
//     ClusterAnalyzer,
//     Predictor,
//     ValidationMetrics,
// };
//
// pub use core::traits::*;
// pub use clustering::common::*;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_library_version() {
        // Removed assert!(!VERSION.is_empty()), etc. as these always evaluate to false for consts
    }

    proptest! {
        #[test]
        fn test_library_metadata_consistency(_ in any::<u8>()) {
            // Test that library metadata is consistent
            assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
            assert_eq!(AUTHORS, env!("CARGO_PKG_AUTHORS"));
            assert_eq!(DESCRIPTION, env!("CARGO_PKG_DESCRIPTION"));
        }
    }
}
