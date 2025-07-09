//! DBSCAN configuration and builder pattern
//!
//! This module provides configuration options for DBSCAN clustering algorithms
//! with support for multiple variants and optimization strategies.

use crate::core::FormicaXError;
use std::time::Duration;

/// DBSCAN algorithm variants
#[derive(Debug, Clone, PartialEq)]
pub enum DBSCANVariant {
    /// Standard DBSCAN with KD-tree spatial indexing
    Standard,
    /// Parallel DBSCAN with lock-free region queries
    Parallel,
    /// Incremental DBSCAN for streaming data
    Incremental,
    /// Approximate DBSCAN for large-scale datasets
    Approximate,
}

/// DBSCAN configuration with builder pattern
#[derive(Debug, Clone)]
pub struct DBSCANConfig {
    /// Epsilon (ε) - maximum distance between points in the same cluster
    pub epsilon: f64,
    /// Minimum number of points required to form a cluster
    pub min_points: usize,
    /// Algorithm variant to use
    pub variant: DBSCANVariant,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Whether to use SIMD optimizations
    pub simd: bool,
    /// Maximum number of iterations for convergence
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to use adaptive epsilon calculation
    pub adaptive_epsilon: bool,
    /// Number of threads for parallel processing
    pub num_threads: usize,
    /// Buffer size for incremental processing
    pub buffer_size: usize,
    /// Timeout for processing
    pub timeout: Option<Duration>,
}

impl Default for DBSCANConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            min_points: 5,
            variant: DBSCANVariant::Standard,
            parallel: false,
            simd: false,
            max_iterations: 1000,
            tolerance: 1e-6,
            adaptive_epsilon: false,
            num_threads: num_cpus::get(),
            buffer_size: 1000,
            timeout: None,
        }
    }
}

impl DBSCANConfig {
    /// Create a new builder for DBSCAN configuration
    pub fn builder() -> DBSCANConfigBuilder {
        DBSCANConfigBuilder::default()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), FormicaXError> {
        if self.epsilon <= 0.0 {
            return Err(FormicaXError::Config(
                crate::core::ConfigError::InvalidValue {
                    field: "epsilon".to_string(),
                    message: "Epsilon must be positive".to_string(),
                },
            ));
        }

        if self.min_points < 2 {
            return Err(FormicaXError::Config(
                crate::core::ConfigError::InvalidValue {
                    field: "min_points".to_string(),
                    message: "Minimum points must be at least 2".to_string(),
                },
            ));
        }

        if self.max_iterations == 0 {
            return Err(FormicaXError::Config(
                crate::core::ConfigError::InvalidValue {
                    field: "max_iterations".to_string(),
                    message: "Max iterations must be positive".to_string(),
                },
            ));
        }

        if self.tolerance <= 0.0 {
            return Err(FormicaXError::Config(
                crate::core::ConfigError::InvalidValue {
                    field: "tolerance".to_string(),
                    message: "Tolerance must be positive".to_string(),
                },
            ));
        }

        if self.num_threads == 0 {
            return Err(FormicaXError::Config(
                crate::core::ConfigError::InvalidValue {
                    field: "num_threads".to_string(),
                    message: "Number of threads must be positive".to_string(),
                },
            ));
        }

        if self.buffer_size == 0 {
            return Err(FormicaXError::Config(
                crate::core::ConfigError::InvalidValue {
                    field: "buffer_size".to_string(),
                    message: "Buffer size must be positive".to_string(),
                },
            ));
        }

        Ok(())
    }
}

/// Builder for DBSCAN configuration
#[derive(Debug, Default)]
pub struct DBSCANConfigBuilder {
    config: DBSCANConfig,
}

impl DBSCANConfigBuilder {
    /// Set epsilon (maximum distance between points in the same cluster)
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Set minimum number of points required to form a cluster
    pub fn min_points(mut self, min_points: usize) -> Self {
        self.config.min_points = min_points;
        self
    }

    /// Set algorithm variant
    pub fn variant(mut self, variant: DBSCANVariant) -> Self {
        self.config.variant = variant;
        self
    }

    /// Enable or disable parallel processing
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }

    /// Enable or disable SIMD optimizations
    pub fn simd(mut self, simd: bool) -> Self {
        self.config.simd = simd;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.config.tolerance = tolerance;
        self
    }

    /// Enable or disable adaptive epsilon calculation
    pub fn adaptive_epsilon(mut self, adaptive_epsilon: bool) -> Self {
        self.config.adaptive_epsilon = adaptive_epsilon;
        self
    }

    /// Set number of threads for parallel processing
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.config.num_threads = num_threads;
        self
    }

    /// Set buffer size for incremental processing
    pub fn buffer_size(mut self, buffer_size: usize) -> Self {
        self.config.buffer_size = buffer_size;
        self
    }

    /// Set timeout for processing
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = Some(timeout);
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<DBSCANConfig, FormicaXError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbscan_config_default() {
        let config = DBSCANConfig::default();
        assert_eq!(config.epsilon, 1.0);
        assert_eq!(config.min_points, 5);
        assert_eq!(config.variant, DBSCANVariant::Standard);
        assert!(!config.parallel);
        assert!(!config.simd);
    }

    #[test]
    fn test_dbscan_config_builder() {
        let config = DBSCANConfig::builder()
            .epsilon(0.5)
            .min_points(10)
            .variant(DBSCANVariant::Parallel)
            .parallel(true)
            .simd(true)
            .max_iterations(500)
            .tolerance(1e-8)
            .adaptive_epsilon(true)
            .num_threads(8)
            .buffer_size(2000)
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        assert_eq!(config.epsilon, 0.5);
        assert_eq!(config.min_points, 10);
        assert_eq!(config.variant, DBSCANVariant::Parallel);
        assert!(config.parallel);
        assert!(config.simd);
        assert_eq!(config.max_iterations, 500);
        assert_eq!(config.tolerance, 1e-8);
        assert!(config.adaptive_epsilon);
        assert_eq!(config.num_threads, 8);
        assert_eq!(config.buffer_size, 2000);
        assert_eq!(config.timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_dbscan_config_validation_positive_epsilon() {
        let config = DBSCANConfig::builder().epsilon(-1.0).build();
        assert!(config.is_err());
    }

    #[test]
    fn test_dbscan_config_validation_min_points() {
        let config = DBSCANConfig::builder().min_points(1).build();
        assert!(config.is_err());
    }

    #[test]
    fn test_dbscan_config_validation_max_iterations() {
        let config = DBSCANConfig::builder().max_iterations(0).build();
        assert!(config.is_err());
    }

    #[test]
    fn test_dbscan_config_validation_tolerance() {
        let config = DBSCANConfig::builder().tolerance(-1e-6).build();
        assert!(config.is_err());
    }

    #[test]
    fn test_dbscan_config_validation_num_threads() {
        let config = DBSCANConfig::builder().num_threads(0).build();
        assert!(config.is_err());
    }

    #[test]
    fn test_dbscan_config_validation_buffer_size() {
        let config = DBSCANConfig::builder().buffer_size(0).build();
        assert!(config.is_err());
    }

    #[test]
    fn test_dbscan_config_validation_success() {
        let config = DBSCANConfig::builder()
            .epsilon(0.5)
            .min_points(5)
            .max_iterations(100)
            .tolerance(1e-6)
            .num_threads(4)
            .buffer_size(1000)
            .build();
        assert!(config.is_ok());
    }
}
