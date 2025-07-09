use crate::core::{ConfigError, FormicaXError};
use std::time::Duration;

/// Linkage methods for hierarchical clustering
#[derive(Debug, Clone, PartialEq, Default)]
pub enum LinkageMethod {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage (average distance)
    Average,
    /// Ward's method (minimize within-cluster variance)
    #[default]
    Ward,
    /// Centroid linkage (distance between centroids)
    Centroid,
}

/// Configuration for hierarchical clustering algorithm
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Number of clusters to form
    pub n_clusters: usize,
    /// Linkage method
    pub linkage: LinkageMethod,
    /// Distance metric
    pub distance_metric: DistanceMetric,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Number of threads for parallel processing
    pub num_threads: usize,
    /// Timeout for algorithm execution
    pub timeout: Option<Duration>,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            n_clusters: 3,
            linkage: LinkageMethod::Ward,
            distance_metric: DistanceMetric::Euclidean,
            parallel: false,
            num_threads: num_cpus::get(),
            timeout: Some(Duration::from_secs(30)),
        }
    }
}

/// Distance metrics for hierarchical clustering
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DistanceMetric {
    /// Euclidean distance
    #[default]
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
}

impl HierarchicalConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), FormicaXError> {
        if self.n_clusters == 0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "n_clusters must be greater than 0".to_string(),
            }));
        }
        if self.num_threads == 0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "num_threads must be greater than 0".to_string(),
            }));
        }
        Ok(())
    }
}

/// Builder for HierarchicalConfig
#[derive(Debug, Default)]
pub struct HierarchicalConfigBuilder {
    config: HierarchicalConfig,
}

impl HierarchicalConfigBuilder {
    /// Create a new HierarchicalConfigBuilder
    pub fn new() -> Self {
        Self {
            config: HierarchicalConfig::default(),
        }
    }

    /// Set number of clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.config.n_clusters = n_clusters;
        self
    }

    /// Set linkage method
    pub fn linkage(mut self, linkage: LinkageMethod) -> Self {
        self.config.linkage = linkage;
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, distance_metric: DistanceMetric) -> Self {
        self.config.distance_metric = distance_metric;
        self
    }

    /// Enable/disable parallel processing
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }

    /// Set number of threads
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.config.num_threads = num_threads;
        self
    }

    /// Set timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = Some(timeout);
        self
    }

    /// Build the HierarchicalConfig
    pub fn build(self) -> Result<HierarchicalConfig, FormicaXError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl HierarchicalConfig {
    /// Create a new HierarchicalConfigBuilder
    pub fn builder() -> HierarchicalConfigBuilder {
        HierarchicalConfigBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_config_default() {
        let config = HierarchicalConfig::default();
        assert_eq!(config.n_clusters, 3);
        assert_eq!(config.linkage, LinkageMethod::Ward);
        assert_eq!(config.distance_metric, DistanceMetric::Euclidean);
        assert!(!config.parallel);
    }

    #[test]
    fn test_hierarchical_config_builder() {
        let config = HierarchicalConfig::builder()
            .n_clusters(5)
            .linkage(LinkageMethod::Complete)
            .distance_metric(DistanceMetric::Manhattan)
            .parallel(true)
            .num_threads(4)
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();

        assert_eq!(config.n_clusters, 5);
        assert_eq!(config.linkage, LinkageMethod::Complete);
        assert_eq!(config.distance_metric, DistanceMetric::Manhattan);
        assert!(config.parallel);
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_hierarchical_config_validation_success() {
        let config = HierarchicalConfig::builder()
            .n_clusters(3)
            .num_threads(1)
            .build()
            .unwrap();

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_hierarchical_config_validation_zero_clusters() {
        let config = HierarchicalConfig::builder().n_clusters(0).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_hierarchical_config_validation_zero_threads() {
        let config = HierarchicalConfig::builder().num_threads(0).build();

        assert!(config.is_err());
    }
}
