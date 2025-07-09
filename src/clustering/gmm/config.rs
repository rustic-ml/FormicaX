use crate::core::{ConfigError, FormicaXError};
use std::time::Duration;

/// GMM algorithm variants
#[derive(Debug, Clone, PartialEq, Default)]
pub enum GMMVariant {
    /// Standard Expectation-Maximization algorithm
    #[default]
    EM,
    /// Variational Bayesian GMM
    VariationalBayes,
    /// Robust GMM using t-distributions
    Robust,
}

/// Configuration for GMM clustering algorithm
#[derive(Debug, Clone)]
pub struct GMMConfig {
    /// Number of Gaussian components
    pub n_components: usize,
    /// Algorithm variant
    pub variant: GMMVariant,
    /// Maximum iterations for EM algorithm
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Covariance type
    pub covariance_type: CovarianceType,
    /// Regularization parameter for covariance matrices
    pub regularization: f64,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Number of threads for parallel processing
    pub num_threads: usize,
    /// Timeout for algorithm execution
    pub timeout: Option<Duration>,
}

impl Default for GMMConfig {
    fn default() -> Self {
        Self {
            n_components: 3,
            variant: GMMVariant::EM,
            max_iterations: 100,
            tolerance: 1e-6,
            random_seed: None,
            covariance_type: CovarianceType::Full,
            regularization: 1e-6,
            parallel: false,
            num_threads: num_cpus::get(),
            timeout: Some(Duration::from_secs(30)),
        }
    }
}

/// Covariance matrix types for GMM
#[derive(Debug, Clone, PartialEq, Default)]
pub enum CovarianceType {
    /// Full covariance matrix
    #[default]
    Full,
    /// Diagonal covariance matrix
    Diagonal,
    /// Tied covariance matrix (shared across components)
    Tied,
    /// Spherical covariance (isotropic)
    Spherical,
}

impl GMMConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), FormicaXError> {
        if self.n_components == 0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "n_components must be greater than 0".to_string(),
            }));
        }
        if self.max_iterations == 0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "max_iterations must be greater than 0".to_string(),
            }));
        }
        if self.tolerance <= 0.0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "tolerance must be positive".to_string(),
            }));
        }
        if self.regularization < 0.0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "regularization must be non-negative".to_string(),
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

/// Builder for GMMConfig
#[derive(Debug, Default)]
pub struct GMMConfigBuilder {
    config: GMMConfig,
}

impl GMMConfigBuilder {
    /// Create a new GMMConfigBuilder
    pub fn new() -> Self {
        Self {
            config: GMMConfig::default(),
        }
    }

    /// Set number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set algorithm variant
    pub fn variant(mut self, variant: GMMVariant) -> Self {
        self.config.variant = variant;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.config.tolerance = tolerance;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, random_seed: u64) -> Self {
        self.config.random_seed = Some(random_seed);
        self
    }

    /// Set covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.config.covariance_type = covariance_type;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.config.regularization = regularization;
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

    /// Build the GMMConfig
    pub fn build(self) -> Result<GMMConfig, FormicaXError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl GMMConfig {
    /// Create a new GMMConfigBuilder
    pub fn builder() -> GMMConfigBuilder {
        GMMConfigBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gmm_config_default() {
        let config = GMMConfig::default();
        assert_eq!(config.n_components, 3);
        assert_eq!(config.variant, GMMVariant::EM);
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.tolerance, 1e-6);
        assert!(config.random_seed.is_none());
        assert_eq!(config.covariance_type, CovarianceType::Full);
        assert_eq!(config.regularization, 1e-6);
        assert!(!config.parallel);
    }

    #[test]
    fn test_gmm_config_builder() {
        let config = GMMConfig::builder()
            .n_components(5)
            .variant(GMMVariant::VariationalBayes)
            .max_iterations(200)
            .tolerance(1e-8)
            .random_seed(42)
            .covariance_type(CovarianceType::Diagonal)
            .regularization(1e-5)
            .parallel(true)
            .num_threads(4)
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();

        assert_eq!(config.n_components, 5);
        assert_eq!(config.variant, GMMVariant::VariationalBayes);
        assert_eq!(config.max_iterations, 200);
        assert_eq!(config.tolerance, 1e-8);
        assert_eq!(config.random_seed, Some(42));
        assert_eq!(config.covariance_type, CovarianceType::Diagonal);
        assert_eq!(config.regularization, 1e-5);
        assert!(config.parallel);
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_gmm_config_validation_success() {
        let config = GMMConfig::builder()
            .n_components(3)
            .max_iterations(100)
            .tolerance(1e-6)
            .regularization(1e-6)
            .num_threads(1)
            .build()
            .unwrap();

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gmm_config_validation_n_components_zero() {
        let config = GMMConfig::builder().n_components(0).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_gmm_config_validation_max_iterations_zero() {
        let config = GMMConfig::builder().max_iterations(0).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_gmm_config_validation_negative_tolerance() {
        let config = GMMConfig::builder().tolerance(-1.0).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_gmm_config_validation_negative_regularization() {
        let config = GMMConfig::builder().regularization(-1.0).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_gmm_config_validation_zero_threads() {
        let config = GMMConfig::builder().num_threads(0).build();

        assert!(config.is_err());
    }
}
