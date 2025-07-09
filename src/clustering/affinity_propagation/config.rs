use crate::core::{ConfigError, FormicaXError};
use std::time::Duration;

/// Preference methods for affinity propagation
#[derive(Debug, Clone, PartialEq)]
pub enum PreferenceMethod {
    /// Use median of similarities as preference
    Median,
    /// Use minimum of similarities as preference
    Minimum,
    /// Use custom preference value
    Custom(f64),
}

impl Default for PreferenceMethod {
    fn default() -> Self {
        PreferenceMethod::Median
    }
}

/// Configuration for Affinity Propagation clustering algorithm
#[derive(Debug, Clone)]
pub struct AffinityPropagationConfig {
    /// Damping factor for message updates
    pub damping: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Preference method
    pub preference: PreferenceMethod,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Number of threads for parallel processing
    pub num_threads: usize,
    /// Timeout for algorithm execution
    pub timeout: Option<Duration>,
}

impl Default for AffinityPropagationConfig {
    fn default() -> Self {
        Self {
            damping: 0.5,
            max_iterations: 200,
            tolerance: 1e-6,
            preference: PreferenceMethod::Median,
            parallel: false,
            num_threads: num_cpus::get(),
            timeout: Some(Duration::from_secs(30)),
        }
    }
}

impl AffinityPropagationConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), FormicaXError> {
        if self.damping < 0.0 || self.damping >= 1.0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "damping must be in [0, 1)".to_string(),
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
        if self.num_threads == 0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "num_threads must be greater than 0".to_string(),
            }));
        }
        Ok(())
    }
}

/// Builder for AffinityPropagationConfig
#[derive(Debug, Default)]
pub struct AffinityPropagationConfigBuilder {
    config: AffinityPropagationConfig,
}

impl AffinityPropagationConfigBuilder {
    /// Create a new AffinityPropagationConfigBuilder
    pub fn new() -> Self {
        Self {
            config: AffinityPropagationConfig::default(),
        }
    }

    /// Set damping factor
    pub fn damping(mut self, damping: f64) -> Self {
        self.config.damping = damping;
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

    /// Set preference method
    pub fn preference(mut self, preference: PreferenceMethod) -> Self {
        self.config.preference = preference;
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

    /// Build the AffinityPropagationConfig
    pub fn build(self) -> Result<AffinityPropagationConfig, FormicaXError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl AffinityPropagationConfig {
    /// Create a new AffinityPropagationConfigBuilder
    pub fn builder() -> AffinityPropagationConfigBuilder {
        AffinityPropagationConfigBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affinity_propagation_config_default() {
        let config = AffinityPropagationConfig::default();
        assert_eq!(config.damping, 0.5);
        assert_eq!(config.max_iterations, 200);
        assert_eq!(config.tolerance, 1e-6);
        assert_eq!(config.preference, PreferenceMethod::Median);
        assert!(!config.parallel);
    }

    #[test]
    fn test_affinity_propagation_config_builder() {
        let config = AffinityPropagationConfig::builder()
            .damping(0.7)
            .max_iterations(300)
            .tolerance(1e-8)
            .preference(PreferenceMethod::Minimum)
            .parallel(true)
            .num_threads(4)
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();

        assert_eq!(config.damping, 0.7);
        assert_eq!(config.max_iterations, 300);
        assert_eq!(config.tolerance, 1e-8);
        assert_eq!(config.preference, PreferenceMethod::Minimum);
        assert!(config.parallel);
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_affinity_propagation_config_validation_success() {
        let config = AffinityPropagationConfig::builder()
            .damping(0.5)
            .max_iterations(200)
            .tolerance(1e-6)
            .num_threads(1)
            .build()
            .unwrap();

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_affinity_propagation_config_validation_invalid_damping() {
        let config = AffinityPropagationConfig::builder().damping(1.5).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_affinity_propagation_config_validation_zero_iterations() {
        let config = AffinityPropagationConfig::builder()
            .max_iterations(0)
            .build();

        assert!(config.is_err());
    }

    #[test]
    fn test_affinity_propagation_config_validation_negative_tolerance() {
        let config = AffinityPropagationConfig::builder().tolerance(-1.0).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_affinity_propagation_config_validation_zero_threads() {
        let config = AffinityPropagationConfig::builder().num_threads(0).build();

        assert!(config.is_err());
    }
}
