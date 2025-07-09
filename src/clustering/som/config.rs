use crate::core::{ConfigError, FormicaXError};
use std::time::Duration;

/// Topology types for SOM
#[derive(Debug, Clone, PartialEq)]
pub enum TopologyType {
    /// Rectangular grid topology
    Rectangular,
    /// Hexagonal grid topology
    Hexagonal,
}

impl Default for TopologyType {
    fn default() -> Self {
        TopologyType::Rectangular
    }
}

/// Configuration for SOM clustering algorithm
#[derive(Debug, Clone)]
pub struct SOMConfig {
    /// Grid width
    pub width: usize,
    /// Grid height
    pub height: usize,
    /// Topology type
    pub topology: TopologyType,
    /// Learning rate
    pub learning_rate: f64,
    /// Neighborhood radius
    pub neighborhood_radius: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Number of threads for parallel processing
    pub num_threads: usize,
    /// Timeout for algorithm execution
    pub timeout: Option<Duration>,
}

impl Default for SOMConfig {
    fn default() -> Self {
        Self {
            width: 5,
            height: 5,
            topology: TopologyType::Rectangular,
            learning_rate: 0.1,
            neighborhood_radius: 2.0,
            epochs: 100,
            parallel: false,
            num_threads: num_cpus::get(),
            timeout: Some(Duration::from_secs(30)),
        }
    }
}

impl SOMConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), FormicaXError> {
        if self.width == 0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "width must be greater than 0".to_string(),
            }));
        }
        if self.height == 0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "height must be greater than 0".to_string(),
            }));
        }
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "learning_rate must be in (0, 1]".to_string(),
            }));
        }
        if self.neighborhood_radius <= 0.0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "neighborhood_radius must be positive".to_string(),
            }));
        }
        if self.epochs == 0 {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "epochs must be greater than 0".to_string(),
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

/// Builder for SOMConfig
#[derive(Debug, Default)]
pub struct SOMConfigBuilder {
    config: SOMConfig,
}

impl SOMConfigBuilder {
    /// Create a new SOMConfigBuilder
    pub fn new() -> Self {
        Self {
            config: SOMConfig::default(),
        }
    }

    /// Set grid width
    pub fn width(mut self, width: usize) -> Self {
        self.config.width = width;
        self
    }

    /// Set grid height
    pub fn height(mut self, height: usize) -> Self {
        self.config.height = height;
        self
    }

    /// Set topology type
    pub fn topology(mut self, topology: TopologyType) -> Self {
        self.config.topology = topology;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set neighborhood radius
    pub fn neighborhood_radius(mut self, neighborhood_radius: f64) -> Self {
        self.config.neighborhood_radius = neighborhood_radius;
        self
    }

    /// Set number of epochs
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.config.epochs = epochs;
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

    /// Build the SOMConfig
    pub fn build(self) -> Result<SOMConfig, FormicaXError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl SOMConfig {
    /// Create a new SOMConfigBuilder
    pub fn builder() -> SOMConfigBuilder {
        SOMConfigBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_som_config_default() {
        let config = SOMConfig::default();
        assert_eq!(config.width, 5);
        assert_eq!(config.height, 5);
        assert_eq!(config.topology, TopologyType::Rectangular);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.neighborhood_radius, 2.0);
        assert_eq!(config.epochs, 100);
        assert!(!config.parallel);
    }

    #[test]
    fn test_som_config_builder() {
        let config = SOMConfig::builder()
            .width(10)
            .height(8)
            .topology(TopologyType::Hexagonal)
            .learning_rate(0.05)
            .neighborhood_radius(3.0)
            .epochs(200)
            .parallel(true)
            .num_threads(4)
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();

        assert_eq!(config.width, 10);
        assert_eq!(config.height, 8);
        assert_eq!(config.topology, TopologyType::Hexagonal);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.neighborhood_radius, 3.0);
        assert_eq!(config.epochs, 200);
        assert!(config.parallel);
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_som_config_validation_success() {
        let config = SOMConfig::builder()
            .width(5)
            .height(5)
            .learning_rate(0.1)
            .neighborhood_radius(2.0)
            .epochs(100)
            .num_threads(1)
            .build()
            .unwrap();

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_som_config_validation_zero_width() {
        let config = SOMConfig::builder().width(0).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_som_config_validation_zero_height() {
        let config = SOMConfig::builder().height(0).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_som_config_validation_invalid_learning_rate() {
        let config = SOMConfig::builder().learning_rate(1.5).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_som_config_validation_zero_neighborhood_radius() {
        let config = SOMConfig::builder().neighborhood_radius(0.0).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_som_config_validation_zero_epochs() {
        let config = SOMConfig::builder().epochs(0).build();

        assert!(config.is_err());
    }

    #[test]
    fn test_som_config_validation_zero_threads() {
        let config = SOMConfig::builder().num_threads(0).build();

        assert!(config.is_err());
    }
}
