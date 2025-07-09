//! Configuration and builder for KMeans clustering

use crate::core::{ConfigError, FormicaXError};

/// Supported KMeans algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMeansVariant {
    Lloyd,
    Elkan,
    Hamerly,
    MiniBatch,
}

impl Default for KMeansVariant {
    fn default() -> Self {
        Self::Lloyd
    }
}

/// Builder for KMeans configuration
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters
    pub k: usize,
    /// Algorithm variant
    pub variant: KMeansVariant,
    /// Enable parallel processing
    pub parallel: bool,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Enable SIMD optimization
    pub simd: bool,
}

impl KMeansConfig {
    /// Start building a new KMeansConfig
    pub fn builder() -> KMeansConfigBuilder {
        KMeansConfigBuilder::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct KMeansConfigBuilder {
    k: Option<usize>,
    variant: Option<KMeansVariant>,
    parallel: Option<bool>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    random_seed: Option<u64>,
}

impl KMeansConfigBuilder {
    /// Set the number of clusters
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Set the K-means variant
    pub fn variant(mut self, variant: KMeansVariant) -> Self {
        self.variant = Some(variant);
        self
    }

    /// Enable parallel processing
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = Some(parallel);
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, random_seed: u64) -> Self {
        self.random_seed = Some(random_seed);
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<KMeansConfig, FormicaXError> {
        let k = self
            .k
            .ok_or(FormicaXError::Config(ConfigError::MissingField {
                field: "k".to_string(),
            }))?;

        if k == 0 {
            return Err(FormicaXError::Config(ConfigError::InvalidValue {
                field: "k".to_string(),
                message: "must be greater than 0".to_string(),
            }));
        }

        Ok(KMeansConfig {
            k,
            variant: self.variant.unwrap_or(KMeansVariant::Lloyd),
            parallel: self.parallel.unwrap_or(false),
            max_iterations: self.max_iterations.unwrap_or(100),
            tolerance: self.tolerance.unwrap_or(1e-8),
            random_seed: self.random_seed,
            simd: false,
        })
    }
}
