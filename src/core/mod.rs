//! Core data structures and interfaces for FormicaX
//!
//! This module contains the fundamental data structures, error handling,
//! and trait definitions that form the foundation of the FormicaX library.

pub mod data_loader;
pub mod data_models;
pub mod data_validator;
pub mod traits;

// Re-export main types
pub use data_loader::DataLoader;
pub use data_models::{ClusterResult, OHLCV};
pub use data_validator::DataValidator;
pub use traits::ClusteringAlgorithm;

// Error handling
use thiserror::Error;

/// Main error type for FormicaX operations
#[derive(Error, Debug)]
pub enum FormicaXError {
    /// Data loading and validation errors
    #[error("Data error: {0}")]
    Data(#[from] DataError),

    /// Clustering algorithm errors
    #[error("Clustering error: {0}")]
    Clustering(#[from] ClusteringError),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    /// Performance and optimization errors
    #[error("Performance error: {0}")]
    Performance(#[from] PerformanceError),

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// CSV parsing errors
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Chrono datetime parsing errors
    #[error("DateTime error: {0}")]
    DateTime(#[from] chrono::ParseError),
}

/// Data-related errors
#[derive(Error, Debug)]
pub enum DataError {
    /// Invalid CSV format
    #[error("Invalid CSV format: {message}")]
    InvalidCsvFormat { message: String },

    /// Missing required columns
    #[error("Missing required column: {column}")]
    MissingColumn { column: String },

    /// Invalid data type
    #[error("Invalid data type for column {column}: expected {expected}, got {actual}")]
    InvalidDataType {
        column: String,
        expected: String,
        actual: String,
    },

    /// Data validation failure
    #[error("Data validation failed: {message}")]
    ValidationFailed { message: String },

    /// Empty dataset
    #[error("Dataset is empty")]
    EmptyDataset,

    /// Insufficient data for clustering
    #[error(
        "Insufficient data for clustering: need at least {min_points} points, got {actual_points}"
    )]
    InsufficientData {
        min_points: usize,
        actual_points: usize,
    },

    /// Invalid OHLCV data (logical inconsistencies)
    #[error("Invalid OHLCV data: {message}")]
    InvalidOHLCV { message: String },
}

/// Clustering algorithm errors
#[derive(Error, Debug)]
pub enum ClusteringError {
    /// Algorithm failed to converge
    #[error("Algorithm failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },

    /// Invalid parameters for clustering
    #[error("Invalid clustering parameters: {message}")]
    InvalidParameters { message: String },

    /// Algorithm-specific errors
    #[error("Algorithm error: {message}")]
    AlgorithmError { message: String },

    /// Memory allocation failure
    #[error("Memory allocation failed: {message}")]
    MemoryError { message: String },

    /// Numerical computation errors
    #[error("Numerical computation error: {message}")]
    NumericalError { message: String },
}

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    /// Invalid configuration value
    #[error("Invalid configuration value for {field}: {message}")]
    InvalidValue { field: String, message: String },

    /// Missing required configuration
    #[error("Missing required configuration: {field}")]
    MissingField { field: String },

    /// Configuration validation failure
    #[error("Configuration validation failed: {message}")]
    ValidationFailed { message: String },
}

/// Performance-related errors
#[derive(Error, Debug)]
pub enum PerformanceError {
    /// SIMD operation failed
    #[error("SIMD operation failed: {message}")]
    SimdError { message: String },

    /// Parallel processing error
    #[error("Parallel processing error: {message}")]
    ParallelError { message: String },

    /// Memory management error
    #[error("Memory management error: {message}")]
    MemoryError { message: String },

    /// Performance timeout
    #[error("Operation timed out after {duration:?}")]
    Timeout { duration: std::time::Duration },
}

// Result type alias for convenience
pub type FormicaXResult<T> = Result<T, FormicaXError>;

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_error_conversions() {
        // Test IO error conversion
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let formicax_error: FormicaXError = io_error.into();
        assert!(matches!(formicax_error, FormicaXError::Io(_)));

        // Test CSV error conversion
        let csv_error = csv::Error::from(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "CSV error",
        ));
        let formicax_error: FormicaXError = csv_error.into();
        assert!(matches!(formicax_error, FormicaXError::Csv(_)));
    }

    proptest! {
        #[test]
        fn test_data_error_creation(
            column in "[a-zA-Z_][a-zA-Z0-9_]*",
            message in "[a-zA-Z0-9 ]+"
        ) {
            let error = DataError::MissingColumn { column: column.clone() };
            assert!(error.to_string().contains(&column));

            let error = DataError::ValidationFailed { message: message.clone() };
            assert!(error.to_string().contains(&message));
        }
    }

    #[test]
    fn test_clustering_error_creation() {
        let error = ClusteringError::ConvergenceFailure { iterations: 100 };
        assert!(error.to_string().contains("100"));

        let error = ClusteringError::InvalidParameters {
            message: "k must be > 0".to_string(),
        };
        assert!(error.to_string().contains("k must be > 0"));
    }

    #[test]
    fn test_config_error_creation() {
        let error = ConfigError::InvalidValue {
            field: "k".to_string(),
            message: "must be positive".to_string(),
        };
        assert!(error.to_string().contains("k"));
        assert!(error.to_string().contains("must be positive"));
    }

    #[test]
    fn test_performance_error_creation() {
        let duration = std::time::Duration::from_secs(30);
        let error = PerformanceError::Timeout { duration };
        assert!(error.to_string().contains("30"));
    }
}
