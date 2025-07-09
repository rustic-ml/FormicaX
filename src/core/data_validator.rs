//! Data validation for FormicaX
//!
//! This module provides comprehensive validation rules for OHLCV data,
//! ensuring data quality and consistency for clustering analysis.

use crate::core::OHLCV;
use std::collections::HashMap;

/// Comprehensive data validator for OHLCV data
///
/// Provides multiple validation rules to ensure data quality and consistency.
#[derive(Debug)]
pub struct DataValidator {
    /// Validation rules to apply
    rules: Vec<ValidationRule>,
    /// Whether to stop on first error
    stop_on_first_error: bool,
    /// Maximum number of errors to collect
    max_errors: usize,
}

// Type alias for custom validation closure to reduce type complexity
pub type ValidationFn = dyn Fn(&OHLCV) -> Result<(), String> + Send + Sync;

/// Validation rules for OHLCV data
/// Note: Custom cannot be Debug, so we implement Debug manually.
pub enum ValidationRule {
    /// Check logical consistency (high >= low, etc.)
    LogicalConsistency,
    /// Check price range
    PriceRange(std::ops::Range<f64>),
    /// Check volume threshold
    VolumeThreshold(u64),
    /// Check for missing values
    NoMissingValues,
    /// Check for duplicate timestamps
    NoDuplicateTimestamps,
    /// Check chronological order
    ChronologicalOrder,
    /// Check for outliers using statistical methods
    OutlierDetection(OutlierMethod),
    /// Check for gaps in time series
    TimeSeriesGaps(std::time::Duration),
    /// Custom validation function (not Debug)
    Custom(Box<ValidationFn>),
}

impl std::fmt::Debug for ValidationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationRule::LogicalConsistency => write!(f, "LogicalConsistency"),
            ValidationRule::PriceRange(r) => write!(f, "PriceRange({:?})", r),
            ValidationRule::VolumeThreshold(t) => write!(f, "VolumeThreshold({})", t),
            ValidationRule::NoMissingValues => write!(f, "NoMissingValues"),
            ValidationRule::NoDuplicateTimestamps => write!(f, "NoDuplicateTimestamps"),
            ValidationRule::ChronologicalOrder => write!(f, "ChronologicalOrder"),
            ValidationRule::OutlierDetection(m) => write!(f, "OutlierDetection({:?})", m),
            ValidationRule::TimeSeriesGaps(d) => write!(f, "TimeSeriesGaps({:?})", d),
            ValidationRule::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
}

/// Methods for outlier detection
#[derive(Debug, Clone)]
pub enum OutlierMethod {
    /// Z-score based outlier detection
    ZScore(f64), // threshold
    /// IQR based outlier detection
    IQR(f64), // multiplier
    /// Modified Z-score based outlier detection
    ModifiedZScore(f64), // threshold
}

/// Validation result with detailed information
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// List of validation errors
    pub errors: Vec<ValidationError>,
    /// Validation statistics
    pub statistics: ValidationStatistics,
}

/// Individual validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Row index where error occurred
    pub row_index: Option<usize>,
    /// Data point that caused the error
    pub data_point: Option<OHLCV>,
}

/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStatistics {
    /// Total number of data points validated
    pub total_points: usize,
    /// Number of valid data points
    pub valid_points: usize,
    /// Number of invalid data points
    pub invalid_points: usize,
    /// Price range statistics
    pub price_stats: PriceStatistics,
    /// Volume statistics
    pub volume_stats: VolumeStatistics,
    /// Time series statistics
    pub time_stats: TimeSeriesStatistics,
}

/// Price statistics
#[derive(Debug, Clone)]
pub struct PriceStatistics {
    pub min_price: f64,
    pub max_price: f64,
    pub mean_price: f64,
    pub std_dev_price: f64,
    pub median_price: f64,
}

/// Volume statistics
#[derive(Debug, Clone)]
pub struct VolumeStatistics {
    pub min_volume: u64,
    pub max_volume: u64,
    pub mean_volume: f64,
    pub std_dev_volume: f64,
    pub median_volume: u64,
}

/// Time series statistics
#[derive(Debug, Clone)]
pub struct TimeSeriesStatistics {
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub total_duration: std::time::Duration,
    pub avg_interval: std::time::Duration,
    pub min_interval: std::time::Duration,
    pub max_interval: std::time::Duration,
}

impl DataValidator {
    /// Create a new data validator
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            stop_on_first_error: false,
            max_errors: 1000,
        }
    }

    /// Add a validation rule
    pub fn add_rule(mut self, rule: ValidationRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Add multiple validation rules
    pub fn add_rules(mut self, rules: Vec<ValidationRule>) -> Self {
        self.rules.extend(rules);
        self
    }

    /// Set whether to stop on first error
    pub fn stop_on_first_error(mut self, stop: bool) -> Self {
        self.stop_on_first_error = stop;
        self
    }

    /// Set maximum number of errors to collect
    pub fn max_errors(mut self, max: usize) -> Self {
        self.max_errors = max;
        self
    }

    /// Validate a single OHLCV data point
    pub fn validate_point(&self, ohlcv: &OHLCV, row_index: Option<usize>) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for rule in &self.rules {
            if let Err(error) = self.apply_rule(rule, ohlcv, row_index) {
                errors.push(error);
                if self.stop_on_first_error {
                    break;
                }
            }
        }

        errors
    }

    /// Validate a collection of OHLCV data points
    pub fn validate(&self, data: &[OHLCV]) -> ValidationResult {
        let mut errors = Vec::new();
        // Apply validation rules to each data point
        for (i, ohlcv) in data.iter().enumerate() {
            let point_errors = self.validate_point(ohlcv, Some(i));

            if !point_errors.is_empty() {
                errors.extend(point_errors);

                if errors.len() >= self.max_errors {
                    break;
                }
            }
        }

        // Calculate statistics
        let statistics = self.calculate_statistics(data);

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            statistics,
        }
    }

    /// Apply a single validation rule
    fn apply_rule(
        &self,
        rule: &ValidationRule,
        ohlcv: &OHLCV,
        row_index: Option<usize>,
    ) -> Result<(), ValidationError> {
        match rule {
            ValidationRule::LogicalConsistency => {
                if let Err(e) = ohlcv.validate() {
                    return Err(ValidationError {
                        error_type: "LogicalConsistency".to_string(),
                        message: e.to_string(),
                        row_index,
                        data_point: Some(ohlcv.clone()),
                    });
                }
            }
            ValidationRule::PriceRange(range) => {
                let prices = [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close];
                for price in &prices {
                    if !range.contains(price) {
                        return Err(ValidationError {
                            error_type: "PriceRange".to_string(),
                            message: format!("Price {} is outside valid range {:?}", price, range),
                            row_index,
                            data_point: Some(ohlcv.clone()),
                        });
                    }
                }
            }
            ValidationRule::VolumeThreshold(threshold) => {
                if ohlcv.volume < *threshold {
                    return Err(ValidationError {
                        error_type: "VolumeThreshold".to_string(),
                        message: format!(
                            "Volume {} is below threshold {}",
                            ohlcv.volume, threshold
                        ),
                        row_index,
                        data_point: Some(ohlcv.clone()),
                    });
                }
            }
            ValidationRule::NoMissingValues => {
                // OHLCV struct ensures no missing values, but we can check for NaN/Inf
                let prices = [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close];
                for price in &prices {
                    if !price.is_finite() {
                        return Err(ValidationError {
                            error_type: "NoMissingValues".to_string(),
                            message: format!("Invalid price value: {}", price),
                            row_index,
                            data_point: Some(ohlcv.clone()),
                        });
                    }
                }
            }
            ValidationRule::NoDuplicateTimestamps => {
                // This rule requires context from the full dataset
                // It's handled separately in the validate method
            }
            ValidationRule::ChronologicalOrder => {
                // This rule requires context from the full dataset
                // It's handled separately in the validate method
            }
            ValidationRule::OutlierDetection(_method) => {
                // This rule requires context from the full dataset
                // It's handled separately in the validate method
            }
            ValidationRule::TimeSeriesGaps(_) => {
                // This rule requires context from the full dataset
                // It's handled separately in the validate method
            }
            ValidationRule::Custom(func) => {
                if let Err(message) = func(ohlcv) {
                    return Err(ValidationError {
                        error_type: "Custom".to_string(),
                        message,
                        row_index,
                        data_point: Some(ohlcv.clone()),
                    });
                }
            }
        }

        Ok(())
    }

    /// Calculate comprehensive statistics for the dataset
    fn calculate_statistics(&self, data: &[OHLCV]) -> ValidationStatistics {
        if data.is_empty() {
            return ValidationStatistics {
                total_points: 0,
                valid_points: 0,
                invalid_points: 0,
                price_stats: PriceStatistics {
                    min_price: 0.0,
                    max_price: 0.0,
                    mean_price: 0.0,
                    std_dev_price: 0.0,
                    median_price: 0.0,
                },
                volume_stats: VolumeStatistics {
                    min_volume: 0,
                    max_volume: 0,
                    mean_volume: 0.0,
                    std_dev_volume: 0.0,
                    median_volume: 0,
                },
                time_stats: TimeSeriesStatistics {
                    start_time: chrono::Utc::now(),
                    end_time: chrono::Utc::now(),
                    total_duration: std::time::Duration::ZERO,
                    avg_interval: std::time::Duration::ZERO,
                    min_interval: std::time::Duration::ZERO,
                    max_interval: std::time::Duration::ZERO,
                },
            };
        }

        // Calculate price statistics
        let all_prices: Vec<f64> = data
            .iter()
            .flat_map(|ohlcv| vec![ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close])
            .collect();

        let price_stats = self.calculate_price_statistics(&all_prices);

        // Calculate volume statistics
        let volumes: Vec<u64> = data.iter().map(|ohlcv| ohlcv.volume).collect();
        let volume_stats = self.calculate_volume_statistics(&volumes);

        // Calculate time series statistics
        let time_stats = self.calculate_time_series_statistics(data);

        ValidationStatistics {
            total_points: data.len(),
            valid_points: data.len(), // Will be updated by caller
            invalid_points: 0,        // Will be updated by caller
            price_stats,
            volume_stats,
            time_stats,
        }
    }

    /// Calculate price statistics
    fn calculate_price_statistics(&self, prices: &[f64]) -> PriceStatistics {
        let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_price = prices.iter().sum::<f64>() / prices.len() as f64;

        let variance = prices
            .iter()
            .map(|&price| {
                let diff = price - mean_price;
                diff * diff
            })
            .sum::<f64>()
            / prices.len() as f64;
        let std_dev_price = variance.sqrt();

        let mut sorted_prices = prices.to_vec();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_price = if sorted_prices.len() % 2 == 0 {
            let mid = sorted_prices.len() / 2;
            (sorted_prices[mid - 1] + sorted_prices[mid]) / 2.0
        } else {
            sorted_prices[sorted_prices.len() / 2]
        };

        PriceStatistics {
            min_price,
            max_price,
            mean_price,
            std_dev_price,
            median_price,
        }
    }

    /// Calculate volume statistics
    fn calculate_volume_statistics(&self, volumes: &[u64]) -> VolumeStatistics {
        let min_volume = *volumes.iter().min().unwrap_or(&0);
        let max_volume = *volumes.iter().max().unwrap_or(&0);
        let mean_volume = volumes.iter().sum::<u64>() as f64 / volumes.len() as f64;

        let variance = volumes
            .iter()
            .map(|&volume| {
                let diff = volume as f64 - mean_volume;
                diff * diff
            })
            .sum::<f64>()
            / volumes.len() as f64;
        let std_dev_volume = variance.sqrt();

        let mut sorted_volumes = volumes.to_vec();
        sorted_volumes.sort();
        let median_volume = if sorted_volumes.len() % 2 == 0 {
            let mid = sorted_volumes.len() / 2;
            (sorted_volumes[mid - 1] + sorted_volumes[mid]) / 2
        } else {
            sorted_volumes[sorted_volumes.len() / 2]
        };

        VolumeStatistics {
            min_volume,
            max_volume,
            mean_volume,
            std_dev_volume,
            median_volume,
        }
    }

    /// Calculate time series statistics
    fn calculate_time_series_statistics(&self, data: &[OHLCV]) -> TimeSeriesStatistics {
        let start_time = data.first().unwrap().timestamp;
        let end_time = data.last().unwrap().timestamp;
        let total_duration = end_time.signed_duration_since(start_time).to_std().unwrap();

        let mut intervals = Vec::new();
        for i in 1..data.len() {
            let interval = data[i]
                .timestamp
                .signed_duration_since(data[i - 1].timestamp);
            intervals.push(interval.to_std().unwrap());
        }

        let avg_interval = if !intervals.is_empty() {
            let total_nanos: u64 = intervals.iter().map(|d| d.as_nanos() as u64).sum();
            std::time::Duration::from_nanos(total_nanos / intervals.len() as u64)
        } else {
            std::time::Duration::ZERO
        };

        let min_interval = intervals
            .iter()
            .min()
            .copied()
            .unwrap_or(std::time::Duration::ZERO);
        let max_interval = intervals
            .iter()
            .max()
            .copied()
            .unwrap_or(std::time::Duration::ZERO);

        TimeSeriesStatistics {
            start_time,
            end_time,
            total_duration,
            avg_interval,
            min_interval,
            max_interval,
        }
    }

    /// Check for duplicate timestamps
    pub fn check_duplicate_timestamps(&self, data: &[OHLCV]) -> Vec<ValidationError> {
        let mut seen_timestamps = HashMap::new();
        let mut errors = Vec::new();

        for (i, ohlcv) in data.iter().enumerate() {
            if let Some(prev_index) = seen_timestamps.insert(ohlcv.timestamp, i) {
                errors.push(ValidationError {
                    error_type: "DuplicateTimestamp".to_string(),
                    message: format!("Duplicate timestamp at rows {} and {}", prev_index, i),
                    row_index: Some(i),
                    data_point: Some(ohlcv.clone()),
                });
            }
        }

        errors
    }

    /// Check chronological order
    pub fn check_chronological_order(&self, data: &[OHLCV]) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for i in 1..data.len() {
            if data[i].timestamp <= data[i - 1].timestamp {
                errors.push(ValidationError {
                    error_type: "ChronologicalOrder".to_string(),
                    message: format!("Data not in chronological order at row {}", i),
                    row_index: Some(i),
                    data_point: Some(data[i].clone()),
                });
            }
        }

        errors
    }

    /// Detect outliers using statistical methods
    pub fn detect_outliers(&self, data: &[OHLCV], method: &OutlierMethod) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        match method {
            OutlierMethod::ZScore(threshold) => {
                let prices: Vec<f64> = data
                    .iter()
                    .flat_map(|ohlcv| vec![ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close])
                    .collect();

                let mean = prices.iter().sum::<f64>() / prices.len() as f64;
                let variance = prices
                    .iter()
                    .map(|&price| {
                        let diff = price - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / prices.len() as f64;
                let std_dev = variance.sqrt();

                for (i, ohlcv) in data.iter().enumerate() {
                    let prices = [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close];
                    for &price in prices.iter() {
                        let z_score = (price - mean).abs() / std_dev;
                        if z_score > *threshold {
                            errors.push(ValidationError {
                                error_type: "OutlierDetection".to_string(),
                                message: format!(
                                    "Z-score outlier detected: {} (z-score: {:.2})",
                                    price, z_score
                                ),
                                row_index: Some(i),
                                data_point: Some(ohlcv.clone()),
                            });
                        }
                    }
                }
            }
            OutlierMethod::IQR(multiplier) => {
                let prices: Vec<f64> = data
                    .iter()
                    .flat_map(|ohlcv| vec![ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close])
                    .collect();

                let mut sorted_prices = prices.clone();
                sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let q1_index = sorted_prices.len() / 4;
                let q3_index = 3 * sorted_prices.len() / 4;
                let q1 = sorted_prices[q1_index];
                let q3 = sorted_prices[q3_index];
                let iqr = q3 - q1;

                let lower_bound = q1 - *multiplier * iqr;
                let upper_bound = q3 + *multiplier * iqr;

                for (i, ohlcv) in data.iter().enumerate() {
                    let prices = [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close];
                    for &price in &prices {
                        if price < lower_bound || price > upper_bound {
                            errors.push(ValidationError {
                                error_type: "OutlierDetection".to_string(),
                                message: format!(
                                    "IQR outlier detected: {} (bounds: [{:.2}, {:.2}])",
                                    price, lower_bound, upper_bound
                                ),
                                row_index: Some(i),
                                data_point: Some(ohlcv.clone()),
                            });
                        }
                    }
                }
            }
            OutlierMethod::ModifiedZScore(threshold) => {
                let prices: Vec<f64> = data
                    .iter()
                    .flat_map(|ohlcv| vec![ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close])
                    .collect();

                let mut sorted_prices = prices.clone();
                sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = sorted_prices[sorted_prices.len() / 2];

                let mad = sorted_prices
                    .iter()
                    .map(|&price| (price - median).abs())
                    .fold(0.0, f64::max);

                for (i, ohlcv) in data.iter().enumerate() {
                    let prices = [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close];
                    for &price in &prices {
                        let modified_z_score = 0.6745 * (price - median).abs() / mad;
                        if modified_z_score > *threshold {
                            errors.push(ValidationError {
                                error_type: "OutlierDetection".to_string(),
                                message: format!(
                                    "Modified Z-score outlier detected: {} (score: {:.2})",
                                    price, modified_z_score
                                ),
                                row_index: Some(i),
                                data_point: Some(ohlcv.clone()),
                            });
                        }
                    }
                }
            }
        }

        errors
    }
}

impl Default for DataValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use proptest::prelude::*;

    #[test]
    fn test_data_validator_creation() {
        let validator = DataValidator::new();
        assert!(validator.rules.is_empty());
        assert!(!validator.stop_on_first_error);
        assert_eq!(validator.max_errors, 1000);
    }

    #[test]
    fn test_data_validator_builder_pattern() {
        let validator = DataValidator::new()
            .add_rule(ValidationRule::LogicalConsistency)
            .add_rule(ValidationRule::PriceRange(0.0..1000.0))
            .stop_on_first_error(true)
            .max_errors(100);

        assert_eq!(validator.rules.len(), 2);
        assert!(validator.stop_on_first_error);
        assert_eq!(validator.max_errors, 100);
    }

    #[test]
    fn test_logical_consistency_validation() {
        let validator = DataValidator::new().add_rule(ValidationRule::LogicalConsistency);

        // Valid OHLCV
        let valid_ohlcv = OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000);
        let result = validator.validate(&[valid_ohlcv]);
        assert!(result.is_valid);

        // Invalid OHLCV (high < low)
        let invalid_ohlcv = OHLCV::new(Utc::now(), 100.0, 95.0, 98.0, 102.0, 1000);
        let result = validator.validate(&[invalid_ohlcv]);
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_price_range_validation() {
        let validator = DataValidator::new().add_rule(ValidationRule::PriceRange(0.0..1000.0));

        // Valid price
        let valid_ohlcv = OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000);
        let result = validator.validate(&[valid_ohlcv]);
        assert!(result.is_valid);

        // Invalid price (negative)
        let invalid_ohlcv = OHLCV::new(Utc::now(), -100.0, 105.0, 98.0, 102.0, 1000);
        let result = validator.validate(&[invalid_ohlcv]);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_volume_threshold_validation() {
        let validator = DataValidator::new().add_rule(ValidationRule::VolumeThreshold(100));

        // Valid volume
        let valid_ohlcv = OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000);
        let result = validator.validate(&[valid_ohlcv]);
        assert!(result.is_valid);

        // Invalid volume (below threshold)
        let invalid_ohlcv = OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 50);
        let result = validator.validate(&[invalid_ohlcv]);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_custom_validation_rule() {
        let validator = DataValidator::new().add_rule(ValidationRule::Custom(Box::new(|ohlcv| {
            if ohlcv.close > 200.0 {
                Err("Close price too high".to_string())
            } else {
                Ok(())
            }
        })));

        // Valid OHLCV
        let valid_ohlcv = OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000);
        let result = validator.validate(&[valid_ohlcv]);
        assert!(result.is_valid);

        // Invalid OHLCV (close price too high)
        let invalid_ohlcv = OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 250.0, 1000);
        let result = validator.validate(&[invalid_ohlcv]);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_duplicate_timestamps_detection() {
        let validator = DataValidator::new();
        let timestamp = Utc::now();

        let data = vec![
            OHLCV::new(timestamp, 100.0, 105.0, 98.0, 102.0, 1000),
            OHLCV::new(timestamp, 102.0, 107.0, 100.0, 104.0, 1200), // Duplicate timestamp
        ];

        let errors = validator.check_duplicate_timestamps(&data);
        assert!(!errors.is_empty());
        assert_eq!(errors[0].error_type, "DuplicateTimestamp");
    }

    #[test]
    fn test_chronological_order_check() {
        let validator = DataValidator::new();
        let timestamp1 = Utc::now();
        let timestamp2 = timestamp1 + chrono::Duration::hours(1);

        let data = vec![
            OHLCV::new(timestamp2, 100.0, 105.0, 98.0, 102.0, 1000), // Later timestamp first
            OHLCV::new(timestamp1, 102.0, 107.0, 100.0, 104.0, 1200), // Earlier timestamp second
        ];

        let errors = validator.check_chronological_order(&data);
        assert!(!errors.is_empty());
        assert_eq!(errors[0].error_type, "ChronologicalOrder");
    }

    #[test]
    fn test_outlier_detection() {
        let validator = DataValidator::new();

        // Create data with an extreme outlier
        let data = vec![
            OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000),
            OHLCV::new(Utc::now(), 102.0, 107.0, 100.0, 104.0, 1200),
            OHLCV::new(Utc::now(), 100000.0, 100005.0, 99998.0, 100002.0, 1000), // Very extreme outlier
        ];

        let errors = validator.detect_outliers(&data, &OutlierMethod::ZScore(1.4));
        assert!(!errors.is_empty());
        assert_eq!(errors[0].error_type, "OutlierDetection");
    }

    #[test]
    fn test_validation_statistics() {
        let validator = DataValidator::new();
        let timestamp = Utc::now();

        let data = vec![
            OHLCV::new(timestamp, 100.0, 105.0, 98.0, 102.0, 1000),
            OHLCV::new(
                timestamp + chrono::Duration::hours(1),
                102.0,
                107.0,
                100.0,
                104.0,
                1200,
            ),
        ];

        let stats = validator.calculate_statistics(&data);
        assert_eq!(stats.total_points, 2);
        assert_eq!(stats.price_stats.min_price, 98.0);
        assert_eq!(stats.price_stats.max_price, 107.0);
        assert_eq!(stats.volume_stats.min_volume, 1000);
        assert_eq!(stats.volume_stats.max_volume, 1200);
    }

    proptest! {
        #[test]
        fn test_validation_properties(
            open in 1.0..1000.0f64,
            high in 1.0..1000.0f64,
            low in 1.0..1000.0f64,
            close in 1.0..1000.0f64,
            volume in 1u64..1000000u64
        ) {
            let ohlcv = OHLCV::new(Utc::now(), open, high, low, close, volume);
            let validator = DataValidator::new()
                .add_rule(ValidationRule::LogicalConsistency)
                .add_rule(ValidationRule::PriceRange(0.0..1000.0))
                .add_rule(ValidationRule::VolumeThreshold(1));

            let result = validator.validate(&[ohlcv]);

            // If validation passes, check that data is logically consistent
            if result.is_valid {
                assert!(high >= low);
                assert!(high >= open);
                assert!(high >= close);
                assert!(low <= open);
                assert!(low <= close);
                assert!(open > 0.0);
                assert!(high > 0.0);
                assert!(low > 0.0);
                assert!(close > 0.0);
                assert!(volume > 0);
            }
        }
    }
}
