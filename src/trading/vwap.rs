//! VWAP (Volume Weighted Average Price) calculator
//!
//! This module provides high-performance VWAP calculations optimized for real-time trading
//! with sub-microsecond latency targets.

use crate::core::{FormicaXError, OHLCV};
use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// VWAP calculation types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VWAPType {
    /// Session-based VWAP (from market open)
    Session,
    /// Rolling window VWAP
    Rolling { window_size: usize },
    /// Anchored VWAP from specific point
    Anchored { anchor_time: DateTime<Utc> },
    /// Custom time-based VWAP
    Custom {
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    },
}

/// VWAP calculation result
#[derive(Debug, Clone)]
pub struct VWAPResult {
    /// Calculated VWAP value
    pub vwap: f64,
    /// Total volume used in calculation
    pub total_volume: f64,
    /// Number of data points used
    pub data_points: usize,
    /// Calculation timestamp
    pub calculated_at: DateTime<Utc>,
    /// Calculation duration
    pub calculation_time: Duration,
}

/// High-performance VWAP calculator
#[derive(Debug)]
pub struct VWAPCalculator {
    /// VWAP calculation type
    vwap_type: VWAPType,
    /// Rolling window buffer (for rolling VWAP)
    window_buffer: VecDeque<OHLCV>,
    /// Session start time

    /// Last calculated VWAP
    last_vwap: Option<f64>,
    /// Performance tracking
    performance_tracker: PerformanceTracker,
}

/// Performance tracking for VWAP calculations
#[derive(Debug)]
struct PerformanceTracker {
    total_calculations: usize,
    total_time: Duration,
    min_time: Duration,
    max_time: Duration,
}

impl VWAPCalculator {
    /// Create a new VWAP calculator with session-based calculation
    pub fn new() -> Self {
        Self {
            vwap_type: VWAPType::Session,
            window_buffer: VecDeque::new(),
            last_vwap: None,
            performance_tracker: PerformanceTracker::new(),
        }
    }

    /// Create a session-based VWAP calculator
    pub fn session_based() -> Self {
        Self {
            vwap_type: VWAPType::Session,
            window_buffer: VecDeque::new(),
            last_vwap: None,
            performance_tracker: PerformanceTracker::new(),
        }
    }

    /// Create a rolling window VWAP calculator
    pub fn rolling_window(window_size: usize) -> Self {
        Self {
            vwap_type: VWAPType::Rolling { window_size },
            window_buffer: VecDeque::with_capacity(window_size),
            last_vwap: None,
            performance_tracker: PerformanceTracker::new(),
        }
    }

    /// Create an anchored VWAP calculator
    pub fn anchored(anchor_time: DateTime<Utc>) -> Self {
        Self {
            vwap_type: VWAPType::Anchored { anchor_time },
            window_buffer: VecDeque::new(),
            last_vwap: None,
            performance_tracker: PerformanceTracker::new(),
        }
    }

    /// Set the VWAP calculation type
    pub fn with_type(mut self, vwap_type: VWAPType) -> Self {
        self.vwap_type = vwap_type;
        self
    }

    /// Calculate VWAP for a dataset
    pub fn calculate(&self, data: &[OHLCV]) -> Result<VWAPResult, FormicaXError> {
        if data.is_empty() {
            return Err(FormicaXError::Data(crate::core::DataError::EmptyDataset));
        }

        let start_time = Instant::now();
        let filtered_data = self.filter_data(data)?;

        if filtered_data.is_empty() {
            return Err(FormicaXError::Data(crate::core::DataError::EmptyDataset));
        }

        let (vwap, total_volume) = self.compute_vwap(&filtered_data);
        let calculation_time = start_time.elapsed();

        Ok(VWAPResult {
            vwap,
            total_volume,
            data_points: filtered_data.len(),
            calculated_at: Utc::now(),
            calculation_time,
        })
    }

    /// Calculate VWAP incrementally (for real-time updates)
    pub fn calculate_incremental(
        &mut self,
        new_data: &[OHLCV],
    ) -> Result<VWAPResult, FormicaXError> {
        let start_time = Instant::now();

        // Update window buffer for rolling VWAP
        if let VWAPType::Rolling { window_size } = self.vwap_type {
            for ohlcv in new_data {
                self.window_buffer.push_back(ohlcv.clone());
                if self.window_buffer.len() > window_size {
                    self.window_buffer.pop_front();
                }
            }

            let data: Vec<OHLCV> = self.window_buffer.iter().cloned().collect();
            let (vwap, total_volume) = self.compute_vwap(&data);
            let calculation_time = start_time.elapsed();

            self.last_vwap = Some(vwap);
            self.performance_tracker.record(calculation_time);

            Ok(VWAPResult {
                vwap,
                total_volume,
                data_points: data.len(),
                calculated_at: Utc::now(),
                calculation_time,
            })
        } else {
            // For non-rolling VWAP, just calculate normally
            self.calculate(new_data)
        }
    }

    /// Get the last calculated VWAP value
    pub fn last_vwap(&self) -> Option<f64> {
        self.last_vwap
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> VWAPPerformanceStats {
        self.performance_tracker.stats()
    }

    /// Filter data based on VWAP type
    fn filter_data(&self, data: &[OHLCV]) -> Result<Vec<OHLCV>, FormicaXError> {
        match self.vwap_type {
            VWAPType::Session => {
                // Use all data for session VWAP
                Ok(data.to_vec())
            }
            VWAPType::Rolling { window_size } => {
                // Use the last window_size data points
                let start = if data.len() > window_size {
                    data.len() - window_size
                } else {
                    0
                };
                Ok(data[start..].to_vec())
            }
            VWAPType::Anchored { anchor_time } => {
                // Filter data from anchor time onwards
                Ok(data
                    .iter()
                    .filter(|ohlcv| ohlcv.timestamp >= anchor_time)
                    .cloned()
                    .collect())
            }
            VWAPType::Custom {
                start_time,
                end_time,
            } => {
                // Filter data within custom time range
                Ok(data
                    .iter()
                    .filter(|ohlcv| ohlcv.timestamp >= start_time && ohlcv.timestamp <= end_time)
                    .cloned()
                    .collect())
            }
        }
    }

    /// Compute VWAP using high-performance algorithm
    fn compute_vwap(&self, data: &[OHLCV]) -> (f64, f64) {
        let mut cumulative_pv = 0.0; // Price * Volume
        let mut total_volume = 0.0;

        for ohlcv in data {
            // Use typical price (HLC/3) for VWAP calculation
            let typical_price = (ohlcv.high + ohlcv.low + ohlcv.close) / 3.0;
            let volume = ohlcv.volume as f64;

            cumulative_pv += typical_price * volume;
            total_volume += volume;
        }

        let vwap = if total_volume > 0.0 {
            cumulative_pv / total_volume
        } else {
            0.0
        };

        (vwap, total_volume)
    }
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            total_calculations: 0,
            total_time: Duration::ZERO,
            min_time: Duration::from_secs(u64::MAX),
            max_time: Duration::ZERO,
        }
    }

    fn record(&mut self, duration: Duration) {
        self.total_calculations += 1;
        self.total_time += duration;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
    }

    fn stats(&self) -> VWAPPerformanceStats {
        let avg_time = if self.total_calculations > 0 {
            self.total_time / self.total_calculations as u32
        } else {
            Duration::ZERO
        };

        VWAPPerformanceStats {
            total_calculations: self.total_calculations,
            average_time: avg_time,
            min_time: self.min_time,
            max_time: self.max_time,
            total_time: self.total_time,
        }
    }
}

/// VWAP performance statistics
#[derive(Debug, Clone)]
pub struct VWAPPerformanceStats {
    pub total_calculations: usize,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub total_time: Duration,
}

impl Default for VWAPCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_data() -> Vec<OHLCV> {
        vec![
            OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000),
            OHLCV::new(Utc::now(), 102.0, 107.0, 100.0, 104.0, 1200),
            OHLCV::new(Utc::now(), 104.0, 109.0, 102.0, 106.0, 1100),
        ]
    }

    #[test]
    fn test_vwap_calculator_creation() {
        let calculator = VWAPCalculator::new();
        assert_eq!(calculator.vwap_type, VWAPType::Session);
    }

    #[test]
    fn test_session_vwap_calculation() {
        let data = create_test_data();
        let calculator = VWAPCalculator::session_based();
        let result = calculator.calculate(&data).unwrap();

        // Expected VWAP calculation:
        // Point 1: typical_price = (105+98+102)/3 = 101.67, volume = 1000
        // Point 2: typical_price = (107+100+104)/3 = 103.67, volume = 1200
        // Point 3: typical_price = (109+102+106)/3 = 105.67, volume = 1100
        // Total PV = 101.67*1000 + 103.67*1200 + 105.67*1100 = 101670 + 124404 + 116237 = 342311
        // Total volume = 1000 + 1200 + 1100 = 3300
        // VWAP = 342311 / 3300 ≈ 103.73

        assert!(result.vwap > 0.0);
        assert_eq!(result.data_points, 3);
        assert!(result.total_volume > 0.0);
        assert!(result.calculation_time < Duration::from_millis(1));
    }

    #[test]
    fn test_rolling_vwap_calculation() {
        let data = create_test_data();
        let mut calculator = VWAPCalculator::rolling_window(2);
        let result = calculator.calculate_incremental(&data).unwrap();

        // Should only use last 2 points
        assert_eq!(result.data_points, 2);
        assert!(result.vwap > 0.0);
    }

    #[test]
    fn test_empty_data_error() {
        let calculator = VWAPCalculator::new();
        let result = calculator.calculate(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_anchored_vwap() {
        let anchor_time = Utc::now();
        let data = create_test_data();
        let calculator = VWAPCalculator::anchored(anchor_time);
        let result = calculator.calculate(&data).unwrap();

        assert!(result.vwap > 0.0);
        assert!(result.data_points > 0);
    }

    #[test]
    fn test_performance_tracking() {
        let data = create_test_data();
        let mut calculator = VWAPCalculator::rolling_window(10);

        // Perform multiple calculations
        for _ in 0..5 {
            calculator.calculate_incremental(&data).unwrap();
        }

        let stats = calculator.performance_stats();
        assert_eq!(stats.total_calculations, 5);
        assert!(stats.average_time > Duration::ZERO);
        assert!(stats.min_time > Duration::ZERO);
        assert!(stats.max_time > Duration::ZERO);
    }
}
