//! Trading signal generation
//!
//! This module provides high-performance trading signal generation optimized for real-time trading
//! with sub-100 microsecond latency targets.

use crate::core::{FormicaXError, OHLCV};
use crate::trading::vwap::{VWAPCalculator, VWAPResult};
use chrono::{DateTime, Utc};
use std::time::{Duration, Instant};

/// Trading signal types
#[derive(Debug, Clone, PartialEq)]
pub enum SignalType {
    /// Buy signal
    Buy { strength: f64, reason: String },
    /// Sell signal
    Sell { strength: f64, reason: String },
    /// Hold signal (no action)
    Hold { reason: String },
    /// Exit signal (close position)
    Exit { reason: String },
}

/// Trading signal with metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal type
    pub signal_type: SignalType,
    /// Timestamp when signal was generated
    pub timestamp: DateTime<Utc>,
    /// Price at signal generation
    pub price: f64,
    /// Volume at signal generation
    pub volume: u64,
    /// VWAP at signal generation
    pub vwap: Option<f64>,
    /// Signal confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Generation latency
    pub generation_time: Duration,
}

/// High-performance trading signal generator
#[derive(Debug)]
pub struct SignalGenerator {
    /// VWAP calculator
    vwap_calculator: VWAPCalculator,
    /// Signal thresholds
    thresholds: SignalThresholds,
    /// Performance tracking
    performance_tracker: SignalPerformanceTracker,
}

/// Signal generation thresholds
#[derive(Debug, Clone)]
pub struct SignalThresholds {
    /// VWAP deviation threshold for buy signals
    pub vwap_buy_threshold: f64,
    /// VWAP deviation threshold for sell signals
    pub vwap_sell_threshold: f64,
    /// Volume spike threshold
    pub volume_threshold: f64,
    /// Price change threshold
    pub price_change_threshold: f64,
    /// Minimum confidence for actionable signals
    pub min_confidence: f64,
}

/// Performance tracking for signal generation
#[derive(Debug)]
struct SignalPerformanceTracker {
    total_signals: usize,
    total_time: Duration,
    min_time: Duration,
    max_time: Duration,
    buy_signals: usize,
    sell_signals: usize,
    hold_signals: usize,
}

impl SignalGenerator {
    /// Create a new signal generator with default thresholds
    pub fn new() -> Self {
        Self {
            vwap_calculator: VWAPCalculator::session_based(),
            thresholds: SignalThresholds::default(),
            performance_tracker: SignalPerformanceTracker::new(),
        }
    }

    /// Create a signal generator with custom thresholds
    pub fn with_thresholds(thresholds: SignalThresholds) -> Self {
        Self {
            vwap_calculator: VWAPCalculator::session_based(),
            thresholds,
            performance_tracker: SignalPerformanceTracker::new(),
        }
    }

    /// Set VWAP calculator
    pub fn with_vwap_calculator(mut self, calculator: VWAPCalculator) -> Self {
        self.vwap_calculator = calculator;
        self
    }

    /// Generate trading signal from OHLCV data
    pub fn generate_signal(&mut self, data: &[OHLCV]) -> Result<TradingSignal, FormicaXError> {
        if data.is_empty() {
            return Err(FormicaXError::Data(crate::core::DataError::EmptyDataset));
        }

        let start_time = Instant::now();
        let latest = &data[data.len() - 1];

        // Calculate VWAP
        let vwap_result = self.vwap_calculator.calculate(data)?;
        let vwap = vwap_result.vwap;

        // Generate signal based on VWAP deviation
        let signal_type = self.analyze_vwap_deviation(latest, vwap);

        // Calculate confidence based on multiple factors
        let confidence = self.calculate_confidence(latest, &vwap_result, data);

        let generation_time = start_time.elapsed();

        // Update performance tracking
        self.performance_tracker
            .record(signal_type.clone(), generation_time);

        Ok(TradingSignal {
            signal_type,
            timestamp: latest.timestamp,
            price: latest.close,
            volume: latest.volume,
            vwap: Some(vwap),
            confidence,
            generation_time,
        })
    }

    /// Generate signal from single OHLCV point (for real-time)
    pub fn generate_signal_incremental(
        &mut self,
        ohlcv: &OHLCV,
    ) -> Result<TradingSignal, FormicaXError> {
        let start_time = Instant::now();

        // Update VWAP calculator incrementally
        let vwap_result = self
            .vwap_calculator
            .calculate_incremental(&[ohlcv.clone()])?;
        let vwap = vwap_result.vwap;

        // Generate signal
        let signal_type = self.analyze_vwap_deviation(ohlcv, vwap);
        let confidence = self.calculate_confidence_single(ohlcv, vwap);

        let generation_time = start_time.elapsed();

        // Update performance tracking
        self.performance_tracker
            .record(signal_type.clone(), generation_time);

        Ok(TradingSignal {
            signal_type,
            timestamp: ohlcv.timestamp,
            price: ohlcv.close,
            volume: ohlcv.volume,
            vwap: Some(vwap),
            confidence,
            generation_time,
        })
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> SignalPerformanceStats {
        self.performance_tracker.stats()
    }

    /// Analyze VWAP deviation to determine signal type
    fn analyze_vwap_deviation(&self, ohlcv: &OHLCV, vwap: f64) -> SignalType {
        let price = ohlcv.close;
        let deviation = (price - vwap) / vwap;

        if deviation > self.thresholds.vwap_buy_threshold {
            SignalType::Buy {
                strength: deviation.abs(),
                reason: format!("Price {:.2}% above VWAP", deviation * 100.0),
            }
        } else if deviation < -self.thresholds.vwap_sell_threshold {
            SignalType::Sell {
                strength: deviation.abs(),
                reason: format!("Price {:.2}% below VWAP", deviation.abs() * 100.0),
            }
        } else {
            SignalType::Hold {
                reason: "Price within VWAP threshold".to_string(),
            }
        }
    }

    /// Calculate signal confidence based on multiple factors
    fn calculate_confidence(&self, ohlcv: &OHLCV, vwap_result: &VWAPResult, data: &[OHLCV]) -> f64 {
        let mut confidence = 0.5; // Base confidence

        // VWAP deviation factor
        let deviation = ((ohlcv.close - vwap_result.vwap) / vwap_result.vwap).abs();
        confidence += (deviation * 10.0).min(0.3); // Max 0.3 from deviation

        // Volume factor
        if data.len() > 1 {
            let avg_volume: f64 =
                data.iter().map(|d| d.volume as f64).sum::<f64>() / data.len() as f64;
            let volume_ratio = ohlcv.volume as f64 / avg_volume;
            if volume_ratio > self.thresholds.volume_threshold {
                confidence += 0.2; // Volume spike bonus
            }
        }

        // Price momentum factor
        if data.len() > 1 {
            let prev_close = data[data.len() - 2].close;
            let price_change = (ohlcv.close - prev_close) / prev_close;
            if price_change.abs() > self.thresholds.price_change_threshold {
                confidence += 0.1; // Price momentum bonus
            }
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Calculate confidence for single OHLCV point
    fn calculate_confidence_single(&self, ohlcv: &OHLCV, vwap: f64) -> f64 {
        let deviation = ((ohlcv.close - vwap) / vwap).abs();
        let confidence = 0.5 + (deviation * 10.0).min(0.4);
        confidence.clamp(0.0, 1.0)
    }
}

impl Default for SignalThresholds {
    fn default() -> Self {
        Self {
            vwap_buy_threshold: 0.001,     // 0.1% above VWAP
            vwap_sell_threshold: 0.001,    // 0.1% below VWAP
            volume_threshold: 1.5,         // 50% above average volume
            price_change_threshold: 0.005, // 0.5% price change
            min_confidence: 0.6,           // 60% minimum confidence
        }
    }
}

impl SignalPerformanceTracker {
    fn new() -> Self {
        Self {
            total_signals: 0,
            total_time: Duration::ZERO,
            min_time: Duration::from_secs(u64::MAX),
            max_time: Duration::ZERO,
            buy_signals: 0,
            sell_signals: 0,
            hold_signals: 0,
        }
    }

    fn record(&mut self, signal_type: SignalType, duration: Duration) {
        self.total_signals += 1;
        self.total_time += duration;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);

        match signal_type {
            SignalType::Buy { .. } => self.buy_signals += 1,
            SignalType::Sell { .. } => self.sell_signals += 1,
            SignalType::Hold { .. } => self.hold_signals += 1,
            SignalType::Exit { .. } => self.sell_signals += 1,
        }
    }

    fn stats(&self) -> SignalPerformanceStats {
        let avg_time = if self.total_signals > 0 {
            self.total_time / self.total_signals as u32
        } else {
            Duration::ZERO
        };

        SignalPerformanceStats {
            total_signals: self.total_signals,
            average_time: avg_time,
            min_time: self.min_time,
            max_time: self.max_time,
            total_time: self.total_time,
            buy_signals: self.buy_signals,
            sell_signals: self.sell_signals,
            hold_signals: self.hold_signals,
        }
    }
}

/// Signal generation performance statistics
#[derive(Debug, Clone)]
pub struct SignalPerformanceStats {
    pub total_signals: usize,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub total_time: Duration,
    pub buy_signals: usize,
    pub sell_signals: usize,
    pub hold_signals: usize,
}

impl Default for SignalGenerator {
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
    fn test_signal_generator_creation() {
        let generator = SignalGenerator::new();
        assert_eq!(generator.thresholds.vwap_buy_threshold, 0.001);
    }

    #[test]
    fn test_signal_generation() {
        let data = create_test_data();
        let mut generator = SignalGenerator::new();
        let signal = generator.generate_signal(&data).unwrap();

        assert!(signal.confidence > 0.0);
        assert!(signal.confidence <= 1.0);
        assert!(signal.generation_time < Duration::from_millis(1));
        assert!(signal.vwap.is_some());
    }

    #[test]
    fn test_buy_signal_generation() {
        let mut data = create_test_data();
        // Create a strong buy signal by making price well above VWAP
        data.push(OHLCV::new(Utc::now(), 110.0, 115.0, 108.0, 112.0, 1500));

        let mut generator = SignalGenerator::with_thresholds(SignalThresholds {
            vwap_buy_threshold: 0.01, // 1% threshold
            ..Default::default()
        });

        let signal = generator.generate_signal(&data).unwrap();

        match signal.signal_type {
            SignalType::Buy { .. } => {
                // Expected buy signal
            }
            _ => panic!("Expected buy signal, got {:?}", signal.signal_type),
        }
    }

    #[test]
    fn test_sell_signal_generation() {
        let mut data = create_test_data();
        // Create a strong sell signal by making price well below VWAP
        data.push(OHLCV::new(Utc::now(), 90.0, 95.0, 88.0, 92.0, 1500));

        let mut generator = SignalGenerator::with_thresholds(SignalThresholds {
            vwap_sell_threshold: 0.01, // 1% threshold
            ..Default::default()
        });

        let signal = generator.generate_signal(&data).unwrap();

        match signal.signal_type {
            SignalType::Sell { .. } => {
                // Expected sell signal
            }
            _ => panic!("Expected sell signal, got {:?}", signal.signal_type),
        }
    }

    #[test]
    fn test_incremental_signal_generation() {
        let data = create_test_data();
        let mut generator = SignalGenerator::new();

        for ohlcv in &data {
            let signal = generator.generate_signal_incremental(ohlcv).unwrap();
            assert!(signal.confidence > 0.0);
            assert!(signal.confidence <= 1.0);
        }
    }

    #[test]
    fn test_performance_tracking() {
        let data = create_test_data();
        let mut generator = SignalGenerator::new();

        // Generate multiple signals
        for _ in 0..5 {
            generator.generate_signal(&data).unwrap();
        }

        let stats = generator.performance_stats();
        assert_eq!(stats.total_signals, 5);
        assert!(stats.average_time > Duration::ZERO);
        assert!(stats.min_time > Duration::ZERO);
        assert!(stats.max_time > Duration::ZERO);
    }

    #[test]
    fn test_empty_data_error() {
        let mut generator = SignalGenerator::new();
        let result = generator.generate_signal(&[]);
        assert!(result.is_err());
    }
}
