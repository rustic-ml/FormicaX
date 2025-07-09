//! Trading strategies implementation
//!
//! This module provides high-performance trading strategies optimized for real-time trading
//! with configurable parameters and performance monitoring.

use crate::core::{FormicaXError, OHLCV};
use crate::trading::performance::{PerformanceConfig, PerformanceMonitor};
use crate::trading::signals::{SignalGenerator, SignalThresholds, TradingSignal};
use crate::trading::vwap::{VWAPCalculator, VWAPType};
use chrono::{DateTime, Utc};
use std::time::{Duration, Instant};

/// Trading strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Strategy name
    pub name: String,
    /// VWAP calculation type
    pub vwap_type: VWAPType,
    /// Signal thresholds
    pub signal_thresholds: SignalThresholds,
    /// Performance monitoring configuration
    pub performance_config: PerformanceConfig,
    /// Enable real-time processing
    pub real_time: bool,
    /// Maximum position size
    pub max_position_size: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
}

/// Trading strategy trait
pub trait TradingStrategy {
    /// Strategy configuration type
    type Config;

    /// Execute strategy on market data
    fn execute(&mut self, data: &[OHLCV]) -> Result<Vec<TradingSignal>, FormicaXError>;

    /// Execute strategy incrementally (for real-time)
    fn execute_incremental(
        &mut self,
        ohlcv: &OHLCV,
    ) -> Result<Option<TradingSignal>, FormicaXError>;

    /// Get strategy performance metrics
    fn get_performance(&self) -> &PerformanceMonitor;

    /// Validate strategy configuration
    fn validate_config(&self) -> Result<(), FormicaXError>;
}

/// VWAP-based trading strategy
#[derive(Debug)]
pub struct VWAPStrategy {
    /// Strategy configuration
    config: StrategyConfig,
    /// VWAP calculator
    vwap_calculator: VWAPCalculator,
    /// Signal generator
    signal_generator: SignalGenerator,
    /// Performance monitor
    performance_monitor: PerformanceMonitor,
    /// Current position
    current_position: Option<Position>,
    /// Strategy state
    state: StrategyState,
}

/// Trading position
#[derive(Debug, Clone)]
pub struct Position {
    /// Position type
    pub position_type: PositionType,
    /// Entry price
    pub entry_price: f64,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Position size
    pub size: f64,
    /// Stop loss price
    pub stop_loss: f64,
    /// Take profit price
    pub take_profit: f64,
}

/// Position type
#[derive(Debug, Clone)]
pub enum PositionType {
    /// Long position
    Long,
    /// Short position
    Short,
}

/// Strategy state
#[derive(Debug)]
struct StrategyState {
    /// Last signal timestamp
    last_signal_time: Option<DateTime<Utc>>,
    /// Signal cooldown period
    signal_cooldown: Duration,
    /// Strategy start time

    /// Total signals generated
    total_signals: usize,
}

impl VWAPStrategy {
    /// Create a new VWAP strategy with default configuration
    pub fn new() -> Self {
        Self {
            config: StrategyConfig::default(),
            vwap_calculator: VWAPCalculator::session_based(),
            signal_generator: SignalGenerator::new(),
            performance_monitor: PerformanceMonitor::new(),
            current_position: None,
            state: StrategyState::new(),
        }
    }

    /// Create a VWAP strategy with custom configuration
    pub fn with_config(config: StrategyConfig) -> Self {
        let vwap_calculator = VWAPCalculator::new().with_type(config.vwap_type);
        let signal_generator = SignalGenerator::with_thresholds(config.signal_thresholds.clone());
        let performance_monitor =
            PerformanceMonitor::with_config(config.performance_config.clone());

        Self {
            config,
            vwap_calculator,
            signal_generator,
            performance_monitor,
            current_position: None,
            state: StrategyState::new(),
        }
    }

    /// Set strategy name
    pub fn with_name(mut self, name: String) -> Self {
        self.config.name = name;
        self
    }

    /// Set VWAP type
    pub fn with_vwap_type(mut self, vwap_type: VWAPType) -> Self {
        self.config.vwap_type = vwap_type;
        self.vwap_calculator = VWAPCalculator::new().with_type(vwap_type);
        self
    }

    /// Set signal thresholds
    pub fn with_signal_thresholds(mut self, thresholds: SignalThresholds) -> Self {
        self.config.signal_thresholds = thresholds.clone();
        self.signal_generator = SignalGenerator::with_thresholds(thresholds);
        self
    }

    /// Get current position
    pub fn get_position(&self) -> Option<&Position> {
        self.current_position.as_ref()
    }

    /// Check if position should be closed
    pub fn should_close_position(&self, current_price: f64) -> bool {
        if let Some(position) = &self.current_position {
            match position.position_type {
                PositionType::Long => {
                    current_price <= position.stop_loss || current_price >= position.take_profit
                }
                PositionType::Short => {
                    current_price >= position.stop_loss || current_price <= position.take_profit
                }
            }
        } else {
            false
        }
    }

    /// Update position with current market data
    pub fn update_position(
        &mut self,
        ohlcv: &OHLCV,
    ) -> Result<Option<TradingSignal>, FormicaXError> {
        if let Some(_position) = &mut self.current_position {
            let current_price = ohlcv.close;

            if self.should_close_position(current_price) {
                // Generate exit signal
                let exit_signal = TradingSignal {
                    signal_type: crate::trading::signals::SignalType::Exit {
                        reason: "Stop loss or take profit hit".to_string(),
                    },
                    timestamp: ohlcv.timestamp,
                    price: current_price,
                    volume: ohlcv.volume,
                    vwap: None,
                    confidence: 1.0,
                    generation_time: Duration::ZERO,
                };

                // Record exit
                self.performance_monitor
                    .record_trade_exit(current_price, ohlcv.timestamp)?;
                self.current_position = None;

                Ok(Some(exit_signal))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
}

impl TradingStrategy for VWAPStrategy {
    type Config = StrategyConfig;

    fn execute(&mut self, data: &[OHLCV]) -> Result<Vec<TradingSignal>, FormicaXError> {
        if data.is_empty() {
            return Err(FormicaXError::Data(crate::core::DataError::EmptyDataset));
        }

        let start_time = Instant::now();
        let mut signals = Vec::new();

        // Generate initial signal
        let signal = self.signal_generator.generate_signal(data)?;
        signals.push(signal.clone());

        // Record signal
        self.performance_monitor.record_signal(&signal)?;
        self.state.total_signals += 1;
        self.state.last_signal_time = Some(signal.timestamp);

        // Process signal and potentially open position
        self.process_signal(&signal, data.last().unwrap())?;

        let execution_time = start_time.elapsed();

        // Log execution time if it exceeds threshold
        if execution_time > Duration::from_millis(10) {
            eprintln!("Warning: Strategy execution took {:?}", execution_time);
        }

        Ok(signals)
    }

    fn execute_incremental(
        &mut self,
        ohlcv: &OHLCV,
    ) -> Result<Option<TradingSignal>, FormicaXError> {
        let start_time = Instant::now();

        // Check if we should close existing position
        if let Some(exit_signal) = self.update_position(ohlcv)? {
            return Ok(Some(exit_signal));
        }

        // Check signal cooldown
        if let Some(last_signal) = self.state.last_signal_time {
            let time_since_last = ohlcv
                .timestamp
                .signed_duration_since(last_signal)
                .to_std()
                .unwrap_or(Duration::ZERO);
            if time_since_last < self.state.signal_cooldown {
                return Ok(None);
            }
        }

        // Generate new signal
        let signal = self.signal_generator.generate_signal_incremental(ohlcv)?;

        // Record signal
        self.performance_monitor.record_signal(&signal)?;
        self.state.total_signals += 1;
        self.state.last_signal_time = Some(signal.timestamp);

        // Process signal
        self.process_signal(&signal, ohlcv)?;

        let execution_time = start_time.elapsed();

        // Log execution time if it exceeds threshold
        if execution_time > Duration::from_millis(1) {
            eprintln!("Warning: Incremental execution took {:?}", execution_time);
        }

        Ok(Some(signal))
    }

    fn get_performance(&self) -> &PerformanceMonitor {
        &self.performance_monitor
    }

    fn validate_config(&self) -> Result<(), FormicaXError> {
        if self.config.name.is_empty() {
            return Err(FormicaXError::Config(
                crate::core::ConfigError::InvalidValue {
                    field: "name".to_string(),
                    message: "Strategy name cannot be empty".to_string(),
                },
            ));
        }

        if self.config.max_position_size <= 0.0 {
            return Err(FormicaXError::Config(
                crate::core::ConfigError::InvalidValue {
                    field: "max_position_size".to_string(),
                    message: "Max position size must be positive".to_string(),
                },
            ));
        }

        if self.config.stop_loss_pct <= 0.0 || self.config.take_profit_pct <= 0.0 {
            return Err(FormicaXError::Config(
                crate::core::ConfigError::InvalidValue {
                    field: "stop_loss_pct/take_profit_pct".to_string(),
                    message: "Stop loss and take profit percentages must be positive".to_string(),
                },
            ));
        }

        Ok(())
    }
}

impl VWAPStrategy {
    /// Process trading signal and potentially open position
    fn process_signal(
        &mut self,
        signal: &TradingSignal,
        ohlcv: &OHLCV,
    ) -> Result<(), FormicaXError> {
        // Only process if we don't have an open position
        if self.current_position.is_none() {
            match &signal.signal_type {
                crate::trading::signals::SignalType::Buy { strength, .. } => {
                    if signal.confidence >= self.config.signal_thresholds.min_confidence {
                        self.open_position(
                            PositionType::Long,
                            ohlcv.close,
                            ohlcv.timestamp,
                            *strength,
                        )?;
                    }
                }
                crate::trading::signals::SignalType::Sell { strength, .. } => {
                    if signal.confidence >= self.config.signal_thresholds.min_confidence {
                        self.open_position(
                            PositionType::Short,
                            ohlcv.close,
                            ohlcv.timestamp,
                            *strength,
                        )?;
                    }
                }
                _ => {
                    // Hold or Exit signals - no action needed
                }
            }
        }

        Ok(())
    }

    /// Open a new trading position
    fn open_position(
        &mut self,
        position_type: PositionType,
        price: f64,
        timestamp: DateTime<Utc>,
        strength: f64,
    ) -> Result<(), FormicaXError> {
        let position_size = self.config.max_position_size * strength.min(1.0);

        let (stop_loss, take_profit) = match position_type {
            PositionType::Long => {
                let stop_loss = price * (1.0 - self.config.stop_loss_pct);
                let take_profit = price * (1.0 + self.config.take_profit_pct);
                (stop_loss, take_profit)
            }
            PositionType::Short => {
                let stop_loss = price * (1.0 + self.config.stop_loss_pct);
                let take_profit = price * (1.0 - self.config.take_profit_pct);
                (stop_loss, take_profit)
            }
        };

        self.current_position = Some(Position {
            position_type,
            entry_price: price,
            entry_time: timestamp,
            size: position_size,
            stop_loss,
            take_profit,
        });

        Ok(())
    }
}

impl StrategyState {
    fn new() -> Self {
        Self {
            last_signal_time: None,
            signal_cooldown: Duration::from_secs(60), // 1 minute cooldown
            total_signals: 0,
        }
    }
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            name: "VWAP Strategy".to_string(),
            vwap_type: VWAPType::Session,
            signal_thresholds: SignalThresholds::default(),
            performance_config: PerformanceConfig::default(),
            real_time: true,
            max_position_size: 10000.0,
            stop_loss_pct: 0.02,   // 2% stop loss
            take_profit_pct: 0.04, // 4% take profit
        }
    }
}

impl Default for VWAPStrategy {
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
    fn test_vwap_strategy_creation() {
        let strategy = VWAPStrategy::new();
        assert_eq!(strategy.config.name, "VWAP Strategy");
    }

    #[test]
    fn test_strategy_execution() {
        let data = create_test_data();
        let mut strategy = VWAPStrategy::new();
        let signals = strategy.execute(&data).unwrap();

        assert!(!signals.is_empty());
        assert!(strategy.state.total_signals > 0);
    }

    #[test]
    fn test_incremental_execution() {
        let data = create_test_data();
        let mut strategy = VWAPStrategy::new();

        for ohlcv in &data {
            let _signal = strategy.execute_incremental(ohlcv).unwrap();
            // Signal might be None due to cooldown
        }

        assert!(strategy.state.total_signals > 0);
    }

    #[test]
    fn test_position_management() {
        let mut strategy = VWAPStrategy::new();
        let ohlcv = OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000);

        // Initially no position
        assert!(strategy.get_position().is_none());

        // Create a buy signal
        let signal = TradingSignal {
            signal_type: crate::trading::signals::SignalType::Buy {
                strength: 0.8,
                reason: "Test buy".to_string(),
            },
            timestamp: ohlcv.timestamp,
            price: ohlcv.close,
            volume: ohlcv.volume,
            vwap: Some(100.0),
            confidence: 0.8,
            generation_time: Duration::ZERO,
        };

        strategy.process_signal(&signal, &ohlcv).unwrap();

        // Should now have a position
        assert!(strategy.get_position().is_some());
    }

    #[test]
    fn test_config_validation() {
        let mut strategy = VWAPStrategy::new();

        // Valid config should pass
        assert!(strategy.validate_config().is_ok());

        // Invalid config
        strategy.config.name = "".to_string();
        assert!(strategy.validate_config().is_err());
    }

    #[test]
    fn test_performance_monitoring() {
        let data = create_test_data();
        let mut strategy = VWAPStrategy::new();

        strategy.execute(&data).unwrap();

        let performance = strategy.get_performance();
        let _metrics = performance.get_metrics();
        // Performance monitoring is working
    }
}
