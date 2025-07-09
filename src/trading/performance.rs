//! Performance monitoring for trading systems
//!
//! This module provides high-performance monitoring and metrics collection for trading systems
//! with real-time performance tracking and alerting capabilities.

use crate::core::FormicaXError;
use crate::trading::signals::{SignalType, TradingSignal};
use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Trading performance metrics
#[derive(Debug, Clone)]
pub struct TradingMetrics {
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Win rate (0.0 to 1.0)
    pub win_rate: f64,
    /// Total profit/loss
    pub total_pnl: f64,
    /// Average trade P&L
    pub avg_trade_pnl: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Current drawdown
    pub current_drawdown: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Average signal generation time
    pub avg_signal_time: Duration,
    /// Average VWAP calculation time
    pub avg_vwap_time: Duration,
    /// Peak equity
    pub peak_equity: f64,
    /// Current equity
    pub current_equity: f64,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

/// Performance alert types
#[derive(Debug, Clone)]
pub enum PerformanceAlert {
    /// Drawdown exceeded threshold
    DrawdownExceeded { current: f64, threshold: f64 },
    /// Win rate below threshold
    WinRateLow { current: f64, threshold: f64 },
    /// P&L below threshold
    PnLLow { current: f64, threshold: f64 },
    /// Latency exceeded threshold
    LatencyHigh {
        current: Duration,
        threshold: Duration,
    },
    /// Signal generation failed
    SignalGenerationFailed { error: String },
}

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Drawdown alert threshold
    pub drawdown_threshold: f64,
    /// Win rate alert threshold
    pub win_rate_threshold: f64,
    /// P&L alert threshold
    pub pnl_threshold: f64,
    /// Latency alert threshold
    pub latency_threshold: Duration,
    /// Maximum history size
    pub max_history_size: usize,
    /// Enable real-time monitoring
    pub real_time_monitoring: bool,
}

/// High-performance trading performance monitor
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Performance configuration
    config: PerformanceConfig,
    /// Trading metrics
    metrics: TradingMetrics,
    /// Trade history
    trade_history: VecDeque<TradeRecord>,
    /// Performance alerts
    alerts: VecDeque<PerformanceAlert>,
    /// Performance tracking
    performance_tracker: MonitorPerformanceTracker,
}

/// Individual trade record
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Trade timestamp
    pub timestamp: DateTime<Utc>,
    /// Signal type
    pub signal_type: SignalType,
    /// Entry price
    pub entry_price: f64,
    /// Exit price (if closed)
    pub exit_price: Option<f64>,
    /// Trade P&L
    pub pnl: Option<f64>,
    /// Trade duration
    pub duration: Option<Duration>,
    /// Signal generation time
    pub signal_time: Duration,
    /// VWAP calculation time
    pub vwap_time: Duration,
}

/// Performance tracking for the monitor itself
#[derive(Debug)]
struct MonitorPerformanceTracker {
    total_updates: usize,
    total_time: Duration,
    min_time: Duration,
    max_time: Duration,
}

impl PerformanceMonitor {
    /// Create a new performance monitor with default configuration
    pub fn new() -> Self {
        Self {
            config: PerformanceConfig::default(),
            metrics: TradingMetrics::default(),
            trade_history: VecDeque::new(),
            alerts: VecDeque::new(),
            performance_tracker: MonitorPerformanceTracker::new(),
        }
    }

    /// Create a performance monitor with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        Self {
            config,
            metrics: TradingMetrics::default(),
            trade_history: VecDeque::new(),
            alerts: VecDeque::new(),
            performance_tracker: MonitorPerformanceTracker::new(),
        }
    }

    /// Record a new trading signal
    pub fn record_signal(&mut self, signal: &TradingSignal) -> Result<(), FormicaXError> {
        let start_time = Instant::now();

        // Create trade record
        let trade_record = TradeRecord {
            timestamp: signal.timestamp,
            signal_type: signal.signal_type.clone(),
            entry_price: signal.price,
            exit_price: None,
            pnl: None,
            duration: None,
            signal_time: signal.generation_time,
            vwap_time: Duration::ZERO, // Will be updated when VWAP time is available
        };

        // Add to history
        self.trade_history.push_back(trade_record);
        if self.trade_history.len() > self.config.max_history_size {
            self.trade_history.pop_front();
        }

        // Update metrics
        self.update_metrics();

        // Check for alerts
        self.check_alerts();

        let update_time = start_time.elapsed();
        self.performance_tracker.record(update_time);

        Ok(())
    }

    /// Record trade exit
    pub fn record_trade_exit(
        &mut self,
        exit_price: f64,
        exit_time: DateTime<Utc>,
    ) -> Result<(), FormicaXError> {
        if let Some(last_trade) = self.trade_history.back_mut() {
            last_trade.exit_price = Some(exit_price);
            last_trade.pnl = Some(exit_price - last_trade.entry_price);
            last_trade.duration = Some(
                exit_time
                    .signed_duration_since(last_trade.timestamp)
                    .to_std()
                    .unwrap_or(Duration::ZERO),
            );

            // Update metrics
            self.update_metrics();

            // Check for alerts
            self.check_alerts();
        }

        Ok(())
    }

    /// Get current trading metrics
    pub fn get_metrics(&self) -> &TradingMetrics {
        &self.metrics
    }

    /// Get recent alerts
    pub fn get_alerts(&self, count: usize) -> Vec<PerformanceAlert> {
        self.alerts.iter().rev().take(count).cloned().collect()
    }

    /// Get trade history
    pub fn get_trade_history(&self, count: usize) -> Vec<TradeRecord> {
        self.trade_history
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }

    /// Get monitor performance statistics
    pub fn get_monitor_stats(&self) -> MonitorStats {
        self.performance_tracker.stats()
    }

    /// Update trading metrics
    fn update_metrics(&mut self) {
        let mut total_pnl = 0.0;
        let mut winning_trades = 0;
        let mut losing_trades = 0;
        let mut total_signal_time = Duration::ZERO;
        let mut total_vwap_time = Duration::ZERO;
        let mut signal_count = 0;
        let mut vwap_count = 0;

        for trade in &self.trade_history {
            if let Some(pnl) = trade.pnl {
                total_pnl += pnl;
                if pnl > 0.0 {
                    winning_trades += 1;
                } else if pnl < 0.0 {
                    losing_trades += 1;
                }
            }

            total_signal_time += trade.signal_time;
            signal_count += 1;

            if trade.vwap_time > Duration::ZERO {
                total_vwap_time += trade.vwap_time;
                vwap_count += 1;
            }
        }

        let total_trades = winning_trades + losing_trades;
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let avg_trade_pnl = if total_trades > 0 {
            total_pnl / total_trades as f64
        } else {
            0.0
        };

        let avg_signal_time = if signal_count > 0 {
            total_signal_time / signal_count as u32
        } else {
            Duration::ZERO
        };

        let avg_vwap_time = if vwap_count > 0 {
            total_vwap_time / vwap_count as u32
        } else {
            Duration::ZERO
        };

        // Calculate drawdown
        let (max_drawdown, current_drawdown, peak_equity, current_equity) =
            self.calculate_drawdown(total_pnl);

        // Calculate risk-adjusted returns
        let sharpe_ratio = self.calculate_sharpe_ratio();
        let sortino_ratio = self.calculate_sortino_ratio();

        self.metrics = TradingMetrics {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            total_pnl,
            avg_trade_pnl,
            max_drawdown,
            current_drawdown,
            sharpe_ratio,
            sortino_ratio,
            avg_signal_time,
            avg_vwap_time,
            peak_equity,
            current_equity,
            last_update: Utc::now(),
        };
    }

    /// Calculate drawdown metrics
    fn calculate_drawdown(&self, current_pnl: f64) -> (f64, f64, f64, f64) {
        let mut peak_equity = 0.0;
        let mut max_drawdown = 0.0;
        let mut running_pnl = 0.0;

        for trade in &self.trade_history {
            if let Some(pnl) = trade.pnl {
                running_pnl += pnl;
                if running_pnl > peak_equity {
                    peak_equity = running_pnl;
                }

                let drawdown = (peak_equity - running_pnl) / peak_equity.max(1.0);
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
            }
        }

        let current_equity = current_pnl;
        let current_drawdown = if peak_equity > 0.0 {
            (peak_equity - current_equity) / peak_equity
        } else {
            0.0
        };

        (max_drawdown, current_drawdown, peak_equity, current_equity)
    }

    /// Calculate Sharpe ratio
    fn calculate_sharpe_ratio(&self) -> f64 {
        // Simplified Sharpe ratio calculation
        if self.metrics.total_trades == 0 {
            return 0.0;
        }

        let returns: Vec<f64> = self
            .trade_history
            .iter()
            .filter_map(|trade| trade.pnl)
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - avg_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;

        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            0.0
        } else {
            avg_return / std_dev
        }
    }

    /// Calculate Sortino ratio
    fn calculate_sortino_ratio(&self) -> f64 {
        // Simplified Sortino ratio calculation
        if self.metrics.total_trades == 0 {
            return 0.0;
        }

        let returns: Vec<f64> = self
            .trade_history
            .iter()
            .filter_map(|trade| trade.pnl)
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if downside_returns.is_empty() {
            return avg_return; // No downside risk
        }

        let downside_variance =
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;

        let downside_deviation = downside_variance.sqrt();

        if downside_deviation == 0.0 {
            avg_return
        } else {
            avg_return / downside_deviation
        }
    }

    /// Check for performance alerts
    fn check_alerts(&mut self) {
        // Check drawdown
        if self.metrics.current_drawdown > self.config.drawdown_threshold {
            self.alerts.push_back(PerformanceAlert::DrawdownExceeded {
                current: self.metrics.current_drawdown,
                threshold: self.config.drawdown_threshold,
            });
        }

        // Check win rate
        if self.metrics.win_rate < self.config.win_rate_threshold && self.metrics.total_trades > 10
        {
            self.alerts.push_back(PerformanceAlert::WinRateLow {
                current: self.metrics.win_rate,
                threshold: self.config.win_rate_threshold,
            });
        }

        // Check P&L
        if self.metrics.total_pnl < self.config.pnl_threshold {
            self.alerts.push_back(PerformanceAlert::PnLLow {
                current: self.metrics.total_pnl,
                threshold: self.config.pnl_threshold,
            });
        }

        // Check latency
        if self.metrics.avg_signal_time > self.config.latency_threshold {
            self.alerts.push_back(PerformanceAlert::LatencyHigh {
                current: self.metrics.avg_signal_time,
                threshold: self.config.latency_threshold,
            });
        }

        // Keep only recent alerts
        if self.alerts.len() > 100 {
            self.alerts.drain(0..50);
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            drawdown_threshold: 0.1,                     // 10% drawdown
            win_rate_threshold: 0.5,                     // 50% win rate
            pnl_threshold: -1000.0,                      // -$1000 P&L
            latency_threshold: Duration::from_millis(1), // 1ms latency
            max_history_size: 10000,
            real_time_monitoring: true,
        }
    }
}

impl Default for TradingMetrics {
    fn default() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            total_pnl: 0.0,
            avg_trade_pnl: 0.0,
            max_drawdown: 0.0,
            current_drawdown: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            avg_signal_time: Duration::ZERO,
            avg_vwap_time: Duration::ZERO,
            peak_equity: 0.0,
            current_equity: 0.0,
            last_update: Utc::now(),
        }
    }
}

impl MonitorPerformanceTracker {
    fn new() -> Self {
        Self {
            total_updates: 0,
            total_time: Duration::ZERO,
            min_time: Duration::from_secs(u64::MAX),
            max_time: Duration::ZERO,
        }
    }

    fn record(&mut self, duration: Duration) {
        self.total_updates += 1;
        self.total_time += duration;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
    }

    fn stats(&self) -> MonitorStats {
        let avg_time = if self.total_updates > 0 {
            self.total_time / self.total_updates as u32
        } else {
            Duration::ZERO
        };

        MonitorStats {
            total_updates: self.total_updates,
            average_time: avg_time,
            min_time: self.min_time,
            max_time: self.max_time,
            total_time: self.total_time,
        }
    }
}

/// Monitor performance statistics
#[derive(Debug, Clone)]
pub struct MonitorStats {
    pub total_updates: usize,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub total_time: Duration,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_signal() -> TradingSignal {
        TradingSignal {
            signal_type: SignalType::Buy {
                strength: 0.01,
                reason: "Test signal".to_string(),
            },
            timestamp: Utc::now(),
            price: 100.0,
            volume: 1000,
            vwap: Some(99.0),
            confidence: 0.8,
            generation_time: Duration::from_micros(100),
        }
    }

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        assert_eq!(monitor.metrics.total_trades, 0);
    }

    #[test]
    fn test_signal_recording() {
        let mut monitor = PerformanceMonitor::new();
        let signal = create_test_signal();

        monitor.record_signal(&signal).unwrap();

        assert_eq!(monitor.metrics.total_trades, 0); // No completed trades yet
        assert!(!monitor.trade_history.is_empty());
    }

    #[test]
    fn test_trade_exit_recording() {
        let mut monitor = PerformanceMonitor::new();
        let signal = create_test_signal();

        monitor.record_signal(&signal).unwrap();
        monitor.record_trade_exit(105.0, Utc::now()).unwrap();

        assert_eq!(monitor.metrics.total_trades, 1);
        assert_eq!(monitor.metrics.winning_trades, 1);
        assert!(monitor.metrics.total_pnl > 0.0);
    }

    #[test]
    fn test_metrics_calculation() {
        let mut monitor = PerformanceMonitor::new();

        // Record winning trade
        let signal = create_test_signal();
        monitor.record_signal(&signal).unwrap();
        monitor.record_trade_exit(105.0, Utc::now()).unwrap();

        // Record losing trade
        let signal2 = TradingSignal {
            signal_type: SignalType::Sell {
                strength: 0.01,
                reason: "Test sell".to_string(),
            },
            timestamp: Utc::now(),
            price: 100.0,
            volume: 1000,
            vwap: Some(101.0),
            confidence: 0.8,
            generation_time: Duration::from_micros(100),
        };
        monitor.record_signal(&signal2).unwrap();
        monitor.record_trade_exit(98.0, Utc::now()).unwrap();

        assert_eq!(monitor.metrics.total_trades, 2);
        assert_eq!(monitor.metrics.winning_trades, 1);
        assert_eq!(monitor.metrics.losing_trades, 1);
        assert_eq!(monitor.metrics.win_rate, 0.5);
    }

    #[test]
    fn test_alert_generation() {
        let config = PerformanceConfig {
            drawdown_threshold: 0.05, // 5% drawdown
            win_rate_threshold: 0.6,  // 60% win rate
            pnl_threshold: -10.0,     // -$10 P&L (lower threshold to trigger)
            latency_threshold: Duration::from_millis(1),
            max_history_size: 1000,
            real_time_monitoring: true,
        };

        let mut monitor = PerformanceMonitor::with_config(config);

        // Record losing trades to trigger alerts
        for _ in 0..5 {
            let signal = create_test_signal();
            monitor.record_signal(&signal).unwrap();
            monitor.record_trade_exit(95.0, Utc::now()).unwrap(); // $5 loss per trade
        }

        let alerts = monitor.get_alerts(10);
        // With 5 trades at $5 loss each = -$25 total P&L, should trigger P&L alert
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_performance_stats() {
        let mut monitor = PerformanceMonitor::new();

        // Record some signals
        for _ in 0..5 {
            let signal = create_test_signal();
            monitor.record_signal(&signal).unwrap();
        }

        let stats = monitor.get_monitor_stats();
        assert_eq!(stats.total_updates, 5);
        assert!(stats.average_time > Duration::ZERO);
    }
}
