//! Trading-specific utilities and algorithms for FormicaX
//!
//! This module provides high-performance trading tools including:
//! - VWAP (Volume Weighted Average Price) calculations
//! - Real-time trading signal generation
//! - Performance monitoring and optimization
//! - Trading strategy implementations

pub mod performance;
pub mod signals;
pub mod strategies;
pub mod vwap;

// Re-export main trading types
pub use performance::{PerformanceMonitor, TradingMetrics};
pub use signals::{SignalGenerator, SignalType, TradingSignal};
pub use strategies::{StrategyConfig, TradingStrategy};
pub use vwap::{VWAPCalculator, VWAPResult, VWAPType};
