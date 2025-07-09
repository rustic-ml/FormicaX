//! VWAP-based trading strategy example for FormicaX
//! Usage: cargo run --example vwap_strategy

use formica::{
    core::DataLoader,
    trading::strategies::{StrategyConfig, VWAPStrategy},
    trading::vwap::VWAPType,
    trading::TradingStrategy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;

    // Configure VWAP strategy
    let config = StrategyConfig {
        name: "VWAP Example".to_string(),
        vwap_type: VWAPType::Session,
        max_position_size: 10000.0,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.04,
        ..Default::default()
    };

    let mut strategy = VWAPStrategy::with_config(config);
    let signals = strategy.execute(&ohlcv_data)?;

    println!("VWAP Strategy Signals:");
    for signal in &signals {
        println!("{}: {:?}", signal.timestamp, signal.signal_type);
    }

    let performance = strategy.get_performance();
    let metrics = performance.get_metrics();
    println!("\nSummary Metrics:");
    println!("Total trades: {}", metrics.total_trades);
    println!("Win rate: {:.2}%", metrics.win_rate * 100.0);
    println!("Total P&L: ${:.2}", metrics.total_pnl);
    Ok(())
}
