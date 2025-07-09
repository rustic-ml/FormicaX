//! Performance monitoring and alerting example for FormicaX
//! Usage: cargo run --example performance_monitor

use formica::{
    core::DataLoader, trading::performance::PerformanceMonitor, trading::signals::SignalGenerator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;

    // Create performance monitor and signal generator
    let mut monitor = PerformanceMonitor::new();
    let mut signal_gen = SignalGenerator::new();

    for ohlcv in &ohlcv_data {
        let signal = signal_gen.generate_signal_incremental(ohlcv)?;
        monitor.record_signal(&signal)?;
    }

    let metrics = monitor.get_metrics();
    println!("Performance Metrics: {:#?}", metrics);
    let alerts = monitor.get_alerts(10);
    println!("Recent Alerts: {:#?}", alerts);
    Ok(())
}
