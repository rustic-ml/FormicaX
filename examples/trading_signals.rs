//! Trading signal generation example for FormicaX
//! Usage: cargo run --example trading_signals

use formica::{core::DataLoader, trading::signals::SignalGenerator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;

    // Create signal generator
    let mut signal_gen = SignalGenerator::new();

    println!("Trading signals:");
    for ohlcv in &ohlcv_data {
        let signal = signal_gen.generate_signal_incremental(ohlcv)?;
        println!("{}: {:?}", ohlcv.timestamp, signal.signal_type);
    }
    Ok(())
}
