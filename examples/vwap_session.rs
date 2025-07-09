//! Session-based VWAP calculation example for FormicaX
//! Usage: cargo run --example vwap_session

use formica::{core::DataLoader, trading::vwap::VWAPCalculator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;

    // Create session-based VWAP calculator
    let vwap_calc = VWAPCalculator::session_based();
    let vwap_result = vwap_calc.calculate(&ohlcv_data)?;

    println!("Session VWAP: {:.4}", vwap_result.vwap);
    println!("Total Volume: {:.0}", vwap_result.total_volume);
    Ok(())
}
