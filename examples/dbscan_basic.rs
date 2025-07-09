//! Basic DBSCAN clustering example for FormicaX
//! Usage: cargo run --example dbscan_basic

use formica::{
    clustering::dbscan::algorithm::DBSCAN, clustering::dbscan::config::DBSCANConfig,
    core::DataLoader,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;

    // Configure DBSCAN
    let config = DBSCANConfig::builder().epsilon(0.5).min_points(5).build()?;

    // Fit DBSCAN
    let mut dbscan = DBSCAN::with_config(config);
    dbscan.fit(&ohlcv_data)?;

    println!("DBSCAN clustering completed.");
    println!("Assignments: {:#?}", dbscan.predict(&ohlcv_data)?);
    println!("Noise points: {:#?}", dbscan.noise_points());
    Ok(())
}
