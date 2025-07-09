//! Basic KMeans clustering example for FormicaX
//! Usage: cargo run --example kmeans_basic

use formica::{
    clustering::kmeans::algorithm::KMeans,
    clustering::kmeans::config::{KMeansConfig, KMeansVariant},
    core::DataLoader,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;

    // Configure KMeans with builder
    let config = KMeansConfig::builder()
        .k(3)
        .variant(KMeansVariant::Lloyd)
        .max_iterations(100)
        .build()?;

    // Fit KMeans
    let mut kmeans = KMeans::with_config(config);
    let result = kmeans.fit(&ohlcv_data)?;

    println!("KMeans clustering completed.");
    println!("Centroids: {:#?}", kmeans.get_centroids());
    println!("Assignments: {:#?}", kmeans.get_assignments());
    println!("Result: {:#?}", result);
    Ok(())
}
