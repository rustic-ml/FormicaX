//! Basic GMM clustering example for FormicaX
//! Usage: cargo run --example gmm_basic

use formica::clustering::gmm::config::CovarianceType;
use formica::{
    clustering::{GMMConfig, GMMVariant, GMM},
    DataLoader,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;

    // Configure GMM with builder
    let config = GMMConfig::builder()
        .n_components(3)
        .variant(GMMVariant::EM)
        .max_iterations(100)
        .tolerance(1e-6)
        .covariance_type(CovarianceType::Diagonal)
        .regularization(1e-6)
        .random_seed(42)
        .build()?;

    // Fit GMM
    let mut gmm = GMM::with_config(config);
    let result = gmm.fit(&ohlcv_data)?;

    println!("GMM clustering completed.");
    println!("Algorithm: {}", result.algorithm_name);
    println!("Clusters: {}", result.n_clusters);
    println!("Iterations: {}", result.iterations);
    println!("Converged: {}", result.converged);
    println!("Silhouette Score: {:.3}", result.silhouette_score);
    println!("Execution Time: {:?}", result.execution_time);

    // Print model parameters
    if let Some(means) = gmm.get_means() {
        println!("\nComponent Means:");
        for (i, mean) in means.iter().enumerate() {
            println!("  Cluster {}: {:?}", i, mean);
        }
    }

    if let Some(weights) = gmm.get_weights() {
        println!("\nComponent Weights:");
        for (i, weight) in weights.iter().enumerate() {
            println!("  Cluster {}: {:.3}", i, weight);
        }
    }

    if let Some(log_likelihood) = gmm.get_log_likelihood() {
        println!("\nFinal Log-Likelihood: {:.3}", log_likelihood);
    }

    Ok(())
}
