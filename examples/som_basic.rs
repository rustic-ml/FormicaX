//! Basic SOM clustering example for FormicaX
//! Usage: cargo run --example som_basic

use formica::{
    clustering::{SOMConfig, TopologyType, SOM},
    DataLoader,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;

    // Configure SOM with builder
    let config = SOMConfig::builder()
        .width(5)
        .height(5)
        .topology(TopologyType::Rectangular)
        .learning_rate(0.1)
        .neighborhood_radius(2.0)
        .epochs(100)
        .build()?;

    // Fit SOM
    let mut som = SOM::with_config(config);
    let result = som.fit(&ohlcv_data)?;

    println!("SOM clustering completed.");
    println!("Algorithm: {}", result.algorithm_name);
    println!("Clusters: {}", result.n_clusters);
    println!("Iterations: {}", result.iterations);
    println!("Converged: {}", result.converged);
    println!("Silhouette Score: {:.3}", result.silhouette_score);
    println!("Execution Time: {:?}", result.execution_time);

    // Print grid structure
    if let Some(grid_positions) = som.get_grid_positions() {
        println!("\nSOM Grid Structure ({}x{}):", som.width(), som.height());
        for (i, &(row, col)) in grid_positions.iter().enumerate() {
            println!("  Neuron {}: Position ({}, {})", i, row, col);
        }
    }

    // Print neuron weights
    if let Some(weights) = som.get_weights() {
        println!("\nNeuron Weights (first 5 neurons):");
        for (i, neuron_weights) in weights.iter().take(5).enumerate() {
            println!("  Neuron {}: {:?}", i, neuron_weights);
        }
    }

    // Print cluster assignments
    if let Some(assignments) = som.get_assignments() {
        println!("\nCluster Assignments (first 10):");
        for (i, &cluster) in assignments.iter().take(10).enumerate() {
            println!("  Data point {} -> Neuron/Cluster {}", i, cluster);
        }
    }

    Ok(())
}
