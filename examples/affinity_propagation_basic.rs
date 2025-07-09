//! Basic Affinity Propagation clustering example for FormicaX
//! Usage: cargo run --example affinity_propagation_basic

use formica::{
    clustering::{AffinityPropagation, AffinityPropagationConfig, PreferenceMethod},
    DataLoader,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;

    // Configure Affinity Propagation with builder
    let config = AffinityPropagationConfig::builder()
        .damping(0.5)
        .max_iterations(200)
        .tolerance(1e-6)
        .preference(PreferenceMethod::Median)
        .build()?;

    // Fit Affinity Propagation
    let mut ap = AffinityPropagation::with_config(config);
    let result = ap.fit(&ohlcv_data)?;

    println!("Affinity Propagation clustering completed.");
    println!("Algorithm: {}", result.algorithm_name);
    println!("Clusters: {}", result.n_clusters);
    println!("Iterations: {}", result.iterations);
    println!("Converged: {}", result.converged);
    println!("Silhouette Score: {:.3}", result.silhouette_score);
    println!("Execution Time: {:?}", result.execution_time);

    // Print exemplars
    if let Some(exemplars) = ap.get_exemplars() {
        println!("\nExemplars (Cluster Centers):");
        for (i, &exemplar) in exemplars.iter().enumerate() {
            println!("  Cluster {}: Data point {}", i, exemplar);
        }
    }

    // Print cluster assignments
    if let Some(assignments) = ap.get_assignments() {
        println!("\nCluster Assignments (first 10):");
        for (i, &cluster) in assignments.iter().take(10).enumerate() {
            println!("  Data point {} -> Cluster {}", i, cluster);
        }
    }

    Ok(())
}
