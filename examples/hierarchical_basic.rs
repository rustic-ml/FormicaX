//! Basic Hierarchical clustering example for FormicaX
//! Usage: cargo run --example hierarchical_basic

use formica::clustering::hierarchical::config::DistanceMetric;
use formica::{
    clustering::{Hierarchical, HierarchicalConfig, LinkageMethod},
    DataLoader,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;

    // Configure Hierarchical clustering with builder
    let config = HierarchicalConfig::builder()
        .n_clusters(3)
        .linkage(LinkageMethod::Ward)
        .distance_metric(DistanceMetric::Euclidean)
        .build()?;

    // Fit Hierarchical clustering
    let mut hierarchical = Hierarchical::with_config(config);
    let result = hierarchical.fit(&ohlcv_data)?;

    println!("Hierarchical clustering completed.");
    println!("Algorithm: {}", result.algorithm_name);
    println!("Clusters: {}", result.n_clusters);
    println!("Iterations: {}", result.iterations);
    println!("Converged: {}", result.converged);
    println!("Silhouette Score: {:.3}", result.silhouette_score);
    println!("Execution Time: {:?}", result.execution_time);

    // Print cluster centers
    if let Some(cluster_centers) = result.cluster_centers {
        println!("\nCluster Centers:");
        for (i, center) in cluster_centers.iter().enumerate() {
            println!("  Cluster {}: {:?}", i, center);
        }
    }

    // Print dendrogram
    if let Some(dendrogram) = hierarchical.get_dendrogram() {
        println!("\nDendrogram (first 5 merges):");
        for (i, (cluster1, cluster2, distance)) in dendrogram.iter().take(5).enumerate() {
            println!(
                "  Merge {}: Clusters {} and {} at distance {:.3}",
                i, cluster1, cluster2, distance
            );
        }
    }

    Ok(())
}
