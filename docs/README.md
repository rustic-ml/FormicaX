# FormicaX - High-Performance Clustering Library for Stock Market Analysis

## Overview

FormicaX is a high-performance Rust library designed specifically for clustering analysis of financial data. It implements advanced machine learning clustering algorithms optimized for OHLCV (Open, High, Low, Close, Volume) data to identify market patterns, regimes, and trading opportunities.

## Why FormicaX for Clustering?

- **⚡ High Performance**: Built in Rust for speed and memory safety
- **🎯 Clustering Focused**: Specialized clustering algorithms for financial data
- **📊 Multiple Algorithms**: Six different clustering approaches
- **🔒 Memory Safe**: Rust's safety guarantees for critical analysis
- **📈 Market Regime Detection**: Identify different market states and patterns

## Quick Start

### Installation

Add FormicaX to your `Cargo.toml`:

```toml
[dependencies]
formica_x = "0.1.0"
```

### Basic Usage

```rust
use formica_x::{
    DataLoader,
    clustering::{KMeans, KMeansConfig, KMeansVariant},
    core::ClusteringAlgorithm
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;

    // Configure K-Means with modern builder pattern
    let config = KMeansConfig::builder()
        .k(3)                                // 3 clusters
        .variant(KMeansVariant::Elkan)       // Use optimized Elkan's algorithm
        .max_iterations(100)                 // Maximum iterations
        .parallel(true)                      // Enable parallel processing
        .simd(true)                         // Enable SIMD optimization
        .build()?;

    // Initialize and fit the clustering algorithm
    let mut kmeans = KMeans::with_config(config);
    let result = kmeans.fit(&ohlcv_data)?;

    // Display comprehensive results
    println!("Clustering completed successfully!");
    println!("- Algorithm: {}", result.algorithm_name);
    println!("- Clusters: {}", result.n_clusters);
    println!("- Iterations: {}", result.iterations);
    println!("- Converged: {}", result.converged);
    println!("- Silhouette Score: {:.3}", result.silhouette_score);
    println!("- Execution Time: {:?}", result.execution_time);
    
    // Predict clusters for new data
    let new_data = &ohlcv_data[0..10];
    let predictions = kmeans.predict(new_data)?;
    
    for (i, cluster) in predictions.iter().enumerate() {
        println!("Data point {} belongs to cluster {}", i, cluster);
    }

    Ok(())
}
```

### Data Format

Your CSV file should have the following format:

```csv
timestamp,open,high,low,close,volume
2025-07-01T09:30:00,100.5,102.0,99.8,101.2,100000
2025-07-01T09:31:00,101.3,103.5,100.7,102.8,120000
```

**Required Fields:**
- **timestamp**: ISO 8601 formatted datetime string
- **open/high/low/close**: Price data (positive numbers)
- **volume**: Trading volume (non-negative integers)

## Core Features

### Modern Clustering Algorithms
- **K-Means Variants**: Lloyd's, Elkan's, Hamerly's, and Mini-batch implementations
- **DBSCAN Family**: Standard, parallel, incremental, and approximate variants
- **Gaussian Mixture Models**: Standard EM, Variational Bayes, and robust variants
- **Hierarchical Methods**: Agglomerative clustering with optimized linkage methods
- **Affinity Propagation**: Message passing with adaptive damping and acceleration
- **Self-Organizing Maps**: Standard, growing, and hierarchical SOM implementations

### Performance Features
- **SIMD Optimization**: AVX2/AVX-512 vectorized operations for 8-16x speedup
- **Parallel Processing**: Work-stealing algorithms with near-linear scaling
- **Memory Efficiency**: Zero-copy operations and streaming for large datasets
- **Smart Algorithm Selection**: Automatic algorithm recommendation based on data

### Advanced Capabilities
- **Unified Interface**: Common API across all clustering algorithms
- **Ensemble Methods**: Consensus clustering for improved robustness
- **Quality Assessment**: Comprehensive validation metrics and stability analysis
- **Real-time Processing**: Incremental algorithms for streaming data

## Performance Tips

### Large Dataset Processing
```rust
use formica_x::{StreamingDataLoader};

fn process_large_dataset() -> Result<(), Box<dyn std::error::Error>> {
    let loader = StreamingDataLoader::new()
        .chunk_size(1000)
        .parallel(true);
    
    loader.process_file("large_data.csv")?;
    Ok(())
}
```

### Parallel Processing
```rust
use formica_x::{DataLoader};
use rayon::prelude::*;

fn parallel_clustering() -> Result<(), Box<dyn std::error::Error>> {
    let data = load_ohlcv_data()?;
    
    let results: Vec<_> = data
        .par_iter()
        .map(|ohlcv| perform_clustering(ohlcv))
        .collect();
    
    Ok(())
}
```

## Error Handling

FormicaX uses Rust's Result type for robust error handling:

```rust
use formica_x::{DataLoader, FormicaXError};

fn process_data() -> Result<(), FormicaXError> {
    let mut loader = DataLoader::new("data.csv")?;
    
    for result in loader.records() {
        match result {
            Ok(ohlcv) => println!("Valid: {:?}", ohlcv),
            Err(e) => eprintln!("Invalid data: {}", e),
        }
    }
    
    Ok(())
}
```

## Documentation

- **[Architecture Guide](ARCHITECTURE.md)** - System design, implementation plan, and configuration
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Data format specification and development details
- **[Clustering Guide](CLUSTERING_GUIDE.md)** - Clustering algorithms and optimization techniques

## Performance Targets

### Throughput and Latency
- **Data Processing**: 1GB CSV files in < 5 seconds with streaming
- **Clustering Predictions**: Sub-100 microsecond latency with SIMD
- **Memory Efficiency**: < 1.5x memory overhead with zero-copy operations
- **Scalability**: >85% parallel efficiency across CPU cores

### Algorithm Quality
- **K-Means**: Silhouette score > 0.7, convergence < 100 iterations
- **DBSCAN**: Noise detection > 95%, cluster purity > 0.9
- **GMM**: Log-likelihood tolerance < 1e-6, BIC/AIC accuracy > 90%
- **Overall Accuracy**: Within 0.5% of reference implementations

## 🎯 **Cursor Implementation Rules**

### **MANDATORY Development Ethos**

When implementing any feature in FormicaX, **ALWAYS** follow these non-negotiable rules:

#### **1. Code Coverage First**
```bash
# BEFORE implementing:
cargo tarpaulin --out Html --output-dir coverage

# AFTER implementing:
cargo tarpaulin --out Html --output-dir coverage --fail-under 95
```

**Requirements:**
- [ ] **95% minimum coverage** for all new code
- [ ] **100% coverage** for public APIs and error paths
- [ ] **Property-based tests** with `proptest`
- [ ] **Integration tests** for end-to-end workflows

#### **2. Stop and Review**
After every implementation:

**Review Checklist:**
- [ ] **Coverage**: `cargo tarpaulin --fail-under 95`
- [ ] **No Duplication**: `cargo clippy --all-targets --all-features`
- [ ] **Clean Code**: Modular, testable, readable
- [ ] **Performance**: `cargo bench` - no regressions
- [ ] **Documentation**: Examples using `examples/csv/` data

#### **3. Latest Dependencies**
```bash
# Weekly:
cargo outdated
cargo update
cargo audit
cargo check --all-features
cargo test --all-features
```

**Rules:**
- [ ] **Latest stable versions** from crates.io
- [ ] **No pinned versions** unless absolutely necessary
- [ ] **Weekly updates** with compatibility verification

#### **4. Clean, Modular Code**
```rust
// ✅ GOOD: Clean, testable, modular
pub trait ClusteringAlgorithm {
    type Config;
    fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError>;
}

// ❌ BAD: Duplicated, hard to test
fn kmeans_inline(data: &[OHLCV], k: usize) -> Vec<usize> {
    // 200 lines of inline algorithm
}
```

### **Quality Gates**

**Every implementation must pass:**

| Gate | Command | Target |
|------|---------|--------|
| **Coverage** | `cargo tarpaulin` | > 95% |
| **Quality** | `cargo clippy` | 0 warnings |
| **Security** | `cargo audit` | 0 issues |
| **Tests** | `cargo test` | All pass |
| **Performance** | `cargo bench` | No regression |

**These rules are MANDATORY and non-negotiable.**

## Contributing

See [CONTRIBUTING.md](contributing.md) for development guidelines. 

## License

Apache-2.0 