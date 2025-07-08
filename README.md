FormicaX: A Rust-Based Stock Trading Prediction Library
Overview
FormicaX is a high-performance, Rust-based library designed for stock market analysis and prediction using OHLCV (Open, High, Low, Close, Volume) data. Leveraging the power of Rust's safety and speed, FormicaX implements a suite of advanced machine learning clustering algorithms to generate predictive insights for stock trading. The library is tailored for developers and data scientists building trading applications or conducting financial research.
The library supports the following clustering algorithms:

K-Means Clustering: Partitions data into K clusters by minimizing variance within clusters.
Hierarchical Clustering: Builds a hierarchy of clusters using a bottom-up or top-down approach.
DBSCAN: Groups data points into clusters based on density, effectively handling noise and outliers.
Gaussian Mixture Models (GMM): Models data as a mixture of Gaussian distributions for probabilistic clustering.
Affinity Propagation: Identifies exemplars among data points to form clusters without requiring a predefined number of clusters.
Self-Organizing Maps (SOM): Uses neural network-based dimensionality reduction to map high-dimensional data into a 2D grid for pattern detection.

FormicaX is designed to process large datasets efficiently, making it suitable for real-time or near-real-time trading applications. It provides a modular and extensible framework for integrating with existing trading systems or data pipelines.
Features

High Performance: Built in Rust for speed and memory safety, optimized for large-scale OHLCV datasets.
Flexible Input: Supports OHLCV data in CSV, JSON, or custom formats via a configurable data loader.
Multiple Algorithms: Implements six clustering algorithms for diverse analytical approaches.
Customizable Parameters: Fine-tune algorithm hyperparameters to suit specific trading strategies.
Prediction Outputs: Generates cluster-based predictions, including trend identification and anomaly detection.
Extensibility: Modular design allows for easy integration of new algorithms or data preprocessing steps.
Cross-Platform: Compatible with Linux, macOS, and Windows (experimental).

Installation
Prerequisites

Rust: Ensure you have Rust installed (version 1.80.0 or later). Install via rustup.
Dependencies: FormicaX relies on external crates for numerical computations and data handling. These are automatically resolved via Cargo.toml.

Steps

Clone the repository:
git clone https://github.com/rustic-ml/FormicaX.git
cd FormicaX


Build the library:
cargo build --release


Add FormicaX as a dependency in your Cargo.toml:
[dependencies]
formica_x = { path = "./FormicaX" }


(Optional) Run tests to verify installation:
cargo test



Usage
Basic Example
Below is a simple example of using FormicaX to cluster OHLCV data using K-Means and generate predictions.
use formica_x::{DataLoader, KMeans, Predictor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from a CSV file
    let mut loader = DataLoader::new("data/stock_data.csv");
    let ohlcv_data = loader.load_csv()?;

    // Initialize K-Means with 3 clusters
    let mut kmeans = KMeans::new(3, 100); // 3 clusters, 100 iterations
    kmeans.fit(&ohlcv_data)?;

    // Create a predictor instance
    let predictor = Predictor::new(kmeans);
    
    // Predict clusters for new data
    let new_data = ohlcv_data[0..10].to_vec();
    let predictions = predictor.predict(&new_data)?;

    // Output predictions
    for (i, cluster) in predictions.iter().enumerate() {
        println!("Data point {} belongs to cluster {}", i, cluster);
    }

    Ok(())
}

Data Format
FormicaX expects OHLCV data in a structured format. A typical CSV file should look like this:
timestamp,open,high,low,close,volume
2025-07-01T09:30:00,100.5,102.0,99.8,101.2,100000
2025-07-01T09:31:00,101.3,103.5,100.7,102.8,120000
...

You can also implement custom data loaders for other formats (e.g., JSON, Parquet) by extending the DataLoader trait.
Algorithm Configuration
Each algorithm in FormicaX can be configured via hyperparameters. For example:

K-Means:
let kmeans = KMeans::new(5, 200); // 5 clusters, 200 iterations


DBSCAN:
let dbscan = DBSCAN::new(0.5, 5); // Epsilon = 0.5, MinPts = 5


GMM:
let gmm = GaussianMixture::new(4, 1e-6); // 4 components, convergence threshold



Refer to the API documentation for detailed parameter options.
Supported Algorithms
K-Means Clustering
Divides data into K clusters by iteratively assigning points to the nearest centroid and updating centroids. Suitable for identifying distinct market regimes.
Hierarchical Clustering
Builds a tree of clusters by merging or splitting data points based on their similarity. Useful for visualizing hierarchical relationships in stock data.
DBSCAN
Identifies clusters based on density, marking outliers as noise. Ideal for detecting anomalous trading patterns.
Gaussian Mixture Models (GMM)
Fits data to a mixture of Gaussian distributions, providing probabilistic cluster assignments. Effective for modeling complex market behaviors.
Affinity Propagation
Automatically determines the number of clusters by selecting exemplars. Good for exploratory analysis without predefined cluster counts.
Self-Organizing Maps (SOM)
Maps high-dimensional OHLCV data to a 2D grid, preserving topological relationships. Useful for visualizing patterns and trends.
Building a Trading Strategy
FormicaX can be integrated into a trading pipeline as follows:

Data Preprocessing: Normalize OHLCV data using FormicaX's Preprocessor module.
Clustering: Apply one or more clustering algorithms to segment the data.
Prediction: Use the Predictor module to assign new data points to clusters.
Strategy Logic: Map clusters to trading signals (e.g., buy, sell, hold) based on historical performance.
Backtesting: Evaluate the strategy using FormicaX's Backtester utility (optional).

Example strategy:
let clusters = kmeans.predict(&new_ohlcv)?;
if clusters[0] == 1 {
    println!("Buy signal: Cluster 1 indicates upward trend.");
} else if clusters[0] == 2 {
    println!("Sell signal: Cluster 2 indicates downward trend.");
}

Contributing
We welcome contributions! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please read the contribution guide for details. Report issues or suggest features via GitHub Issues.
Building and Testing

Build: cargo build --release
Test: cargo test
Documentation: cargo doc --open

API Documentation
Full API documentation is available in the docs/ directory or generated via:
cargo doc --open

Limitations

Beta Status: FormicaX is in active development and not yet recommended for production trading systems.
Windows Support: Experimental due to Rust ecosystem limitations.
Data Size: Large datasets may require significant memory; optimize using data streaming for scalability.

License
FormicaX is licensed under the Apache-2.0 License. See LICENSE for details.
Contact

GitHub: rustic-ml/FormicaX
Issues: GitHub Issues
Discord: Join our community Discord for discussions and support.
