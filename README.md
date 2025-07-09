# FormicaX: Rust-Based Clustering Library for Stock Market Analysis.

![FormicaX Logo](https://github.com/rustic-ml/FormicaX/blob/main/FormicaX.png)

## Overview

[FormicaX](https://github.com/rustic-ml/FormicaX) is a high-performance, Rust-based library designed for stock market analysis and prediction using OHLCV (Open, High, Low, Close, Volume) data. Leveraging Rust's safety and speed, FormicaX implements advanced machine learning clustering algorithms to generate predictive insights for stock trading. The library is tailored for developers and data scientists building trading applications or conducting financial research.

The name **FormicaX**, derived from "Formica" (Latin for ant), reflects the library's design principles:
- **Collaboration**: Like ants in a colony, FormicaX's algorithms work together to process data efficiently.
- **Adaptability**: Ants adapt to complex environments; FormicaX adapts to diverse market patterns.
- **Resilience**: Ant colonies are robust; FormicaX handles large datasets with Rust's performance.
- **Exploration**: Ants explore for resources; FormicaX uncovers hidden patterns in data.
- **Simplicity**: Individual ants are simple, yet powerful collectively; FormicaX offers a modular, user-friendly API.

The "X" signifies excellence, exploration, and extensibility, highlighting the library’s advanced and flexible capabilities.

Supported clustering algorithms:
- **K-Means Clustering**: Partitions data into K clusters by minimizing variance within clusters.
- **Hierarchical Clustering**: Builds a hierarchy of clusters using a bottom-up or top-down approach.
- **DBSCAN**: Groups data points into clusters based on density, handling noise and outliers.
- **Gaussian Mixture Models (GMM)**: Models data as a mixture of Gaussian distributions for probabilistic clustering.
- **Affinity Propagation**: Identifies exemplars to form clusters without a predefined number of clusters.
- **Self-Organizing Maps (SOM)**: Uses neural network-based dimensionality reduction to map data into a 2D grid.

FormicaX processes large datasets efficiently, making it suitable for real-time or near-real-time trading applications. It offers a modular framework for integration with trading systems or data pipelines.

## Features

- **High Performance**: Built in Rust for speed and memory safety, optimized for large OHLCV datasets.
- **Advanced Trading**: VWAP-based strategies, real-time signal generation, and performance monitoring.
- **Flexible Input**: Supports OHLCV data in CSV, JSON, or custom formats via a configurable data loader.
- **Multiple Algorithms**: Implements six clustering algorithms for diverse analytical approaches.
- **Customizable Parameters**: Fine-tune algorithm hyperparameters for specific trading strategies.
- **Prediction Outputs**: Generates cluster-based predictions, including trend identification and anomaly detection.
- **Extensibility**: Modular design allows integration of new algorithms or preprocessing steps.
- **Cross-Platform**: Compatible with Linux, macOS, and Windows (experimental).

## Installation

### Prerequisites
- **Rust**: Version 1.80.0 or later, installed via [rustup](https://rustup.rs/).
- **Dependencies**: Managed automatically via `Cargo.toml`.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/rustic-ml/FormicaX.git
   cd FormicaX
   ```

2. Build the library:
   ```bash
   cargo build --release
   ```

3. Add [FormicaX](https://github.com/rustic-ml/FormicaX) as a dependency in your `Cargo.toml`:
   ```toml
   [dependencies]
   formica_x = { path = "./FormicaX" }
   ```

4. (Optional) Run tests:
   ```bash
   cargo test
   ```

## Usage

### Basic Clustering Example
Cluster OHLCV data using K-Means and generate predictions:

```rust
use formica_x::{DataLoader, KMeans, Predictor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV
    let mut loader = DataLoader::new("data/stock_data.csv");
    let ohlcv_data = loader.load_csv()?;

    // Initialize K-Means with 3 clusters
    let mut kmeans = KMeans::new(3, 100); // 3 clusters, 100 iterations
    kmeans.fit(&ohlcv_data)?;

    // Create predictor
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
```

### Trading Strategy Example
Implement a VWAP-based trading strategy:

```rust
use formica_x::{
    DataLoader, 
    trading::{VWAPCalculator, SignalGenerator, VWAPStrategy, StrategyConfig}
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load market data
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let data = loader.load_csv()?;

    // Create VWAP-based trading strategy
    let config = StrategyConfig {
        name: "VWAP Strategy".to_string(),
        vwap_type: VWAPType::Session,
        max_position_size: 10000.0,
        stop_loss_pct: 0.02,  // 2% stop loss
        take_profit_pct: 0.04, // 4% take profit
        ..Default::default()
    };

    let mut strategy = VWAPStrategy::with_config(config);

    // Execute strategy on historical data
    let signals = strategy.execute(&data)?;

    // Process trading signals
    for signal in signals {
        match signal.signal_type {
            SignalType::Buy { strength, reason } => {
                println!("BUY: {} (confidence: {:.2})", reason, signal.confidence);
            }
            SignalType::Sell { strength, reason } => {
                println!("SELL: {} (confidence: {:.2})", reason, signal.confidence);
            }
            SignalType::Hold { reason } => {
                println!("HOLD: {}", reason);
            }
            _ => {}
        }
    }

    // Get performance metrics
    let performance = strategy.get_performance();
    let metrics = performance.get_metrics();
    println!("Total trades: {}", metrics.total_trades);
    println!("Win rate: {:.2}%", metrics.win_rate * 100.0);
    println!("Total P&L: ${:.2}", metrics.total_pnl);

    Ok(())
}
```

### Data Format
OHLCV data should be structured, e.g., in CSV:

```csv
timestamp,open,high,low,close,volume
2025-07-01T09:30:00,100.5,102.0,99.8,101.2,100000
2025-07-01T09:31:00,101.3,103.5,100.7,102.8,120000
...
```

Custom data loaders for formats like JSON can be implemented via the `DataLoader` trait.

### Algorithm Configuration
Configure hyperparameters, e.g.:

- **K-Means**:
  ```rust
  let kmeans = KMeans::new(5, 200); // 5 clusters, 200 iterations
  ```

- **DBSCAN**:
  ```rust
  let dbscan = DBSCAN::new(0.5, 5); // Epsilon = 0.5, MinPts = 5
  ```

- **GMM**:
  ```rust
  let gmm = GaussianMixture::new(4, 1e-6); // 4 components, convergence threshold
  ```

See [API documentation](#api-documentation) for details.

## Supported Algorithms

- **K-Means Clustering**: Identifies market regimes by partitioning data.
- **Hierarchical Clustering**: Visualizes hierarchical relationships in stock data.
- **DBSCAN**: Detects anomalous trading patterns via density-based clustering.
- **Gaussian Mixture Models (GMM)**: Models complex market behaviors probabilistically.
- **Affinity Propagation**: Exploratory analysis without predefined cluster counts.
- **Self-Organizing Maps (SOM)**: Visualizes patterns via 2D mapping.

## Building Trading Strategies

### Clustering-Based Strategy
Integrate clustering algorithms into a trading pipeline:
1. **Preprocess**: Normalize data using `Preprocessor`.
2. **Cluster**: Apply clustering algorithms.
3. **Predict**: Assign new data to clusters.
4. **Trade**: Map clusters to signals (buy/sell/hold).
5. **Backtest**: Use `Backtester` (optional).

Example:
```rust
let clusters = kmeans.predict(&new_ohlcv)?;
if clusters[0] == 1 {
    println!("Buy signal: Cluster 1 indicates upward trend.");
} else if clusters[0] == 2 {
    println!("Sell signal: Cluster 2 indicates downward trend.");
}
```

### VWAP-Based Strategy
Use VWAP (Volume Weighted Average Price) for real-time trading:

```rust
use formica_x::trading::{VWAPCalculator, SignalGenerator};

// Create VWAP calculator
let mut vwap_calc = VWAPCalculator::session_based();

// Create signal generator
let mut signal_gen = SignalGenerator::new();

// Process real-time data
for ohlcv in real_time_data {
    // Calculate VWAP incrementally
    let vwap_result = vwap_calc.calculate_incremental(&[ohlcv.clone()])?;
    
    // Generate trading signal
    let signal = signal_gen.generate_signal_incremental(&ohlcv)?;
    
    // Execute based on signal
    match signal.signal_type {
        SignalType::Buy { .. } => execute_buy_order(ohlcv.close),
        SignalType::Sell { .. } => execute_sell_order(ohlcv.close),
        _ => {}
    }
}
```

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push branch (`git push origin feature/your-feature`).
5. Open a pull request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details. Report issues at [GitHub Issues](https://github.com/rustic-ml/FormicaX/issues).

## Building and Testing
- **Build**: `cargo build --release`
- **Test**: `cargo test`
- **Docs**: `cargo doc --open`

## API Documentation
Generated via:
```bash
cargo doc --open
```

## Limitations
- **Beta Status**: Not recommended for production trading.
- **Windows**: Experimental support.
- **Data Size**: Large datasets may require streaming for scalability.

## License
Apache-2.0. See [LICENSE](LICENSE).

## Contact
- **GitHub**: [rustic-ml/FormicaX](https://github.com/rustic-ml/FormicaX)
- **Issues**: [GitHub Issues](https://github.com/rustic-ml/FormicaX/issues)
- **Discord**: [Community Discord](https://discord.gg/rustic-ml)
