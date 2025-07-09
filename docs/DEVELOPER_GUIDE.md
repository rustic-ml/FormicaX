# FormicaX Developer Guide

## Data Format Specification

### OHLCV Data Format

FormicaX expects OHLCV (Open, High, Low, Close, Volume) data in a specific CSV format for clustering analysis.

#### Flexible Column Format

FormicaX supports flexible CSV column ordering and case-insensitive column names. The library automatically detects and maps columns regardless of their position or case.

**Default Expected Order:**
```
timestamp,open,high,low,close,volume
```

**Supported Column Variations:**
- **Timestamp**: `timestamp`, `Timestamp`, `TIME`, `time`, `date`, `Date`, `datetime`, `DateTime`
- **Open**: `open`, `Open`, `OPEN`, `o`, `O`
- **High**: `high`, `High`, `HIGH`, `h`, `H`
- **Low**: `low`, `Low`, `LOW`, `l`, `L`
- **Close**: `close`, `Close`, `CLOSE`, `c`, `C`
- **Volume**: `volume`, `Volume`, `VOLUME`, `vol`, `Vol`, `VOL`

**Example - Different Column Orders:**
```csv
# Standard order
timestamp,open,high,low,close,volume

# Alternative order 1
open,high,low,close,volume,timestamp

# Alternative order 2
volume,close,low,high,open,timestamp

# With different case
Timestamp,Open,High,Low,Close,Volume
```

#### Field Descriptions

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| timestamp | ISO 8601 string | Date and time of the data point | `2020-07-08T04:00:00Z` |
| open | float | Opening price for the period | `10.600000` |
| high | float | Highest price during the period | `10.600000` |
| low | float | Lowest price during the period | `10.340000` |
| close | float | Closing price for the period | `10.570000` |
| volume | integer | Trading volume for the period | `44` |

#### Data Requirements

1. **Timestamp Format**: Must be ISO 8601 compliant
   - Format: `YYYY-MM-DDTHH:MM:SSZ`
   - Timezone: UTC (Z suffix)
   - Example: `2020-07-08T04:00:00Z`

2. **Price Fields**: 
   - Must be positive numbers
   - Precision: Up to 6 decimal places
   - Range: Reasonable market prices (typically 0.01 to 1000000)

3. **Volume Field**:
   - Must be non-negative integers
   - Represents number of shares/contracts traded

#### Example Data

```csv
timestamp,open,high,low,close,volume
2020-07-08T04:00:00Z,10.600000,10.600000,10.340000,10.570000,44
2020-07-09T04:00:00Z,10.530000,10.750000,10.470000,10.545000,27
2020-07-10T04:00:00Z,10.590000,10.600000,10.360000,10.360000,22
```

#### Data Validation Rules

1. **Logical Consistency**:
   - `high >= low`
   - `high >= open` and `high >= close`
   - `low <= open` and `low <= close`

2. **Data Integrity**:
   - No missing values
   - No duplicate timestamps
   - Chronological order (ascending timestamps)

3. **Format Validation**:
   - Valid CSV format
   - Correct number of columns (6)
   - Proper data types for each column

#### Common Issues and Solutions

##### Issue: Unrecognized Column Names
**Problem**: Column names don't match any supported variations
**Solution**: Use standard column names or add custom column mapping

##### Issue: Missing Required Columns
**Problem**: CSV missing one or more required columns
**Solution**: Ensure all 6 columns (timestamp, open, high, low, close, volume) are present

##### Issue: Invalid Timestamp Format
**Problem**: Timestamps not in ISO 8601 format
**Solution**: Convert to proper format: `YYYY-MM-DDTHH:MM:SSZ`

##### Issue: Non-UTC Timezone
**Problem**: Timestamps in local timezone
**Solution**: Convert to UTC before processing

## Quick Start Guide

### Installation

Add FormicaX to your `Cargo.toml`:

```toml
[dependencies]
formica_x = "0.1.0"
```

### Basic Usage

#### Loading OHLCV Data

```rust
use formica_x::{DataLoader, OHLCV};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load OHLCV data from CSV file in examples/csv/ folder
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;
    
    // Process data points
    for ohlcv in &ohlcv_data {
        println!("Timestamp: {}, Close: {}", ohlcv.timestamp, ohlcv.close);
    }
    
    Ok(())
}
```

#### K-Means Clustering

```rust
use formica_x::{DataLoader, KMeans, Predictor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from examples/csv/ folder
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;
    
    // Initialize K-Means with 3 clusters
    let mut kmeans = KMeans::new(3, 100); // 3 clusters, 100 iterations
    kmeans.fit(&ohlcv_data)?;
    
    // Create predictor
    let predictor = Predictor::new(kmeans);
    
    // Predict clusters for new data
    let new_data = ohlcv_data[0..10].to_vec();
    let predictions = predictor.predict(&new_data)?;
    
    println!("Predictions: {:?}", predictions);
    
    Ok(())
}
```

#### DBSCAN Clustering

```rust
use formica_x::{DataLoader, DBSCAN, Predictor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from examples/csv/ folder
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let ohlcv_data = loader.load_csv()?;
    
    // Initialize DBSCAN
    let mut dbscan = DBSCAN::new(0.5, 5); // epsilon = 0.5, min_pts = 5
    dbscan.fit(&ohlcv_data)?;
    
    // Get clustering results
    let clusters = dbscan.get_clusters()?;
    let noise_points = dbscan.get_noise_points()?;
    
    println!("Found {} clusters", clusters.len());
    println!("Noise points: {}", noise_points.len());
    
    Ok(())
}
```

#### Trading Module Usage

```rust
use formica_x::{
    DataLoader,
    trading::{
        VWAPCalculator,
        SignalGenerator,
        VWAPStrategy,
        StrategyConfig,
        VWAPType,
        PerformanceMonitor
    }
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load market data from examples/csv/ folder
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let data = loader.load_csv()?;
    
    // Create VWAP calculator
    let mut vwap_calc = VWAPCalculator::session_based();
    let vwap_result = vwap_calc.calculate(&data)?;
    
    println!("Session VWAP: ${:.2}", vwap_result.vwap);
    println!("Total Volume: {:.0}", vwap_result.total_volume);
    
    // Create trading strategy
    let config = StrategyConfig {
        name: "VWAP Strategy".to_string(),
        vwap_type: VWAPType::Session,
        max_position_size: 10000.0,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.04,
        ..Default::default()
    };
    
    let mut strategy = VWAPStrategy::with_config(config);
    let signals = strategy.execute(&data)?;
    
    // Create performance monitor
    let mut monitor = PerformanceMonitor::new()
        .track_vwap_deviations(true)
        .track_volume_spikes(true)
        .track_price_movements(true);
    
    // Process signals and track performance
    for signal in signals {
        match signal.signal_type {
            SignalType::Buy { strength, reason } => {
                println!("BUY: {} (strength: {:.3})", reason, strength);
            }
            SignalType::Sell { strength, reason } => {
                println!("SELL: {} (strength: {:.3})", reason, strength);
            }
            SignalType::Hold { reason } => {
                println!("HOLD: {}", reason);
            }
            _ => {}
        }
        
        // Update performance metrics
        monitor.update_performance(&signal)?;
    }
    
    // Get performance metrics
    let performance = monitor.get_performance();
    let metrics = performance.get_metrics();
    
    println!("Total trades: {}", metrics.total_trades);
    println!("Win rate: {:.2}%", metrics.win_rate * 100.0);
    println!("Total P&L: ${:.2}", metrics.total_pnl);
    
    Ok(())
}
```

### Error Handling

FormicaX uses Rust's Result type for robust error handling:

```rust
use formica_x::{DataLoader, FormicaXError};

fn process_data() -> Result<(), FormicaXError> {
    let mut loader = DataLoader::new("data.csv")?;
    
    for result in loader.records() {
        match result {
            Ok(ohlcv) => {
                // Process valid data
                println!("Valid: {:?}", ohlcv);
            }
            Err(e) => {
                // Handle validation errors
                eprintln!("Invalid data: {}", e);
            }
        }
    }
    
    Ok(())
}
```

## Advanced Usage

### Streaming Large Files

```rust
use formica_x::{StreamingDataLoader};

fn process_large_file() -> Result<(), Box<dyn std::error::Error>> {
    let loader = StreamingDataLoader::new()
        .chunk_size(1000)
        .parallel(true);
    
    // Process large CSV file from examples/csv/ folder
    loader.process_file("examples/csv/daily.csv")?;
    
    Ok(())
}
```

### Parallel Processing

```rust
use formica_x::{DataLoader};
use rayon::prelude::*;

fn parallel_clustering() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from examples/csv/ folder
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let data = loader.load_csv()?;
    
    // Process multiple clustering algorithms in parallel
    let results: Vec<_> = data
        .par_iter()
        .map(|ohlcv| {
            // Perform clustering analysis
            perform_clustering(ohlcv)
        })
        .collect();
    
    Ok(())
}
```

### Flexible CSV Parsing

```rust
use formica_x::{FlexibleDataLoader, ColumnMapping};

fn flexible_csv_parsing() -> Result<(), Box<dyn std::error::Error>> {
    // Auto-detect column mapping using data from examples/csv/ folder
    let mut loader = FlexibleDataLoader::from_path("examples/csv/daily.csv")?;
    
    // Or specify custom column mapping
    let mapping = ColumnMapping::new()
        .timestamp("DateTime")
        .open("O")
        .high("H")
        .low("L")
        .close("C")
        .volume("Vol");
    
    let loader_with_mapping = FlexibleDataLoader::from_path_with_mapping("examples/csv/daily.csv", mapping)?;
    
    // Process data regardless of column order
    for result in loader.records() {
        let ohlcv: OHLCV = result?;
        println!("Processed: {:?}", ohlcv);
    }
    
    Ok(())
}
```

### Custom Data Validation

```rust
use formica_x::{DataLoader, DataValidator, ValidationRule};

fn custom_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from examples/csv/ folder for validation
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let data = loader.load_csv()?;
    
    let validator = DataValidator::new()
        .add_rule(ValidationRule::PriceRange(0.01..1000000.0))
        .add_rule(ValidationRule::VolumeThreshold(100))
        .add_rule(ValidationRule::LogicalConsistency);
    
    let validation_result = validator.validate(&data)?;
    
    if !validation_result.is_valid() {
        for error in validation_result.errors() {
            eprintln!("Validation error: {}", error);
        }
    }
    
    Ok(())
}
```

### Data Transformation

```rust
use formica_x::{DataLoader, DataTransformer, Timeframe, KMeans};

fn transform_data() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from examples/csv/ folder
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let one_min_data = loader.load_csv()?;
    
    let transformer = DataTransformer::new();
    
    // Convert 1-minute data to 5-minute bars
    let five_min_data = transformer.aggregate(&one_min_data, Timeframe::Minute(5))?;
    
    // Perform clustering on aggregated data
    let mut kmeans = KMeans::new(3, 100);
    kmeans.fit(&five_min_data)?;
    
    Ok(())
}
```

## Unified Clustering Interface

FormicaX provides a unified interface for all clustering algorithms through the `ClusteringAlgorithm` trait:

```rust
use formica_x::{
    core::{ClusteringAlgorithm, ClusterResult},
    common::{DistanceMetric, ValidationMetrics}
};

pub trait ClusteringAlgorithm {
    type Config;
    
    fn new(config: Self::Config) -> Self;
    fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError>;
    fn predict(&self, data: &[OHLCV]) -> Result<Vec<usize>, FormicaXError>;
    fn get_cluster_centers(&self) -> Option<Vec<Vec<f64>>>;
    fn validate(&self, data: &[OHLCV]) -> ValidationMetrics;
}

pub struct ClusterResult {
    pub algorithm_name: String,
    pub n_clusters: usize,
    pub cluster_assignments: Vec<usize>,
    pub cluster_centers: Option<Vec<Vec<f64>>>,
    pub inertia: Option<f64>,
    pub silhouette_score: f64,
    pub iterations: usize,
    pub converged: bool,
    pub execution_time: Duration,
    pub noise_points: Vec<usize>,
    pub core_points: Vec<usize>,
    pub border_points: Vec<usize>,
}
```

### Algorithm Selection Guide

Choose the appropriate clustering algorithm based on your data characteristics:

| Algorithm | Use Case | Data Size | Cluster Shape | Noise Handling |
|-----------|----------|-----------|---------------|----------------|
| **K-Means** | Well-separated spherical clusters | Large | Spherical | Poor |
| **DBSCAN** | Arbitrary shapes with noise | Medium-Large | Arbitrary | Excellent |
| **GMM** | Probabilistic clustering | Medium | Elliptical | Good |
| **Hierarchical** | Hierarchy discovery | Small-Medium | Arbitrary | Fair |
| **Affinity Propagation** | Unknown cluster count | Small-Medium | Arbitrary | Good |
| **SOM** | Visualization and topology | Medium | Grid-based | Fair |

## Clustering Algorithms

### K-Means Clustering

```rust
use formica_x::{
    DataLoader, 
    clustering::{KMeans, KMeansConfig, KMeansVariant},
    core::ClusteringAlgorithm,
    common::{DistanceMetric, Initialization}
};

fn kmeans_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from examples/csv/ folder
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let data = loader.load_csv()?;
    
    // Configure K-Means with builder pattern
    let config = KMeansConfig::builder()
        .k(5)                                    // Number of clusters
        .variant(KMeansVariant::Elkan)           // Use Elkan's algorithm
        .max_iterations(200)                     // Maximum iterations
        .tolerance(1e-8)                         // Convergence tolerance
        .distance_metric(DistanceMetric::Euclidean) // Distance function
        .initialization(Initialization::KMeansPlusPlus) // Initialization method
        .parallel(true)                          // Enable parallel processing
        .simd(true)                             // Enable SIMD optimization
        .build()?;
    
    let mut kmeans = KMeans::with_config(config);
    
    // Fit the model using the unified clustering interface
    let result = kmeans.fit(&data)?;
    
    // Access clustering results with metadata
    println!("K-Means completed:");
    println!("- Algorithm: {}", result.algorithm_name);
    println!("- Clusters: {}", result.n_clusters);
    println!("- Iterations: {}", result.iterations);
    println!("- Inertia: {:.6}", result.inertia);
    println!("- Silhouette Score: {:.3}", result.silhouette_score);
    println!("- Execution Time: {:?}", result.execution_time);
    
    Ok(())
}
```

### DBSCAN Clustering

```rust
use formica_x::{
    DataLoader,
    clustering::{DBSCAN, DBSCANConfig, DBSCANVariant},
    core::ClusteringAlgorithm,
    common::{DistanceMetric, SpatialIndex}
};

fn dbscan_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from examples/csv/ folder
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let data = loader.load_csv()?;
    
    // Configure DBSCAN with builder pattern
    let config = DBSCANConfig::builder()
        .epsilon(0.5)                            // Neighborhood radius
        .min_pts(5)                              // Minimum points for core
        .variant(DBSCANVariant::Parallel)        // Use parallel implementation
        .distance_metric(DistanceMetric::Euclidean) // Distance function
        .spatial_index(SpatialIndex::KDTree)     // Spatial indexing method
        .adaptive_epsilon(true)                  // Auto-tune epsilon
        .parallel(true)                          // Enable parallel processing
        .build()?;
    
    let mut dbscan = DBSCAN::with_config(config);
    
    // Fit the model using the unified interface
    let result = dbscan.fit(&data)?;
    
    // Access comprehensive results
    println!("DBSCAN clustering completed:");
    println!("- Algorithm: {}", result.algorithm_name);
    println!("- Clusters found: {}", result.n_clusters);
    println!("- Noise points: {}", result.noise_points.len());
    println!("- Core points: {}", result.core_points.len());
    println!("- Border points: {}", result.border_points.len());
    println!("- Silhouette Score: {:.3}", result.silhouette_score);
    println!("- Execution Time: {:?}", result.execution_time);
    
    Ok(())
}
```

### Gaussian Mixture Models

```rust
use formica_x::{DataLoader, GMM, GMMConfig};

fn gmm_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from examples/csv/ folder
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let data = loader.load_csv()?;
    
    // Configure GMM
    let config = GMMConfig::new()
        .n_components(4)         // Number of components
        .max_iterations(100)     // Maximum EM iterations
        .tolerance(1e-6)         // Convergence tolerance
        .covariance_type(CovarianceType::Full)
        .initialization(GMMInit::KMeans);
    
    let mut gmm = GMM::with_config(config);
    gmm.fit(&data)?;
    
    // Get results
    let clusters = gmm.predict(&data)?;
    let probabilities = gmm.predict_proba(&data)?;
    
    println!("GMM clustering completed");
    
    Ok(())
}
```

### Hierarchical Clustering

```rust
use formica_x::{DataLoader, Hierarchical, HierarchicalConfig};

fn hierarchical_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from examples/csv/ folder
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let data = loader.load_csv()?;
    
    // Configure Hierarchical Clustering
    let config = HierarchicalConfig::new()
        .linkage(LinkageMethod::Ward)
        .distance_metric(DistanceMetric::Euclidean)
        .n_clusters(5);          // Number of clusters to extract
    
    let mut hierarchical = Hierarchical::with_config(config);
    hierarchical.fit(&data)?;
    
    // Get results
    let clusters = hierarchical.get_clusters()?;
    let dendrogram = hierarchical.get_dendrogram()?;
    
    println!("Hierarchical clustering completed");
    
    Ok(())
}
```

## Data Conversion Tools

The library includes utilities for:

### Converting from Other Formats

```rust
use formica_x::{DataConverter, Format};

fn convert_data() -> Result<(), Box<dyn std::error::Error>> {
    let converter = DataConverter::new();
    
    // Convert from JSON format
    let ohlcv_data = converter.from_format(&json_data, Format::JSON)?;
    
    // Convert to Parquet format
    converter.to_format(&ohlcv_data, Format::Parquet, "output.parquet")?;
    
    Ok(())
}
```

### Feature Engineering

```rust
use formica_x::{FeatureEngineer, TechnicalFeatures};

fn engineer_features() -> Result<(), Box<dyn std::error::Error>> {
    let engineer = FeatureEngineer::new();
    
    // Add technical indicators as features
    let features = engineer.add_features(&ohlcv_data, vec![
        TechnicalFeatures::Returns,
        TechnicalFeatures::Volatility,
        TechnicalFeatures::VolumeRatio,
        TechnicalFeatures::PriceRange,
    ])?;
    
    // Use engineered features for clustering
    let mut kmeans = KMeans::new(3, 100);
    kmeans.fit(&features)?;
    
    Ok(())
}
```

### Timestamp Conversion

```rust
use formica_x::{TimestampConverter, Timezone};

fn convert_timestamps() -> Result<(), Box<dyn std::error::Error>> {
    let converter = TimestampConverter::new();
    
    // Convert from local timezone to UTC
    let utc_data = converter.to_utc(&local_data, Timezone::EST)?;
    
    // Convert from Unix timestamp to ISO 8601
    let iso_data = converter.from_unix(&unix_data)?;
    
    Ok(())
}
```

## Performance Optimization

### Algorithm-Specific Optimizations

#### K-Means Performance
```rust
use formica_x::clustering::{KMeansConfig, KMeansVariant};

// Choose the best variant for your data
let config = KMeansConfig::builder()
    .variant(match data_characteristics {
        DataCharacteristics::HighDimensional => KMeansVariant::Hamerly,
        DataCharacteristics::LargeClusters => KMeansVariant::Elkan,
        DataCharacteristics::Streaming => KMeansVariant::MiniBatch,
        _ => KMeansVariant::Lloyd,
    })
    .simd(true)                    // Enable SIMD acceleration
    .parallel(true)                // Enable parallel processing
    .memory_efficient(true)        // Optimize for memory usage
    .build()?;
```

#### DBSCAN Performance
```rust
use formica_x::clustering::{DBSCANConfig, SpatialIndex};

let config = DBSCANConfig::builder()
    .spatial_index(match dimensions {
        dims if dims <= 10 => SpatialIndex::KDTree,
        dims if dims <= 50 => SpatialIndex::BallTree,
        _ => SpatialIndex::LSH,        // Locality-sensitive hashing for high dimensions
    })
    .parallel(true)
    .memory_streaming(data.len() > 1_000_000) // Stream for large datasets
    .build()?;
```

### Memory Management
- **Streaming Processing**: For datasets > 1GB, use streaming to process data in chunks
- **Memory Pooling**: Reuse allocated memory for repeated clustering operations
- **Zero-Copy Operations**: Use references and slices to avoid unnecessary data copying
- **SIMD Alignment**: Ensure data is properly aligned for vectorized operations

### CPU Optimization
- **SIMD Acceleration**: Enable vectorized operations for 4-16x speedup
- **Parallel Processing**: Utilize all CPU cores with work-stealing algorithms
- **Cache Locality**: Use Structure-of-Arrays (SoA) layout for better cache performance
- **Branch Prediction**: Minimize branching in hot loops

### Benchmarking and Profiling
```rust
use formica_x::performance::{Profiler, PerformanceMetrics};

let profiler = Profiler::new()
    .track_memory(true)
    .track_cpu_usage(true)
    .track_cache_misses(true);

let _guard = profiler.start();
let result = algorithm.fit(&data)?;
let metrics = profiler.stop();

println!("Performance Metrics:");
println!("- Peak Memory: {} MB", metrics.peak_memory_mb);
println!("- CPU Usage: {:.1}%", metrics.avg_cpu_usage);
println!("- Cache Miss Rate: {:.2}%", metrics.cache_miss_rate);
println!("- SIMD Usage: {:.1}%", metrics.simd_utilization);
```

## Common Issues

### CSV Format Errors
- Ensure your CSV has all required columns (timestamp, open, high, low, close, volume)
- Check that timestamps are in ISO 8601 format
- Verify all numeric fields are valid numbers
- Column names are case-insensitive and order-independent

### Memory Issues
- Use streaming for large files
- Enable parallel processing for better performance
- Consider chunking data for very large datasets

### Performance Issues
- Use SIMD-optimized functions where available
- Enable parallel processing
- Profile your specific use case

## Best Practices

### Modern Clustering Architecture

#### 1. Use the Unified Interface
```rust
// Good: Use the common ClusteringAlgorithm trait
fn cluster_data<T: ClusteringAlgorithm>(
    algorithm: &mut T, 
    data: &[OHLCV]
) -> Result<ClusterResult, FormicaXError> {
    algorithm.fit(data)
}

// Bad: Algorithm-specific implementations
// fn cluster_with_kmeans(kmeans: &mut KMeans, data: &[OHLCV]) { ... }
// fn cluster_with_dbscan(dbscan: &mut DBSCAN, data: &[OHLCV]) { ... }
```

#### 2. Builder Pattern for Configuration
```rust
// Good: Use builder pattern for clean configuration
let config = KMeansConfig::builder()
    .k(5)
    .variant(KMeansVariant::Elkan)
    .parallel(true)
    .simd(true)
    .build()?;

// Bad: Direct struct initialization
// let config = KMeansConfig { k: 5, variant: ..., parallel: true, ... };
```

#### 3. Modular Component Reuse
```rust
// Good: Reuse common components
use formica_x::common::{DistanceMetric, Initialization, Convergence};

let distance = DistanceMetric::Euclidean.with_simd();
let init = Initialization::KMeansPlusPlus.with_parallel();
let convergence = Convergence::builder()
    .tolerance(1e-8)
    .max_iterations(200)
    .early_stopping(true)
    .build();
```

### Performance Best Practices

#### 1. Algorithm Selection
```rust
use formica_x::analysis::AlgorithmSelector;

// Automatically select the best algorithm for your data
let selector = AlgorithmSelector::new()
    .analyze_data(&data)
    .consider_performance(true)
    .consider_accuracy(true);

let best_algorithm = selector.recommend()?;
```

#### 2. Memory-Efficient Processing
```rust
// For large datasets, use streaming
if data.len() > 1_000_000 {
    let streaming_config = config.with_streaming(true)
                                .chunk_size(10_000);
}

// Use memory pooling for repeated operations
let pool = MemoryPool::new().initial_size(data.len());
let config = config.with_memory_pool(pool);
```

#### 3. Validation and Quality Assurance
```rust
use formica_x::validation::{ClusterValidator, QualityMetrics};

let validator = ClusterValidator::new()
    .silhouette_analysis(true)
    .stability_analysis(true)
    .statistical_tests(true);

let quality = validator.assess(&result)?;
if quality.overall_score < 0.7 {
    // Consider different algorithm or parameters
    warn!("Low clustering quality: {:.2}", quality.overall_score);
}
```

### Error Handling and Robustness

#### 1. Comprehensive Error Types
```rust
use formica_x::error::{FormicaXError, ClusteringError};

match algorithm.fit(&data) {
    Ok(result) => { /* process result */ },
    Err(FormicaXError::Clustering(ClusteringError::ConvergenceFailure { .. })) => {
        // Handle convergence issues
        warn!("Algorithm failed to converge, trying with different parameters");
    },
    Err(FormicaXError::Data(data_error)) => {
        // Handle data validation errors
        error!("Data validation failed: {}", data_error);
    },
    Err(e) => return Err(e),
}
```

#### 2. Graceful Degradation
```rust
// Fallback strategies for robustness
let primary_config = KMeansConfig::builder()
    .variant(KMeansVariant::Elkan)
    .build()?;

let fallback_config = KMeansConfig::builder()
    .variant(KMeansVariant::Lloyd)  // More stable variant
    .max_iterations(500)            // More iterations
    .build()?;

let result = try_with_fallback(primary_config, fallback_config, &data)?;
```

### Testing and Validation

#### 1. Property-Based Testing
```rust
#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn kmeans_always_converges(
            data in generate_valid_ohlcv_data(),
            k in 2..10usize
        ) {
            let config = KMeansConfig::builder().k(k).build().unwrap();
            let mut kmeans = KMeans::with_config(config);
            let result = kmeans.fit(&data).unwrap();
            
            assert!(result.converged);
            assert_eq!(result.n_clusters, k);
        }
    }
}
```

#### 2. Benchmark-Driven Development
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_kmeans(c: &mut Criterion) {
    let data = load_test_data();
    
    c.bench_function("kmeans_elkan", |b| {
        b.iter(|| {
            let config = KMeansConfig::builder()
                .variant(KMeansVariant::Elkan)
                .build().unwrap();
            let mut kmeans = KMeans::with_config(config);
            black_box(kmeans.fit(&data))
        })
    });
}
```

This developer guide provides all the information needed to effectively use FormicaX for clustering analysis of financial data.

## 🎯 **Cursor Implementation Rules**

### **MANDATORY Development Ethos**

When contributing to FormicaX, **ALWAYS** follow these non-negotiable rules:

#### **Rule 1: Code Coverage First**
```bash
# BEFORE writing any code:
cargo tarpaulin --out Html --output-dir coverage

# AFTER implementing:
cargo tarpaulin --out Html --output-dir coverage --fail-under 95
```

**Coverage Requirements:**
- [ ] **95% minimum coverage** for all new code
- [ ] **100% coverage** for public APIs and error paths
- [ ] **Property-based tests** for all algorithms
- [ ] **Integration tests** for end-to-end workflows
- [ ] **Benchmark tests** for performance-critical code

#### **Rule 2: Stop and Review**
After every significant implementation:

**Review Checklist:**
- [ ] **Coverage Check**: `cargo tarpaulin --fail-under 95`
- [ ] **No Duplication**: `cargo clippy --all-targets --all-features`
- [ ] **Clean Code**: Self-documenting, modular, testable
- [ ] **Performance**: `cargo bench` - no regressions
- [ ] **Documentation**: Examples updated with `examples/csv/` data

#### **Rule 3: Latest Dependencies**
```bash
# Weekly dependency check:
cargo outdated
cargo update
cargo audit
cargo check --all-features
cargo test --all-features
```

**Dependency Rules:**
- [ ] **Latest stable versions** from crates.io
- [ ] **No pinned versions** unless absolutely necessary
- [ ] **Weekly updates** with compatibility verification
- [ ] **Security audit** on every update

#### **Rule 4: Clean, Modular Code**
```rust
// ✅ GOOD: Clean, testable, modular
pub trait ClusteringAlgorithm {
    type Config;
    fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError>;
}

// ✅ GOOD: Builder pattern
let config = KMeansConfig::builder()
    .k(5)
    .parallel(true)
    .simd(true)
    .build()?;

// ❌ BAD: Duplicated, hard to test
fn kmeans_inline(data: &[OHLCV], k: usize) -> Vec<usize> {
    // 200 lines of inline algorithm
}

// ❌ BAD: Outdated dependencies
[dependencies]
serde = "1.0.100"  # Pinned old version
```

### **Implementation Workflow**

#### **Phase 1: Test-Driven Development**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    #[test]
    fn test_new_feature() {
        // Write test FIRST
        let expected = expected_result();
        let actual = new_feature(input_data);
        assert_eq!(actual, expected);
    }
    
    proptest! {
        #[test]
        fn test_new_feature_properties(data in generate_test_data()) {
            // Property-based tests
            let result = new_feature(&data);
            assert!(validate_properties(&result));
        }
    }
}
```

#### **Phase 2: Implementation**
```rust
// Implement with clean, modular code
pub fn new_feature(data: &[OHLCV]) -> Result<FeatureResult, FormicaXError> {
    // Clean, readable implementation
    // Comprehensive error handling
    // No code duplication
}
```

#### **Phase 3: Validation**
```bash
# Run all quality checks
cargo test --all-features
cargo tarpaulin --fail-under 95
cargo clippy --all-targets --all-features -- -D warnings
cargo bench
cargo audit
```

#### **Phase 4: Documentation**
```rust
/// Example using examples/csv/ data
/// 
/// # Example
/// ```rust
/// use formica_x::{DataLoader, NewFeature};
/// 
/// let mut loader = DataLoader::new("examples/csv/daily.csv");
/// let data = loader.load_csv()?;
/// let result = NewFeature::new().process(&data)?;
/// ```
pub struct NewFeature {
    // Implementation
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
| **Dependencies** | `cargo outdated` | Latest stable |

### **Failure Recovery**

#### **Coverage Below 95%**
```bash
# Identify uncovered code
cargo tarpaulin --out Html --output-dir coverage

# Add missing tests
# Re-run until > 95%
cargo tarpaulin --fail-under 95
```

#### **Code Duplication**
```bash
# Detect duplication
cargo clippy --all-targets --all-features

# Refactor into shared modules
# Update tests
cargo test --all-features
```

#### **Outdated Dependencies**
```bash
# Update dependencies
cargo update

# Check compatibility
cargo check --all-features
cargo test --all-features

# If breaking changes, update code
```

### **Development Tools**

#### **Pre-commit Hook**
```bash
#!/bin/bash
# .git/hooks/pre-commit

set -e

echo "Running pre-commit quality gates..."

# Coverage check
cargo tarpaulin --out Html --output-dir coverage --fail-under 95

# Dependency check
cargo outdated --exit-code 1

# Security audit
cargo audit

# Code quality
cargo clippy --all-targets --all-features -- -D warnings

# Tests
cargo test --all-features

echo "✅ All quality gates passed!"
```

#### **Daily Development Routine**
```bash
# Start of day
cargo update
cargo audit
cargo outdated

# Before committing
cargo test --all-features
cargo tarpaulin --fail-under 95
cargo clippy --all-targets --all-features -- -D warnings

# End of day
cargo bench
cargo doc --open
```

**Remember: These rules are MANDATORY. Every line of code must follow this ethos to maintain FormicaX's high quality standards.**

## Code Coverage and Examples

### Code Coverage Requirements

FormicaX maintains strict code coverage standards to ensure reliability and quality:

#### Coverage Standards
- **Minimum Coverage**: > 95% code coverage for all public interfaces
- **Coverage Tool**: Uses `cargo-tarpaulin` for accurate coverage reporting
- **Coverage Reports**: HTML reports generated in `coverage/` directory
- **CI Integration**: Coverage checks included in continuous integration pipeline

#### Running Coverage Analysis
```bash
# Install coverage tool
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage

# View coverage report
open coverage/tarpaulin-report.html
```

#### Coverage Targets
- **Public APIs**: 100% coverage for all public functions and methods
- **Error Handling**: 100% coverage for all error paths
- **Edge Cases**: > 90% coverage for boundary conditions
- **Integration Tests**: > 95% coverage for end-to-end workflows

### Examples Structure

All examples are organized in the `examples/` folder with the following structure:

```
examples/
├── basic_usage/
│   ├── data_loading.rs          # Basic CSV loading examples
│   ├── kmeans_clustering.rs     # K-Means clustering examples
│   └── dbscan_clustering.rs     # DBSCAN clustering examples
├── advanced_usage/
│   ├── streaming_processing.rs  # Large file processing
│   ├── parallel_clustering.rs   # Parallel algorithm usage
│   └── custom_validation.rs     # Custom validation rules
├── clustering_algorithms/
│   ├── kmeans_examples.rs       # Comprehensive K-Means examples
│   ├── dbscan_examples.rs       # Comprehensive DBSCAN examples
│   ├── gmm_examples.rs          # Gaussian Mixture Model examples
│   ├── hierarchical_examples.rs # Hierarchical clustering examples
│   ├── affinity_propagation_examples.rs # Affinity propagation examples
│   └── som_examples.rs          # Self-Organizing Map examples
├── data_processing/
│   ├── csv_parsing.rs           # CSV parsing and validation
│   ├── data_validation.rs       # Data validation examples
│   └── feature_engineering.rs   # Feature engineering examples
└── csv/
    ├── daily.csv                # Daily OHLCV data
    ├── hourly.csv               # Hourly OHLCV data (to be added)
    └── minute.csv               # Minute OHLCV data (to be added)
```

### Example Requirements

#### CSV Data Source
- **All examples must use CSV files from `examples/csv/` folder**
- **Primary data source**: `examples/csv/daily.csv` for most examples
- **Additional sources**: `hourly.csv` and `minute.csv` for time-specific examples
- **Data format**: Standard OHLCV format with flexible column detection

#### Example Standards
- **External Interfaces**: Every public API must have at least one working example
- **Error Handling**: Include examples showing proper error handling and recovery
- **Documentation**: Each example must have clear comments explaining the code
- **Runnable**: All examples must compile and run successfully
- **Realistic Data**: Use realistic financial data scenarios in examples

#### Example File Naming Convention
- Use snake_case for example file names
- Include the main functionality in the filename
- Group related examples in subdirectories
- Use descriptive names that indicate the example's purpose

#### Example Documentation Standards
- **Header Comments**: Each example file must have a header explaining its purpose
- **Inline Comments**: Include comments explaining complex operations
- **Output Examples**: Show expected output where relevant
- **Error Scenarios**: Demonstrate error handling with realistic scenarios
- **Performance Notes**: Include performance considerations where applicable

### Running Examples

```bash
# Run a specific example
cargo run --example data_loading

# Run all examples (if implemented)
cargo run --examples

# Build examples without running
cargo build --examples
```

### Example Development Guidelines

#### Creating New Examples
1. **Use CSV Data**: Always use files from `examples/csv/` folder
2. **Follow Structure**: Place examples in appropriate subdirectories
3. **Include Documentation**: Add comprehensive comments and explanations
4. **Test Thoroughly**: Ensure examples work with current CSV data
5. **Handle Errors**: Include proper error handling in examples

#### Example Content Requirements
- **Data Loading**: Show how to load and validate CSV data
- **Algorithm Usage**: Demonstrate clustering algorithm configuration and usage
- **Result Processing**: Show how to interpret and use clustering results
- **Error Handling**: Demonstrate proper error handling and recovery
- **Performance**: Include performance considerations and optimizations

#### Example Testing
- **Compilation**: All examples must compile without warnings
- **Execution**: All examples must run successfully with provided CSV data
- **Output Validation**: Examples should produce expected, meaningful output
- **Error Scenarios**: Test error handling with malformed data

## Trading Module Development

### Trading Module Overview

The FormicaX trading module provides comprehensive tools for implementing VWAP-based trading strategies, real-time signal generation, and performance monitoring.

### Core Trading Components

#### VWAP Calculator
```rust
use formica_x::trading::VWAPCalculator;

// Session-based VWAP calculation
let mut vwap_calc = VWAPCalculator::session_based();
let vwap_result = vwap_calc.calculate(&data)?;

// Incremental VWAP updates
let vwap_result = vwap_calc.calculate_incremental(&[new_ohlcv])?;
```

#### Signal Generator
```rust
use formica_x::trading::SignalGenerator;

// Create signal generator with custom thresholds
let mut signal_gen = SignalGenerator::with_thresholds(SignalThresholds {
    vwap_buy_threshold: 0.001,   // 0.1% above VWAP
    vwap_sell_threshold: 0.001,  // 0.1% below VWAP
    volume_threshold: 1.5,       // 50% above average volume
    price_change_threshold: 0.005, // 0.5% price change
    min_confidence: 0.6,         // 60% minimum confidence
});

// Generate signals incrementally
let signal = signal_gen.generate_signal_incremental(&ohlcv)?;
```

#### Performance Monitor
```rust
use formica_x::trading::PerformanceMonitor;

// Create performance monitor
let mut monitor = PerformanceMonitor::new()
    .track_vwap_deviations(true)
    .track_volume_spikes(true)
    .track_price_movements(true)
    .track_trade_performance(true);

// Update performance metrics
let performance = monitor.update_performance(&ohlcv)?;

// Get performance metrics
let metrics = performance.get_metrics();
println!("Win rate: {:.2}%", metrics.win_rate * 100.0);
```

#### Alert Generator
```rust
use formica_x::trading::AlertGenerator;

// Create alert generator
let mut alert_gen = AlertGenerator::new()
    .vwap_deviation_threshold(0.02)  // 2% VWAP deviation
    .volume_spike_threshold(2.0)     // 2x average volume
    .price_movement_threshold(0.01); // 1% price movement

// Check for alerts
let alerts = alert_gen.check_alerts(&performance)?;
```

### Trading Strategy Implementation

#### Basic VWAP Strategy
```rust
use formica_x::trading::{VWAPStrategy, StrategyConfig, VWAPType};

// Create strategy configuration
let config = StrategyConfig {
    name: "VWAP Strategy".to_string(),
    vwap_type: VWAPType::Session,
    max_position_size: 10000.0,
    stop_loss_pct: 0.02,      // 2% stop loss
    take_profit_pct: 0.04,    // 4% take profit
    max_drawdown: 0.10,       // 10% max drawdown
    risk_per_trade: 0.01,     // 1% risk per trade
    ..Default::default()
};

// Create and execute strategy
let mut strategy = VWAPStrategy::with_config(config);
let signals = strategy.execute(&data)?;
```

#### Real-Time Trading System
```rust
use formica_x::trading::RealTimeProcessor;
use std::time::Duration;

// Create real-time processor
let mut processor = RealTimeProcessor::new()
    .update_frequency(Duration::from_millis(100))
    .buffer_size(1000)
    .parallel_processing(true);

// Start real-time processing
processor.start_processing(|ohlcv| {
    // Process each tick
    let vwap_result = vwap_calc.calculate_incremental(&[ohlcv.clone()])?;
    let signal = signal_gen.generate_signal_incremental(ohlcv)?;
    
    // Execute signal if actionable
    if signal.is_actionable() {
        match signal.signal_type {
            SignalType::Buy { .. } => execute_buy_order(ohlcv.close),
            SignalType::Sell { .. } => execute_sell_order(ohlcv.close),
            _ => {}
        }
    }
    
    Ok(())
})?;
```

### Trading Module Testing

#### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use formica_x::{DataLoader, trading::*};

    #[test]
    fn test_vwap_calculation() {
        let mut loader = DataLoader::new("examples/csv/daily.csv");
        let data = loader.load_csv().unwrap();
        
        let mut vwap_calc = VWAPCalculator::session_based();
        let result = vwap_calc.calculate(&data).unwrap();
        
        assert!(result.vwap > 0.0);
        assert!(result.total_volume > 0.0);
    }

    #[test]
    fn test_signal_generation() {
        let mut loader = DataLoader::new("examples/csv/daily.csv");
        let data = loader.load_csv().unwrap();
        
        let mut signal_gen = SignalGenerator::new();
        let signal = signal_gen.generate_signal_incremental(&data[0]).unwrap();
        
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
    }
}
```

#### Integration Tests
```rust
#[test]
fn test_complete_trading_strategy() {
    let mut loader = DataLoader::new("examples/csv/daily.csv");
    let data = loader.load_csv().unwrap();
    
    let config = StrategyConfig {
        name: "Test Strategy".to_string(),
        vwap_type: VWAPType::Session,
        max_position_size: 10000.0,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.04,
        ..Default::default()
    };
    
    let mut strategy = VWAPStrategy::with_config(config);
    let signals = strategy.execute(&data).unwrap();
    
    assert!(!signals.is_empty());
    assert!(signals.iter().all(|s| s.is_valid()));
}
```

### Trading Module Performance

#### Performance Targets
- **VWAP Calculation**: < 1 microsecond per update
- **Signal Generation**: < 100 microseconds per signal
- **Performance Monitoring**: < 10 microseconds per update
- **Alert Generation**: < 50 microseconds per alert

#### Performance Optimization
```rust
// Use incremental calculations for real-time performance
let vwap_result = vwap_calc.calculate_incremental(&[new_ohlcv])?;

// Enable parallel processing for large datasets
let mut processor = RealTimeProcessor::new()
    .parallel_processing(true)
    .buffer_size(1000);

// Use memory pooling for frequent allocations
let mut monitor = PerformanceMonitor::new()
    .memory_pooling(true)
    .preallocate_buffers(true);
```

### Trading Module Examples

All trading examples should use data from `examples/csv/` folder and demonstrate:

1. **Data Loading**: Load market data from CSV files
2. **VWAP Calculation**: Calculate VWAP using different methods
3. **Signal Generation**: Generate trading signals with custom thresholds
4. **Performance Monitoring**: Track and analyze trading performance
5. **Alert Generation**: Generate alerts for market conditions
6. **Strategy Execution**: Execute complete trading strategies
7. **Backtesting**: Backtest strategies with historical data
8. **Real-Time Processing**: Process real-time market data

### Trading Module Documentation

#### API Documentation
- **VWAPCalculator**: Methods for calculating VWAP
- **SignalGenerator**: Methods for generating trading signals
- **PerformanceMonitor**: Methods for tracking performance metrics
- **AlertGenerator**: Methods for generating market alerts
- **VWAPStrategy**: Complete trading strategy implementation
- **RealTimeProcessor**: Real-time data processing capabilities

#### Example Documentation
- **Basic Usage**: Simple VWAP calculation and signal generation
- **Advanced Usage**: Complex trading strategies and performance analysis
- **Real-Time Usage**: Real-time trading system implementation
- **Backtesting**: Strategy backtesting and performance analysis 