# FormicaX Architecture & Implementation

## System Architecture

### Overview

FormicaX is designed as a modular, high-performance clustering library with a focus on financial data analysis using OHLCV data and advanced clustering algorithms.

### Core Architecture

#### 1. Data Layer
- **DataLoader**: Handles CSV file parsing and data validation
- **DataValidator**: Ensures data integrity and format compliance
- **DataTransformer**: Converts between different data formats

#### 2. Common Traits and Interfaces
- **ClusteringAlgorithm**: Unified trait for all clustering algorithms
- **ClusteringConfig**: Builder pattern for algorithm configuration
- **DistanceMetric**: Pluggable distance functions with SIMD optimization
- **Initialization**: Common initialization strategies (K-means++, random, etc.)
- **Convergence**: Unified convergence criteria across algorithms

#### 3. Clustering Layer (Modular Design)
- **KMeans Module**: Lloyd's, Elkan's, Hamerly's, and Mini-batch variants
- **DBSCAN Module**: Standard, parallel, incremental, and approximate variants
- **GMM Module**: Standard EM, Variational Bayes, and robust variants
- **Hierarchical Module**: Agglomerative methods with multiple linkage options
- **AffinityPropagation Module**: Standard and hierarchical message passing
- **SOM Module**: Standard, growing, and hierarchical self-organizing maps

#### 4. Performance Layer
- **SIMD Operations**: Vectorized distance calculations and matrix operations
- **Memory Management**: Zero-copy, memory pooling, and streaming
- **Parallel Processing**: Lock-free concurrent algorithms with work-stealing

#### 5. Analysis Layer
- **ClusterAnalyzer**: Statistical analysis and pattern recognition
- **ValidationMetrics**: Unified clustering quality metrics
- **Predictor**: Ensemble prediction with confidence estimation
- **Visualization**: Cluster visualization and dimensionality reduction

### Data Flow

```
CSV Input → DataLoader → DataValidator → Clustering Algorithm → ClusterAnalyzer → Predictions
```

### Key Design Principles

1. **Modularity**: Each clustering algorithm is self-contained and replaceable
2. **Performance**: Optimized for large financial datasets
3. **Accuracy**: Precise clustering with maximum accuracy implementations
4. **Extensibility**: Easy to add new clustering algorithms
5. **Validation**: Comprehensive clustering validation and error handling

## Project Configuration

### Cargo.toml Configuration

```toml
[package]
name = "formica_x"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
description = "High-performance Rust clustering library for financial data analysis"
license = "Apache-2.0"
repository = "https://github.com/yourusername/formica_x"
documentation = "https://docs.rs/formica_x"
readme = "README.md"
keywords = ["clustering", "machine-learning", "finance", "ohlcv", "trading"]
categories = ["data-structures", "science", "financial", "machine-learning"]

[dependencies]
# Core dependencies
serde = { version = "1.0", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
thiserror = "1.0"

# CSV and data processing
csv = "1.3"
ndarray = { version = "0.15", features = ["blas"] }

# Parallel processing and concurrency
rayon = "1.7"
crossbeam = "0.8"
parking_lot = "0.12"
dashmap = "5.4"

# Memory management
memmap2 = "0.9"

# Numeric operations
num-traits = "0.2"

# Clustering algorithms
ndarray-stats = "0.5"
rand = "0.8"
rand_distr = "0.4"
statrs = "0.16"

[dev-dependencies]
# Testing
criterion = "0.5"
proptest = "1.3"
tempfile = "3.8"

# Documentation
rustdoc-stripper = "0.1"

[features]
default = ["std"]
std = []
simd = ["packed_simd"]
parallel = ["rayon"]

# Enable all features for development
dev = ["simd", "parallel"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

[profile.dev]
opt-level = 0
debug = true

[profile.bench]
opt-level = 3
lto = true
codegen-units = 1

[[bench]]
name = "kmeans_benchmark"
harness = false

[[bench]]
name = "dbscan_benchmark"
harness = false

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]

[package.metadata.release]
pre-release-commit-message = "chore: release {{version}}"
tag-message = "{{project_name}} {{version}}"
tag-name = "{{version}}"
```

### Technology Stack

- **Language**: Rust 2021 edition
- **Core Dependencies**: 
  - `csv` for CSV file parsing with flexible column detection
  - `serde` for data serialization/deserialization
  - `chrono` for datetime handling with timezone support
  - `thiserror` for ergonomic error handling
  - `criterion` for performance benchmarking
  - `rayon` for parallel processing with work-stealing
  - `ndarray` for SIMD-optimized numerical operations
  - `parking_lot` for high-performance synchronization primitives
  - `crossbeam` for lock-free data structures
  - `memmap2` for zero-copy file access
  - `dashmap` for concurrent hash maps
  - `num-traits` for generic numeric operations

## File Structure

```
src/
├── lib.rs
├── core/
│   ├── mod.rs
│   ├── data_models.rs
│   ├── data_loader.rs
│   ├── data_validator.rs
│   └── traits.rs                    # Common clustering traits
├── clustering/
│   ├── mod.rs
│   ├── common/                      # Shared clustering components
│   │   ├── mod.rs
│   │   ├── distance.rs              # Distance metrics (SIMD optimized)
│   │   ├── initialization.rs        # K-means++ and other initializers
│   │   └── convergence.rs           # Convergence criteria
│   ├── kmeans/                      # K-Means module
│   │   ├── mod.rs
│   │   ├── algorithm.rs             # Core K-Means with Elkan's optimization
│   │   ├── parallel.rs              # Parallel implementation
│   │   └── config.rs                # Configuration and builder
│   ├── dbscan/                      # DBSCAN module
│   │   ├── mod.rs
│   │   ├── algorithm.rs             # Core DBSCAN with KD-tree
│   │   ├── spatial.rs               # Spatial indexing structures
│   │   └── config.rs                # Configuration and builder
│   ├── gmm/                         # Gaussian Mixture Models
│   │   ├── mod.rs
│   │   ├── algorithm.rs             # EM algorithm with numerical stability
│   │   ├── covariance.rs            # Covariance matrix handling
│   │   └── config.rs                # Configuration and builder
│   ├── hierarchical/                # Hierarchical clustering
│   │   ├── mod.rs
│   │   ├── algorithm.rs             # Linkage methods
│   │   ├── dendrogram.rs            # Dendrogram operations
│   │   └── config.rs                # Configuration and builder
│   ├── affinity_propagation/        # Affinity Propagation
│   │   ├── mod.rs
│   │   ├── algorithm.rs             # Message passing algorithm
│   │   └── config.rs                # Configuration and builder
│   ├── som/                         # Self-Organizing Maps
│   │   ├── mod.rs
│   │   ├── algorithm.rs             # SOM training algorithm
│   │   ├── topology.rs              # Grid topology handling
│   │   └── config.rs                # Configuration and builder
│   └── validation.rs                # Clustering validation metrics
├── analysis/
│   ├── mod.rs
│   ├── cluster_analyzer.rs
│   ├── predictor.rs
│   └── ensemble.rs                  # Ensemble clustering methods
├── performance/                     # Performance optimization utilities
│   ├── mod.rs
│   ├── simd.rs                      # SIMD operations
│   ├── memory.rs                    # Memory management
│   └── parallel.rs                  # Parallel processing utilities
└── utils/
    ├── mod.rs
    ├── helpers.rs
    └── constants.rs
```

## Implementation Plan

### Phase 1: Project Setup and Core Infrastructure (Weeks 1-2)

#### 1.1 Project Initialization
- [ ] Initialize Rust project with workspace structure
- [ ] Set up `Cargo.toml` with optimized dependencies and features
- [ ] Configure modular workspace with separate crates for algorithms
- [ ] Set up development tools (rustfmt, clippy, cargo-audit, criterion)

#### 1.2 Core Traits and Interfaces
- [ ] Define `ClusteringAlgorithm` trait with unified interface
- [ ] Implement `DistanceMetric` trait with SIMD support
- [ ] Create `ClusteringConfig` trait for builder patterns
- [ ] Add `Convergence` trait for stopping criteria
- [ ] Define `Initialization` trait for centroid/exemplar initialization

#### 1.3 Core Data Structures and Error Handling
- [ ] Define `OHLCV` struct with efficient memory layout
- [ ] Implement `ClusterResult` with metadata and quality metrics
- [ ] Create comprehensive error hierarchy using `thiserror`
- [ ] Add zero-cost abstractions for performance-critical paths
- [ ] Implement streaming data structures for large datasets

### Phase 2: Data Loading and Validation (Weeks 3-4)

#### 2.1 Flexible CSV Parser
- [ ] Implement intelligent column detection with case-insensitive matching
- [ ] Add support for arbitrary column ordering
- [ ] Implement streaming support for large files with memory mapping
- [ ] Add progress reporting and error recovery for malformed data
- [ ] Support custom column mapping for non-standard formats

#### 2.2 Data Validation
- [ ] Create validation rules for OHLCV data
- [ ] Implement logical consistency checks
- [ ] Add data type validation
- [ ] Create validation result types

#### 2.3 Data Transformation
- [ ] Implement data format conversion utilities
- [ ] Add timestamp normalization
- [ ] Create data cleaning functions
- [ ] Implement data aggregation methods

### Phase 3: Common Clustering Infrastructure (Weeks 5-6)

#### 3.1 Shared Components
- [ ] Implement SIMD-optimized distance metrics (Euclidean, Manhattan, Cosine)
- [ ] Create K-means++ initialization with parallel seeding
- [ ] Build convergence monitoring system with multiple criteria
- [ ] Implement memory-efficient spatial data structures

#### 3.2 Performance Infrastructure
- [ ] Create SIMD operation wrappers for vectorized calculations
- [ ] Implement memory pooling system for cluster data
- [ ] Build lock-free parallel processing utilities
- [ ] Create streaming data processing framework

### Phase 4: K-Means Implementation (Weeks 7-8)

#### 4.1 K-Means Variants
- [ ] Implement Lloyd's algorithm with SIMD optimization
- [ ] Add Elkan's algorithm with triangle inequality optimization
- [ ] Implement Hamerly's algorithm for memory efficiency
- [ ] Create Mini-batch K-means for streaming data

#### 4.2 K-Means Advanced Features
- [ ] Add intelligent initialization selection
- [ ] Implement adaptive convergence criteria
- [ ] Create parallel processing with work-stealing
- [ ] Add cluster quality assessment and validation

### Phase 5: DBSCAN Implementation (Weeks 9-10)

#### 5.1 DBSCAN Core Algorithm
- [ ] Implement standard DBSCAN with KD-tree spatial indexing
- [ ] Add R*-tree support for complex spatial queries
- [ ] Implement efficient border point classification
- [ ] Create robust noise detection with statistical validation

#### 5.2 DBSCAN Advanced Variants
- [ ] Implement parallel DBSCAN with lock-free region queries
- [ ] Add incremental DBSCAN for streaming data
- [ ] Create approximate DBSCAN for large-scale data
- [ ] Implement adaptive epsilon calculation with k-distance graph

### Phase 6: Advanced Clustering Algorithms (Weeks 11-12)

#### 6.1 Gaussian Mixture Models
- [ ] Implement numerically stable EM algorithm with log-space computations
- [ ] Add Variational Bayesian GMM with automatic component selection
- [ ] Implement multiple covariance types (diagonal, full, tied)
- [ ] Create online EM for streaming data with concept drift handling
- [ ] Add robust GMM variants using t-distributions

#### 6.2 Hierarchical Clustering
- [ ] Implement SLINK/CLINK algorithms for linear-time linkage
- [ ] Add optimized linkage methods with SIMD acceleration
- [ ] Create memory-efficient union-find data structures
- [ ] Implement parallel hierarchical clustering
- [ ] Add BIRCH-style clustering trees for massive datasets

### Phase 7: Specialized Clustering Algorithms (Weeks 13-14)

#### 7.1 Affinity Propagation
- [ ] Implement message passing with adaptive damping
- [ ] Add sparse similarity matrix operations for scalability
- [ ] Create parallel responsibility and availability updates
- [ ] Implement convergence acceleration with momentum methods
- [ ] Add hierarchical affinity propagation for multi-scale clustering

#### 7.2 Self-Organizing Maps
- [ ] Implement SOM with advanced neighborhood functions
- [ ] Add multiple learning rate schedules (exponential, polynomial, adaptive)
- [ ] Support multi-grid topologies (hexagonal, cylindrical, toroidal)
- [ ] Create growing SOM with dynamic grid expansion
- [ ] Implement hierarchical SOM for complex data structures

### Phase 8: Analysis and Ensemble Methods (Weeks 15-16)

#### 8.1 Unified Analysis Framework
- [ ] Implement comprehensive clustering validation metrics
- [ ] Create cluster quality assessment with multiple criteria
- [ ] Add cluster stability analysis across parameter ranges
- [ ] Implement automated clustering algorithm selection

#### 8.2 Ensemble and Prediction Systems
- [ ] Create ensemble clustering with consensus methods
- [ ] Implement cluster prediction with confidence estimation
- [ ] Add active learning for clustering improvement
- [ ] Create cluster visualization and interpretation tools

### Phase 9: Testing, Optimization, and Documentation (Weeks 17-18)

#### 8.1 Unit Testing
- [ ] Write unit tests for all clustering algorithms
- [ ] Add property-based testing with `proptest`
- [ ] Implement integration tests
- [ ] Add performance regression tests
- [ ] Achieve > 95% code coverage for all public interfaces
- [ ] Test all error conditions and edge cases

#### 8.2 Benchmarking
- [ ] Set up benchmarks using `criterion`
- [ ] Compare performance with other clustering libraries
- [ ] Profile memory usage
- [ ] Optimize based on benchmark results

#### 8.3 Documentation and Examples
- [ ] Create comprehensive examples for all external interfaces
- [ ] Use CSV files from `examples/csv/` folder in all examples
- [ ] Place all examples in `examples/` folder with clear structure
- [ ] Ensure each public API has working code examples
- [ ] Include examples for error handling and edge cases

## Most Efficient Implementation Approaches

### Research-Based Optimizations

Based on analysis of high-performance Rust clustering libraries:

### Clustering Algorithm Accuracy Optimizations

For maximum accuracy in clustering algorithms, FormicaX implements the following research-based optimizations:

#### 1. K-Means Clustering - State-of-the-Art Implementation
- **K-Means++ Initialization**: Use k-means++ for optimal initial centroid placement with parallel seeding
- **Elkan's Algorithm**: Implement Elkan's triangle inequality optimization for 3-5x speedup
- **Hamerly's Algorithm**: Alternative to Elkan's with lower memory overhead for high-dimensional data
- **Mini-Batch K-Means**: For streaming/large datasets with configurable batch sizes
- **Lloyd's Algorithm**: Classic implementation with SIMD-optimized distance calculations
- **Convergence Criteria**: Multiple criteria - relative change < 1e-8, centroid movement, iteration limit
- **Adaptive Tolerance**: Dynamic convergence tolerance based on data characteristics
- **Parallel Processing**: Lock-free parallel implementation using work-stealing

#### 2. DBSCAN - High-Performance Density Clustering
- **Spatial Indexing**: Multi-level approach - KD-tree for low dimensions, LSH for high dimensions
- **R*-Tree Integration**: For complex spatial queries and range searches
- **Parallel DBSCAN**: Lock-free parallel region queries with atomic cluster assignment
- **Adaptive Epsilon**: Auto-tuning using k-distance graph and knee detection
- **Border Point Optimization**: Efficient border point classification with early termination
- **Memory-Efficient**: Streaming implementation for datasets larger than memory
- **Incremental DBSCAN**: Support for data stream processing with cluster evolution
- **Approximate DBSCAN**: Trade-off precision for speed in large-scale scenarios

#### 3. Gaussian Mixture Models (GMM) - Advanced Probabilistic Clustering
- **Numerically Stable EM**: Log-space computations to prevent underflow in high dimensions
- **Variational Bayesian GMM**: Automatic component count selection with regularization
- **Diagonal/Full/Tied Covariance**: Multiple covariance types with adaptive selection
- **Online EM**: Streaming EM algorithm for large datasets with concept drift handling
- **Robust GMM**: Outlier-resistant variants using t-distributions or trimmed likelihood
- **Parallel EM**: Distributed E-step and M-step computation with data parallelism
- **Model Selection**: Integrated BIC/AIC/ICL with cross-validation for robust selection
- **Initialization Strategies**: Smart initialization using k-means++, random, and spectral methods

#### 4. Hierarchical Clustering - Scalable Agglomerative Methods
- **Optimized Linkage Methods**: Single, complete, average, Ward, and centroid linkage with SIMD
- **SLINK/CLINK Algorithms**: Linear-time single/complete linkage for efficiency
- **Memory-Efficient Union-Find**: Disjoint set data structure for large-scale clustering
- **Parallel Hierarchical**: Multi-threaded distance matrix computation and merging
- **Approximate Methods**: BIRCH-style clustering trees for massive datasets
- **Dynamic Programming**: Optimal dendrogram cutting with multiple criteria
- **Streaming Hierarchical**: Online agglomerative clustering for data streams
- **Quality Metrics**: Cophenetic correlation, silhouette analysis, and cluster stability

#### 5. Affinity Propagation - Advanced Message Passing
- **Adaptive Damping**: Dynamic damping factor adjustment based on oscillation detection
- **Sparse Similarity Matrix**: Memory-efficient sparse matrix operations for large datasets
- **Parallel Message Passing**: Lock-free parallel responsibility and availability updates
- **Multiple Preference Strategies**: Median, quantile-based, and input preferences
- **Convergence Acceleration**: Momentum-based and Nesterov acceleration methods
- **Hierarchical AP**: Multi-level affinity propagation for scalability
- **Robust Exemplar Selection**: Stability-based exemplar validation and refinement
- **Early Stopping**: Intelligent termination with oscillation and plateau detection

#### 6. Self-Organizing Maps (SOM) - Modern Neural Clustering
- **Advanced Neighborhood Functions**: Gaussian, bubble, Mexican hat, and adaptive neighborhoods
- **Learning Rate Schedules**: Exponential, polynomial, and adaptive decay strategies
- **Multi-Grid Topologies**: Hexagonal, rectangular, cylindrical, and toroidal grids
- **Parallel SOM**: Multi-threaded BMU search and weight updates with vectorization
- **Growing SOM**: Dynamic grid expansion based on quantization error thresholds
- **Hierarchical SOM**: Multi-level SOMs for complex data structure discovery
- **Batch/Online Hybrid**: Optimal batch size selection for convergence and speed
- **Advanced Metrics**: Quantization error, topographic error, and trustworthiness measures

#### 1. Zero-Copy Data Processing
- **Memory Mapping**: Use `memmap2` for direct file access without copying data into memory
- **Borrowed References**: Process data using references instead of owned values where possible
- **Streaming Parsers**: Parse CSV data in chunks to minimize memory footprint

#### 2. SIMD-Optimized Calculations
- **Auto-Vectorization**: Use compiler intrinsics with runtime CPU feature detection
- **AVX-512/AVX2 Support**: Vectorized distance calculations with 16x/8x parallel processing
- **FMA Operations**: Fused multiply-add for improved precision and performance
- **Horizontal Operations**: Efficient reduction operations for aggregated calculations
- **Memory Alignment**: 64-byte aligned data structures for optimal cache and SIMD performance
- **Portable SIMD**: Fallback implementations for different CPU architectures

#### 3. Lock-Free Concurrency
- **Work-Stealing Scheduler**: Rayon-based work distribution with minimal overhead
- **Lock-Free Data Structures**: Crossbeam channels, DashMap, and atomic collections
- **Compare-and-Swap Loops**: Atomic updates for cluster assignments and centroids
- **Memory Ordering**: Optimized memory ordering for different access patterns
- **NUMA Awareness**: Thread and memory locality optimization for multi-socket systems
- **Parallel Iterators**: Zero-cost parallel abstractions with automatic load balancing

#### 4. Cache-Optimized Data Layout
- **Structure of Arrays (SoA)**: Separate arrays for each feature dimension with optimal stride
- **Cache Line Alignment**: 64-byte aligned data structures with padding elimination
- **Prefetching Strategies**: Software and hardware prefetching for predictable access patterns
- **Memory Hierarchy Optimization**: L1/L2/L3 cache-aware data organization
- **False Sharing Elimination**: Padding and alignment to prevent cache line contention
- **Temporal Locality**: Data layout optimized for access patterns in clustering algorithms

#### 5. High-Performance CSV Parsing
- **Flexible Column Detection**: Use hash maps for O(1) column name matching
- **Streaming Parsers**: Parse large files without loading entire content
- **Error Recovery**: Continue processing after encountering malformed rows

## Performance Considerations

### Memory Management
- **Memory Mapping**: Use `memmap2` for zero-copy file access to large datasets
- **Memory Pooling**: Implement object pools for clustering data structures to avoid allocation overhead
- **Cache Optimization**: Structure of Arrays (SoA) layout for better cache locality and SIMD operations
- **Lazy Loading**: Load data on-demand with intelligent prefetching
- **Arena Allocation**: Use bump allocators for temporary calculations

### Calculation Speed
- **SIMD Vectorization**: Use `ndarray` with AVX2/AVX-512 for 8x-16x speedup on distance calculations
- **Lock-Free Parallelism**: Use `crossbeam` channels and `dashmap` for concurrent data structures
- **Incremental Updates**: Update clusters incrementally with rolling window optimizations
- **Branch Prediction**: Use `likely!`/`unlikely!` hints and profile-guided optimization
- **CPU Cache Optimization**: Align data structures to cache lines and use prefetching

### Accuracy
- **Numerical Precision**: Use double-precision floating point for clustering calculations
- **Error Handling**: Comprehensive validation with `thiserror` and context-aware error messages
- **Data Integrity**: Ensure logical consistency of OHLCV data with runtime checks
- **Precision Control**: Configurable precision with compile-time guarantees
- **Numerical Stability**: Use Kahan summation for distance calculations to prevent accumulation errors

### Clustering Accuracy
- **Numerical Precision**: Use double-precision floating point for clustering calculations
- **Convergence Monitoring**: Implement robust convergence criteria for all iterative algorithms
- **Validation Metrics**: Include silhouette score, cophenetic correlation, and other quality measures
- **Parameter Tuning**: Automatic parameter estimation using statistical methods
- **Reproducibility**: Deterministic algorithms with configurable random seeds

## Success Metrics

### Performance Targets
1. **Throughput**: Process 1GB CSV file in < 5 seconds with streaming and SIMD
2. **Memory Efficiency**: Use < 1.5x memory of input file size with zero-copy operations
3. **Latency**: Sub-100 microsecond clustering predictions with optimized algorithms
4. **Scalability**: Near-linear scaling with CPU cores (>85% efficiency)

### Quality and Accuracy
5. **Algorithm Accuracy**: Clustering quality within 0.5% of reference implementations
6. **Numerical Stability**: Robust convergence across diverse datasets and parameters
7. **Reproducibility**: Deterministic results with configurable random seeds

### Code Quality
8. **Safety**: Zero unsafe code with comprehensive error handling and validation
9. **Coverage**: > 95% test coverage with property-based and integration testing
10. **Modularity**: Clean separation of concerns with reusable components

### Documentation and Usability
11. **API Coverage**: 100% of external interfaces with working examples using `examples/csv/` data
12. **Examples**: Comprehensive examples in `examples/` folder with clear documentation
13. **Performance**: Detailed benchmarks and optimization guides

### Clustering Algorithm Accuracy Targets

8. **K-Means Accuracy**: 
   - Silhouette score > 0.7 for well-separated clusters
   - Convergence in < 100 iterations for 95% of datasets
   - Inertia reduction > 99% from initial to final state

9. **DBSCAN Accuracy**:
   - Noise detection accuracy > 95% on synthetic datasets
   - Cluster purity > 0.9 for labeled datasets
   - Adaptive epsilon estimation within 10% of optimal

10. **GMM Accuracy**:
    - Log-likelihood convergence with relative tolerance < 1e-6
    - BIC/AIC model selection accuracy > 90%
    - Component overlap detection with precision > 0.95

11. **Hierarchical Clustering Accuracy**:
    - Cophenetic correlation > 0.8 for dendrogram quality
    - Linkage method accuracy > 95% compared to reference implementations
    - Memory usage < O(n²) for large datasets

12. **Affinity Propagation Accuracy**:
    - Exemplar selection stability > 90% across multiple runs
    - Convergence rate > 95% within 200 iterations
    - Silhouette analysis for exemplar quality validation

13. **SOM Accuracy**:
    - Quantization error < 0.1 for well-structured data
    - Topographic error < 0.05 for topology preservation
    - Training convergence in < 1000 epochs

## Development Setup

1. **Install Rust**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. **Install tools**: `cargo install cargo-audit cargo-tarpaulin cargo-update`
3. **Setup pre-commit**: Configure git hooks for code quality
4. **Run tests**: `cargo test --all-features`
5. **Run benchmarks**: `cargo bench`
6. **Check code quality**: `cargo clippy --all-targets --all-features`
7. **Check coverage**: `cargo tarpaulin --out Html --output-dir coverage`
8. **Run examples**: `cargo run --example <example_name>`
9. **Update dependencies**: `cargo update` (run regularly)

## Cursor Implementation Rules

### 🎯 **Project Ethos Enforcement**

When implementing any feature or component, **ALWAYS** follow these mandatory rules:

#### 1. **Code Coverage First (MANDATORY)**
```bash
# BEFORE implementing any feature:
cargo tarpaulin --out Html --output-dir coverage

# AFTER implementing:
cargo tarpaulin --out Html --output-dir coverage
# Verify: > 95% coverage for new code
```

**Implementation Checklist:**
- [ ] Write unit tests BEFORE implementing the feature
- [ ] Add integration tests for public APIs
- [ ] Include property-based tests with `proptest`
- [ ] Test all error conditions and edge cases
- [ ] Verify coverage > 95% for new code
- [ ] Add benchmarks for performance-critical code

#### 2. **Stop and Review (MANDATORY)**
After implementing any significant feature:

**Review Checklist:**
- [ ] **Code Coverage**: Run `cargo tarpaulin` and verify > 95%
- [ ] **No Duplication**: Check for code duplication using `cargo clippy`
- [ ] **Modularity**: Ensure clean separation of concerns
- [ ] **Readability**: Code is self-documenting with clear naming
- [ ] **Performance**: Run benchmarks and verify no regressions
- [ ] **Documentation**: Update examples and documentation

#### 3. **Latest Dependencies (MANDATORY)**
```bash
# Check for outdated dependencies
cargo outdated

# Update to latest versions
cargo update

# Verify compatibility
cargo check --all-features
cargo test --all-features
```

**Dependency Rules:**
- [ ] Use latest stable versions from crates.io
- [ ] No pinned versions unless absolutely necessary
- [ ] Regular dependency updates (weekly)
- [ ] Security audit: `cargo audit`
- [ ] Verify no breaking changes after updates

#### 4. **Clean, Modular Code (MANDATORY)**

**Code Quality Standards:**
```rust
// ✅ GOOD: Clean, modular, testable
pub trait ClusteringAlgorithm {
    type Config;
    fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError>;
}

// ✅ GOOD: Builder pattern for configuration
pub struct KMeansConfig {
    k: usize,
    variant: KMeansVariant,
    parallel: bool,
}

impl KMeansConfig {
    pub fn builder() -> KMeansConfigBuilder {
        KMeansConfigBuilder::default()
    }
}

// ❌ BAD: Duplicated code, hard to test
fn kmeans_algorithm(data: &[OHLCV], k: usize) -> Vec<usize> {
    // 100 lines of inline algorithm
}

// ❌ BAD: Outdated dependencies
[dependencies]
serde = "1.0.100"  # Pinned old version
```

**Modularity Rules:**
- [ ] **Single Responsibility**: Each module has one clear purpose
- [ ] **Dependency Inversion**: Depend on abstractions, not concretions
- [ ] **Interface Segregation**: Small, focused traits
- [ ] **Open/Closed**: Open for extension, closed for modification
- [ ] **DRY Principle**: No code duplication
- [ ] **SOLID Principles**: Follow all SOLID design principles

### 🔄 **Implementation Workflow**

#### Phase 1: Planning
1. **Define Requirements**: Clear, testable requirements
2. **Design Interface**: Define traits and public APIs
3. **Plan Tests**: Write test specifications first
4. **Check Dependencies**: Ensure latest versions

#### Phase 2: Implementation
1. **Write Tests First**: TDD approach
2. **Implement Feature**: Following clean code principles
3. **Run Coverage**: Verify > 95% coverage
4. **Code Review**: Self-review against checklist

#### Phase 3: Validation
1. **Run All Tests**: `cargo test --all-features`
2. **Check Coverage**: `cargo tarpaulin`
3. **Run Benchmarks**: `cargo bench`
4. **Update Dependencies**: `cargo update`
5. **Security Audit**: `cargo audit`

#### Phase 4: Documentation
1. **Update Examples**: Add working examples using `examples/csv/`
2. **Update Documentation**: Keep docs in sync with code
3. **Performance Notes**: Document performance characteristics
4. **Migration Guide**: If breaking changes

### 🛠️ **Development Tools Configuration**

#### Pre-commit Hooks
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running pre-commit checks..."

# Check code coverage
cargo tarpaulin --out Html --output-dir coverage --fail-under 95

# Check for outdated dependencies
cargo outdated --exit-code 1

# Run security audit
cargo audit

# Check code quality
cargo clippy --all-targets --all-features -- -D warnings

# Run tests
cargo test --all-features

echo "All checks passed!"
```

#### CI/CD Pipeline
```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      # Check for outdated dependencies
      - name: Check outdated dependencies
        run: cargo outdated --exit-code 1
      
      # Security audit
      - name: Security audit
        run: cargo audit
      
      # Run tests with coverage
      - name: Test with coverage
        run: cargo tarpaulin --out Html --output-dir coverage --fail-under 95
      
      # Upload coverage report
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: coverage/tarpaulin-report.html
```

### 📊 **Quality Metrics Dashboard**

**Required Metrics for Every Implementation:**

| Metric | Target | Tool | Frequency |
|--------|--------|------|-----------|
| **Code Coverage** | > 95% | `cargo tarpaulin` | Every commit |
| **Dependency Freshness** | Latest stable | `cargo outdated` | Weekly |
| **Security Issues** | 0 | `cargo audit` | Every commit |
| **Code Quality** | 0 warnings | `cargo clippy` | Every commit |
| **Performance** | No regression | `cargo bench` | Every PR |
| **Documentation** | 100% API coverage | Manual review | Every PR |

### 🚨 **Failure Modes and Recovery**

#### Coverage Below 95%
```bash
# Identify uncovered code
cargo tarpaulin --out Html --output-dir coverage

# Add missing tests
# Re-run until > 95%
cargo tarpaulin --out Html --output-dir coverage --fail-under 95
```

#### Outdated Dependencies
```bash
# Update dependencies
cargo update

# Check for breaking changes
cargo check --all-features
cargo test --all-features

# If breaking changes, update code or pin version temporarily
```

#### Code Duplication
```bash
# Use clippy to detect duplication
cargo clippy --all-targets --all-features

# Refactor duplicated code into shared modules
# Update tests to cover shared code
```

### 📝 **Implementation Templates**

#### New Clustering Algorithm
```rust
// 1. Define trait implementation
impl ClusteringAlgorithm for NewAlgorithm {
    type Config = NewAlgorithmConfig;
    
    fn new(config: Self::Config) -> Self {
        // Implementation
    }
    
    fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError> {
        // Implementation with comprehensive error handling
    }
}

// 2. Write tests FIRST
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    #[test]
    fn test_new_algorithm_basic() {
        // Test implementation
    }
    
    proptest! {
        #[test]
        fn test_new_algorithm_properties(data in generate_test_data()) {
            // Property-based tests
        }
    }
}

// 3. Add benchmarks
#[cfg(test)]
mod benches {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn bench_new_algorithm(c: &mut Criterion) {
        c.bench_function("new_algorithm", |b| {
            b.iter(|| {
                // Benchmark implementation
            })
        });
    }
}
```

#### New Module Structure
```
src/clustering/new_algorithm/
├── mod.rs              # Public API and trait implementations
├── algorithm.rs        # Core algorithm implementation
├── config.rs           # Configuration and builder pattern
├── parallel.rs         # Parallel implementation (if applicable)
└── tests/              # Comprehensive test suite
    ├── mod.rs
    ├── unit_tests.rs
    ├── integration_tests.rs
    └── property_tests.rs
```

**Remember: These rules are MANDATORY and non-negotiable. Every implementation must follow this ethos to maintain the high quality standards of FormicaX.**

## Examples and Documentation Requirements

### Code Coverage Standards
- **Minimum Coverage**: > 95% code coverage for all public interfaces
- **Coverage Tools**: Use `cargo-tarpaulin` for coverage reporting
- **Coverage Reports**: Generate HTML reports in `coverage/` directory
- **Coverage CI**: Include coverage checks in CI/CD pipeline

### Example Structure
All examples must be placed in the `examples/` folder with the following structure:

```
examples/
├── basic_usage/
│   ├── data_loading.rs
│   ├── kmeans_clustering.rs
│   └── dbscan_clustering.rs
├── advanced_usage/
│   ├── streaming_processing.rs
│   ├── parallel_clustering.rs
│   └── custom_validation.rs
├── clustering_algorithms/
│   ├── kmeans_examples.rs
│   ├── dbscan_examples.rs
│   ├── gmm_examples.rs
│   ├── hierarchical_examples.rs
│   ├── affinity_propagation_examples.rs
│   └── som_examples.rs
├── data_processing/
│   ├── csv_parsing.rs
│   ├── data_validation.rs
│   └── feature_engineering.rs
└── csv/
    ├── daily.csv
    ├── hourly.csv
    └── minute.csv
```

### Example Requirements
- **CSV Data Source**: All examples must use CSV files from `examples/csv/` folder
- **External Interfaces**: Every public API must have at least one working example
- **Error Handling**: Include examples showing proper error handling
- **Documentation**: Each example must have clear comments explaining the code
- **Runnable**: All examples must compile and run successfully
- **Realistic Data**: Use realistic financial data scenarios in examples

### Example File Naming Convention
- Use snake_case for example file names
- Include the main functionality in the filename
- Group related examples in subdirectories
- Use descriptive names that indicate the example's purpose

### Example Documentation Standards
- **Header Comments**: Each example file must have a header explaining its purpose
- **Inline Comments**: Include comments explaining complex operations
- **Output Examples**: Show expected output where relevant
- **Error Scenarios**: Demonstrate error handling with realistic scenarios
- **Performance Notes**: Include performance considerations where applicable 