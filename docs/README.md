# FormicaX - High-Performance Financial Data Analysis

## Overview

FormicaX is a high-performance Rust library designed specifically for traders and quantitative analysts. It provides lightning-fast OHLCV (Open, High, Low, Close, Volume) data processing with advanced VWAP (Volume Weighted Average Price) calculations, enabling real-time trading strategies and backtesting with institutional-grade performance.

## Why FormicaX for Trading?

- **⚡ Real-time Performance**: Process millions of data points in seconds
- **🎯 VWAP-Focused**: Specialized tools for volume-weighted analysis
- **🔄 Backtesting Ready**: Seamless integration with trading strategies
- **📊 Multi-timeframe**: Support for tick, minute, hourly, and daily data
- **🔒 Memory Safe**: Rust's safety guarantees for critical trading operations
- **📈 Technical Indicators**: Built-in indicators optimized for trading

## Quick Start

### Installation

Add FormicaX to your `Cargo.toml`:

```toml
[dependencies]
formicax = "0.1.0"
```

### Basic Usage

```rust
use formicax::{OHLCVReader, VWAPCalculator, OHLCV};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read OHLCV data from CSV file
    let mut reader = OHLCVReader::from_path("data/daily.csv")?;
    
    // Calculate VWAP
    let vwap_calc = VWAPCalculator::new();
    let data: Vec<OHLCV> = reader.records().collect::<Result<Vec<_>, _>>()?;
    let vwap = vwap_calc.calculate(&data)?;
    
    println!("VWAP: {}", vwap);
    Ok(())
}
```

### Data Format

FormicaX supports flexible CSV formats with case-insensitive column names and any column order.

**Default Format:**
```csv
timestamp,open,high,low,close,volume,vwap
2020-07-08T04:00:00Z,10.600000,10.600000,10.340000,10.570000,44,10.493488
2020-07-09T04:00:00Z,10.530000,10.750000,10.470000,10.545000,27,10.604113
```

**Alternative Formats (all supported):**
```csv
# Different order
open,high,low,close,volume,vwap,timestamp

# Different case
Timestamp,Open,High,Low,Close,Volume,VWAP

# Abbreviated names
time,o,h,l,c,vol,vw
```

**Required Fields:**
- **timestamp**: ISO 8601 formatted datetime string (UTC)
- **open/high/low/close**: Price data (positive numbers)
- **volume**: Trading volume (non-negative integers)
- **vwap**: Volume Weighted Average Price

## Core Features

### Trading Tools
- **VWAP Analysis**: Session, anchored, and rolling VWAP calculations
- **Volume Profile**: Advanced volume-weighted price analysis
- **Multi-timeframe Support**: Seamless data aggregation and analysis
- **Real-time Processing**: Sub-millisecond data processing for live trading

### Technical Indicators
- **Moving Averages**: SMA, EMA, WMA with VWAP integration
- **Momentum Indicators**: RSI, MACD, Stochastic with volume confirmation
- **Volatility Tools**: Bollinger Bands, ATR with volume context
- **Volume Indicators**: OBV, Volume Rate of Change, Volume Profile

### Trading Strategy Support
- **Backtesting Engine**: High-performance strategy testing
- **Signal Generation**: VWAP-based entry/exit signals
- **Risk Management**: Position sizing and stop-loss calculations
- **Performance Analytics**: Sharpe ratio, drawdown, and other metrics

## Performance Tips

### Streaming Large Files
```rust
use formicax::{StreamingProcessor};

fn process_large_file() -> Result<(), Box<dyn std::error::Error>> {
    let processor = StreamingProcessor::new()
        .chunk_size(1000)
        .parallel(true);
    
    processor.process_file("large_data.csv")?;
    Ok(())
}
```

### Parallel Processing
```rust
use formicax::{OHLCVReader};
use rayon::prelude::*;

fn parallel_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let data = load_ohlcv_data()?;
    
    let results: Vec<_> = data
        .par_iter()
        .map(|ohlcv| calculate_indicators(ohlcv))
        .collect();
    
    Ok(())
}
```

## Error Handling

FormicaX uses Rust's Result type for robust error handling:

```rust
use formicax::{OHLCVReader, FormicaXError};

fn process_data() -> Result<(), FormicaXError> {
    let mut reader = OHLCVReader::from_path("data.csv")?;
    
    for result in reader.records() {
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
- **[Trading Guide](TRADING_GUIDE.md)** - Trading strategies, use cases, and performance optimization
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Data format specification and development details

## Performance Targets

- **VWAP Calculation**: < 1 microsecond
- **Signal Generation**: < 100 microseconds
- **Data Processing**: < 10 microseconds per tick
- **Memory Usage**: < 2x input data size
- **Throughput**: > 1,000,000 ticks/second

## Contributing

See [CONTRIBUTING.md](contributing.md) for development guidelines.

## License

MIT OR Apache-2.0 