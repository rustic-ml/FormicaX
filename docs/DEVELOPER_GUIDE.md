# FormicaX Developer Guide

## Data Format Specification

### OHLCV Data Format

FormicaX expects OHLCV (Open, High, Low, Close, Volume) data with VWAP in a specific CSV format.

#### Flexible Column Format

FormicaX supports flexible CSV column ordering and case-insensitive column names. The library automatically detects and maps columns regardless of their position or case.

**Default Expected Order:**
```
timestamp,open,high,low,close,volume,vwap
```

**Supported Column Variations:**
- **Timestamp**: `timestamp`, `Timestamp`, `TIME`, `time`, `date`, `Date`, `datetime`, `DateTime`
- **Open**: `open`, `Open`, `OPEN`, `o`, `O`
- **High**: `high`, `High`, `HIGH`, `h`, `H`
- **Low**: `low`, `Low`, `LOW`, `l`, `L`
- **Close**: `close`, `Close`, `CLOSE`, `c`, `C`
- **Volume**: `volume`, `Volume`, `VOLUME`, `vol`, `Vol`, `VOL`
- **VWAP**: `vwap`, `VWAP`, `Vwap`, `vw`, `VW`

**Example - Different Column Orders:**
```csv
# Standard order
timestamp,open,high,low,close,volume,vwap

# Alternative order 1
open,high,low,close,volume,vwap,timestamp

# Alternative order 2
vwap,volume,close,low,high,open,timestamp

# With different case
Timestamp,Open,High,Low,Close,Volume,VWAP
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
| vwap | float | Volume Weighted Average Price | `10.493488` |

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

4. **VWAP Field**:
   - Must be positive numbers
   - Should be within the high-low range for the period
   - Precision: Up to 6 decimal places

#### Example Data

```csv
timestamp,open,high,low,close,volume,vwap
2020-07-08T04:00:00Z,10.600000,10.600000,10.340000,10.570000,44,10.493488
2020-07-09T04:00:00Z,10.530000,10.750000,10.470000,10.545000,27,10.604113
2020-07-10T04:00:00Z,10.590000,10.600000,10.360000,10.360000,22,10.385251
```

#### Data Validation Rules

1. **Logical Consistency**:
   - `high >= low`
   - `high >= open` and `high >= close`
   - `low <= open` and `low <= close`
   - `vwap` should be between `low` and `high`

2. **Data Integrity**:
   - No missing values
   - No duplicate timestamps
   - Chronological order (ascending timestamps)

3. **Format Validation**:
   - Valid CSV format
   - Correct number of columns (7)
   - Proper data types for each column

#### Common Issues and Solutions

##### Issue: Unrecognized Column Names
**Problem**: Column names don't match any supported variations
**Solution**: Use standard column names or add custom column mapping

##### Issue: Missing VWAP Column
**Problem**: CSV only has 6 columns (timestamp,open,high,low,close,volume)
**Solution**: Calculate VWAP or add placeholder values

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
formicax = "0.1.0"
```

### Basic Usage

#### Loading OHLCV Data

```rust
use formicax::{OHLCVReader, OHLCV};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read OHLCV data from CSV file
    let mut reader = OHLCVReader::from_path("data/daily.csv")?;
    
    // Process data points
    for result in reader.records() {
        let ohlcv: OHLCV = result?;
        println!("Timestamp: {}, Close: {}", ohlcv.timestamp, ohlcv.close);
    }
    
    Ok(())
}
```

#### VWAP Calculation

```rust
use formicax::{VWAPCalculator, OHLCV};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = load_ohlcv_data()?;
    
    // Calculate standard VWAP
    let vwap_calc = VWAPCalculator::new();
    let vwap = vwap_calc.calculate(&data)?;
    
    println!("VWAP: {}", vwap);
    
    // Calculate rolling VWAP with 20-period window
    let rolling_vwap = vwap_calc.rolling_vwap(&data, 20)?;
    
    Ok(())
}
```

#### Technical Indicators

```rust
use formicax::{indicators::MovingAverage, OHLCV};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = load_ohlcv_data()?;
    
    // Calculate Simple Moving Average
    let sma = MovingAverage::simple(&data.close_prices(), 20)?;
    
    // Calculate Exponential Moving Average
    let ema = MovingAverage::exponential(&data.close_prices(), 20)?;
    
    Ok(())
}
```

### Error Handling

FormicaX uses Rust's Result type for robust error handling:

```rust
use formicax::{OHLCVReader, FormicaXError};

fn process_data() -> Result<(), FormicaXError> {
    let mut reader = OHLCVReader::from_path("data.csv")?;
    
    for result in reader.records() {
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
use formicax::{OHLCVReader, StreamingProcessor};

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
use formicax::{OHLCVReader, ParallelProcessor};
use rayon::prelude::*;

fn parallel_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let data = load_ohlcv_data()?;
    
    // Process multiple indicators in parallel
    let results: Vec<_> = data
        .par_iter()
        .map(|ohlcv| {
            // Perform calculations
            calculate_indicators(ohlcv)
        })
        .collect();
    
    Ok(())
}
```

### Flexible CSV Parsing

```rust
use formicax::{FlexibleCSVReader, ColumnMapping};

fn flexible_csv_parsing() -> Result<(), Box<dyn std::error::Error>> {
    // Auto-detect column mapping
    let mut reader = FlexibleCSVReader::from_path("data.csv")?;
    
    // Or specify custom column mapping
    let mapping = ColumnMapping::new()
        .timestamp("DateTime")
        .open("O")
        .high("H")
        .low("L")
        .close("C")
        .volume("Vol")
        .vwap("VWAP");
    
    let reader_with_mapping = FlexibleCSVReader::from_path_with_mapping("data.csv", mapping)?;
    
    // Process data regardless of column order
    for result in reader.records() {
        let ohlcv: OHLCV = result?;
        println!("Processed: {:?}", ohlcv);
    }
    
    Ok(())
}
```

### Custom Data Validation

```rust
use formicax::{DataValidator, ValidationRule};

fn custom_validation() -> Result<(), Box<dyn std::error::Error>> {
    let validator = DataValidator::new()
        .add_rule(ValidationRule::PriceRange(0.01..1000000.0))
        .add_rule(ValidationRule::VolumeThreshold(100))
        .add_rule(ValidationRule::VWAPConsistency);
    
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
use formicax::{DataTransformer, Timeframe};

fn transform_data() -> Result<(), Box<dyn std::error::Error>> {
    let transformer = DataTransformer::new();
    
    // Convert 1-minute data to 5-minute bars
    let five_min_data = transformer.aggregate(&one_min_data, Timeframe::Minute(5))?;
    
    // Calculate VWAP for aggregated data
    let vwap = VWAPCalculator::new().calculate(&five_min_data)?;
    
    Ok(())
}
```

## Data Conversion Tools

The library includes utilities for:

### Converting from Other Formats

```rust
use formicax::{DataConverter, Format};

fn convert_data() -> Result<(), Box<dyn std::error::Error>> {
    let converter = DataConverter::new();
    
    // Convert from JSON format
    let ohlcv_data = converter.from_format(&json_data, Format::JSON)?;
    
    // Convert to Parquet format
    converter.to_format(&ohlcv_data, Format::Parquet, "output.parquet")?;
    
    Ok(())
}
```

### Calculating VWAP from Price/Volume Data

```rust
use formicax::{VWAPCalculator, PriceVolumeData};

fn calculate_vwap_from_data() -> Result<(), Box<dyn std::error::Error>> {
    let price_volume_data = PriceVolumeData::new()
        .add_tick(10.50, 100)
        .add_tick(10.55, 200)
        .add_tick(10.52, 150);
    
    let vwap = VWAPCalculator::new().calculate_from_ticks(&price_volume_data)?;
    
    println!("Calculated VWAP: {}", vwap);
    
    Ok(())
}
```

### Timestamp Conversion

```rust
use formicax::{TimestampConverter, Timezone};

fn convert_timestamps() -> Result<(), Box<dyn std::error::Error>> {
    let converter = TimestampConverter::new();
    
    // Convert from local timezone to UTC
    let utc_data = converter.to_utc(&local_data, Timezone::EST)?;
    
    // Convert from Unix timestamp to ISO 8601
    let iso_data = converter.from_unix(&unix_data)?;
    
    Ok(())
}
```

## Performance Considerations

### Memory Usage
- **Large files (>1GB)**: Use streaming processing
- **Memory usage**: Scales with data size
- **Chunking**: Consider chunking for very large datasets
- **Parallel processing**: Available for data validation

### Processing Speed
- **Vectorized operations**: Use SIMD where possible
- **Parallel processing**: Utilize all CPU cores
- **Caching**: Cache frequently accessed calculations
- **Lazy evaluation**: Load data on-demand

### Data Quality
- **Validation**: Always validate data before processing
- **Error handling**: Robust error handling for malformed data
- **Logging**: Comprehensive logging for debugging
- **Monitoring**: Track processing performance

## Common Issues

### CSV Format Errors
- Ensure your CSV has all required columns (timestamp, open, high, low, close, volume, vwap)
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

### Data Preparation
1. **Validate data format** before processing
2. **Convert timestamps** to UTC
3. **Check data consistency** (high >= low, etc.)
4. **Handle missing values** appropriately

### Performance Optimization
1. **Use streaming** for large datasets
2. **Enable parallel processing** for better throughput
3. **Cache results** for repeated calculations
4. **Profile performance** to identify bottlenecks

### Error Handling
1. **Validate input data** thoroughly
2. **Handle errors gracefully** with proper error types
3. **Log errors** for debugging
4. **Provide meaningful error messages**

### Code Organization
1. **Separate concerns** (data loading, processing, analysis)
2. **Use builder patterns** for complex configurations
3. **Implement traits** for extensibility
4. **Write comprehensive tests** for all functionality

This developer guide provides all the information needed to effectively use FormicaX for financial data analysis and trading applications. 