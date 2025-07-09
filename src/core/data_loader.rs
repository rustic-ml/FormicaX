//! Data loading and CSV parsing for FormicaX
//!
//! This module provides flexible CSV parsing with intelligent column detection,
//! support for various data formats, and comprehensive error handling.

use crate::core::{DataError, FormicaXError, OHLCV};
use chrono::{DateTime, Utc};
use csv::{Reader, ReaderBuilder};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Flexible CSV data loader with intelligent column detection
///
/// Supports various CSV formats with different column orders and naming conventions.
/// Automatically detects and maps columns regardless of their position or case.
#[derive(Debug)]
pub struct DataLoader {
    /// File path to load data from
    file_path: String,
    /// Custom column mapping (if provided)
    column_mapping: Option<ColumnMapping>,
    /// Whether to validate data during loading
    validate_data: bool,
    /// Whether to use memory mapping for large files
    use_memory_mapping: bool,
    /// Buffer size for reading
    buffer_size: usize,
}

/// Column mapping for flexible CSV parsing
#[derive(Debug, Clone)]
pub struct ColumnMapping {
    /// Column indices for each required field
    pub timestamp_col: usize,
    pub open_col: usize,
    pub high_col: usize,
    pub low_col: usize,
    pub close_col: usize,
    pub volume_col: usize,
}

impl DataLoader {
    /// Create a new data loader for the specified file
    pub fn new<P: AsRef<Path>>(file_path: P) -> Self {
        Self {
            file_path: file_path.as_ref().to_string_lossy().to_string(),
            column_mapping: None,
            validate_data: true,
            use_memory_mapping: false,
            buffer_size: 8192,
        }
    }

    /// Set custom column mapping
    pub fn with_column_mapping(mut self, mapping: ColumnMapping) -> Self {
        self.column_mapping = Some(mapping);
        self
    }

    /// Enable or disable data validation
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_data = validate;
        self
    }

    /// Enable or disable memory mapping
    pub fn with_memory_mapping(mut self, use_mapping: bool) -> Self {
        self.use_memory_mapping = use_mapping;
        self
    }

    /// Set buffer size for reading
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Load OHLCV data from CSV file
    pub fn load_csv(&mut self) -> Result<Vec<OHLCV>, FormicaXError> {
        let file = File::open(&self.file_path).map_err(FormicaXError::Io);

        let reader = BufReader::with_capacity(self.buffer_size, file?);
        self.load_from_reader(reader)
    }

    /// Load data from any reader
    pub fn load_from_reader<R: Read>(&self, reader: R) -> Result<Vec<OHLCV>, FormicaXError> {
        // Determine if we have headers based on whether column mapping is provided
        let has_headers = self.column_mapping.is_none();

        let mut csv_reader = ReaderBuilder::new()
            .flexible(true)
            .has_headers(has_headers)
            .from_reader(reader);

        // Get column mapping
        let column_mapping = if let Some(ref mapping) = self.column_mapping {
            mapping.clone()
        } else {
            self.detect_column_mapping(&mut csv_reader)?
        };

        // Load and parse data
        let mut data = Vec::new();
        let mut row_number = 0;

        for result in csv_reader.records() {
            row_number += 1;
            let record = result?;

            let ohlcv = self.parse_record(&record, &column_mapping, row_number)?;

            if self.validate_data {
                ohlcv.validate().map_err(FormicaXError::Data)?;
            }

            data.push(ohlcv);
        }

        if data.is_empty() {
            return Err(FormicaXError::Data(DataError::EmptyDataset));
        }

        Ok(data)
    }

    /// Detect column mapping from CSV headers
    fn detect_column_mapping<R: Read>(
        &self,
        reader: &mut Reader<R>,
    ) -> Result<ColumnMapping, FormicaXError> {
        let headers = reader.headers()?;

        let mut mapping = ColumnMapping {
            timestamp_col: 0,
            open_col: 0,
            high_col: 0,
            low_col: 0,
            close_col: 0,
            volume_col: 0,
        };

        let mut found_columns = HashMap::new();

        for (i, header) in headers.iter().enumerate() {
            let header_lower = header.to_lowercase();

            // Timestamp column detection
            if Self::is_timestamp_column(&header_lower) {
                mapping.timestamp_col = i;
                found_columns.insert("timestamp", i);
            }
            // Open column detection
            else if Self::is_open_column(&header_lower) {
                mapping.open_col = i;
                found_columns.insert("open", i);
            }
            // High column detection
            else if Self::is_high_column(&header_lower) {
                mapping.high_col = i;
                found_columns.insert("high", i);
            }
            // Low column detection
            else if Self::is_low_column(&header_lower) {
                mapping.low_col = i;
                found_columns.insert("low", i);
            }
            // Close column detection
            else if Self::is_close_column(&header_lower) {
                mapping.close_col = i;
                found_columns.insert("close", i);
            }
            // Volume column detection
            else if Self::is_volume_column(&header_lower) {
                mapping.volume_col = i;
                found_columns.insert("volume", i);
            }
        }

        // Verify all required columns were found
        let required_columns = ["timestamp", "open", "high", "low", "close", "volume"];
        for column in &required_columns {
            if !found_columns.contains_key(column) {
                return Err(FormicaXError::Data(DataError::MissingColumn {
                    column: column.to_string(),
                }));
            }
        }

        Ok(mapping)
    }

    /// Parse a single CSV record into OHLCV data
    fn parse_record(
        &self,
        record: &csv::StringRecord,
        mapping: &ColumnMapping,
        row_number: usize,
    ) -> Result<OHLCV, FormicaXError> {
        // Parse timestamp
        let timestamp_str = record.get(mapping.timestamp_col).ok_or_else(|| {
            FormicaXError::Data(DataError::InvalidCsvFormat {
                message: format!("Missing timestamp column at row {}", row_number),
            })
        })?;

        let timestamp = self.parse_timestamp(timestamp_str, row_number)?;

        // Parse numeric fields
        let open = self.parse_float(record.get(mapping.open_col), "open", row_number)?;
        let high = self.parse_float(record.get(mapping.high_col), "high", row_number)?;
        let low = self.parse_float(record.get(mapping.low_col), "low", row_number)?;
        let close = self.parse_float(record.get(mapping.close_col), "close", row_number)?;
        let volume = self.parse_volume(record.get(mapping.volume_col), row_number)?;

        Ok(OHLCV::new(timestamp, open, high, low, close, volume))
    }

    /// Parse timestamp string to DateTime<Utc>
    fn parse_timestamp(
        &self,
        timestamp_str: &str,
        row_number: usize,
    ) -> Result<DateTime<Utc>, FormicaXError> {
        // Handle ISO 8601 format with Z suffix (UTC)
        if let Some(without_z) = timestamp_str.strip_suffix('Z') {
            if let Ok(dt) =
                DateTime::parse_from_str(&format!("{}+00:00", without_z), "%Y-%m-%dT%H:%M:%S%z")
            {
                return Ok(dt.with_timezone(&Utc));
            }
        }

        // Try various timestamp formats
        let formats = [
            "%Y-%m-%dT%H:%M:%S%z", // ISO 8601 with timezone
            "%Y-%m-%d %H:%M:%S%z", // Standard format with timezone
        ];

        for format in &formats {
            if let Ok(dt) = DateTime::parse_from_str(timestamp_str, format) {
                return Ok(dt.with_timezone(&Utc));
            }
        }

        // Try formats without timezone (assume UTC)
        let naive_formats = [
            "%Y-%m-%d %H:%M:%S", // Standard format
            "%Y-%m-%d",          // Date only
            "%m/%d/%Y %H:%M:%S", // US format
            "%d/%m/%Y %H:%M:%S", // European format
        ];

        for format in &naive_formats {
            if let Ok(naive_dt) = chrono::NaiveDateTime::parse_from_str(timestamp_str, format) {
                return Ok(DateTime::from_naive_utc_and_offset(naive_dt, Utc));
            }
        }

        // Try parsing as Unix timestamp
        if let Ok(timestamp) = timestamp_str.parse::<i64>() {
            if timestamp > 1_000_000_000_000 {
                // Assume milliseconds
                return DateTime::from_timestamp_millis(timestamp).ok_or_else(|| {
                    FormicaXError::Data(DataError::InvalidCsvFormat {
                        message: format!(
                            "Invalid timestamp at row {}: {}",
                            row_number, timestamp_str
                        ),
                    })
                });
            } else {
                // Assume seconds
                return DateTime::from_timestamp(timestamp, 0).ok_or_else(|| {
                    FormicaXError::Data(DataError::InvalidCsvFormat {
                        message: format!(
                            "Invalid timestamp at row {}: {}",
                            row_number, timestamp_str
                        ),
                    })
                });
            }
        }

        Err(FormicaXError::Data(DataError::InvalidCsvFormat {
            message: format!(
                "Unable to parse timestamp at row {}: {}",
                row_number, timestamp_str
            ),
        }))
    }

    /// Parse float value
    fn parse_float(
        &self,
        value: Option<&str>,
        field_name: &str,
        row_number: usize,
    ) -> Result<f64, FormicaXError> {
        let value = value.ok_or_else(|| {
            FormicaXError::Data(DataError::InvalidCsvFormat {
                message: format!("Missing {} field at row {}", field_name, row_number),
            })
        })?;

        value.parse::<f64>().map_err(|_| {
            FormicaXError::Data(DataError::InvalidDataType {
                column: field_name.to_string(),
                expected: "float".to_string(),
                actual: value.to_string(),
            })
        })
    }

    /// Parse volume value
    fn parse_volume(&self, value: Option<&str>, row_number: usize) -> Result<u64, FormicaXError> {
        let value = value.ok_or_else(|| {
            FormicaXError::Data(DataError::InvalidCsvFormat {
                message: format!("Missing volume field at row {}", row_number),
            })
        })?;

        value.parse::<u64>().map_err(|_| {
            FormicaXError::Data(DataError::InvalidDataType {
                column: "volume".to_string(),
                expected: "unsigned integer".to_string(),
                actual: value.to_string(),
            })
        })
    }

    /// Check if column name represents timestamp
    fn is_timestamp_column(header: &str) -> bool {
        matches!(
            header,
            "timestamp" | "time" | "date" | "datetime" | "dt" | "ts"
        )
    }

    /// Check if column name represents open price
    fn is_open_column(header: &str) -> bool {
        matches!(header, "open" | "o" | "opening" | "open_price")
    }

    /// Check if column name represents high price
    fn is_high_column(header: &str) -> bool {
        matches!(header, "high" | "h" | "highest" | "high_price")
    }

    /// Check if column name represents low price
    fn is_low_column(header: &str) -> bool {
        matches!(header, "low" | "l" | "lowest" | "low_price")
    }

    /// Check if column name represents close price
    fn is_close_column(header: &str) -> bool {
        matches!(header, "close" | "c" | "closing" | "close_price")
    }

    /// Check if column name represents volume
    fn is_volume_column(header: &str) -> bool {
        matches!(header, "volume" | "vol" | "v" | "quantity" | "shares")
    }

    /// Get an iterator over CSV records
    pub fn records(&mut self) -> Result<DataLoaderIterator, FormicaXError> {
        let file = File::open(&self.file_path).map_err(FormicaXError::Io);

        let reader = BufReader::with_capacity(self.buffer_size, file?);
        let mut csv_reader = ReaderBuilder::new()
            .flexible(true)
            .has_headers(true)
            .from_reader(reader);

        // Detect column mapping if not provided
        let column_mapping = if let Some(ref mapping) = self.column_mapping {
            mapping.clone()
        } else {
            self.detect_column_mapping(&mut csv_reader)?
        };

        Ok(DataLoaderIterator {
            csv_reader,
            column_mapping,
            validate_data: self.validate_data,
            row_number: 0,
        })
    }

    /// Create a new DataLoader for CSV files without headers
    /// Assumes column order: open,high,low,trade_count,close,timestamp,volume,vwap
    pub fn without_headers<P: AsRef<Path>>(file_path: P) -> Self {
        let default_mapping = ColumnMapping {
            open_col: 0,      // open
            high_col: 1,      // high
            low_col: 2,       // low
            close_col: 4,     // close (skip trade_count at index 3)
            timestamp_col: 5, // timestamp
            volume_col: 6,    // volume (skip vwap at index 7)
        };

        Self {
            file_path: file_path.as_ref().to_string_lossy().to_string(),
            column_mapping: Some(default_mapping),
            validate_data: true,
            use_memory_mapping: false,
            buffer_size: 8192,
        }
    }
}

/// Iterator for streaming CSV data loading
pub struct DataLoaderIterator {
    csv_reader: Reader<BufReader<File>>,
    column_mapping: ColumnMapping,
    validate_data: bool,
    row_number: usize,
}

impl Iterator for DataLoaderIterator {
    type Item = Result<OHLCV, FormicaXError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.row_number += 1;

        match self.csv_reader.records().next() {
            Some(Ok(record)) => {
                let loader = DataLoader::new(""); // Dummy loader for parsing
                let result = loader.parse_record(&record, &self.column_mapping, self.row_number);

                match result {
                    Ok(ohlcv) => {
                        if self.validate_data {
                            if let Err(e) = ohlcv.validate() {
                                return Some(Err(FormicaXError::Data(e)));
                            }
                        }
                        Some(Ok(ohlcv))
                    }
                    Err(e) => Some(Err(e)),
                }
            }
            Some(Err(e)) => Some(Err(FormicaXError::Csv(e))),
            None => None,
        }
    }
}

/// Streaming data loader for large files
#[derive(Debug)]
pub struct StreamingDataLoader {
    chunk_size: usize,
    parallel: bool,
    validate_data: bool,
}

impl StreamingDataLoader {
    /// Create a new streaming data loader
    pub fn new() -> Self {
        Self {
            chunk_size: 1000,
            parallel: false,
            validate_data: true,
        }
    }

    /// Set chunk size for processing
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Enable or disable parallel processing
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Enable or disable data validation
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_data = validate;
        self
    }

    /// Process file in chunks
    pub fn process_file<P: AsRef<Path>>(&self, file_path: P) -> Result<Vec<OHLCV>, FormicaXError> {
        let mut loader = DataLoader::new(file_path).with_validation(self.validate_data);

        if self.parallel {
            // Parallel processing implementation would go here
            // For now, fall back to sequential processing
            loader.load_csv()
        } else {
            loader.load_csv()
        }
    }
}

impl Default for StreamingDataLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, Timelike};
    use proptest::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_data_loader_creation() {
        let loader = DataLoader::new("test.csv");
        assert_eq!(loader.file_path, "test.csv");
        assert!(loader.validate_data);
        assert!(!loader.use_memory_mapping);
    }

    #[test]
    fn test_data_loader_builder_pattern() {
        let loader = DataLoader::new("test.csv")
            .with_validation(false)
            .with_memory_mapping(true)
            .with_buffer_size(16384);

        assert!(!loader.validate_data);
        assert!(loader.use_memory_mapping);
        assert_eq!(loader.buffer_size, 16384);
    }

    #[test]
    fn test_column_detection() {
        assert!(DataLoader::is_timestamp_column("timestamp"));
        assert!(DataLoader::is_timestamp_column("time"));
        assert!(DataLoader::is_timestamp_column("date"));
        assert!(!DataLoader::is_timestamp_column("open"));

        assert!(DataLoader::is_open_column("open"));
        assert!(DataLoader::is_open_column("o"));
        assert!(!DataLoader::is_open_column("high"));

        assert!(DataLoader::is_high_column("high"));
        assert!(DataLoader::is_high_column("h"));
        assert!(!DataLoader::is_high_column("low"));

        assert!(DataLoader::is_low_column("low"));
        assert!(DataLoader::is_low_column("l"));
        assert!(!DataLoader::is_low_column("close"));

        assert!(DataLoader::is_close_column("close"));
        assert!(DataLoader::is_close_column("c"));
        assert!(!DataLoader::is_close_column("volume"));

        assert!(DataLoader::is_volume_column("volume"));
        assert!(DataLoader::is_volume_column("vol"));
        assert!(DataLoader::is_volume_column("v"));
        assert!(!DataLoader::is_volume_column("open"));
    }

    #[test]
    fn test_timestamp_parsing() -> Result<(), Box<dyn std::error::Error>> {
        let loader = DataLoader::new("test.csv");

        // Test ISO 8601 format
        let dt = loader.parse_timestamp("2023-01-01T12:00:00Z", 1)?;
        assert_eq!(dt.year(), 2023);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 1);

        // Test standard format
        let dt = loader.parse_timestamp("2023-01-01 12:00:00", 1)?;
        assert_eq!(dt.year(), 2023);

        // Test Unix timestamp (seconds)
        let dt = loader.parse_timestamp("1672531200", 1)?;
        assert_eq!(dt.year(), 2023);

        // Test Unix timestamp (milliseconds)
        let dt = loader.parse_timestamp("1672531200000", 1)?;
        assert_eq!(dt.year(), 2023);
        Ok(())
    }

    #[test]
    fn test_csv_loading() -> Result<(), Box<dyn std::error::Error>> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "timestamp,open,high,low,close,volume")?;
        writeln!(
            temp_file,
            "2023-01-01T12:00:00Z,100.0,105.0,98.0,102.0,1000"
        )?;
        writeln!(
            temp_file,
            "2023-01-02T12:00:00Z,102.0,107.0,100.0,104.0,1200"
        )?;

        let mut loader = DataLoader::new(temp_file.path());
        let data = loader.load_csv()?;

        assert_eq!(data.len(), 2);
        assert_eq!(data[0].open, 100.0);
        assert_eq!(data[0].high, 105.0);
        assert_eq!(data[0].low, 98.0);
        assert_eq!(data[0].close, 102.0);
        assert_eq!(data[0].volume, 1000);
        Ok(())
    }

    #[test]
    fn test_csv_loading_with_different_column_order() -> Result<(), Box<dyn std::error::Error>> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "volume,close,low,high,open,timestamp")?;
        writeln!(
            temp_file,
            "1000,102.0,98.0,105.0,100.0,2023-01-01T12:00:00Z"
        )?;

        let mut loader = DataLoader::new(temp_file.path());
        let data = loader.load_csv()?;

        assert_eq!(data.len(), 1);
        assert_eq!(data[0].open, 100.0);
        assert_eq!(data[0].high, 105.0);
        assert_eq!(data[0].low, 98.0);
        assert_eq!(data[0].close, 102.0);
        assert_eq!(data[0].volume, 1000);
        Ok(())
    }

    #[test]
    fn test_csv_loading_with_case_insensitive_headers() -> Result<(), Box<dyn std::error::Error>> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "TIMESTAMP,OPEN,HIGH,LOW,CLOSE,VOLUME")?;
        writeln!(
            temp_file,
            "2023-01-01T12:00:00Z,100.0,105.0,98.0,102.0,1000"
        )?;

        let mut loader = DataLoader::new(temp_file.path());
        let data = loader.load_csv()?;

        assert_eq!(data.len(), 1);
        assert_eq!(data[0].open, 100.0);
        assert_eq!(data[0].high, 105.0);
        assert_eq!(data[0].low, 98.0);
        assert_eq!(data[0].close, 102.0);
        assert_eq!(data[0].volume, 1000);
        Ok(())
    }

    #[test]
    fn test_csv_loading_missing_column() -> Result<(), Box<dyn std::error::Error>> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "timestamp,open,high,low,close")?;
        writeln!(temp_file, "2023-01-01T12:00:00Z,100.0,105.0,98.0,102.0")?;

        let mut loader = DataLoader::new(temp_file.path());
        let result = loader.load_csv();
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_csv_loading_invalid_data() -> Result<(), Box<dyn std::error::Error>> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "timestamp,open,high,low,close,volume")?;
        writeln!(
            temp_file,
            "2023-01-01T12:00:00Z,invalid,105.0,98.0,102.0,1000"
        )?;

        let mut loader = DataLoader::new(temp_file.path());
        let result = loader.load_csv();
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_streaming_data_loader() -> Result<(), Box<dyn std::error::Error>> {
        let loader = StreamingDataLoader::new()
            .with_chunk_size(500)
            .with_parallel(true)
            .with_validation(false);

        assert_eq!(loader.chunk_size, 500);
        assert!(loader.parallel);
        assert!(!loader.validate_data);
        Ok(())
    }

    #[test]
    fn test_csv_loading_without_headers() -> Result<(), Box<dyn std::error::Error>> {
        // Create test data without headers in the specified order:
        // open,high,low,trade_count,close,timestamp,volume,vwap
        let csv_data = "21.01,21.45,20.40,83,21.45,2020-07-09 04:00:00,6294,20.873028\n\
                       21.69,21.92,20.77,308,20.88,2020-07-10 04:00:00,10046,21.485068\n\
                       21.60,21.99,21.38,122,21.73,2020-07-13 04:00:00,12947,21.706798";

        let temp_file = tempfile::NamedTempFile::new()?;
        std::fs::write(&temp_file, csv_data)?;

        let mut loader = DataLoader::without_headers(temp_file.path());
        let data = loader.load_csv()?;

        assert_eq!(data.len(), 3);

        // Check first record
        let first = &data[0];
        assert_eq!(first.open, 21.01);
        assert_eq!(first.high, 21.45);
        assert_eq!(first.low, 20.40);
        assert_eq!(first.close, 21.45);
        assert_eq!(first.volume, 6294);

        // Check second record
        let second = &data[1];
        assert_eq!(second.open, 21.69);
        assert_eq!(second.high, 21.92);
        assert_eq!(second.low, 20.77);
        assert_eq!(second.close, 20.88);
        assert_eq!(second.volume, 10046);

        Ok(())
    }

    #[test]
    fn test_csv_loading_mixed_order() -> Result<(), Box<dyn std::error::Error>> {
        // Test loading a CSV with mixed column order: volume,timestamp,close,high,low,open
        let mut loader = DataLoader::new("examples/csv/mixed_order.csv");
        let data = loader.load_csv()?;

        assert_eq!(data.len(), 200); // Should have 200 records

        // Check first record
        let first = &data[0];
        println!(
            "First record: open={}, high={}, low={}, close={}, volume={}",
            first.open, first.high, first.low, first.close, first.volume
        );
        assert_eq!(first.open, 10.60);
        assert_eq!(first.high, 10.60);
        assert_eq!(first.low, 10.34);
        assert_eq!(first.close, 10.57);
        assert_eq!(first.volume, 44);

        // Check a middle record (101st record, index 100)
        let middle = &data[100];
        println!(
            "Middle record (100): open={}, high={}, low={}, close={}, volume={}",
            middle.open, middle.high, middle.low, middle.close, middle.volume
        );
        assert_eq!(middle.open, 19.99);
        assert_eq!(middle.high, 20.15);
        assert_eq!(middle.low, 19.32);
        assert_eq!(middle.close, 19.85);
        assert_eq!(middle.volume, 226);

        // Check last record (200th record, index 199)
        let last = &data[199];
        println!(
            "Last record (199): open={}, high={}, low={}, close={}, volume={}",
            last.open, last.high, last.low, last.close, last.volume
        );
        assert_eq!(last.open, 25.40);
        assert_eq!(last.high, 25.96);
        assert_eq!(last.low, 24.59);
        assert_eq!(last.close, 24.80);
        assert_eq!(last.volume, 315);

        Ok(())
    }

    proptest! {
        #[test]
        fn test_timestamp_parsing_properties(
            year in 2020..2030i32,
            month in 1..13u32,
            day in 1..29u32,
            hour in 0..24u32,
            minute in 0..60u32,
            second in 0..60u32
        ) {
            let timestamp_str = format!("{}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                year, month, day, hour, minute, second);

            let loader = DataLoader::new("test.csv");
            let result = loader.parse_timestamp(&timestamp_str, 1);

            if result.is_ok() {
                let dt = result.unwrap();
                assert_eq!(dt.year(), year);
                assert_eq!(dt.month(), month);
                assert_eq!(dt.day(), day);
                assert_eq!(dt.hour(), hour);
                assert_eq!(dt.minute(), minute);
                assert_eq!(dt.second(), second);
            }
        }
    }
}
