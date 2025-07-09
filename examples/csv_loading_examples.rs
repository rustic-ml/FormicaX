//! CSV Loading Examples
//!
//! This example demonstrates how to load CSV files with and without headers
//! using FormicaX's DataLoader.

use formica::{core::data_loader::ColumnMapping, DataLoader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CSV Loading Examples ===\n");

    // Example 1: Load CSV with headers (default behavior)
    println!("1. Loading CSV with headers:");
    let mut loader_with_headers = DataLoader::new("examples/csv/daily.csv");
    let data_with_headers = loader_with_headers.load_csv()?;
    println!("   Loaded {} records with headers", data_with_headers.len());
    println!(
        "   First record: Open=${:.2}, Close=${:.2}, Volume={}",
        data_with_headers[0].open, data_with_headers[0].close, data_with_headers[0].volume
    );
    println!();

    // Example 2: Load CSV without headers using the new method
    println!("2. Loading CSV without headers:");
    let mut loader_without_headers =
        DataLoader::without_headers("examples/csv/daily_without_headers.csv");
    let data_without_headers = loader_without_headers.load_csv()?;
    println!(
        "   Loaded {} records without headers",
        data_without_headers.len()
    );
    println!(
        "   First record: Open=${:.2}, Close=${:.2}, Volume={}",
        data_without_headers[0].open, data_without_headers[0].close, data_without_headers[0].volume
    );
    println!();

    // Example 3: Custom column mapping for different column order
    println!("3. Loading CSV with custom column mapping:");
    let custom_mapping = ColumnMapping {
        open_col: 1,      // open is in column 1
        high_col: 2,      // high is in column 2
        low_col: 3,       // low is in column 3
        close_col: 0,     // close is in column 0
        timestamp_col: 4, // timestamp is in column 4
        volume_col: 5,    // volume is in column 5
    };

    let mut loader_custom =
        DataLoader::new("examples/csv/daily.csv").with_column_mapping(custom_mapping);

    // Note: This will likely fail with the current data, but demonstrates the concept
    match loader_custom.load_csv() {
        Ok(data) => {
            println!("   Loaded {} records with custom mapping", data.len());
        }
        Err(e) => {
            println!("   Custom mapping failed (expected): {}", e);
        }
    }
    println!();

    // Example 4: Compare data from both files
    println!("4. Comparing data from both files:");
    if data_with_headers.len() > 0 && data_without_headers.len() > 0 {
        let first_with_headers = &data_with_headers[0];
        let first_without_headers = &data_without_headers[0];

        println!(
            "   With headers - Open: ${:.2}, Close: ${:.2}",
            first_with_headers.open, first_with_headers.close
        );
        println!(
            "   Without headers - Open: ${:.2}, Close: ${:.2}",
            first_without_headers.open, first_without_headers.close
        );

        if (first_with_headers.open - first_without_headers.open).abs() < 0.01 {
            println!("   ✅ Data matches between files!");
        } else {
            println!("   ❌ Data differs between files");
        }
    }
    println!();

    // Example 5: Data validation
    println!("5. Data validation:");
    let mut loader_validation =
        DataLoader::without_headers("examples/csv/daily_without_headers.csv").with_validation(true);

    match loader_validation.load_csv() {
        Ok(data) => {
            println!("   ✅ All {} records passed validation", data.len());

            // Check for any anomalies
            let invalid_records: Vec<_> = data
                .iter()
                .filter(|record| {
                    record.high < record.low
                        || record.open < 0.0
                        || record.close < 0.0
                        || record.volume == 0
                })
                .collect();

            if invalid_records.is_empty() {
                println!("   ✅ No data anomalies detected");
            } else {
                println!(
                    "   ⚠️  Found {} records with potential anomalies",
                    invalid_records.len()
                );
            }
        }
        Err(e) => {
            println!("   ❌ Validation failed: {}", e);
        }
    }

    Ok(())
}
