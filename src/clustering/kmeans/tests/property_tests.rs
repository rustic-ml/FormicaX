//! Property-based tests for KMeans using proptest

#[cfg(test)]
mod tests {
    use super::super::config::*;
    use super::super::algorithm::*;
    use crate::core::OHLCV;
    use proptest::prelude::*;
    use chrono::{TimeZone, Utc};

    // Strategy to generate a valid OHLCV
    fn arb_ohlcv() -> impl Strategy<Value = OHLCV> {
        // Prices between 10.0 and 200.0, volume between 100 and 10_000
        (
            10.0..200.0, // open
            10.0..200.0, // high
            10.0..200.0, // low
            10.0..200.0, // close
            100u64..10_000u64, // volume
        ).prop_map(|(open, high, low, close, volume)| {
            let (high, low) = if high < low { (low, high) } else { (high, low) };
            OHLCV {
                timestamp: Utc.timestamp_opt(1_600_000_000, 0).unwrap(),
                open,
                high,
                low,
                close,
                volume,
            }
        })
    }

    proptest! {
        #[test]
        fn test_kmeans_property_valid_data(
            (k, n) in (2usize..6, 10usize..50),
            data in prop::collection::vec(arb_ohlcv(), 10..50)
        ) {
            // Only run if n <= data.len()
            prop_assume!(n <= data.len());
            let config = KMeansConfig::builder()
                .k(k)
                .max_iterations(50)
                .tolerance(1e-6)
                .build()
                .unwrap();
            let mut kmeans = KMeans::with_config(config);
            let result = kmeans.fit(&data).unwrap();
            // Check cluster assignment length
            prop_assert_eq!(result.cluster_assignments.len(), data.len());
            // Check number of clusters
            prop_assert_eq!(result.n_clusters, k);
            // All assignments are valid cluster indices
            prop_assert!(result.cluster_assignments.iter().all(|&c| c < k));
            // Should have at least one point in each cluster
            let mut cluster_counts = vec![0; k];
            for &c in &result.cluster_assignments {
                cluster_counts[c] += 1;
            }
            prop_assert!(cluster_counts.iter().all(|&count| count > 0));
        }

        #[test]
        fn test_kmeans_property_insufficient_data(
            k in 2usize..10,
            n in 1usize..5
        ) {
            let data: Vec<OHLCV> = (0..n).map(|_| OHLCV {
                timestamp: Utc.timestamp_opt(1_600_000_000, 0).unwrap(),
                open: 100.0,
                high: 105.0,
                low: 98.0,
                close: 103.0,
                volume: 1000,
            }).collect();
            let config = KMeansConfig::builder()
                .k(k)
                .max_iterations(10)
                .tolerance(1e-6)
                .build()
                .unwrap();
            let mut kmeans = KMeans::with_config(config);
            let result = kmeans.fit(&data);
            prop_assert!(result.is_err());
        }
    }
} 