//! Integration tests for KMeans public API

#[cfg(test)]
mod tests {
    use super::super::config::*;
    use super::super::algorithm::*;
    use crate::core::{OHLCV, ClusterResult, FormicaXError};
    use chrono::{DateTime, Utc};

    fn create_test_data() -> Vec<OHLCV> {
        vec![
            OHLCV {
                timestamp: DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z").unwrap().with_timezone(&Utc),
                open: 100.0,
                high: 105.0,
                low: 98.0,
                close: 103.0,
                volume: 1000,
            },
            OHLCV {
                timestamp: DateTime::parse_from_rfc3339("2023-01-02T00:00:00Z").unwrap().with_timezone(&Utc),
                open: 103.0,
                high: 108.0,
                low: 101.0,
                close: 106.0,
                volume: 1200,
            },
            OHLCV {
                timestamp: DateTime::parse_from_rfc3339("2023-01-03T00:00:00Z").unwrap().with_timezone(&Utc),
                open: 106.0,
                high: 110.0,
                low: 104.0,
                close: 109.0,
                volume: 1100,
            },
            OHLCV {
                timestamp: DateTime::parse_from_rfc3339("2023-01-04T00:00:00Z").unwrap().with_timezone(&Utc),
                open: 109.0,
                high: 115.0,
                low: 107.0,
                close: 113.0,
                volume: 1300,
            },
            OHLCV {
                timestamp: DateTime::parse_from_rfc3339("2023-01-05T00:00:00Z").unwrap().with_timezone(&Utc),
                open: 113.0,
                high: 118.0,
                low: 111.0,
                close: 116.0,
                volume: 1400,
            },
        ]
    }

    #[test]
    fn test_kmeans_public_api() {
        let data = create_test_data();
        let config = KMeansConfig::builder()
            .k(2)
            .variant(KMeansVariant::Lloyd)
            .max_iterations(100)
            .tolerance(1e-6)
            .build()
            .unwrap();

        let mut kmeans = KMeans::with_config(config);
        let result = kmeans.fit(&data).unwrap();

        // Verify basic result properties
        assert_eq!(result.algorithm_name, "KMeans-Lloyd");
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.cluster_assignments.len(), data.len());
        assert!(result.iterations > 0);
        assert!(result.iterations <= 100);
        assert!(result.inertia.is_some());
        assert!(result.inertia.unwrap() > 0.0);
        assert!(result.cluster_centers.is_some());
        assert_eq!(result.cluster_centers.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_kmeans_error_handling() {
        let config = KMeansConfig::builder()
            .k(2)
            .build()
            .unwrap();

        let mut kmeans = KMeans::with_config(config);

        // Test empty data
        let empty_data: Vec<OHLCV> = vec![];
        let result = kmeans.fit(&empty_data);
        assert!(matches!(
            result,
            Err(FormicaXError::Data(crate::core::DataError::EmptyDataset))
        ));

        // Test insufficient data
        let small_data = create_test_data();
        let config_large_k = KMeansConfig::builder()
            .k(10) // More clusters than data points
            .build()
            .unwrap();
        let mut kmeans_large_k = KMeans::with_config(config_large_k);
        let result = kmeans_large_k.fit(&small_data);
        assert!(matches!(
            result,
            Err(FormicaXError::Data(crate::core::DataError::InsufficientData { .. }))
        ));
    }

    #[test]
    fn test_kmeans_convergence() {
        let data = create_test_data();
        let config = KMeansConfig::builder()
            .k(2)
            .max_iterations(50)
            .tolerance(1e-8)
            .build()
            .unwrap();

        let mut kmeans = KMeans::with_config(config);
        let result = kmeans.fit(&data).unwrap();

        // Verify convergence properties
        assert!(result.iterations > 0);
        assert!(result.iterations <= 50);
        assert!(result.converged || result.iterations == 50); // Either converged or hit max iterations
    }

    #[test]
    fn test_kmeans_algorithm_state() {
        let data = create_test_data();
        let config = KMeansConfig::builder()
            .k(3)
            .build()
            .unwrap();

        let mut kmeans = KMeans::with_config(config);
        
        // Before fitting
        assert!(kmeans.get_centroids().is_none());
        assert!(kmeans.get_assignments().is_none());
        assert!(!kmeans.is_converged());
        assert_eq!(kmeans.get_iterations(), 0);

        // After fitting
        let result = kmeans.fit(&data).unwrap();
        
        assert!(kmeans.get_centroids().is_some());
        assert!(kmeans.get_assignments().is_some());
        assert_eq!(kmeans.get_iterations(), result.iterations);
        assert_eq!(kmeans.is_converged(), result.converged);
        
        let centroids = kmeans.get_centroids().unwrap();
        let assignments = kmeans.get_assignments().unwrap();
        
        assert_eq!(centroids.len(), 3);
        assert_eq!(assignments.len(), data.len());
        assert_eq!(assignments, &result.cluster_assignments);
    }
} 