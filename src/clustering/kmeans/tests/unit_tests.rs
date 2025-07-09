//! Unit tests for KMeans config and algorithm

#[cfg(test)]
mod tests {
    use super::super::config::*;
    use super::super::algorithm::*;
    use crate::core::FormicaXError;

    #[test]
    fn test_kmeans_config_builder() {
        // Test basic builder functionality
        let config = KMeansConfig::builder()
            .k(5)
            .variant(KMeansVariant::Elkan)
            .max_iterations(200)
            .tolerance(1e-6)
            .parallel(true)
            .simd(true)
            .build()
            .unwrap();

        assert_eq!(config.k, 5);
        assert_eq!(config.variant, KMeansVariant::Elkan);
        assert_eq!(config.max_iterations, 200);
        assert_eq!(config.tolerance, 1e-6);
        assert!(config.parallel);
        assert!(config.simd);
    }

    #[test]
    fn test_kmeans_config_defaults() {
        // Test default values when not specified
        let config = KMeansConfig::builder()
            .k(3)
            .build()
            .unwrap();

        assert_eq!(config.k, 3);
        assert_eq!(config.variant, KMeansVariant::Lloyd); // Default variant
        assert_eq!(config.max_iterations, 100); // Default max_iterations
        assert_eq!(config.tolerance, 1e-8); // Default tolerance
        assert!(!config.parallel); // Default parallel
        assert!(!config.simd); // Default simd
    }

    #[test]
    fn test_kmeans_config_validation_errors() {
        // Test missing k parameter
        let result = KMeansConfig::builder().build();
        assert!(matches!(
            result,
            Err(FormicaXError::Clustering(crate::core::ClusteringError::InvalidParameters { message }))
            if message.contains("k")
        ));

        // Test k < 2
        let result = KMeansConfig::builder().k(1).build();
        assert!(matches!(
            result,
            Err(FormicaXError::Clustering(crate::core::ClusteringError::InvalidParameters { message }))
            if message.contains("clusters")
        ));

        // Test max_iterations = 0
        let result = KMeansConfig::builder().k(3).max_iterations(0).build();
        assert!(matches!(
            result,
            Err(FormicaXError::Clustering(crate::core::ClusteringError::InvalidParameters { message }))
            if message.contains("iterations")
        ));

        // Test tolerance <= 0
        let result = KMeansConfig::builder().k(3).tolerance(0.0).build();
        assert!(matches!(
            result,
            Err(FormicaXError::Clustering(crate::core::ClusteringError::InvalidParameters { message }))
            if message.contains("Tolerance")
        ));

        let result = KMeansConfig::builder().k(3).tolerance(-1.0).build();
        assert!(matches!(
            result,
            Err(FormicaXError::Clustering(crate::core::ClusteringError::InvalidParameters { message }))
            if message.contains("Tolerance")
        ));
    }

    #[test]
    fn test_kmeans_variant_default() {
        assert_eq!(KMeansVariant::default(), KMeansVariant::Lloyd);
    }

    #[test]
    fn test_kmeans_struct_creation() {
        let config = KMeansConfig::builder()
            .k(4)
            .variant(KMeansVariant::Hamerly)
            .build()
            .unwrap();

        let kmeans = KMeans::with_config(config);
        // For now, just verify the struct can be created
        // The actual fit method is still unimplemented
        assert!(true); // Placeholder assertion
    }
} 