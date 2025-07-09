use crate::clustering::hierarchical::config::{DistanceMetric, HierarchicalConfig, LinkageMethod};
use crate::core::{
    ClusterResult, ClusteringAlgorithm, ConfigError, DataError, FormicaXError, OHLCV,
};
use std::time::Instant;

/// Hierarchical clustering algorithm
#[derive(Debug)]
pub struct Hierarchical {
    /// Configuration for the hierarchical clustering algorithm
    config: HierarchicalConfig,
    /// Cluster assignments
    assignments: Option<Vec<usize>>,
    /// Cluster centers
    cluster_centers: Option<Vec<Vec<f64>>>,
    /// Dendrogram (simplified)
    dendrogram: Option<Vec<(usize, usize, f64)>>,
    /// Indicates whether the algorithm has been fitted
    fitted: bool,
}

impl Default for Hierarchical {
    fn default() -> Self {
        Self::new()
    }
}

impl Hierarchical {
    /// Create a new Hierarchical with default configuration
    pub fn new() -> Self {
        Self::with_config(HierarchicalConfig::default())
    }

    /// Create a new Hierarchical with custom configuration
    pub fn with_config(config: HierarchicalConfig) -> Self {
        Self {
            config,
            assignments: None,
            cluster_centers: None,
            dendrogram: None,
            fitted: false,
        }
    }

    /// Fit the hierarchical clustering to the data
    pub fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError> {
        if data.is_empty() {
            return Err(FormicaXError::Data(DataError::EmptyDataset));
        }

        self.config.validate()?;
        let start_time = Instant::now();

        // Extract features from OHLCV data
        let features = self.extract_features(data);

        // Perform hierarchical clustering
        let (assignments, cluster_centers, dendrogram) = self.perform_clustering(&features)?;

        let execution_time = start_time.elapsed();

        self.assignments = Some(assignments.clone());
        self.cluster_centers = Some(cluster_centers.clone());
        self.dendrogram = Some(dendrogram);
        self.fitted = true;

        // Calculate quality metrics
        let silhouette_score = self.calculate_silhouette_score(&features, &assignments)?;
        let inertia = self.calculate_inertia(&features, &assignments)?;

        Ok(ClusterResult::new(
            "Hierarchical".to_string(),
            self.config.n_clusters,
            assignments,
        )
        .with_centers(cluster_centers)
        .with_inertia(inertia)
        .with_silhouette_score(silhouette_score)
        .with_converged(true) // Hierarchical clustering always converges
        .with_iterations(1) // Single pass algorithm
        .with_execution_time(execution_time))
    }

    /// Extract features from OHLCV data
    fn extract_features(&self, data: &[OHLCV]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|ohlcv| {
                vec![
                    ohlcv.open,
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    ohlcv.volume as f64,
                ]
            })
            .collect()
    }

    /// Perform hierarchical clustering
    fn perform_clustering(
        &self,
        features: &[Vec<f64>],
    ) -> Result<(Vec<usize>, Vec<Vec<f64>>, Vec<(usize, usize, f64)>), FormicaXError> {
        let n_samples = features.len();
        let n_clusters = self.config.n_clusters.min(n_samples);

        if n_samples <= n_clusters {
            // If we have fewer samples than clusters, assign each to its own cluster
            let assignments: Vec<usize> = (0..n_samples).collect();
            let cluster_centers = features.to_vec();
            return Ok((assignments, cluster_centers, vec![]));
        }

        // Initialize each point as its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();
        let mut dendrogram = Vec::new();

        // Agglomerative clustering
        while clusters.len() > n_clusters {
            let (i, j, distance) = self.find_closest_clusters(&clusters, features)?;

            // Merge clusters i and j
            let mut merged_cluster = clusters[i].clone();
            merged_cluster.extend_from_slice(&clusters[j]);

            // Remove the two clusters and add the merged one
            clusters.remove(j.max(i));
            clusters.remove(j.min(i));
            clusters.push(merged_cluster);

            dendrogram.push((i, j, distance));
        }

        // Create assignments and cluster centers
        let mut assignments = vec![0; n_samples];
        let mut cluster_centers = Vec::new();

        for (cluster_id, cluster) in clusters.iter().enumerate() {
            // Calculate cluster center
            let mut center = vec![0.0; features[0].len()];
            for &point_idx in cluster {
                for (dim, &value) in features[point_idx].iter().enumerate() {
                    center[dim] += value;
                }
                assignments[point_idx] = cluster_id;
            }

            for dim in 0..center.len() {
                center[dim] /= cluster.len() as f64;
            }
            cluster_centers.push(center);
        }

        Ok((assignments, cluster_centers, dendrogram))
    }

    /// Find the two closest clusters
    fn find_closest_clusters(
        &self,
        clusters: &[Vec<usize>],
        features: &[Vec<f64>],
    ) -> Result<(usize, usize, f64), FormicaXError> {
        let mut min_distance = f64::INFINITY;
        let mut closest_pair = (0, 0);

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let distance =
                    self.calculate_cluster_distance(&clusters[i], &clusters[j], features)?;
                if distance < min_distance {
                    min_distance = distance;
                    closest_pair = (i, j);
                }
            }
        }

        Ok((closest_pair.0, closest_pair.1, min_distance))
    }

    /// Calculate distance between two clusters based on linkage method
    fn calculate_cluster_distance(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        features: &[Vec<f64>],
    ) -> Result<f64, FormicaXError> {
        match self.config.linkage {
            LinkageMethod::Single => self.single_linkage(cluster1, cluster2, features),
            LinkageMethod::Complete => self.complete_linkage(cluster1, cluster2, features),
            LinkageMethod::Average => self.average_linkage(cluster1, cluster2, features),
            LinkageMethod::Ward => self.ward_linkage(cluster1, cluster2, features),
            LinkageMethod::Centroid => self.centroid_linkage(cluster1, cluster2, features),
        }
    }

    /// Single linkage (minimum distance between any two points)
    fn single_linkage(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        features: &[Vec<f64>],
    ) -> Result<f64, FormicaXError> {
        let mut min_distance = f64::INFINITY;

        for &i in cluster1 {
            for &j in cluster2 {
                let distance = self.calculate_distance(&features[i], &features[j])?;
                min_distance = min_distance.min(distance);
            }
        }

        Ok(min_distance)
    }

    /// Complete linkage (maximum distance between any two points)
    fn complete_linkage(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        features: &[Vec<f64>],
    ) -> Result<f64, FormicaXError> {
        let mut max_distance: f64 = 0.0;

        for &i in cluster1 {
            for &j in cluster2 {
                let distance = self.calculate_distance(&features[i], &features[j])?;
                max_distance = max_distance.max(distance);
            }
        }

        Ok(max_distance)
    }

    /// Average linkage (average distance between all pairs)
    fn average_linkage(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        features: &[Vec<f64>],
    ) -> Result<f64, FormicaXError> {
        let mut total_distance = 0.0;
        let mut count = 0;

        for &i in cluster1 {
            for &j in cluster2 {
                let distance = self.calculate_distance(&features[i], &features[j])?;
                total_distance += distance;
                count += 1;
            }
        }

        Ok(total_distance / count as f64)
    }

    /// Ward's linkage (minimize within-cluster variance)
    fn ward_linkage(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        features: &[Vec<f64>],
    ) -> Result<f64, FormicaXError> {
        // Simplified Ward's method - use average linkage for now
        self.average_linkage(cluster1, cluster2, features)
    }

    /// Centroid linkage (distance between cluster centroids)
    fn centroid_linkage(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        features: &[Vec<f64>],
    ) -> Result<f64, FormicaXError> {
        let centroid1 = self.calculate_centroid(cluster1, features)?;
        let centroid2 = self.calculate_centroid(cluster2, features)?;
        self.calculate_distance(&centroid1, &centroid2)
    }

    /// Calculate centroid of a cluster
    fn calculate_centroid(
        &self,
        cluster: &[usize],
        features: &[Vec<f64>],
    ) -> Result<Vec<f64>, FormicaXError> {
        if cluster.is_empty() {
            return Err(FormicaXError::Data(DataError::ValidationFailed {
                message: "Cannot calculate centroid of empty cluster".to_string(),
            }));
        }

        let n_features = features[0].len();
        let mut centroid = vec![0.0; n_features];

        for &point_idx in cluster {
            for (dim, &value) in features[point_idx].iter().enumerate() {
                centroid[dim] += value;
            }
        }

        for dim in 0..n_features {
            centroid[dim] /= cluster.len() as f64;
        }

        Ok(centroid)
    }

    /// Calculate distance between two feature vectors
    fn calculate_distance(&self, a: &[f64], b: &[f64]) -> Result<f64, FormicaXError> {
        match self.config.distance_metric {
            DistanceMetric::Euclidean => Ok(self.euclidean_distance(a, b)),
            DistanceMetric::Manhattan => Ok(self.manhattan_distance(a, b)),
            DistanceMetric::Cosine => Ok(self.cosine_distance(a, b)),
        }
    }

    /// Euclidean distance
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Manhattan distance
    fn manhattan_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f64>()
    }

    /// Cosine distance
    fn cosine_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }

        1.0 - (dot_product / (norm_a * norm_b))
    }

    /// Calculate silhouette score
    fn calculate_silhouette_score(
        &self,
        features: &[Vec<f64>],
        assignments: &[usize],
    ) -> Result<f64, FormicaXError> {
        // Simplified silhouette score calculation
        let n_samples = features.len();
        if n_samples < 2 {
            return Ok(0.0);
        }

        let mut total_silhouette = 0.0;
        let mut valid_samples = 0;

        for i in 0..n_samples {
            let cluster_i = assignments[i];

            // Calculate intra-cluster distance
            let mut intra_dist = 0.0;
            let mut intra_count = 0;

            for j in 0..n_samples {
                if i != j && assignments[j] == cluster_i {
                    intra_dist += self.calculate_distance(&features[i], &features[j])?;
                    intra_count += 1;
                }
            }

            if intra_count == 0 {
                continue;
            }
            intra_dist /= intra_count as f64;

            // Calculate nearest inter-cluster distance
            let mut min_inter_dist = f64::INFINITY;
            for cluster_k in 0..self.config.n_clusters {
                if cluster_k != cluster_i {
                    let mut inter_dist = 0.0;
                    let mut inter_count = 0;

                    for j in 0..n_samples {
                        if assignments[j] == cluster_k {
                            inter_dist += self.calculate_distance(&features[i], &features[j])?;
                            inter_count += 1;
                        }
                    }

                    if inter_count > 0 {
                        inter_dist /= inter_count as f64;
                        min_inter_dist = min_inter_dist.min(inter_dist);
                    }
                }
            }

            if min_inter_dist.is_finite() {
                let silhouette = (min_inter_dist - intra_dist) / intra_dist.max(min_inter_dist);
                total_silhouette += silhouette;
                valid_samples += 1;
            }
        }

        if valid_samples > 0 {
            Ok(total_silhouette / valid_samples as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate inertia (within-cluster sum of squares)
    fn calculate_inertia(
        &self,
        features: &[Vec<f64>],
        assignments: &[usize],
    ) -> Result<f64, FormicaXError> {
        if !self.fitted {
            return Err(FormicaXError::Config(ConfigError::ValidationFailed {
                message: "Hierarchical clustering not fitted yet".to_string(),
            }));
        }

        let cluster_centers = self.cluster_centers.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "Hierarchical clustering not fitted yet".to_string(),
            })
        })?;

        let mut inertia = 0.0;
        for (i, feature) in features.iter().enumerate() {
            let cluster = assignments[i];
            if cluster < cluster_centers.len() {
                inertia += self.euclidean_distance_squared(feature, &cluster_centers[cluster]);
            }
        }

        Ok(inertia)
    }

    /// Squared Euclidean distance
    fn euclidean_distance_squared(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
    }

    /// Get cluster assignments
    pub fn get_assignments(&self) -> Option<&Vec<usize>> {
        self.assignments.as_ref()
    }

    /// Get cluster centers
    pub fn get_cluster_centers(&self) -> Option<&Vec<Vec<f64>>> {
        self.cluster_centers.as_ref()
    }

    /// Get dendrogram
    pub fn get_dendrogram(&self) -> Option<&Vec<(usize, usize, f64)>> {
        self.dendrogram.as_ref()
    }
}

impl ClusteringAlgorithm for Hierarchical {
    type Config = HierarchicalConfig;

    fn new() -> Self {
        Self::new()
    }

    fn with_config(config: Self::Config) -> Self {
        Self::with_config(config)
    }

    fn fit(&mut self, data: &[OHLCV]) -> Result<ClusterResult, FormicaXError> {
        self.fit(data)
    }

    fn predict(&self, data: &[OHLCV]) -> Result<Vec<usize>, FormicaXError> {
        let features = self.extract_features(data);
        let cluster_centers = self.cluster_centers.as_ref().ok_or_else(|| {
            FormicaXError::Config(ConfigError::ValidationFailed {
                message: "Hierarchical clustering not fitted yet".to_string(),
            })
        })?;

        let mut assignments = Vec::new();
        for feature in &features {
            let mut best_cluster = 0;
            let mut best_distance = f64::INFINITY;

            for (cluster_id, center) in cluster_centers.iter().enumerate() {
                let distance = self.calculate_distance(feature, center)?;
                if distance < best_distance {
                    best_distance = distance;
                    best_cluster = cluster_id;
                }
            }

            assignments.push(best_cluster);
        }

        Ok(assignments)
    }

    fn get_cluster_centers(&self) -> Option<Vec<Vec<f64>>> {
        self.cluster_centers.clone()
    }

    fn validate_config(&self, data: &[OHLCV]) -> Result<(), FormicaXError> {
        self.config.validate()?;
        if data.is_empty() {
            return Err(FormicaXError::Data(DataError::EmptyDataset));
        }
        Ok(())
    }

    fn algorithm_name(&self) -> &'static str {
        "Hierarchical"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OHLCV;
    use chrono::Utc;

    fn create_test_data() -> Vec<OHLCV> {
        vec![
            OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000),
            OHLCV::new(Utc::now(), 102.0, 107.0, 100.0, 104.0, 1200),
            OHLCV::new(Utc::now(), 104.0, 109.0, 102.0, 106.0, 1100),
            OHLCV::new(Utc::now(), 90.0, 95.0, 88.0, 92.0, 800),
            OHLCV::new(Utc::now(), 92.0, 97.0, 90.0, 94.0, 900),
            OHLCV::new(Utc::now(), 94.0, 99.0, 92.0, 96.0, 850),
        ]
    }

    #[test]
    fn test_hierarchical_creation() {
        let hierarchical = Hierarchical::new();
        assert_eq!(hierarchical.config.n_clusters, 3);
        assert_eq!(hierarchical.config.linkage, LinkageMethod::Ward);
    }

    #[test]
    fn test_hierarchical_with_config() {
        let config = HierarchicalConfig::builder()
            .n_clusters(5)
            .linkage(LinkageMethod::Complete)
            .distance_metric(DistanceMetric::Manhattan)
            .build()
            .unwrap();

        let hierarchical = Hierarchical::with_config(config);
        assert_eq!(hierarchical.config.n_clusters, 5);
        assert_eq!(hierarchical.config.linkage, LinkageMethod::Complete);
    }

    #[test]
    fn test_hierarchical_fit() {
        let mut hierarchical = Hierarchical::new();
        let data = create_test_data();

        let result = hierarchical.fit(&data).unwrap();

        assert_eq!(result.n_clusters, 3);
        assert_eq!(result.assignments().len(), data.len());
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_hierarchical_empty_data() {
        let mut hierarchical = Hierarchical::new();
        let result = hierarchical.fit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hierarchical_predict() {
        let mut hierarchical = Hierarchical::new();
        let data = create_test_data();

        hierarchical.fit(&data).unwrap();

        let predictions = hierarchical.predict(&data).unwrap();
        assert_eq!(predictions.len(), data.len());
        assert!(predictions.iter().all(|&cluster| cluster < 3));
    }

    #[test]
    fn test_hierarchical_get_parameters() {
        let mut hierarchical = Hierarchical::new();
        let data = create_test_data();

        hierarchical.fit(&data).unwrap();

        assert!(hierarchical.get_assignments().is_some());
        assert!(hierarchical.get_cluster_centers().is_some());
        assert!(hierarchical.get_dendrogram().is_some());

        let cluster_centers = hierarchical.get_cluster_centers().unwrap();
        assert_eq!(cluster_centers.len(), 3);
        assert_eq!(cluster_centers[0].len(), 5); // 5 features
    }
}
