//! Clustering algorithms module for FormicaX

pub mod affinity_propagation;
pub mod dbscan;
pub mod gmm;
pub mod hierarchical;
pub mod kmeans;
pub mod som;

// Re-export main types for convenience
pub use affinity_propagation::algorithm::AffinityPropagation;
pub use affinity_propagation::config::{AffinityPropagationConfig, PreferenceMethod};
pub use dbscan::algorithm::DBSCAN;
pub use dbscan::config::{DBSCANConfig, DBSCANVariant};
pub use gmm::algorithm::GMM;
pub use gmm::config::{GMMConfig, GMMVariant};
pub use hierarchical::algorithm::Hierarchical;
pub use hierarchical::config::{HierarchicalConfig, LinkageMethod};
pub use kmeans::algorithm::KMeans;
pub use kmeans::config::{KMeansConfig, KMeansVariant};
pub use som::algorithm::SOM;
pub use som::config::{SOMConfig, TopologyType};
