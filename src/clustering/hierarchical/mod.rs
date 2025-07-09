//! Hierarchical clustering algorithm

pub mod algorithm;
pub mod config;

pub use algorithm::Hierarchical;
pub use config::{HierarchicalConfig, LinkageMethod};
