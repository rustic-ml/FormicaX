//! Self-Organizing Maps (SOM) clustering algorithm

pub mod algorithm;
pub mod config;

pub use algorithm::SOM;
pub use config::{SOMConfig, TopologyType};
