//! Affinity Propagation clustering algorithm

pub mod algorithm;
pub mod config;

pub use algorithm::AffinityPropagation;
pub use config::{AffinityPropagationConfig, PreferenceMethod};
