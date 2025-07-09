//! Gaussian Mixture Models (GMM) clustering algorithm

pub mod algorithm;
pub mod config;

pub use algorithm::GMM;
pub use config::{GMMConfig, GMMVariant};
