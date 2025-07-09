//! Criterion benchmark for KMeans clustering

#[cfg(test)]
mod benches {
    use super::super::algorithm::*;
    use super::super::config::*;
    use crate::core::OHLCV;
    use chrono::{TimeZone, Utc};
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn generate_synthetic_ohlcv(n: usize) -> Vec<OHLCV> {
        (0..n)
            .map(|i| OHLCV {
                timestamp: Utc.timestamp_opt(1_600_000_000 + i as i64 * 60, 0).unwrap(),
                open: 100.0 + (i % 10) as f64,
                high: 105.0 + (i % 10) as f64,
                low: 98.0 + (i % 10) as f64,
                close: 103.0 + (i % 10) as f64,
                volume: 1000 + (i % 100) as u64,
            })
            .collect()
    }

    fn bench_kmeans(c: &mut Criterion) {
        let data = generate_synthetic_ohlcv(1000);
        let config = KMeansConfig::builder()
            .k(5)
            .max_iterations(50)
            .tolerance(1e-6)
            .build()
            .unwrap();
        c.bench_function("kmeans_1000x5", |b| {
            b.iter(|| {
                let mut kmeans = KMeans::with_config(config.clone());
                let _ = kmeans.fit(black_box(&data)).unwrap();
            })
        });
    }

    criterion_group!(benches, bench_kmeans);
    criterion_main!(benches);
} 