//! Per-frame vision cost: histogram shot detection and saliency tracking on
//! 160x90 analysis frames (the default analysis raster).

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use detent_indexer::RgbFrame;
use detent_indexer::saliency::{SaliencyTracker, SubjectDetector};
use detent_indexer::shots::{ShotConfig, ShotDetector};

fn frames(n: usize) -> Vec<RgbFrame> {
    let mut s: u32 = 0x9E37_79B9;
    (0..n)
        .map(|k| {
            let mut f = RgbFrame::solid(160, 90, [(k * 3 % 255) as u8, 90, 140]);
            for v in &mut f.data {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *v = v.wrapping_add((s % 23) as u8);
            }
            f
        })
        .collect()
}

fn vision(c: &mut Criterion) {
    let fs = frames(300);
    let mut g = c.benchmark_group("vision");
    g.throughput(Throughput::Elements(fs.len() as u64));
    g.bench_function("shot_detector_160x90", |b| {
        b.iter(|| {
            let mut d = ShotDetector::new(ShotConfig::default());
            for f in &fs {
                d.push(black_box(f));
            }
            black_box(d.frames())
        });
    });
    g.bench_function("saliency_160x90", |b| {
        b.iter(|| {
            let mut t = SaliencyTracker::new(0);
            let mut n = 0;
            for (k, f) in fs.iter().enumerate() {
                if let Ok(Some(_)) = t.observe(black_box(f), k as u64 * 33) {
                    n += 1;
                }
            }
            black_box(n)
        });
    });
    g.finish();
}

criterion_group!(benches, vision);
criterion_main!(benches);
