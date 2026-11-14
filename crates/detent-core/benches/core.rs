//! Criterion benchmarks for the hot paths of `detent-core`.
//!
//! * `dsp/*`     — streaming envelope + pause/peak detection throughput.
//! * `solver/*`  — end-to-end `EdlSolver::solve` latency.
//! * `camera/*`  — spring simulation + RDP compression for one segment.
//! * `captions/*` — Knuth–Plass chunking.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use detent_core::EdlSolver;
use detent_core::captions::{TimedWord, chunk};
use detent_core::dsp::{AudioAnalyzer, DspConfig, PauseConfig, analyze};
use detent_core::edl::CaptionGrouping;
use detent_core::testkit::{speech_like_pcm, synthetic_edl, synthetic_index};

fn dsp(c: &mut Criterion) {
    let mut g = c.benchmark_group("dsp");
    for secs in [60.0_f32, 600.0] {
        let pcm = speech_like_pcm(secs, 16_000);
        g.throughput(Throughput::Bytes((pcm.len() * 4) as u64));
        g.sample_size(if secs > 100.0 { 10 } else { 30 });
        g.bench_with_input(
            BenchmarkId::new("envelope+pauses", format!("{secs}s@16k")),
            &pcm,
            |b, pcm| {
                b.iter(|| {
                    let mut a =
                        AudioAnalyzer::new(DspConfig::default()).unwrap_or_else(|e| panic!("{e}"));
                    for block in pcm.chunks(16_384) {
                        a.push(block);
                    }
                    black_box(analyze(a.finish(), &PauseConfig::default()))
                });
            },
        );
    }
    g.finish();
}

fn solver(c: &mut Criterion) {
    let mut g = c.benchmark_group("solver");
    let idx = synthetic_index(600_000, 11);
    for clips in [4_usize, 12, 32] {
        let edl = synthetic_edl(&idx, clips, 5);
        g.bench_with_input(
            BenchmarkId::new("solve", format!("{clips}_clips")),
            &edl,
            |b, edl| {
                let solver = EdlSolver::default();
                b.iter(|| black_box(solver.solve(black_box(edl), &idx)));
            },
        );
    }
    g.finish();
}

fn captions(c: &mut Criterion) {
    let words: Vec<TimedWord> = (0..1_000)
        .map(|i| TimedWord {
            text: [
                "the", "spring", "camera", "of", "a", "talking", "head,", "never", "jitters.",
            ][i % 9]
                .to_owned(),
            start_us: i as i64 * 300_000,
            end_us: i as i64 * 300_000 + 250_000,
        })
        .collect();
    c.bench_function("captions/chunk_1000_words", |b| {
        b.iter(|| {
            black_box(chunk(
                black_box(&words),
                &[],
                &CaptionGrouping::default(),
                250_000,
            ))
        });
    });
}

criterion_group!(benches, dsp, solver, captions);
criterion_main!(benches);
