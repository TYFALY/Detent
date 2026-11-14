//! `detent bench`: reproducible end-to-end numbers for the README.
//!
//! Synthetic stages use the deterministic generators in
//! `detent_core::testkit`, so they are comparable across machines. With
//! `--input`, the real-file stages measure FFmpeg decode + analysis and
//! rendering on that file.

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use detent_core::EdlSolver;
use detent_core::captions::{TimedWord, chunk};
use detent_core::dsp::{AudioAnalyzer, DspConfig, PauseConfig, analyze};
use detent_core::edl::{AspectPreset, CaptionGrouping, EdlClip, OutputSpec, SemanticEdl};
use detent_core::testkit::{speech_like_pcm, synthetic_edl, synthetic_index};
use detent_indexer::ffmpeg::FfmpegTools;
use detent_indexer::{IndexOptions, NoTranscript, SaliencyTracker};
use detent_renderer::{
    Encoder, EncoderChoice, Quality, RenderMode, RenderOptions, Renderer, Tools,
};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn time<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64())
}

fn cpu_name() -> String {
    if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo")
        && let Some(l) = s.lines().find(|l| l.starts_with("model name"))
    {
        return l.split(':').nth(1).unwrap_or("").trim().to_owned();
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown CPU".into())
}

pub fn run(input: Option<&Path>, seconds: f32) -> Result<()> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!(
        "Machine: {} | {} hardware threads | {}-{} | release={}\n",
        cpu_name(),
        threads,
        std::env::consts::OS,
        std::env::consts::ARCH,
        !cfg!(debug_assertions)
    );
    println!("| Stage | Workload | Result |");
    println!("|---|---|---|");

    // 1. Audio DSP (single thread).
    let pcm = speech_like_pcm(seconds, 16_000);
    let bytes = (pcm.len() * 4) as f64;
    let mut walls = Vec::new();
    let mut pauses = 0;
    for _ in 0..3 {
        let (tel, wall) = time(|| -> Result<_> {
            let mut a = AudioAnalyzer::new(DspConfig::default())?;
            for b in pcm.chunks(16_384) {
                a.push(b);
            }
            Ok(analyze(a.finish(), &PauseConfig::default()))
        });
        pauses = tel?.pauses.len();
        walls.push(wall);
    }
    let w = median(walls);
    println!(
        "| Audio DSP (RMS, ZCR, 512-pt FFT flux, pauses, peaks) | {seconds:.0} s mono f32 @ 16 kHz ({:.1} MB), 1 thread | **{:.0} MB/s**, **{:.0}x realtime** ({pauses} pauses) |",
        bytes / 1e6,
        bytes / 1e6 / w,
        f64::from(seconds) / w
    );

    // 2. Solver latency.
    let idx = synthetic_index(600_000, 11);
    let solver = EdlSolver::default();
    for clips in [4_usize, 12, 32] {
        let edl = synthetic_edl(&idx, clips, 5);
        let frames: i64 = solver
            .solve(&edl, &idx)?
            .timeline
            .segments
            .iter()
            .map(|s| s.video.frames())
            .sum();
        let samples: Vec<f64> = (0..200)
            .map(|_| time(|| black_box(solver.solve(black_box(&edl), &idx))).1 * 1e6)
            .collect();
        println!(
            "| EdlSolver::solve (snap, scene guard, J/L, crossfades, springs, captions) | {clips} clips / {frames} frames, 10-min index | **{:.0} µs** median |",
            median(samples)
        );
    }

    // 3. Caption chunking.
    let words: Vec<TimedWord> = (0..1_000)
        .map(|i| TimedWord {
            text: [
                "the", "camera", "of", "a", "talking", "head,", "never", "jitters.",
            ][i % 8]
                .into(),
            start_us: i as i64 * 300_000,
            end_us: i as i64 * 300_000 + 250_000,
        })
        .collect();
    let samples: Vec<f64> = (0..200)
        .map(|_| {
            time(|| black_box(chunk(&words, &[], &CaptionGrouping::default(), 250_000))).1 * 1e6
        })
        .collect();
    println!(
        "| Caption DP chunker | 1,000 words | **{:.0} µs** median |",
        median(samples)
    );

    let Some(input) = input else {
        println!("\n(pass --input <file> to benchmark real decode + render)");
        return Ok(());
    };

    // 4. Real-file indexing.
    let tools = FfmpegTools::from_env();
    let out = tempfile::tempdir()?;
    let opts = IndexOptions {
        contact_sheet_path: Some(out.path().join("grid.jpg")),
        ..IndexOptions::default()
    };
    let indexed = detent_indexer::index_file(
        &tools,
        input,
        &mut NoTranscript,
        &mut SaliencyTracker::new(200),
        &opts,
    )?;
    let i = &indexed.index;
    let v = i.source.video.as_ref();
    println!(
        "| Index real file (ffmpeg decode → DSP + shots + saliency + contact sheet) | {:.0} s {} | **{:.0}x realtime** ({:.2} s wall, {} frames, {} shots) |",
        i.source.duration_ms as f64 / 1e3,
        v.map_or("audio".into(), |v| format!(
            "{}x{}@{:.2}",
            v.width,
            v.height,
            v.fps.as_f64()
        )),
        indexed.stats.realtime_factor,
        indexed.stats.wall_ms / 1e3,
        indexed.stats.video_frames,
        i.visual.shots.len()
    );

    // 5. Rendering: 3 clips → 9:16 with motion + captions-free (no transcript).
    let dur = i.source.duration_ms;
    let clip = |a: u64, b: u64| EdlClip {
        start_ms: a,
        end_ms: b,
        hook_score: 0.5,
        editorial_note: String::new(),
    };
    let span = (dur / 4).min(20_000);
    let mut edl = SemanticEdl {
        clips: vec![
            clip(dur / 2, dur / 2 + span),
            clip(dur / 8, dur / 8 + span),
            clip(dur * 3 / 4, dur * 3 / 4 + span),
        ],
        motion_cues: vec![],
        audio_transitions: vec![],
        captions: None,
        headline: None,
        highlight_words: Vec::new(),
        output: OutputSpec {
            aspect: AspectPreset::Vertical9x16,
            height: 1920,
            layout: Default::default(),
        },
    };
    edl.motion_cues = synthetic_edl(i, 3, 1).motion_cues;
    let t = EdlSolver::default().solve(&edl, i)?.timeline;
    let renderer = Renderer::new(Tools::default())?;
    let out_s = t.duration_us as f64 / 1e6;
    let mut encoders: Vec<EncoderChoice> = vec![EncoderChoice::Force(Encoder::Libx264)];
    if let Ok(e) = renderer.info().select(EncoderChoice::Auto)
        && e != Encoder::Libx264
    {
        encoders.insert(0, EncoderChoice::Force(e));
    }
    for enc in encoders {
        let m = renderer.render(
            &t,
            &out.path().join("bench.mp4"),
            &RenderOptions {
                quality: Quality::Standard,
                encoder: enc,
                mode: RenderMode::Full,
            },
            &mut |_| {},
        )?;
        println!(
            "| Render 9:16 1080x1920, spring camera, crossfades | {out_s:.0} s output, {} | **{:.1}x realtime** ({:.0} fps) |",
            m.encoder.as_deref().unwrap_or("?"),
            m.realtime_factor,
            m.encode_fps
        );
    }
    let mut prev = edl.clone();
    prev.motion_cues.clear();
    prev.output = OutputSpec {
        aspect: AspectPreset::Source,
        height: v.map_or(1080, |v| v.height),
        layout: Default::default(),
    };
    let t = EdlSolver::default().solve(&prev, i)?.timeline;
    let m = renderer.render(
        &t,
        &out.path().join("preview.mp4"),
        &RenderOptions {
            mode: RenderMode::Preview,
            ..RenderOptions::default()
        },
        &mut |_| {},
    )?;
    println!(
        "| Preview (keyframe-aligned stream copy) | {:.0} s output | **{:.0}x realtime** ({}) |",
        t.duration_us as f64 / 1e6,
        m.realtime_factor,
        m.mode
    );
    Ok(())
}
