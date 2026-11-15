//! Index → solve → render on a synthetic FFmpeg-generated file, then verify
//! the container with ffprobe. Skipped when FFmpeg is not installed.

use std::path::{Path, PathBuf};
use std::process::Command;

use detent_core::edl::{
    AspectPreset, AudioMix, AudioTransition, CaptionSpec, ColorGrade, EdlClip, Intensity,
    MotionCue, MotionPreset, OutputLayout, OutputSpec, SemanticEdl, TransitionKind,
    VideoTransition, VideoTransitionKind,
};
use detent_core::index::{ExcitementPeak, WordToken};
use detent_core::{EdlSolver, ResolvedTimeline};
use detent_indexer::ffmpeg::FfmpegTools;
use detent_indexer::{IndexOptions, SaliencyTracker, StaticTranscript, index_file};
use detent_renderer::{
    Encoder, EncoderChoice, Quality, RenderMode, RenderOptions, Renderer, Tools, copy_eligible,
};
use serde_json::Value;

/// 8 s, 640x360 @ 30 fps busy test pattern with a hard cut at 4 s; speech-like
/// tone bursts separated by silences at 1.8-2.4 s and 5.6-6.2 s.
fn fixture(dir: &Path) -> Option<PathBuf> {
    let tools = FfmpegTools::from_env();
    if !tools.available() {
        eprintln!("skipping: ffmpeg/ffprobe not found");
        return None;
    }
    let out = dir.join("talk.mp4");
    let ok = Command::new(&tools.ffmpeg)
        .args(["-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i"])
        .arg("testsrc2=s=640x360:r=30:d=4[a];smptehdbars=s=640x360:r=30:d=4[b];[a][b]concat=n=2:v=1:a=0")
        .args(["-f", "lavfi", "-i"])
        .arg("aevalsrc='if(between(t,1.8,2.4)+between(t,5.6,6.2),0.0004*random(0),0.25*sin(2*PI*180*t)*(0.6+0.4*sin(2*PI*3*t)))':s=48000:d=8")
        .args(["-c:v", "libx264", "-preset", "ultrafast", "-g", "30", "-pix_fmt", "yuv420p"])
        .args(["-c:a", "aac", "-shortest"])
        .arg(&out)
        .status()
        .ok()?
        .success();
    ok.then_some(out)
}

fn probe(path: &Path) -> Value {
    let out = Command::new(FfmpegTools::from_env().ffprobe)
        .args(["-v", "error", "-count_frames", "-show_entries"])
        .arg("stream=codec_type,width,height,nb_read_frames,duration")
        .args(["-of", "json"])
        .arg(path)
        .output()
        .unwrap();
    serde_json::from_slice(&out.stdout).unwrap()
}

fn stream<'a>(v: &'a Value, kind: &str) -> &'a Value {
    v["streams"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["codec_type"] == kind)
        .unwrap()
}

fn solve(path: &Path, dir: &Path, edl: &SemanticEdl) -> ResolvedTimeline {
    let words = [
        ("Rust", 300, 700),
        ("makes", 750, 1_100),
        ("the", 1_150, 1_300),
        ("renderer", 1_350, 1_750),
        ("deterministic.", 2_450, 3_400),
        ("Every", 6_250, 6_700),
        ("cut", 6_750, 7_100),
        ("lands", 7_150, 7_500),
        ("cleanly.", 7_550, 7_950),
    ]
    .iter()
    .map(|&(w, s, e)| WordToken {
        word: w.into(),
        start_ms: s,
        end_ms: e,
        confidence: 0.9,
    })
    .collect();
    let opts = IndexOptions {
        contact_sheet_path: Some(dir.join("grid.jpg")),
        ..IndexOptions::default()
    };
    let mut indexed = index_file(
        &FfmpegTools::from_env(),
        path,
        &mut StaticTranscript(words),
        &mut SaliencyTracker::new(200),
        &opts,
    )
    .unwrap();
    // Guarantee a punch-in trigger regardless of synthetic DSP output.
    indexed.index.audio.peaks.push(ExcitementPeak {
        t_ms: 6_800,
        score: 3.0,
    });
    indexed.index.audio.peaks.sort_by_key(|p| p.t_ms);
    let res = EdlSolver::default().solve(edl, &indexed.index).unwrap();
    assert!(res.timeline.verify().is_empty());
    res.timeline
}

fn edl() -> SemanticEdl {
    SemanticEdl {
        clips: vec![
            EdlClip {
                start_ms: 6_150,
                end_ms: 8_000,
                hook_score: 0.9,
                editorial_note: "hook".into(),
            },
            EdlClip {
                start_ms: 250,
                end_ms: 3_450,
                hook_score: 0.5,
                editorial_note: "setup".into(),
            },
        ],
        motion_cues: vec![
            MotionCue {
                clip: 0,
                preset: MotionPreset::SpringPunchIn,
                intensity: Intensity::Strong,
            },
            MotionCue {
                clip: 1,
                preset: MotionPreset::SmoothTrackSubject,
                intensity: Intensity::Medium,
            },
        ],
        audio_transitions: vec![AudioTransition {
            after_clip: 0,
            kind: TransitionKind::AutoLCut,
            offset_ms: Some(200),
        }],
        captions: Some(CaptionSpec::default()),
        headline: None,
        highlight_words: Vec::new(),
        headline_duration_ms: None,
        video_transitions: Vec::new(),
        tighten: None,
        output: OutputSpec {
            aspect: AspectPreset::Vertical9x16,
            height: 640,
            layout: Default::default(),
            ..OutputSpec::default()
        },
    }
}

#[test]
fn full_render_is_frame_exact() {
    let dir = tempfile::tempdir().unwrap();
    let Some(src) = fixture(dir.path()) else {
        return;
    };
    let t = solve(&src, dir.path(), &edl());
    let renderer = Renderer::new(Tools::default()).unwrap();
    let out = dir.path().join("out.mp4");
    let mut updates = 0;
    let m = renderer
        .render(
            &t,
            &out,
            &RenderOptions {
                quality: Quality::Draft,
                encoder: EncoderChoice::Force(Encoder::Libx264),
                mode: RenderMode::Full,
            },
            &mut |_| updates += 1,
        )
        .unwrap();
    assert_eq!(m.mode, "filtergraph");
    assert!(updates > 0);

    let p = probe(&out);
    let v = stream(&p, "video");
    assert_eq!(
        (v["width"].as_u64(), v["height"].as_u64()),
        (Some(360), Some(640))
    );
    let expected: i64 = t.segments.iter().map(|s| s.video.frames()).sum();
    let got: i64 = v["nb_read_frames"].as_str().unwrap().parse().unwrap();
    assert_eq!(
        got, expected,
        "output must contain exactly the solved frames"
    );

    let a = stream(&p, "audio");
    let a_dur: f64 = a["duration"].as_str().unwrap().parse().unwrap();
    let v_dur = t.duration_us as f64 / 1e6;
    assert!(
        (a_dur - v_dur).abs() < 0.05,
        "audio {a_dur}s vs video {v_dur}s"
    );
}

#[test]
fn finishing_effects_keep_frames_and_sync() {
    let dir = tempfile::tempdir().unwrap();
    let Some(src) = fixture(dir.path()) else {
        return;
    };
    for layout in [OutputLayout::Crop, OutputLayout::FitBlurTall] {
        let mut e = edl();
        e.headline = Some("The hook".into());
        e.headline_duration_ms = Some(3_000);
        e.video_transitions = vec![VideoTransition {
            after_clip: 0,
            kind: VideoTransitionKind::Flash,
        }];
        e.output.layout = layout;
        e.output.grade = ColorGrade::MobilePop;
        e.output.audio_mix = AudioMix::VoiceForward;
        e.output.opening_pop = true;
        let t = solve(&src, dir.path(), &e);
        assert!(t.splices[0].flash);
        assert!(!copy_eligible(&t), "finishing effects need a re-encode");
        let renderer = Renderer::new(Tools::default()).unwrap();
        let out = dir.path().join(format!("finish-{layout:?}.mp4"));
        renderer
            .render(
                &t,
                &out,
                &RenderOptions {
                    quality: Quality::Draft,
                    encoder: EncoderChoice::Force(Encoder::Libx264),
                    mode: RenderMode::Full,
                },
                &mut |_| {},
            )
            .unwrap();
        let p = probe(&out);
        let v = stream(&p, "video");
        assert_eq!(
            (v["width"].as_u64(), v["height"].as_u64()),
            (Some(360), Some(640))
        );
        let expected: i64 = t.segments.iter().map(|s| s.video.frames()).sum();
        let got: i64 = v["nb_read_frames"].as_str().unwrap().parse().unwrap();
        assert_eq!(got, expected, "{layout:?}");
        let a = stream(&p, "audio");
        let a_dur: f64 = a["duration"].as_str().unwrap().parse().unwrap();
        let v_dur = t.duration_us as f64 / 1e6;
        assert!(
            (a_dur - v_dur).abs() < 0.05,
            "{layout:?}: audio {a_dur}s vs video {v_dur}s"
        );
    }
}

#[test]
fn preview_uses_stream_copy_when_possible() {
    let dir = tempfile::tempdir().unwrap();
    let Some(src) = fixture(dir.path()) else {
        return;
    };
    let mut e = edl();
    e.motion_cues.clear();
    e.captions = None;
    e.output = OutputSpec {
        aspect: AspectPreset::Source,
        height: 360,
        layout: Default::default(),
        ..OutputSpec::default()
    };
    let t = solve(&src, dir.path(), &e);
    assert!(copy_eligible(&t));
    let renderer = Renderer::new(Tools::default()).unwrap();
    let out = dir.path().join("preview.mp4");
    let m = renderer
        .render(
            &t,
            &out,
            &RenderOptions {
                mode: RenderMode::Preview,
                ..RenderOptions::default()
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(m.mode, "stream_copy");
    assert!(m.encoder.is_none());
    // GOP is 1 s, so keyframe alignment adds < 1 s per segment.
    assert!(m.keyframe_drift_ms.unwrap() < 2_000);
    let p = probe(&out);
    let v = stream(&p, "video");
    assert_eq!(v["width"].as_u64(), Some(640));
}

#[test]
fn rejects_tampered_timeline() {
    let dir = tempfile::tempdir().unwrap();
    let Some(src) = fixture(dir.path()) else {
        return;
    };
    let mut t = solve(&src, dir.path(), &edl());
    t.splices[0].crossfade_us = 90_000;
    let renderer = Renderer::new(Tools::default()).unwrap();
    let err = renderer
        .render(
            &t,
            &dir.path().join("x.mp4"),
            &RenderOptions::default(),
            &mut |_| {},
        )
        .unwrap_err();
    assert!(err.to_string().contains("invalid timeline"), "{err}");
}
