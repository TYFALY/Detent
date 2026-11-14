//! End-to-end tests of `EdlSolver` against synthetic media indices, including
//! property-based invariant checks.

use detent_core::edl::{
    AspectPreset, AudioTransition, CaptionSpec, EdlClip, Intensity, MotionCue, MotionPreset,
    OutputSpec, SemanticEdl, TransitionKind,
};
use detent_core::index::{
    AudioStreamInfo, AudioTelemetry, ExcitementPeak, INDEX_VERSION, MediaIndex, PauseKind,
    ShotBoundary, SilenceInterval, SourceInfo, SubjectBox, SubjectSource, VideoStreamInfo,
    VisualTelemetry, WordToken,
};
use detent_core::solver::{Adjustment, EdlSolver, Level};
use detent_core::time::FrameRate;
use detent_core::{CoreError, ResolvedTimeline};
use proptest::prelude::*;

/// Builds an index from `(word, start_ms, end_ms)`; every gap >= 100 ms
/// between words becomes a silence with its valley in the middle.
fn index(words: &[(&str, u64, u64)], duration_ms: u64, fps: FrameRate) -> MediaIndex {
    let transcript: Vec<WordToken> = words
        .iter()
        .map(|&(w, s, e)| WordToken {
            word: w.into(),
            start_ms: s,
            end_ms: e,
            confidence: 0.95,
        })
        .collect();
    let mut pauses = Vec::new();
    let mut prev_end = 0;
    for w in transcript
        .iter()
        .map(|w| (w.start_ms, w.end_ms))
        .chain(std::iter::once((duration_ms, duration_ms)))
    {
        if w.0 >= prev_end + 100 {
            pauses.push(SilenceInterval {
                start_ms: prev_end,
                end_ms: w.0,
                min_dbfs: -62.0,
                valley_ms: (prev_end + w.0) / 2,
                kind: PauseKind::Silence,
            });
        }
        prev_end = prev_end.max(w.1);
    }
    MediaIndex {
        version: INDEX_VERSION,
        source: SourceInfo {
            path: "/media/talk.mp4".into(),
            duration_ms,
            video: Some(VideoStreamInfo {
                width: 1920,
                height: 1080,
                fps,
                codec: "h264".into(),
            }),
            audio: Some(AudioStreamInfo {
                sample_rate: 48_000,
                channels: 2,
                codec: "aac".into(),
            }),
        },
        transcript,
        audio: AudioTelemetry {
            hop_ms: 10,
            analysis_rate: 16_000,
            pauses,
            ..AudioTelemetry::default()
        },
        visual: VisualTelemetry::default(),
    }
}

fn clip(start_ms: u64, end_ms: u64) -> EdlClip {
    EdlClip {
        start_ms,
        end_ms,
        hook_score: 0.5,
        editorial_note: "test".into(),
    }
}

fn edl(clips: Vec<EdlClip>) -> SemanticEdl {
    SemanticEdl {
        clips,
        motion_cues: vec![],
        audio_transitions: vec![],
        captions: None,
        output: OutputSpec::default(),
    }
}

fn fps30() -> FrameRate {
    FrameRate::new(30, 1).unwrap()
}

/// "Hello there friend. ... This is the second sentence." with clear pauses.
fn talk() -> MediaIndex {
    index(
        &[
            ("Hello", 1_000, 1_300),
            ("there", 1_350, 1_700),
            ("friend.", 1_750, 2_200),
            ("This", 3_000, 3_250),
            ("is", 3_300, 3_450),
            ("the", 3_500, 3_600),
            ("second", 3_650, 4_000),
            ("sentence.", 4_050, 4_600),
            ("And", 5_400, 5_600),
            ("a", 5_650, 5_700),
            ("third", 5_750, 6_100),
            ("one.", 6_150, 6_600),
        ],
        8_000,
        fps30(),
    )
}

#[test]
fn snaps_into_silence_valleys() {
    let idx = talk();
    // In-point 60 ms inside "This"; out-point 50 ms inside "sentence.".
    let res = EdlSolver::default()
        .solve(&edl(vec![clip(3_060, 4_550)]), &idx)
        .unwrap();
    let cut = &res.report.cuts[0];
    // Gap before "This" is 2200..3000 (valley 2600) but tolerance is 120 ms,
    // so the cut lands on the silence edge-most point in the window: 2940.
    assert!(
        cut.video_ms.0 <= 3_000 && cut.video_ms.0 >= 2_930,
        "{cut:?}"
    );
    // Gap after "sentence." is 4600..5400; window reaches 4670.
    assert!(
        cut.video_ms.1 >= 4_600 && cut.video_ms.1 <= 4_680,
        "{cut:?}"
    );
    let snaps = res
        .report
        .adjustments
        .iter()
        .filter(|a| matches!(a, Adjustment::BreathSnap { .. }))
        .count();
    assert_eq!(snaps, 2);
    assert!(
        !res.report
            .diagnostics
            .iter()
            .any(|d| d.code == "cut_inside_word"),
        "{:?}",
        res.report.diagnostics
    );
}

#[test]
fn falls_back_to_word_gaps_without_pauses() {
    let mut idx = talk();
    idx.audio.pauses.clear();
    // Inside "there" (1350..1700), gap to "friend." is 1700..1750 → mid 1725.
    let res = EdlSolver::default()
        .solve(&edl(vec![clip(1_000, 1_690)]), &idx)
        .unwrap();
    assert!(
        res.report
            .adjustments
            .iter()
            .any(|a| matches!(a, Adjustment::WordGapSnap { to_ms: 1_725, .. })),
        "{:?}",
        res.report.adjustments
    );
}

#[test]
fn scene_guard_snaps_to_shot_frame() {
    let mut idx = talk();
    // Shot change at frame 85 (2833.3 ms), inside the 2200..3000 silence.
    idx.visual.shots.push(ShotBoundary {
        t_ms: 2_833,
        frame: 85,
        score: 4.0,
    });
    let res = EdlSolver::default()
        .solve(&edl(vec![clip(2_950, 4_550)]), &idx)
        .unwrap();
    let seg = &res.timeline.segments[0];
    assert_eq!(seg.video.start_frame, 85, "{:?}", res.report.adjustments);
    assert!(
        res.report
            .adjustments
            .iter()
            .any(|a| matches!(a, Adjustment::SceneSnap { .. }))
    );
}

#[test]
fn scene_guard_refuses_to_cut_speech() {
    let mut idx = talk();
    // Shot change in the middle of "second" (3650..4000).
    idx.visual.shots.push(ShotBoundary {
        t_ms: 3_800,
        frame: 114,
        score: 4.0,
    });
    let res = EdlSolver::default()
        .solve(&edl(vec![clip(1_000, 3_760)]), &idx)
        .unwrap();
    assert!(
        res.report
            .diagnostics
            .iter()
            .any(|d| d.code == "scene_collision" && d.level == Level::Warning)
    );
    assert_ne!(res.timeline.segments[0].video.end_frame, 114);
}

#[test]
fn j_cut_applied_when_both_sides_are_silent() {
    let idx = talk();
    let mut e = edl(vec![clip(950, 2_300), clip(5_300, 6_700)]);
    e.audio_transitions.push(AudioTransition {
        after_clip: 0,
        kind: TransitionKind::AutoJCut,
        offset_ms: Some(250),
    });
    let res = EdlSolver::default().solve(&e, &idx).unwrap();
    let sp = res.timeline.splices[0];
    assert!(sp.av_offset_us > 0, "{:?}", res.report);
    let applied = res.report.adjustments.iter().find_map(|a| match a {
        Adjustment::JCut { offset_ms, .. } => Some(*offset_ms),
        _ => None,
    });
    assert!(matches!(applied, Some(150..=250)), "{applied:?}");
    // Audio of clip 1 starts before its picture.
    let s1 = &res.timeline.segments[1];
    assert!(s1.audio.start_us < s1.video.start_us);
}

#[test]
fn l_cut_downgraded_when_speech_follows() {
    let idx = talk();
    // Clip 0 ends right before "This" at 3000: extending audio would leak speech.
    let mut e = edl(vec![clip(950, 2_990), clip(5_300, 6_700)]);
    e.audio_transitions.push(AudioTransition {
        after_clip: 0,
        kind: TransitionKind::AutoLCut,
        offset_ms: Some(300),
    });
    let res = EdlSolver::default().solve(&e, &idx).unwrap();
    let out = res.report.cuts[0].video_ms.1;
    if out + 150 > 2_980 {
        assert_eq!(res.timeline.splices[0].av_offset_us, 0);
        assert!(
            res.report
                .diagnostics
                .iter()
                .any(|d| d.code == "jl_downgraded")
        );
    }
}

#[test]
fn every_splice_gets_a_crossfade_and_av_stays_locked() {
    let idx = talk();
    let res = EdlSolver::default()
        .solve(
            &edl(vec![
                clip(950, 2_300),
                clip(2_900, 4_700),
                clip(5_300, 6_700),
            ]),
            &idx,
        )
        .unwrap();
    let t = &res.timeline;
    assert_eq!(t.splices.len(), 2);
    for sp in &t.splices {
        assert_eq!(sp.crossfade_us, 20_000);
    }
    let video: i64 = t.segments.iter().map(|s| s.video.len_us()).sum();
    let audio: i64 = t.segments.iter().map(|s| s.audio.len_us()).sum();
    assert_eq!(audio - 40_000, video);
    assert!(t.verify().is_empty());
}

#[test]
fn validation_reports_every_issue_at_once() {
    let idx = talk();
    let mut e = edl(vec![clip(500, 400), clip(0, 9_000)]);
    e.clips[1].hook_score = 1.5;
    e.motion_cues.push(MotionCue {
        clip: 7,
        preset: MotionPreset::SlowCreep,
        intensity: Intensity::Medium,
    });
    e.audio_transitions.push(AudioTransition {
        after_clip: 0,
        kind: TransitionKind::AutoJCut,
        offset_ms: Some(900),
    });
    match EdlSolver::default().solve(&e, &idx) {
        Err(CoreError::Validation(issues)) => {
            let paths: Vec<_> = issues.iter().map(|i| i.path.as_str()).collect();
            for p in [
                "/clips/0/end_ms",
                "/clips/1/end_ms",
                "/clips/1/hook_score",
                "/motion_cues/0/clip",
                "/audio_transitions/0/offset_ms",
            ] {
                assert!(paths.contains(&p), "missing {p} in {paths:?}");
            }
        }
        other => panic!("expected validation error, got {other:?}"),
    }
}

#[test]
fn captions_follow_output_timeline() {
    let idx = talk();
    let mut e = edl(vec![clip(2_900, 4_700), clip(950, 2_300)]);
    e.captions = Some(CaptionSpec::default());
    let res = EdlSolver::default().solve(&e, &idx).unwrap();
    let caps = res.timeline.captions.as_ref().unwrap();
    let text: Vec<String> = caps
        .blocks
        .iter()
        .flat_map(|b| b.words.iter().map(|w| w.text.clone()))
        .collect();
    assert_eq!(
        text,
        [
            "This",
            "is",
            "the",
            "second",
            "sentence.",
            "Hello",
            "there",
            "friend."
        ]
    );
    // Reordered clip: "Hello" appears after the first segment.
    let hello = caps
        .blocks
        .iter()
        .find(|b| b.words[0].text == "Hello")
        .unwrap();
    assert!(hello.start_us >= res.timeline.segments[1].output_start_us);
}

#[test]
fn solving_is_deterministic() {
    let mut idx = talk();
    idx.audio.peaks.push(ExcitementPeak {
        t_ms: 3_700,
        score: 2.5,
    });
    let mut e = edl(vec![clip(2_900, 4_700), clip(950, 2_300)]);
    e.motion_cues.push(MotionCue {
        clip: 0,
        preset: MotionPreset::SpringPunchIn,
        intensity: Intensity::Strong,
    });
    e.captions = Some(CaptionSpec::default());
    let a = EdlSolver::default().solve(&e, &idx).unwrap();
    let b = EdlSolver::default().solve(&e, &idx).unwrap();
    assert_eq!(
        serde_json::to_string(&a.timeline).unwrap(),
        serde_json::to_string(&b.timeline).unwrap()
    );
    let round_trip =
        ResolvedTimeline::from_json(&serde_json::to_string(&a.timeline).unwrap()).unwrap();
    assert_eq!(round_trip, a.timeline);
}

#[test]
fn tampered_timeline_is_rejected() {
    let idx = talk();
    let res = EdlSolver::default()
        .solve(&edl(vec![clip(950, 2_300), clip(2_900, 4_700)]), &idx)
        .unwrap();
    let mut t = res.timeline;
    t.segments[1].output_start_us += 1;
    assert!(ResolvedTimeline::from_json(&serde_json::to_string(&t).unwrap()).is_err());
}

// ---------------------------------------------------------------------------
// Property-based invariants
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Scenario {
    idx: MediaIndex,
    edl: SemanticEdl,
}

fn arb_fps() -> impl Strategy<Value = FrameRate> {
    prop::sample::select(vec![(24, 1), (25, 1), (30, 1), (30_000, 1_001), (60, 1)])
        .prop_map(|(n, d)| FrameRate::new(n, d).unwrap())
}

prop_compose! {
    fn arb_index()(
        fps in arb_fps(),
        duration_ms in 8_000u64..90_000,
        word_shape in prop::collection::vec((150u64..600, 20u64..800, any::<bool>()), 10..300),
        shot_ms in prop::collection::vec(0u64..90_000, 0..30),
        peak_ms in prop::collection::vec(0u64..90_000, 0..30),
        subj in prop::collection::vec((0.0f32..0.9, 0.0f32..0.9), 0..20),
        has_audio in prop::bool::weighted(0.9),
        has_words in prop::bool::weighted(0.8),
        w in prop::sample::select(vec![640u32, 1280, 1920, 1080, 720, 3840]),
        h in prop::sample::select(vec![480u32, 720, 1080, 1920, 2160]),
    ) -> MediaIndex {
        let mut words = Vec::new();
        let mut t = 200;
        for (i, (len, gap, breath)) in word_shape.iter().enumerate() {
            if t + len + 200 > duration_ms { break; }
            words.push((format!("w{i}"), t, t + len, *breath));
            t += len + gap;
        }
        let refs: Vec<(&str, u64, u64)> = words.iter().map(|(s, a, b, _)| (s.as_str(), *a, *b)).collect();
        let mut idx = index(&refs, duration_ms, fps);
        for (p, (_, _, _, breath)) in idx.audio.pauses.iter_mut().zip(words.iter()) {
            if *breath { p.kind = PauseKind::Breath; }
        }
        if !has_words { idx.transcript.clear(); }
        if !has_audio { idx.source.audio = None; }
        let v = idx.source.video.as_mut().unwrap();
        v.width = w;
        v.height = h;
        let mut shots: Vec<u64> = shot_ms.into_iter().filter(|&s| s < duration_ms).collect();
        shots.sort_unstable();
        shots.dedup();
        idx.visual.shots = shots.into_iter().map(|t_ms| {
            let frame = fps.nearest_frame(t_ms as i64 * 1_000) as u64;
            ShotBoundary { t_ms, frame, score: 3.0 }
        }).collect();
        let mut peaks: Vec<u64> = peak_ms.into_iter().filter(|&p| p < duration_ms).collect();
        peaks.sort_unstable();
        idx.audio.peaks = peaks.into_iter().map(|t_ms| ExcitementPeak { t_ms, score: 2.0 }).collect();
        idx.visual.subjects = subj.iter().enumerate().map(|(i, &(x, y))| SubjectBox {
            t_ms: i as u64 * duration_ms / 20, x, y, w: 0.1, h: 0.2,
            confidence: 0.8, source: SubjectSource::Saliency,
        }).collect();
        idx
    }
}

fn arb_scenario() -> impl Strategy<Value = Scenario> {
    arb_index().prop_flat_map(|idx| {
        let dur = idx.source.duration_ms;
        let has_words = !idx.transcript.is_empty();
        let clips = prop::collection::vec((0..dur - 400, 300u64..12_000, 0.0f32..=1.0), 1..8);
        let transitions = prop::collection::vec(
            (
                0usize..8,
                prop::sample::select(vec![
                    TransitionKind::Hard,
                    TransitionKind::AutoJCut,
                    TransitionKind::AutoLCut,
                ]),
                prop::option::of(150u32..=300),
            ),
            0..8,
        );
        let motion = prop::collection::vec(
            (
                0usize..8,
                prop::sample::select(vec![
                    MotionPreset::None,
                    MotionPreset::SpringPunchIn,
                    MotionPreset::SmoothTrackSubject,
                    MotionPreset::SlowCreep,
                ]),
            ),
            0..8,
        );
        let aspect = prop::sample::select(vec![
            AspectPreset::Vertical9x16,
            AspectPreset::Square1x1,
            AspectPreset::Source,
        ]);
        (
            Just(idx),
            clips,
            transitions,
            motion,
            aspect,
            any::<bool>(),
            Just(has_words),
        )
            .prop_map(
                |(idx, clips, transitions, motion, aspect, caps, has_words)| {
                    let dur = idx.source.duration_ms;
                    let clips: Vec<EdlClip> = clips
                        .into_iter()
                        .map(|(s, len, hook)| EdlClip {
                            start_ms: s,
                            end_ms: (s + len).min(dur),
                            hook_score: hook,
                            editorial_note: String::new(),
                        })
                        .filter(|c| c.end_ms - c.start_ms >= 300)
                        .collect();
                    let n = clips.len();
                    let mut seen = std::collections::HashSet::new();
                    let audio_transitions = transitions
                        .into_iter()
                        .filter(|(k, _, _)| k + 1 < n && seen.insert(*k))
                        .map(|(after_clip, kind, offset_ms)| AudioTransition {
                            after_clip,
                            kind,
                            offset_ms: if kind == TransitionKind::Hard {
                                None
                            } else {
                                offset_ms
                            },
                        })
                        .collect();
                    let mut seen = std::collections::HashSet::new();
                    let motion_cues = motion
                        .into_iter()
                        .filter(|(c, _)| *c < n && seen.insert(*c))
                        .map(|(clip, preset)| MotionCue {
                            clip,
                            preset,
                            intensity: Intensity::Medium,
                        })
                        .collect();
                    Scenario {
                        idx,
                        edl: SemanticEdl {
                            clips,
                            motion_cues,
                            audio_transitions,
                            captions: (caps && has_words).then(CaptionSpec::default),
                            output: OutputSpec {
                                aspect,
                                height: 1920,
                            },
                        },
                    }
                },
            )
    })
}

proptest! {
    // Case count honours PROPTEST_CASES (default 256).
    /// Any schema-valid EDL resolves, and the result satisfies every invariant.
    #[test]
    fn solver_invariants(sc in arb_scenario()) {
        prop_assume!(!sc.edl.clips.is_empty());
        let res = EdlSolver::default().solve(&sc.edl, &sc.idx);
        let res = match res {
            Ok(r) => r,
            Err(e) => return Err(TestCaseError::fail(format!("valid EDL rejected: {e}"))),
        };
        let t = &res.timeline;

        // verify() covers: frame alignment, monotonic output, A/V lock,
        // crossfade bounds, crop bounds/aspect, caption monotonicity.
        let v = t.verify();
        prop_assert!(v.is_empty(), "{v:?}");

        let fps = t.source.fps;
        for (s, c) in t.segments.iter().zip(&sc.edl.clips) {
            // Zero negative-duration clips.
            prop_assert!(s.video.frames() > 0);
            prop_assert!(s.audio.len_us() > 0);
            // Cuts move at most snap tolerance + scene guard + one frame,
            // unless the collapse fallback restored the request.
            let slack = (120 + 150) * 1_000 + fps.frame_duration_us();
            prop_assert!((s.video.start_us - c.start_ms as i64 * 1_000).abs() <= slack);
            prop_assert!((s.video.end_us - (c.end_ms as i64 * 1_000)).abs() <= slack);
        }
        // Monotonic output timestamps.
        for w in t.segments.windows(2) {
            prop_assert!(w[1].output_start_us > w[0].output_start_us);
        }
        // Valid crossfade overlaps: 0..=25ms, never longer than either side.
        for (k, sp) in t.splices.iter().enumerate() {
            prop_assert!((0..=25_000).contains(&sp.crossfade_us));
            if t.source.has_audio {
                prop_assert!(sp.crossfade_us <= t.segments[k].audio.len_us());
                prop_assert!(sp.crossfade_us <= t.segments[k + 1].audio.len_us());
                // J/L offsets within the accepted band.
                let o = sp.av_offset_us.abs();
                prop_assert!(o == 0 || (150_000..=300_000).contains(&o));
            }
        }
        // Bounded crops (sampled every frame).
        let (sw, sh) = (f64::from(t.source.width), f64::from(t.source.height));
        for s in &t.segments {
            for k in 0..s.video.frames() {
                let c = s.camera.sample(fps.frame_to_us(k)).unwrap();
                prop_assert!(c.x >= -1e-6 && c.y >= -1e-6);
                prop_assert!(c.x + c.w <= sw + 1e-6 && c.y + c.h <= sh + 1e-6);
            }
        }
    }

    /// Re-solving the same input is bit-identical.
    #[test]
    fn solver_is_deterministic(sc in arb_scenario()) {
        prop_assume!(!sc.edl.clips.is_empty());
        let a = EdlSolver::default().solve(&sc.edl, &sc.idx);
        let b = EdlSolver::default().solve(&sc.edl, &sc.idx);
        if let (Ok(a), Ok(b)) = (a, b) {
            prop_assert_eq!(a.timeline, b.timeline);
            prop_assert_eq!(a.report, b.report);
        }
    }
}
