//! Deterministic synthetic data for benchmarks, tests and `detent bench`.
//!
//! Nothing here touches the filesystem or a clock, so results are
//! reproducible across machines.

use crate::edl::{
    AudioTransition, CaptionSpec, EdlClip, Intensity, MotionCue, MotionPreset, OutputSpec,
    SemanticEdl, TightenSpec, TransitionKind, VideoTransition, VideoTransitionKind,
};
use crate::index::{
    AudioStreamInfo, AudioTelemetry, ExcitementPeak, INDEX_VERSION, MediaIndex, PauseKind,
    ShotBoundary, SilenceInterval, SourceInfo, SubjectBox, SubjectSource, VideoStreamInfo,
    VisualTelemetry, WordToken,
};
use crate::time::FrameRate;

/// Xorshift32 PRNG (no dependency, stable across platforms).
#[derive(Debug, Clone)]
pub struct Rng(u32);

impl Rng {
    /// Seeded generator (seed 0 is remapped).
    pub fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }

    /// Next `u32`.
    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + u64::from(self.next_u32()) % (hi - lo).max(1)
    }

    /// Uniform in `[-1, 1]`.
    pub fn signed(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
}

/// Speech-like mono PCM: harmonic "syllables" with pitch and loudness
/// movement, breaths and room-tone pauses.
pub fn speech_like_pcm(seconds: f32, sample_rate: u32) -> Vec<f32> {
    let n = (seconds * sample_rate as f32) as usize;
    let mut rng = Rng::new(0x00D3_7E17);
    let mut out = Vec::with_capacity(n);
    let sr = sample_rate as f32;
    while out.len() < n {
        // Phrase: 1-4 s of voiced audio, then a 0.15-0.7 s pause (sometimes a breath).
        let phrase = (rng.range(1_000, 4_000) as f32 / 1_000.0 * sr) as usize;
        let f0 = 110.0 + rng.range(0, 120) as f32;
        let start = out.len();
        for i in 0..phrase.min(n - out.len()) {
            let t = (start + i) as f32 / sr;
            let syllable = (0.5 + 0.5 * (std::f32::consts::TAU * 4.0 * t).sin()).powi(2);
            let mut v = 0.0;
            for h in 1..=6 {
                v += (std::f32::consts::TAU * f0 * h as f32 * t).sin() / h as f32;
            }
            out.push(0.18 * syllable * v + 0.001 * rng.signed());
        }
        let pause = (rng.range(150, 700) as f32 / 1_000.0 * sr) as usize;
        let breath = rng.range(0, 3) == 0;
        for _ in 0..pause.min(n - out.len()) {
            out.push(if breath { 0.01 } else { 0.0004 } * rng.signed());
        }
    }
    out
}

/// A realistic index: `duration_ms` of talking-head telemetry with words,
/// pauses, shots every ~8 s, peaks and subject samples.
pub fn synthetic_index(duration_ms: u64, seed: u32) -> MediaIndex {
    let mut rng = Rng::new(seed);
    let mut transcript = Vec::new();
    let mut pauses = Vec::new();
    let mut t = 300;
    let vocab = [
        "the", "engine", "cuts", "on", "breath", "and", "motion", "not", "on", "text,", "which",
        "is", "why", "it", "feels", "human.", "every", "frame", "lands", "exactly.",
    ];
    let mut w = 0;
    while t + 1_000 < duration_ms {
        let phrase_words = rng.range(3, 12);
        for _ in 0..phrase_words {
            let len = rng.range(140, 520);
            transcript.push(WordToken {
                word: vocab[w % vocab.len()].to_owned(),
                start_ms: t,
                end_ms: t + len,
                confidence: 0.9,
            });
            w += 1;
            t += len + rng.range(20, 70);
        }
        let gap = rng.range(180, 900);
        pauses.push(SilenceInterval {
            start_ms: t,
            end_ms: t + gap,
            min_dbfs: -58.0 - rng.range(0, 10) as f32,
            valley_ms: t + gap / 2,
            kind: if rng.range(0, 4) == 0 {
                PauseKind::Breath
            } else {
                PauseKind::Silence
            },
        });
        t += gap;
    }
    let fps = FrameRate {
        num: 30_000,
        den: 1_001,
    };
    let shots = (1..duration_ms / 8_000)
        .map(|k| {
            let t_ms = k * 8_000 + rng.range(0, 2_000);
            ShotBoundary {
                t_ms,
                frame: fps.nearest_frame(t_ms as i64 * 1_000) as u64,
                score: 3.0,
            }
        })
        .collect();
    let peaks = (0..duration_ms / 2_500)
        .map(|k| ExcitementPeak {
            t_ms: k * 2_500 + rng.range(0, 2_000),
            score: 1.5 + rng.range(0, 30) as f32 / 10.0,
        })
        .collect();
    let subjects = (0..duration_ms / 200)
        .map(|k| {
            let cx = 0.5 + 0.2 * ((k as f32) / 40.0).sin();
            SubjectBox {
                t_ms: k * 200,
                x: cx - 0.1,
                y: 0.2,
                w: 0.2,
                h: 0.4,
                confidence: 0.8,
                source: SubjectSource::Saliency,
            }
        })
        .collect();
    let hops = (duration_ms / 10) as usize;
    MediaIndex {
        version: INDEX_VERSION,
        source: SourceInfo {
            path: "synthetic.mp4".into(),
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
            rms_dbfs: vec![-30.0; hops],
            zcr: vec![0.1; hops],
            flux: vec![0.01; hops],
            noise_floor_dbfs: -62.0,
            speech_level_dbfs: -24.0,
            pauses,
            peaks,
        },
        visual: VisualTelemetry {
            shots,
            subjects,
            contact_sheet: None,
        },
    }
}

/// An EDL of `clips` clips (each ~4-9 s) cycling through every motion preset
/// and transition kind, with captions.
pub fn synthetic_edl(index: &MediaIndex, clips: usize, seed: u32) -> SemanticEdl {
    let mut rng = Rng::new(seed);
    let dur = index.source.duration_ms;
    let presets = [
        MotionPreset::SpringPunchIn,
        MotionPreset::SmoothTrackSubject,
        MotionPreset::SlowCreep,
        MotionPreset::None,
    ];
    let kinds = [
        TransitionKind::AutoJCut,
        TransitionKind::AutoLCut,
        TransitionKind::Hard,
    ];
    let clips_v: Vec<EdlClip> = (0..clips)
        .map(|i| {
            let start = rng.range(0, dur.saturating_sub(10_000).max(1));
            EdlClip {
                start_ms: start,
                end_ms: (start + rng.range(4_000, 9_000)).min(dur),
                hook_score: if i == 0 { 0.9 } else { 0.5 },
                editorial_note: format!("clip {i}"),
            }
        })
        .collect();
    SemanticEdl {
        motion_cues: (0..clips)
            .map(|i| MotionCue {
                clip: i,
                preset: presets[i % presets.len()],
                intensity: Intensity::Medium,
            })
            .collect(),
        audio_transitions: (0..clips.saturating_sub(1))
            .map(|i| {
                let kind = kinds[i % kinds.len()];
                AudioTransition {
                    after_clip: i,
                    kind,
                    offset_ms: (kind != TransitionKind::Hard).then_some(240),
                }
            })
            .collect(),
        captions: Some(CaptionSpec::default()),
        headline: None,
        highlight_words: Vec::new(),
        headline_duration_ms: None,
        video_transitions: (0..clips.saturating_sub(1))
            .step_by(2)
            .map(|i| VideoTransition {
                after_clip: i,
                kind: VideoTransitionKind::Flash,
            })
            .collect(),
        tighten: clips.is_multiple_of(2).then(TightenSpec::default),
        output: OutputSpec::default(),
        clips: clips_v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EdlSolver;

    #[test]
    fn synthetic_fixtures_solve() {
        let idx = synthetic_index(600_000, 7);
        idx.validate().unwrap();
        let edl = synthetic_edl(&idx, 12, 3);
        let res = EdlSolver::default().solve(&edl, &idx).unwrap();
        // Even clip counts enable `tighten`, so clips may be split at pauses.
        let segs = &res.timeline.segments;
        assert!(segs.len() >= 12);
        assert!(segs.windows(2).all(|w| w[0].clip_index <= w[1].clip_index));
        assert_eq!(segs.last().unwrap().clip_index, 11);
        assert!(res.timeline.captions.unwrap().blocks.len() > 20);
    }

    #[test]
    fn pcm_is_deterministic() {
        assert_eq!(speech_like_pcm(1.0, 16_000), speech_like_pcm(1.0, 16_000));
    }
}
