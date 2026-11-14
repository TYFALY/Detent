//! The frame-accurate output of the solver, consumed by the renderer.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::camera::CameraPath;
use crate::captions::CaptionBlock;
use crate::edl::{AspectPreset, CaptionPosition, CaptionStyle};
use crate::time::{FrameRate, UsRange};

/// Current on-disk schema version of [`ResolvedTimeline`].
pub const TIMELINE_VERSION: u32 = 1;

/// Maximum micro-crossfade length the solver may emit (µs).
pub const MAX_CROSSFADE_US: i64 = 25_000;

/// A half-open range of source frames `[start_frame, end_frame)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FrameRange {
    /// First frame (inclusive).
    pub start_frame: i64,
    /// End frame (exclusive).
    pub end_frame: i64,
    /// `start_frame` in µs.
    pub start_us: i64,
    /// `end_frame` in µs.
    pub end_us: i64,
}

impl FrameRange {
    /// Builds a range, deriving exact µs bounds from the frame rate.
    pub fn new(start_frame: i64, end_frame: i64, fps: FrameRate) -> Self {
        Self {
            start_frame,
            end_frame,
            start_us: fps.frame_to_us(start_frame),
            end_us: fps.frame_to_us(end_frame),
        }
    }

    /// Frame count.
    pub fn frames(&self) -> i64 {
        self.end_frame - self.start_frame
    }

    /// Duration in µs.
    pub fn len_us(&self) -> i64 {
        self.end_us - self.start_us
    }
}

/// One clip on the output timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedSegment {
    /// Index of the originating EDL clip.
    pub clip_index: usize,
    /// Source video frames.
    pub video: FrameRange,
    /// Source audio range including crossfade handles (µs).
    pub audio: UsRange,
    /// Where this segment's picture starts on the output timeline (µs).
    pub output_start_us: i64,
    /// Solved virtual camera.
    pub camera: CameraPath,
}

/// Audio crossfade at an internal splice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Splice {
    /// Index of the segment before the splice.
    pub after_segment: usize,
    /// Equal-power crossfade length (µs).
    pub crossfade_us: i64,
    /// Audio offset relative to the picture cut (µs): `> 0` J-cut (audio
    /// leads), `< 0` L-cut (audio lags), `0` straight cut.
    pub av_offset_us: i64,
}

/// Burned-in captions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedCaptions {
    /// Visual preset.
    pub style: CaptionStyle,
    /// Placement.
    pub position: CaptionPosition,
    /// Blocks on the output timeline.
    pub blocks: Vec<CaptionBlock>,
}

/// Output raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct OutputFormat {
    /// Aspect preset requested.
    pub aspect: AspectPreset,
    /// Width (even).
    pub width: u32,
    /// Height (even).
    pub height: u32,
    /// Frame rate (same as source).
    pub fps: FrameRate,
}

/// Source facts the renderer needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TimelineSource {
    /// Path of the source media.
    pub path: String,
    /// Source width.
    pub width: u32,
    /// Source height.
    pub height: u32,
    /// Source frame rate.
    pub fps: FrameRate,
    /// Source duration (µs).
    pub duration_us: i64,
    /// Whether the source has an audio stream.
    pub has_audio: bool,
}

/// A fully resolved, frame-accurate edit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedTimeline {
    /// Schema version.
    pub version: u32,
    /// Source facts.
    pub source: TimelineSource,
    /// Output raster.
    pub output: OutputFormat,
    /// Segments in playback order.
    pub segments: Vec<ResolvedSegment>,
    /// One entry per internal splice (`segments.len() - 1`).
    pub splices: Vec<Splice>,
    /// Captions, if requested.
    pub captions: Option<ResolvedCaptions>,
    /// Total output duration (µs).
    pub duration_us: i64,
}

impl ResolvedTimeline {
    /// True when any segment needs a moving or non-identity crop.
    pub fn needs_reframe(&self) -> bool {
        let (sw, sh) = (f64::from(self.source.width), f64::from(self.source.height));
        let same_raster =
            self.output.width == self.source.width && self.output.height == self.source.height;
        self.segments.iter().any(|s| {
            !s.camera.is_static()
                || s.camera.keyframes.first().is_some_and(|k| {
                    k.x.abs() > 0.5
                        || k.y.abs() > 0.5
                        || (k.w - sw).abs() > 0.5
                        || (k.h - sh).abs() > 0.5
                })
        }) || !same_raster
    }

    /// Parses a timeline and checks every invariant.
    pub fn from_json(s: &str) -> crate::Result<Self> {
        let t: Self = serde_json::from_str(s).map_err(|source| crate::CoreError::Json {
            what: "ResolvedTimeline",
            source,
        })?;
        let violations = t.verify();
        if violations.is_empty() {
            Ok(t)
        } else {
            Err(crate::CoreError::Validation(
                violations
                    .into_iter()
                    .map(|m| crate::ValidationIssue::new("/", m))
                    .collect(),
            ))
        }
    }

    /// Checks all solver invariants. Returns human-readable violations
    /// (empty = valid). Used by the solver itself, the renderer (defence in
    /// depth on untrusted files) and the property tests.
    pub fn verify(&self) -> Vec<String> {
        let mut v = Vec::new();
        let fps = self.source.fps;
        let dur = self.source.duration_us;
        if self.segments.is_empty() {
            v.push("timeline has no segments".to_owned());
        }
        if self.splices.len() != self.segments.len().saturating_sub(1) {
            v.push(format!(
                "expected {} splices, found {}",
                self.segments.len().saturating_sub(1),
                self.splices.len()
            ));
        }
        if !self.output.width.is_multiple_of(2)
            || !self.output.height.is_multiple_of(2)
            || self.output.width == 0
        {
            v.push("output raster must be non-zero and even".to_owned());
        }
        let mut t_out = 0_i64;
        let mut video_total = 0_i64;
        let mut audio_total = 0_i64;
        let (sw, sh) = (f64::from(self.source.width), f64::from(self.source.height));
        let aspect = f64::from(self.output.width) / f64::from(self.output.height.max(1));
        for (i, s) in self.segments.iter().enumerate() {
            let fr = s.video;
            if fr.frames() <= 0 {
                v.push(format!("segment {i}: non-positive frame count"));
            }
            if fr.start_us != fps.frame_to_us(fr.start_frame)
                || fr.end_us != fps.frame_to_us(fr.end_frame)
            {
                v.push(format!("segment {i}: video µs bounds not frame-aligned"));
            }
            if fr.start_us < 0 || fr.end_us > dur {
                v.push(format!("segment {i}: video outside source"));
            }
            if s.output_start_us != t_out {
                v.push(format!(
                    "segment {i}: output start {} != expected {t_out} (non-monotonic)",
                    s.output_start_us
                ));
            }
            t_out += fr.len_us();
            video_total += fr.len_us();
            if self.source.has_audio {
                if s.audio.len_us() <= 0 {
                    v.push(format!("segment {i}: non-positive audio duration"));
                }
                if s.audio.start_us < 0 || s.audio.end_us > dur {
                    v.push(format!("segment {i}: audio outside source"));
                }
                audio_total += s.audio.len_us();
            }
            let eps = 1e-6 * sw.max(sh);
            for (k, kf) in s.camera.keyframes.iter().enumerate() {
                if kf.w <= 0.0 || kf.h <= 0.0 {
                    v.push(format!("segment {i} keyframe {k}: empty crop"));
                }
                if kf.x < -eps || kf.y < -eps || kf.x + kf.w > sw + eps || kf.y + kf.h > sh + eps {
                    v.push(format!("segment {i} keyframe {k}: crop outside source"));
                }
                if ((kf.w / kf.h) - aspect).abs() > 0.01 * aspect {
                    v.push(format!(
                        "segment {i} keyframe {k}: crop aspect != output aspect"
                    ));
                }
                if kf.t_us < 0 || kf.t_us > fr.len_us() {
                    v.push(format!("segment {i} keyframe {k}: time outside segment"));
                }
            }
            if s.camera.keyframes.is_empty() {
                v.push(format!("segment {i}: camera path has no keyframes"));
            }
            if s.camera
                .keyframes
                .windows(2)
                .any(|w| w[1].t_us <= w[0].t_us)
            {
                v.push(format!("segment {i}: keyframes not strictly increasing"));
            }
        }
        if t_out != self.duration_us {
            v.push(format!(
                "duration {} != sum of segments {t_out}",
                self.duration_us
            ));
        }
        let mut xfade_total = 0_i64;
        for (k, sp) in self.splices.iter().enumerate() {
            if sp.after_segment != k {
                v.push(format!("splice {k}: after_segment mismatch"));
            }
            if !(0..=MAX_CROSSFADE_US).contains(&sp.crossfade_us) {
                v.push(format!(
                    "splice {k}: crossfade {}µs out of range",
                    sp.crossfade_us
                ));
            }
            if self.source.has_audio {
                let a = self.segments.get(k).map_or(0, |s| s.audio.len_us());
                let b = self.segments.get(k + 1).map_or(0, |s| s.audio.len_us());
                if sp.crossfade_us > a.min(b) {
                    v.push(format!("splice {k}: crossfade longer than adjacent audio"));
                }
            }
            xfade_total += sp.crossfade_us;
        }
        if self.source.has_audio && audio_total - xfade_total != video_total {
            v.push(format!(
                "A/V drift: audio {audio_total}µs - crossfades {xfade_total}µs != video {video_total}µs"
            ));
        }
        if let Some(c) = &self.captions {
            for (k, b) in c.blocks.iter().enumerate() {
                if b.end_us <= b.start_us || b.start_us < 0 || b.end_us > self.duration_us {
                    v.push(format!("caption {k}: invalid display range"));
                }
            }
            if c.blocks.windows(2).any(|w| w[1].start_us < w[0].end_us) {
                v.push("caption blocks overlap".to_owned());
            }
        }
        v
    }
}
