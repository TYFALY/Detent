//! The frame-accurate output of the solver, consumed by the renderer.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::camera::CameraPath;
use crate::captions::CaptionBlock;
use crate::edl::{
    AspectPreset, AudioMix, CaptionPosition, CaptionStyle, ColorGrade, InsertPlacement, MusicSpec,
    WordColor,
};
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
    /// The next segment opens with a short white flash.
    #[serde(default)]
    pub flash: bool,
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
    /// Hook headline pinned to the top for the whole timeline.
    #[serde(default)]
    pub headline: Option<String>,
    /// When the headline leaves the screen (µs); `None` pins it throughout.
    #[serde(default)]
    pub headline_end_us: Option<i64>,
    /// Words rendered with emphasis.
    #[serde(default)]
    pub highlight_words: Vec<String>,
    /// Per-word accent colours.
    #[serde(default)]
    pub word_colors: Vec<WordColor>,
    /// Section labels on the output timeline, in order, non-overlapping.
    #[serde(default)]
    pub chapters: Vec<ResolvedChapter>,
}

/// A section label on the output timeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedChapter {
    /// Label text.
    pub text: String,
    /// Output start (µs).
    pub start_us: i64,
    /// Output end (µs, exclusive).
    pub end_us: i64,
}

/// An image/video card or B-roll cutaway on the output timeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedInsert {
    /// Image or video path as written in the EDL.
    pub path: String,
    /// Output start (µs).
    pub start_us: i64,
    /// Output end (µs, exclusive).
    pub end_us: i64,
    /// Placement preset.
    pub placement: InsertPlacement,
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
    /// Composition of the picture inside the raster.
    #[serde(default)]
    pub layout: Layout,
    /// Colour finish.
    #[serde(default)]
    pub grade: ColorGrade,
    /// Audio finish.
    #[serde(default)]
    pub audio_mix: AudioMix,
    /// First segment scales in 100% → 108% over 0.4 s.
    #[serde(default)]
    pub opening_pop: bool,
}

/// Resolved composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
pub enum Layout {
    /// Camera crops fill the whole raster.
    #[default]
    Crop,
    /// Camera crops (source aspect) are scaled to `fg_width x fg_height` and
    /// centred over a blurred fill of the raster.
    FitBlur {
        /// Foreground width (even).
        fg_width: u32,
        /// Foreground height (even).
        fg_height: u32,
    },
}

impl OutputFormat {
    /// Size the camera crop is scaled to (the whole raster unless `FitBlur`).
    pub fn picture_size(&self) -> (u32, u32) {
        match self.layout {
            Layout::Crop => (self.width, self.height),
            Layout::FitBlur {
                fg_width,
                fg_height,
            } => (fg_width, fg_height),
        }
    }
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
    /// Image/video cards and B-roll cutaways over the picture.
    #[serde(default)]
    pub inserts: Vec<ResolvedInsert>,
    /// Sound design, if requested.
    #[serde(default)]
    pub sound: Option<ResolvedSound>,
    /// Total output duration (µs).
    pub duration_us: i64,
}

/// One synthesised sound effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum SfxKind {
    /// Airy swoosh.
    Whoosh,
    /// Percussive click.
    Click,
    /// Bubbly pop.
    Pop,
}

impl SfxKind {
    /// How long before its anchor the effect starts (µs): a whoosh swells
    /// into the cut, a click or pop lands on it.
    pub fn lead_us(self) -> i64 {
        match self {
            Self::Whoosh => 250_000,
            Self::Click | Self::Pop => 0,
        }
    }
}

/// A sound effect on the output timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SfxEvent {
    /// Effect.
    pub kind: SfxKind,
    /// The moment it is anchored to (a cut or a card appearing), µs.
    pub at_us: i64,
}

/// Resolved sound design.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedSound {
    /// Effects, sorted by time.
    pub sfx: Vec<SfxEvent>,
    /// Effect level relative to the voice (dB).
    pub sfx_db: f32,
    /// Background music bed.
    pub music: Option<MusicSpec>,
}

impl ResolvedTimeline {
    /// True when any segment needs a moving or non-identity crop.
    pub fn needs_reframe(&self) -> bool {
        if matches!(self.output.layout, Layout::FitBlur { .. }) {
            return true;
        }
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
        let (pw, ph) = self.output.picture_size();
        let aspect = f64::from(pw) / f64::from(ph.max(1));
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
            for (k, ch) in c.chapters.iter().enumerate() {
                if ch.end_us <= ch.start_us || ch.start_us < 0 || ch.end_us > self.duration_us {
                    v.push(format!("chapter {k}: invalid display range"));
                }
            }
            if c.chapters.windows(2).any(|w| w[1].start_us < w[0].end_us) {
                v.push("chapters overlap".to_owned());
            }
        }
        for (k, ins) in self.inserts.iter().enumerate() {
            if ins.end_us <= ins.start_us || ins.start_us < 0 || ins.end_us > self.duration_us {
                v.push(format!("insert {k}: invalid display range"));
            }
        }
        if let Some(s) = &self.sound {
            if s.sfx
                .iter()
                .any(|e| e.at_us < 0 || e.at_us > self.duration_us)
            {
                v.push("sound effect outside the timeline".to_owned());
            }
            if s.sfx.windows(2).any(|w| w[1].at_us < w[0].at_us) {
                v.push("sound effects not sorted".to_owned());
            }
            if !self.source.has_audio {
                v.push("sound design without source audio".to_owned());
            }
        }
        v
    }
}
