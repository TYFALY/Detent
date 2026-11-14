//! The Semantic EDL: the *only* thing an LLM is allowed to emit.
//!
//! It expresses editorial intent (which ranges, in which order, with which
//! camera feel and audio transition) and nothing else. There are no pixel
//! coordinates, no FFmpeg expressions and no frame numbers: the deterministic
//! solver owns all of that. Unknown fields are rejected so schema drift or
//! hallucinated keys fail loudly at the boundary.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result, ValidationIssue};
use crate::index::MediaIndex;

/// Accepted J/L-cut offset range (ms).
pub const JL_OFFSET_RANGE_MS: std::ops::RangeInclusive<u32> = 150..=300;
/// Default J/L-cut offset (ms).
pub const JL_DEFAULT_OFFSET_MS: u32 = 220;
/// Shortest clip the solver will accept (ms).
pub const MIN_CLIP_MS: u64 = 300;

/// An edit decision list expressed as editorial intent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SemanticEdl {
    /// Source ranges in playback order. Ranges may be reordered relative to
    /// the source (e.g. a hook first) and may not overlap each other's output.
    pub clips: Vec<EdlClip>,
    /// Camera intent per clip. Clips without a cue get `None` (static framing).
    #[serde(default)]
    pub motion_cues: Vec<MotionCue>,
    /// Audio transition per splice. Splices without an entry are `Hard`.
    #[serde(default)]
    pub audio_transitions: Vec<AudioTransition>,
    /// Burned-in caption settings. Omit for no captions.
    #[serde(default)]
    pub captions: Option<CaptionSpec>,
    /// Output framing.
    #[serde(default)]
    pub output: OutputSpec,
}

/// One source range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdlClip {
    /// Approximate in-point in source milliseconds. The solver snaps it into
    /// the nearest breath/silence valley within the snap tolerance (±120 ms).
    pub start_ms: u64,
    /// Approximate out-point in source milliseconds (exclusive).
    pub end_ms: u64,
    /// How strongly this clip works as an opening hook, 0.0-1.0.
    #[schemars(range(min = 0.0, max = 1.0))]
    pub hook_score: f32,
    /// One short sentence explaining the editorial choice.
    #[schemars(length(max = 280))]
    pub editorial_note: String,
}

/// Camera motion preset. Purely semantic; geometry is solved deterministically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum MotionPreset {
    /// Static framing centred on the subject at clip start.
    #[default]
    None,
    /// Snappy, overshoot-free zoom-in on vocal emphasis peaks, easing back out.
    SpringPunchIn,
    /// Keep the detected subject centred with a soft, critically damped follow.
    SmoothTrackSubject,
    /// Slow continuous push-in across the clip.
    SlowCreep,
}

/// Strength of a motion preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum Intensity {
    /// Barely perceptible.
    Subtle,
    /// Default.
    #[default]
    Medium,
    /// Pronounced.
    Strong,
}

/// Camera intent for one clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MotionCue {
    /// Index into `clips`.
    pub clip: usize,
    /// Preset to apply.
    pub preset: MotionPreset,
    /// Preset strength.
    #[serde(default)]
    pub intensity: Intensity,
}

/// Audio behaviour at a splice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum TransitionKind {
    /// Audio and video cut together (still receives a micro-crossfade).
    #[default]
    Hard,
    /// Next clip's audio starts *before* the picture cut (audio leads).
    AutoJCut,
    /// Previous clip's audio continues *past* the picture cut (audio lags).
    AutoLCut,
}

/// Audio transition at the splice between `after_clip` and `after_clip + 1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AudioTransition {
    /// Index of the clip *before* the splice.
    pub after_clip: usize,
    /// Transition kind.
    pub kind: TransitionKind,
    /// Target J/L offset in ms (150-300). The solver may shorten it (never
    /// below 150) or downgrade to `Hard` if surrounding audio is not safe.
    #[serde(default)]
    #[schemars(range(min = 150, max = 300))]
    pub offset_ms: Option<u32>,
}

/// Caption visual preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum CaptionStyle {
    /// Heavy white text, thick black outline, active line pops in.
    #[default]
    BoldPop,
    /// Small clean sans-serif with soft shadow.
    Minimal,
    /// Word-by-word highlight synced to speech.
    Karaoke,
}

/// Vertical caption placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum CaptionPosition {
    /// Lower third (safe area above platform UI).
    #[default]
    Lower,
    /// Centre of frame.
    Center,
    /// Upper third.
    Upper,
}

/// Syntactic grouping rules for caption blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CaptionGrouping {
    /// Minimum words per block (soft: pauses and sentence ends may force fewer).
    #[schemars(range(min = 1, max = 8))]
    pub min_words: u8,
    /// Maximum words per block.
    #[schemars(range(min = 1, max = 8))]
    pub max_words: u8,
    /// Maximum characters per block.
    #[schemars(range(min = 8, max = 64))]
    pub max_chars: u8,
    /// A silence at least this long (ms) always starts a new block.
    #[schemars(range(min = 100, max = 2000))]
    pub pause_break_ms: u32,
}

impl Default for CaptionGrouping {
    fn default() -> Self {
        Self {
            min_words: 3,
            max_words: 5,
            max_chars: 32,
            pause_break_ms: 300,
        }
    }
}

/// Caption settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct CaptionSpec {
    /// Visual preset.
    #[serde(default)]
    pub style: CaptionStyle,
    /// Placement.
    #[serde(default)]
    pub position: CaptionPosition,
    /// Grouping rules.
    #[serde(default)]
    pub grouping: CaptionGrouping,
}

/// Output aspect ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum AspectPreset {
    /// 9:16 vertical (Shorts / Reels / TikTok).
    #[default]
    Vertical9x16,
    /// 1:1 square.
    Square1x1,
    /// Keep the source aspect ratio (no reframing unless motion is requested).
    Source,
}

impl AspectPreset {
    /// Width / height ratio, or `None` for `Source`.
    pub fn ratio(self) -> Option<f64> {
        match self {
            Self::Vertical9x16 => Some(9.0 / 16.0),
            Self::Square1x1 => Some(1.0),
            Self::Source => None,
        }
    }
}

/// Output framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    /// Aspect ratio preset.
    #[serde(default)]
    pub aspect: AspectPreset,
    /// Output height in pixels (width is derived; both rounded to even).
    #[serde(default = "default_height")]
    #[schemars(range(min = 144, max = 4320))]
    pub height: u32,
    /// How the source fills the output frame.
    #[serde(default)]
    pub layout: OutputLayout,
}

/// How the source picture fills the output raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum OutputLayout {
    /// Crop a window with the output aspect (camera tracks within the source).
    /// Best for talking heads and vlogs.
    #[default]
    Crop,
    /// Show the whole source frame (letterboxed, full width) over a blurred,
    /// zoomed copy of itself. Best for screen recordings, gameplay and
    /// reaction videos with a facecam inset, where cropping would discard the
    /// content. Captions sit in the lower blurred band.
    FitBlur,
}

fn default_height() -> u32 {
    1920
}

impl Default for OutputSpec {
    fn default() -> Self {
        Self {
            aspect: AspectPreset::default(),
            height: default_height(),
            layout: OutputLayout::default(),
        }
    }
}

impl SemanticEdl {
    /// Parses an EDL from JSON. Unknown fields are a hard error.
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|source| CoreError::Json {
            what: "SemanticEdl",
            source,
        })
    }

    /// The JSON Schema handed to the model.
    pub fn json_schema() -> schemars::Schema {
        schemars::schema_for!(SemanticEdl)
    }

    /// Effective motion preset for `clip`.
    pub fn motion_for(&self, clip: usize) -> (MotionPreset, Intensity) {
        self.motion_cues
            .iter()
            .find(|c| c.clip == clip)
            .map_or((MotionPreset::None, Intensity::Medium), |c| {
                (c.preset, c.intensity)
            })
    }

    /// Effective transition at the splice after `clip`.
    pub fn transition_after(&self, clip: usize) -> (TransitionKind, u32) {
        self.audio_transitions
            .iter()
            .find(|t| t.after_clip == clip)
            .map_or((TransitionKind::Hard, 0), |t| {
                (t.kind, t.offset_ms.unwrap_or(JL_DEFAULT_OFFSET_MS))
            })
    }

    /// Semantic validation against a media index. Collects *every* issue.
    pub fn validate(&self, index: &MediaIndex) -> Result<()> {
        let mut issues = Vec::new();
        let dur = index.source.duration_ms;
        let n = self.clips.len();
        if n == 0 {
            issues.push(ValidationIssue::new(
                "/clips",
                "at least one clip is required",
            ));
        }
        for (i, c) in self.clips.iter().enumerate() {
            let p = format!("/clips/{i}");
            if c.end_ms <= c.start_ms {
                issues.push(ValidationIssue::new(
                    format!("{p}/end_ms"),
                    format!(
                        "end_ms ({}) must be greater than start_ms ({})",
                        c.end_ms, c.start_ms
                    ),
                ));
            } else if c.end_ms - c.start_ms < MIN_CLIP_MS {
                issues.push(ValidationIssue::new(
                    p.clone(),
                    format!(
                        "clip is {} ms; minimum is {MIN_CLIP_MS} ms",
                        c.end_ms - c.start_ms
                    ),
                ));
            }
            if c.end_ms > dur {
                issues.push(ValidationIssue::new(
                    format!("{p}/end_ms"),
                    format!("end_ms ({}) exceeds media duration ({dur} ms)", c.end_ms),
                ));
            }
            if !(0.0..=1.0).contains(&c.hook_score) || c.hook_score.is_nan() {
                issues.push(ValidationIssue::new(
                    format!("{p}/hook_score"),
                    "hook_score must be within [0.0, 1.0]",
                ));
            }
            if c.editorial_note.chars().count() > 280 {
                issues.push(ValidationIssue::new(
                    format!("{p}/editorial_note"),
                    "editorial_note must be <= 280 characters",
                ));
            }
        }
        let mut seen = std::collections::HashSet::new();
        for (i, m) in self.motion_cues.iter().enumerate() {
            if m.clip >= n {
                issues.push(ValidationIssue::new(
                    format!("/motion_cues/{i}/clip"),
                    format!("clip index {} out of range (0..{n})", m.clip),
                ));
            }
            if !seen.insert(m.clip) {
                issues.push(ValidationIssue::new(
                    format!("/motion_cues/{i}"),
                    format!("duplicate motion cue for clip {}", m.clip),
                ));
            }
            if m.preset != MotionPreset::None && index.source.video.is_none() {
                issues.push(ValidationIssue::new(
                    format!("/motion_cues/{i}/preset"),
                    "motion requires a video stream",
                ));
            }
        }
        let mut seen = std::collections::HashSet::new();
        for (i, t) in self.audio_transitions.iter().enumerate() {
            let p = format!("/audio_transitions/{i}");
            if t.after_clip + 1 >= n {
                issues.push(ValidationIssue::new(
                    format!("{p}/after_clip"),
                    format!(
                        "after_clip must reference a splice (0..{})",
                        n.saturating_sub(1)
                    ),
                ));
            }
            if !seen.insert(t.after_clip) {
                issues.push(ValidationIssue::new(
                    p.clone(),
                    format!("duplicate transition after clip {}", t.after_clip),
                ));
            }
            if let Some(o) = t.offset_ms {
                if t.kind == TransitionKind::Hard {
                    issues.push(ValidationIssue::new(
                        format!("{p}/offset_ms"),
                        "offset_ms is only meaningful for AutoJCut / AutoLCut",
                    ));
                } else if !JL_OFFSET_RANGE_MS.contains(&o) {
                    issues.push(ValidationIssue::new(
                        format!("{p}/offset_ms"),
                        format!("offset_ms must be within 150..=300 (got {o})"),
                    ));
                }
            }
        }
        if let Some(c) = &self.captions {
            let g = c.grouping;
            if g.min_words == 0 || g.max_words == 0 || g.min_words > g.max_words || g.max_words > 8
            {
                issues.push(ValidationIssue::new(
                    "/captions/grouping",
                    "require 1 <= min_words <= max_words <= 8",
                ));
            }
            if g.max_chars < 8 {
                issues.push(ValidationIssue::new(
                    "/captions/grouping/max_chars",
                    "max_chars must be >= 8",
                ));
            }
            if index.transcript.is_empty() {
                issues.push(ValidationIssue::new(
                    "/captions",
                    "captions requested but the index has no transcript",
                ));
            }
        }
        if !(144..=4320).contains(&self.output.height) {
            issues.push(ValidationIssue::new(
                "/output/height",
                "height must be within 144..=4320",
            ));
        }
        if index.source.video.is_none() {
            issues.push(ValidationIssue::new(
                "/clips",
                "source has no video stream; detent edits audio-visual media",
            ));
        }
        if issues.is_empty() {
            Ok(())
        } else {
            Err(CoreError::Validation(issues))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_fields_and_raw_ffmpeg() {
        let bad = r#"{"clips":[{"start_ms":0,"end_ms":1000,"hook_score":0.5,"editorial_note":"x","zoompan":"z=1.5"}]}"#;
        let err = SemanticEdl::from_json(bad).unwrap_err().to_string();
        assert!(err.contains("unknown field"), "{err}");
    }

    #[test]
    fn rejects_pixel_coordinates_in_motion() {
        let bad = r#"{"clips":[{"start_ms":0,"end_ms":1000,"hook_score":0.5,"editorial_note":"x"}],
            "motion_cues":[{"clip":0,"preset":"SpringPunchIn","x":120}]}"#;
        assert!(SemanticEdl::from_json(bad).is_err());
    }

    #[test]
    fn defaults_are_applied() {
        let ok =
            r#"{"clips":[{"start_ms":0,"end_ms":1000,"hook_score":0.5,"editorial_note":"x"}]}"#;
        let edl = SemanticEdl::from_json(ok).unwrap();
        assert_eq!(edl.output.aspect, AspectPreset::Vertical9x16);
        assert_eq!(edl.output.height, 1920);
        assert_eq!(edl.motion_for(0).0, MotionPreset::None);
        assert_eq!(edl.transition_after(0).0, TransitionKind::Hard);
    }

    #[test]
    fn schema_exposes_presets_but_no_coordinates() {
        let schema = serde_json::to_string(&SemanticEdl::json_schema()).unwrap();
        for needle in [
            "SpringPunchIn",
            "AutoJCut",
            "Karaoke",
            "hook_score",
            "additionalProperties",
        ] {
            assert!(schema.contains(needle), "schema missing {needle}");
        }
        for forbidden in ["\"x\"", "zoompan", "filter"] {
            assert!(!schema.contains(forbidden), "schema leaks {forbidden}");
        }
    }
}
