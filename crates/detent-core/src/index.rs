//! The `MediaIndex`: compact multi-modal telemetry for one source file.
//!
//! The full index (10 ms audio envelope, every shot boundary, subject boxes)
//! is what the deterministic solver consumes. LLMs never see it directly; they
//! get an [`IndexDigest`], which is bounded in size regardless of media length.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result, ValidationIssue};
use crate::time::{FrameRate, UsRange, ms_to_us};

/// Current on-disk schema version of [`MediaIndex`].
pub const INDEX_VERSION: u32 = 1;

/// Longest plausible single word; longer tokens are rejected by validation.
pub const MAX_WORD_MS: u64 = 10_000;

/// Telemetry for a single source media file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MediaIndex {
    /// Schema version (see [`INDEX_VERSION`]).
    pub version: u32,
    /// Container / stream metadata.
    pub source: SourceInfo,
    /// Word-level transcript, sorted by `start_ms`. Empty when no ASR ran.
    #[serde(default)]
    pub transcript: Vec<WordToken>,
    /// Audio DSP telemetry.
    pub audio: AudioTelemetry,
    /// Visual telemetry.
    pub visual: VisualTelemetry,
}

/// Container-level metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SourceInfo {
    /// Absolute path of the indexed file.
    pub path: String,
    /// Container duration in milliseconds.
    pub duration_ms: u64,
    /// Primary video stream, if any.
    pub video: Option<VideoStreamInfo>,
    /// Primary audio stream, if any.
    pub audio: Option<AudioStreamInfo>,
}

/// Primary video stream metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VideoStreamInfo {
    /// Width in pixels (after rotation metadata is applied).
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Exact frame rate.
    pub fps: FrameRate,
    /// Codec name as reported by the demuxer (e.g. `h264`).
    pub codec: String,
}

/// Primary audio stream metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AudioStreamInfo {
    /// Native sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u32,
    /// Codec name (e.g. `aac`).
    pub codec: String,
}

/// A single recognised word.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WordToken {
    /// Surface form including trailing punctuation (`"world,"`).
    pub word: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Recogniser confidence in `[0, 1]`.
    pub confidence: f32,
}

/// Kind of low-energy interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum PauseKind {
    /// Near noise floor (room tone).
    Silence,
    /// Quiet, broadband, high zero-crossing-rate energy: an audible breath.
    Breath,
}

/// A detected silence or breath valley.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SilenceInterval {
    /// Start in milliseconds.
    pub start_ms: u64,
    /// End in milliseconds.
    pub end_ms: u64,
    /// Quietest RMS level inside the interval (dBFS).
    pub min_dbfs: f32,
    /// Timestamp of the quietest hop: the ideal splice point.
    pub valley_ms: u64,
    /// Silence or breath.
    pub kind: PauseKind,
}

impl SilenceInterval {
    /// The interval as a microsecond range.
    pub fn range_us(&self) -> UsRange {
        UsRange::new(ms_to_us(self.start_ms), ms_to_us(self.end_ms))
    }
}

/// A local maximum of vocal excitement (spectral flux x loudness).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExcitementPeak {
    /// Peak time in milliseconds.
    pub t_ms: u64,
    /// Standard score of the excitement curve at the peak.
    pub score: f32,
}

/// Audio DSP telemetry. Envelope vectors are struct-of-arrays sampled every
/// `hop_ms`; element `i` covers `[i*hop_ms, (i+1)*hop_ms)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
pub struct AudioTelemetry {
    /// Envelope hop size in milliseconds (10 by default).
    pub hop_ms: u32,
    /// Sample rate the analysis ran at (audio is resampled to this).
    pub analysis_rate: u32,
    /// RMS loudness per hop in dBFS (rounded to 0.1 dB).
    pub rms_dbfs: Vec<f32>,
    /// Zero-crossing rate per hop in `[0, 1]`.
    pub zcr: Vec<f32>,
    /// Vocal-band (300 Hz - 3.4 kHz) half-wave rectified spectral flux per hop.
    pub flux: Vec<f32>,
    /// Estimated noise floor (5th percentile of RMS) in dBFS.
    pub noise_floor_dbfs: f32,
    /// Estimated speech level (median of non-silent hops) in dBFS.
    pub speech_level_dbfs: f32,
    /// Detected silence / breath valleys.
    pub pauses: Vec<SilenceInterval>,
    /// Detected vocal excitement peaks.
    pub peaks: Vec<ExcitementPeak>,
}

/// A hard cut / scene change in the source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ShotBoundary {
    /// Time of the first frame of the new shot (ms).
    pub t_ms: u64,
    /// Index of the first frame of the new shot.
    pub frame: u64,
    /// Detector score (histogram distance / adaptive threshold).
    pub score: f32,
}

/// How a subject box was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum SubjectSource {
    /// Pure-Rust motion + contrast saliency centroid.
    Saliency,
    /// ONNX face / person detector.
    Onnx,
}

/// A normalised subject bounding box sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SubjectBox {
    /// Sample time in milliseconds.
    pub t_ms: u64,
    /// Left edge in `[0, 1]`.
    pub x: f32,
    /// Top edge in `[0, 1]`.
    pub y: f32,
    /// Width in `[0, 1]`.
    pub w: f32,
    /// Height in `[0, 1]`.
    pub h: f32,
    /// Detector confidence in `[0, 1]`.
    pub confidence: f32,
    /// Producer of this sample.
    pub source: SubjectSource,
}

impl SubjectBox {
    /// Box centre in normalised coordinates.
    pub fn center(&self) -> (f64, f64) {
        (
            f64::from(self.x) + f64::from(self.w) / 2.0,
            f64::from(self.y) + f64::from(self.h) / 2.0,
        )
    }
}

/// Location and layout of a keyframe contact sheet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ContactSheet {
    /// Path to the JPEG grid.
    pub path: String,
    /// Grid columns.
    pub columns: u32,
    /// Grid rows.
    pub rows: u32,
    /// Tile width in pixels.
    pub tile_width: u32,
    /// Tile height in pixels.
    pub tile_height: u32,
    /// Timestamp (ms) of each tile in row-major order.
    pub tile_ms: Vec<u64>,
}

/// Visual telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
pub struct VisualTelemetry {
    /// Detected hard cuts, sorted by time.
    pub shots: Vec<ShotBoundary>,
    /// Subject samples, sorted by time.
    pub subjects: Vec<SubjectBox>,
    /// Contact sheet, if generated.
    pub contact_sheet: Option<ContactSheet>,
}

impl MediaIndex {
    /// Parses and validates an index from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        let idx: Self = serde_json::from_str(s).map_err(|source| CoreError::Json {
            what: "MediaIndex",
            source,
        })?;
        idx.validate()?;
        Ok(idx)
    }

    /// Checks internal consistency (sortedness, ranges, envelope lengths).
    pub fn validate(&self) -> Result<()> {
        let mut issues = Vec::new();
        if self.version != INDEX_VERSION {
            issues.push(ValidationIssue::new(
                "/version",
                format!("unsupported index version {}", self.version),
            ));
        }
        let n = self.audio.rms_dbfs.len();
        if self.audio.zcr.len() != n || self.audio.flux.len() != n {
            issues.push(ValidationIssue::new(
                "/audio",
                "rms_dbfs, zcr and flux must have equal length",
            ));
        }
        if n > 0 && self.audio.hop_ms == 0 {
            issues.push(ValidationIssue::new("/audio/hop_ms", "must be > 0"));
        }
        for (i, w) in self.transcript.iter().enumerate() {
            if w.end_ms < w.start_ms || w.end_ms - w.start_ms > MAX_WORD_MS {
                issues.push(ValidationIssue::new(
                    format!("/transcript/{i}"),
                    format!("word must satisfy start_ms <= end_ms <= start_ms + {MAX_WORD_MS}"),
                ));
            }
        }
        if self
            .transcript
            .windows(2)
            .any(|p| p[1].start_ms < p[0].start_ms)
        {
            issues.push(ValidationIssue::new(
                "/transcript",
                "words must be sorted by start_ms",
            ));
        }
        for (i, p) in self.audio.pauses.iter().enumerate() {
            if p.end_ms <= p.start_ms || p.valley_ms < p.start_ms || p.valley_ms > p.end_ms {
                issues.push(ValidationIssue::new(
                    format!("/audio/pauses/{i}"),
                    "pause must satisfy start < end and start <= valley <= end",
                ));
            }
        }
        if let Some(v) = &self.source.video
            && (v.width == 0 || v.height == 0 || v.fps.num == 0 || v.fps.den == 0)
        {
            issues.push(ValidationIssue::new(
                "/source/video",
                "zero dimension or frame rate",
            ));
        }
        if issues.is_empty() {
            Ok(())
        } else {
            Err(CoreError::Validation(issues))
        }
    }

    /// Duration in microseconds.
    pub fn duration_us(&self) -> i64 {
        ms_to_us(self.source.duration_ms)
    }

    /// True when any transcript word overlaps `range` (expanded by `guard_us`).
    pub fn words_overlap(&self, range: UsRange, guard_us: i64) -> bool {
        if range.len_us() == 0 && guard_us == 0 {
            return false;
        }
        let probe = UsRange::new(range.start_us - guard_us, range.end_us + guard_us);
        // Words are sorted by start; binary search the first word that could overlap.
        // `validate` guarantees no word is longer than MAX_WORD_MS, which bounds the scan.
        let first = self
            .transcript
            .partition_point(|w| ms_to_us(w.start_ms) < probe.start_us - ms_to_us(MAX_WORD_MS));
        self.transcript[first..]
            .iter()
            .take_while(|w| ms_to_us(w.start_ms) < probe.end_us)
            .any(|w| UsRange::new(ms_to_us(w.start_ms), ms_to_us(w.end_ms)).overlaps(probe))
    }

    /// True when `range` lies entirely inside a single detected pause.
    pub fn inside_pause(&self, range: UsRange) -> bool {
        self.audio.pauses.iter().any(|p| {
            let r = p.range_us();
            r.start_us <= range.start_us && range.end_us <= r.end_us
        })
    }

    /// Whether removing or inserting the source audio in `range` is safe:
    /// it must not contain (part of) a spoken word. When no transcript exists
    /// the range must instead sit inside a detected pause.
    pub fn audio_range_is_safe(&self, range: UsRange, guard_us: i64) -> bool {
        if range.start_us < 0 || range.end_us > self.duration_us() {
            return false;
        }
        if range.len_us() == 0 {
            return true;
        }
        if self.transcript.is_empty() {
            self.inside_pause(range)
        } else {
            !self.words_overlap(range, guard_us)
        }
    }

    /// Linearly interpolated subject centre at `t_ms`, if subjects exist.
    pub fn subject_center_at(&self, t_ms: f64) -> Option<(f64, f64)> {
        let s = &self.visual.subjects;
        let first = s.first()?;
        let idx = s.partition_point(|b| (b.t_ms as f64) <= t_ms);
        if idx == 0 {
            return Some(first.center());
        }
        let a = &s[idx - 1];
        let Some(b) = s.get(idx) else {
            return Some(a.center());
        };
        let span = (b.t_ms.saturating_sub(a.t_ms)) as f64;
        let k = if span <= 0.0 {
            0.0
        } else {
            ((t_ms - a.t_ms as f64) / span).clamp(0.0, 1.0)
        };
        let (ax, ay) = a.center();
        let (bx, by) = b.center();
        Some((ax + (bx - ax) * k, ay + (by - ay) * k))
    }

    /// RMS dBFS at `t_ms` (nearest hop); `None` without an envelope.
    pub fn rms_at(&self, t_ms: u64) -> Option<f32> {
        let hop = u64::from(self.audio.hop_ms.max(1));
        let i = usize::try_from(t_ms / hop).ok()?;
        self.audio.rms_dbfs.get(i).copied()
    }
}

/// A bounded-size, token-efficient view of a [`MediaIndex`] for an LLM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct IndexDigest {
    /// Duration (ms).
    pub duration_ms: u64,
    /// `"WIDTHxHEIGHT@FPS"` or `null` for audio-only.
    pub video: Option<String>,
    /// Transcript grouped into phrases separated by pauses: `[start_ms, end_ms, text]`.
    pub phrases: Vec<(u64, u64, String)>,
    /// Pauses >= 250 ms usable as cut points: `[start_ms, end_ms, kind]`.
    pub cut_windows: Vec<(u64, u64, PauseKind)>,
    /// Strongest excitement peaks (ms), strongest first.
    pub excitement_ms: Vec<u64>,
    /// Shot boundaries (ms).
    pub shots_ms: Vec<u64>,
    /// Loudness every `loudness_step_ms`, integer dBFS.
    pub loudness_dbfs: Vec<i16>,
    /// Step of `loudness_dbfs`.
    pub loudness_step_ms: u64,
    /// Contact sheet path, if any.
    pub contact_sheet: Option<String>,
}

/// Limits for [`MediaIndex::digest`].
#[derive(Debug, Clone, Copy)]
pub struct DigestLimits {
    /// Max loudness samples.
    pub loudness_points: usize,
    /// Max excitement peaks.
    pub peaks: usize,
    /// Max shot boundaries.
    pub shots: usize,
    /// Max pause windows.
    pub cut_windows: usize,
    /// Pause length (ms) that ends a phrase.
    pub phrase_gap_ms: u64,
}

impl Default for DigestLimits {
    fn default() -> Self {
        Self {
            loudness_points: 240,
            peaks: 40,
            shots: 200,
            cut_windows: 400,
            phrase_gap_ms: 350,
        }
    }
}

impl MediaIndex {
    /// Produces the LLM-facing digest. Output size is bounded by `limits`
    /// (plus the transcript text, which is inherently required).
    pub fn digest(&self, limits: DigestLimits) -> IndexDigest {
        let mut phrases = Vec::new();
        let mut cur: Option<(u64, u64, String)> = None;
        for w in &self.transcript {
            match cur.as_mut() {
                Some((_, end, text)) if w.start_ms.saturating_sub(*end) < limits.phrase_gap_ms => {
                    text.push(' ');
                    text.push_str(w.word.trim());
                    *end = w.end_ms;
                }
                _ => {
                    if let Some(p) = cur.take() {
                        phrases.push(p);
                    }
                    cur = Some((w.start_ms, w.end_ms, w.word.trim().to_owned()));
                }
            }
        }
        phrases.extend(cur);

        let mut cut_windows: Vec<_> = self
            .audio
            .pauses
            .iter()
            .filter(|p| p.end_ms - p.start_ms >= 250)
            .map(|p| (p.start_ms, p.end_ms, p.kind))
            .collect();
        cut_windows.truncate(limits.cut_windows);

        let mut peaks: Vec<_> = self.audio.peaks.iter().collect();
        peaks.sort_by(|a, b| b.score.total_cmp(&a.score));
        let excitement_ms = peaks.iter().take(limits.peaks).map(|p| p.t_ms).collect();

        let shots_ms = self
            .visual
            .shots
            .iter()
            .take(limits.shots)
            .map(|s| s.t_ms)
            .collect();

        let n = self.audio.rms_dbfs.len();
        let hop = u64::from(self.audio.hop_ms.max(1));
        let points = limits.loudness_points.max(1);
        let per = n.div_ceil(points).max(1);
        let loudness_dbfs = self
            .audio
            .rms_dbfs
            .chunks(per)
            .map(|c| c.iter().copied().fold(f32::NEG_INFINITY, f32::max).round() as i16)
            .collect();

        IndexDigest {
            duration_ms: self.source.duration_ms,
            video: self
                .source
                .video
                .as_ref()
                .map(|v| format!("{}x{}@{:.3}", v.width, v.height, v.fps.as_f64())),
            phrases,
            cut_windows,
            excitement_ms,
            shots_ms,
            loudness_dbfs,
            loudness_step_ms: per as u64 * hop,
            contact_sheet: self.visual.contact_sheet.as_ref().map(|c| c.path.clone()),
        }
    }
}
