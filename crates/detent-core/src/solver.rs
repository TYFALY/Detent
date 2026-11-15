//! `EdlSolver`: deterministic polish from editorial intent to a frame-exact
//! [`ResolvedTimeline`].
//!
//! Stages, in order:
//! 1. **Breath snapping** — each in/out point moves (±`snap_tolerance_ms`) to
//!    the valley of the nearest silence, or to the leading edge of a breath so
//!    the breath is kept or dropped *whole*. Falls back to the nearest
//!    inter-word gap when the transcript says the cut lands inside a word.
//! 2. **Scene-collision guard** — a cut within `scene_guard_ms` of a shot
//!    boundary moves onto the boundary (preventing 1–4 frame flashes) when
//!    the move does not cross speech.
//! 3. **Frame quantisation** — to the nearest frame boundary, preferring the
//!    neighbour that does not clip a word.
//! 4. **J/L-cut injection** — requested splices shift the audio cut by the
//!    largest offset in `[150, target]` ms whose removed *and* inserted audio
//!    are speech-free. When speech is continuous on one side, the *picture*
//!    cut moves instead (audio untouched), which needs only the extra frames
//!    to be speech-free and inside one shot; otherwise the splice is
//!    downgraded to `Hard`.
//! 5. **Micro-crossfades** — every splice gets an equal-power crossfade
//!    (15–25 ms) centred on the audio cut, using handles taken from the
//!    source so total audio length stays identical to total video length.
//! 6. **Camera** — a critically damped spring path per segment.
//! 7. **Captions** — syntactic chunking on the output timeline.
//!
//! The result is checked with [`ResolvedTimeline::verify`]; an invariant
//! violation is reported as an error rather than handed to the renderer.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::camera::{Framing, SegmentCamera, solve_segment};
use crate::captions::{TimedWord, chunk};
use crate::edl::{
    MIN_CLIP_MS, MotionPreset, OutputLayout, SemanticEdl, TransitionKind, VideoTransitionKind,
};
use crate::error::{CoreError, Result, ValidationIssue};
use crate::index::{MediaIndex, PauseKind};
use crate::time::{FrameRate, US_PER_MS, UsRange, ms_to_us, us_to_ms};
use crate::timeline::{
    FrameRange, Layout, OutputFormat, ResolvedCaptions, ResolvedSegment, ResolvedTimeline, Splice,
    TIMELINE_VERSION, TimelineSource,
};

/// Solver tunables. Defaults encode common editorial practice.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SolverConfig {
    /// Max distance a cut may move to reach a breath/silence valley (ms).
    pub snap_tolerance_ms: u32,
    /// Cuts closer than this to a shot boundary snap onto it (ms).
    pub scene_guard_ms: u32,
    /// Micro-crossfade length; clamped to 15..=25 ms.
    pub crossfade_ms: u32,
    /// Guard band around words when testing whether audio is speech-free (ms).
    pub word_guard_ms: u32,
    /// Step used when searching for a feasible J/L offset (ms).
    pub jl_step_ms: u32,
    /// Minimum audio left in a segment after J/L shifts (ms).
    pub min_audio_ms: u32,
    /// Max camera path error after compression (px).
    pub camera_tolerance_px: f64,
    /// How long captions linger after the last word (ms).
    pub caption_linger_ms: u32,
    /// Extra cost (ms-equivalent) for snapping to a breath instead of silence.
    pub breath_penalty_ms: u32,
    /// Max distance a cut inside a word may move to reach a word boundary (ms).
    pub word_snap_max_ms: u32,
    /// ASR word-edge uncertainty: overlaps up to this long are not "speech" (ms).
    pub word_edge_tolerance_ms: u32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            snap_tolerance_ms: 120,
            scene_guard_ms: 150,
            crossfade_ms: 20,
            word_guard_ms: 20,
            jl_step_ms: 10,
            min_audio_ms: 200,
            camera_tolerance_px: 0.25,
            caption_linger_ms: 250,
            breath_penalty_ms: 30,
            word_snap_max_ms: 350,
            word_edge_tolerance_ms: 60,
        }
    }
}

/// Which side of a clip a cut belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum Edge {
    /// In-point.
    In,
    /// Out-point.
    Out,
}

/// An automatic change the solver made to the requested edit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind")]
pub enum Adjustment {
    /// Cut moved into a silence valley or to a breath's leading edge.
    BreathSnap {
        /// Clip index.
        clip: usize,
        /// Edge.
        edge: Edge,
        /// Requested time (ms).
        from_ms: u64,
        /// New time (ms).
        to_ms: u64,
        /// Pause type used.
        pause: PauseKind,
        /// Quietest level in that pause (dBFS).
        valley_dbfs: f32,
    },
    /// Cut moved out of a word into the nearest inter-word gap.
    WordGapSnap {
        /// Clip index.
        clip: usize,
        /// Edge.
        edge: Edge,
        /// Requested time (ms).
        from_ms: u64,
        /// New time (ms).
        to_ms: u64,
    },
    /// Cut moved onto a nearby shot boundary.
    SceneSnap {
        /// Clip index.
        clip: usize,
        /// Edge.
        edge: Edge,
        /// Time before the move (ms).
        from_ms: u64,
        /// Shot boundary time (ms).
        to_ms: u64,
    },
    /// J-cut applied (audio leads the picture).
    JCut {
        /// Splice index (after clip `splice`).
        splice: usize,
        /// Offset actually applied (ms).
        offset_ms: u32,
        /// Offset requested (ms).
        requested_ms: u32,
        /// True when the picture cut moved instead of the audio cut (only
        /// one side of the splice had a speech-free window).
        #[serde(default)]
        picture_side: bool,
    },
    /// L-cut applied (audio trails the picture).
    LCut {
        /// Splice index.
        splice: usize,
        /// Offset actually applied (ms).
        offset_ms: u32,
        /// Offset requested (ms).
        requested_ms: u32,
        /// True when the picture cut moved instead of the audio cut (only
        /// one side of the splice had a speech-free window).
        #[serde(default)]
        picture_side: bool,
    },
    /// Crossfade shortened because the source ran out of handle.
    CrossfadeShortened {
        /// Splice index.
        splice: usize,
        /// New length (µs).
        crossfade_us: i64,
    },
    /// Camera path solved.
    Camera {
        /// Clip index.
        clip: usize,
        /// Preset.
        preset: MotionPreset,
        /// Keyframes after compression.
        keyframes: usize,
        /// Frames simulated.
        frames: u32,
        /// Punch-ins triggered.
        punch_ins: u32,
        /// Max zoom factor reached.
        max_zoom: f64,
    },
}

/// Severity of a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum Level {
    /// Informational.
    Info,
    /// The edit renders, but a request could not be honoured exactly.
    Warning,
}

/// A message about the edit for the creative director.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Diagnostic {
    /// Severity.
    pub level: Level,
    /// Stable machine-readable code.
    pub code: String,
    /// Clip the message refers to, if any.
    pub clip: Option<usize>,
    /// Explanation.
    pub message: String,
}

/// Exact resolved timing of one clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedCut {
    /// Clip index.
    pub clip: usize,
    /// Requested `[start_ms, end_ms]`.
    pub requested_ms: (u64, u64),
    /// Resolved picture `[in_ms, out_ms]` (frame-aligned).
    pub video_ms: (u64, u64),
    /// Resolved audio `[in_ms, out_ms]` (excluding crossfade handles).
    pub audio_ms: (u64, u64),
    /// Source frames `[start, end)`.
    pub frames: (i64, i64),
    /// Where the clip starts in the output (ms).
    pub output_at_ms: u64,
}

/// Everything the solver did and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
pub struct SolveReport {
    /// Resolved cut list.
    pub cuts: Vec<ResolvedCut>,
    /// Automatic adjustments.
    pub adjustments: Vec<Adjustment>,
    /// Warnings and notes.
    pub diagnostics: Vec<Diagnostic>,
    /// Output duration (ms).
    pub output_duration_ms: u64,
    /// `"WIDTHxHEIGHT@FPS"`.
    pub output: String,
}

impl SolveReport {
    fn diag(&mut self, level: Level, code: &str, clip: Option<usize>, message: String) {
        self.diagnostics.push(Diagnostic {
            level,
            code: code.to_owned(),
            clip,
            message,
        });
    }
}

/// Solver output.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolution {
    /// Frame-accurate timeline.
    pub timeline: ResolvedTimeline,
    /// Explanation of every decision.
    pub report: SolveReport,
}

/// The deterministic EDL solver.
#[derive(Debug, Clone, Copy, Default)]
pub struct EdlSolver {
    /// Configuration.
    pub config: SolverConfig,
}

/// A resolved cut position: a time, or an exact frame (from a shot boundary).
#[derive(Debug, Clone, Copy)]
struct Cut {
    us: i64,
    frame: Option<i64>,
}

impl EdlSolver {
    /// Creates a solver.
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Validates `edl` against `index` and resolves it.
    ///
    /// With `tighten`, clips are first split at long pauses; segment
    /// `clip_index` values still point at the clips of the EDL as written,
    /// while report cuts and diagnostics count the split pieces.
    pub fn solve(&self, edl: &SemanticEdl, index: &MediaIndex) -> Result<Resolution> {
        index.validate()?;
        edl.validate(index)?;
        let Some(tighten) = edl.tighten else {
            return self.solve_validated(edl, index, &[]);
        };
        let (expanded, origin, removed_ms) = edl.tightened(index, tighten.max_silence_ms);
        let pieces = expanded.clips.len();
        // Edges created by dead-air removal are exact: snapping them toward the
        // middle of the pause would put the silence back.
        let pinned: Vec<bool> = origin.windows(2).map(|w| w[0] == w[1]).collect();
        let mut res = self.solve_validated(&expanded, index, &pinned)?;
        for s in &mut res.timeline.segments {
            s.clip_index = origin[s.clip_index];
        }
        res.report.diag(
            Level::Info,
            "dead_air_removed",
            None,
            format!(
                "tighten removed {} pause(s), {removed_ms} ms of dead air: {} clip(s) became {pieces} piece(s); report cuts are numbered by piece",
                pieces - edl.clips.len(),
                edl.clips.len()
            ),
        );
        Ok(res)
    }

    /// `pinned[k]` marks the splice after clip `k` as exact (no snapping).
    fn solve_validated(
        &self,
        edl: &SemanticEdl,
        index: &MediaIndex,
        pinned: &[bool],
    ) -> Result<Resolution> {
        edl.validate(index)?;
        let video = index
            .source
            .video
            .as_ref()
            .ok_or(CoreError::MissingStream("video"))?;
        let fps = video.fps;
        let dur_us = index.duration_us();
        let total_frames = fps.floor_frame(dur_us);
        let has_audio = index.source.audio.is_some();
        let mut rep = SolveReport::default();
        let n = edl.clips.len();

        // 1-3. Cut points.
        let min_frames = fps.nearest_frame(ms_to_us(MIN_CLIP_MS)).max(1);
        let mut video_ranges = Vec::with_capacity(n);
        for (i, clip) in edl.clips.iter().enumerate() {
            let req_in = ms_to_us(clip.start_ms);
            let req_out = ms_to_us(clip.end_ms);
            let pin_in = i > 0 && pinned.get(i - 1).copied().unwrap_or(false);
            let pin_out = pinned.get(i).copied().unwrap_or(false);
            let a = if pin_in {
                Cut {
                    us: req_in,
                    frame: None,
                }
            } else {
                self.resolve_edge(index, fps, i, Edge::In, req_in, &mut rep)
            };
            let b = if pin_out {
                Cut {
                    us: req_out,
                    frame: None,
                }
            } else {
                self.resolve_edge(index, fps, i, Edge::Out, req_out, &mut rep)
            };
            let mut fa = self.quantize(index, fps, a);
            let mut fb = self.quantize(index, fps, b).min(total_frames);
            if fb - fa < min_frames {
                rep.diag(
                    Level::Warning,
                    "snap_collapsed",
                    Some(i),
                    "snapping would shorten the clip below the minimum; using requested bounds"
                        .into(),
                );
                fa = fps.nearest_frame(req_in);
                fb = fps.nearest_frame(req_out).min(total_frames);
            }
            if fb <= fa {
                return Err(CoreError::Validation(vec![ValidationIssue::new(
                    format!("/clips/{i}"),
                    "clip does not contain a whole frame of video",
                )]));
            }
            video_ranges.push(FrameRange::new(fa, fb, fps));
        }

        // 4. J/L offsets per splice (µs, + = J / audio leads, - = L / audio lags).
        let mut offsets = vec![0_i64; n.saturating_sub(1)];
        for k in 0..n.saturating_sub(1) {
            let (kind, target_ms) = edl.transition_after(k);
            if kind == TransitionKind::Hard {
                continue;
            }
            if !has_audio {
                rep.diag(
                    Level::Warning,
                    "jl_no_audio",
                    Some(k),
                    "J/L-cut requested but the source has no audio".into(),
                );
                continue;
            }
            let prev = if k == 0 { 0 } else { offsets[k - 1] };
            let mut picture_side = false;
            let found = self
                .find_jl_offset(index, &video_ranges, k, kind, target_ms, prev)
                .or_else(|| {
                    let (o, a, b) = self.find_picture_shift(
                        index,
                        fps,
                        &video_ranges,
                        k,
                        kind,
                        target_ms,
                        (min_frames, total_frames),
                    )?;
                    video_ranges[k] = a;
                    video_ranges[k + 1] = b;
                    picture_side = true;
                    Some(o)
                });
            match found {
                Some(o) => {
                    offsets[k] = o;
                    let offset_ms = us_to_ms(o.abs()) as u32;
                    rep.adjustments.push(if o > 0 {
                        Adjustment::JCut {
                            splice: k,
                            offset_ms,
                            requested_ms: target_ms,
                            picture_side,
                        }
                    } else {
                        Adjustment::LCut {
                            splice: k,
                            offset_ms,
                            requested_ms: target_ms,
                            picture_side,
                        }
                    });
                }
                None => rep.diag(
                    Level::Warning,
                    "jl_downgraded",
                    Some(k),
                    format!(
                        "{kind:?} after clip {k} downgraded to Hard: no speech-free {}-{target_ms} ms window on either side of the splice (the audio side needs both, the picture side needs the extra frames to be silent and inside one shot); a micro-crossfade is still applied",
                        crate::edl::JL_OFFSET_RANGE_MS.start()
                    ),
                ),
            }
        }

        // 5. Audio ranges, crossfades and handles.
        let off = |k: isize| -> i64 {
            if k < 0 {
                0
            } else {
                offsets.get(k as usize).copied().unwrap_or(0)
            }
        };
        let mut audio: Vec<UsRange> = video_ranges
            .iter()
            .enumerate()
            .map(|(i, v)| {
                UsRange::new(v.start_us - off(i as isize - 1), v.end_us - off(i as isize))
            })
            .collect();
        let mut splices = Vec::with_capacity(n.saturating_sub(1));
        let xf = i64::from(self.config.crossfade_ms.clamp(15, 25)) * US_PER_MS;
        for k in 0..n.saturating_sub(1) {
            let mut c = 0;
            if has_audio {
                let room_tail = dur_us - audio[k].end_us;
                let room_head = audio[k + 1].start_us;
                let min_len = audio[k].len_us().min(audio[k + 1].len_us());
                let h = (xf / 2)
                    .min(room_tail)
                    .min(room_head)
                    .min(min_len / 2)
                    .max(0);
                c = 2 * h;
                audio[k].end_us += h;
                audio[k + 1].start_us -= h;
                if c < xf {
                    rep.adjustments.push(Adjustment::CrossfadeShortened {
                        splice: k,
                        crossfade_us: c,
                    });
                }
            }
            splices.push(Splice {
                after_segment: k,
                crossfade_us: c,
                av_offset_us: offsets[k],
                flash: edl.video_transition_after(k) == VideoTransitionKind::Flash,
            });
        }
        if !has_audio {
            audio = video_ranges
                .iter()
                .map(|v| UsRange::new(v.start_us, v.end_us))
                .collect();
        }

        // Output raster.
        let src_w = f64::from(video.width);
        let src_h = f64::from(video.height);
        let even = |x: f64| ((x / 2.0).round() as u32).max(1) * 2;
        let (out_w, out_h) = match edl.output.aspect.ratio() {
            Some(r) => {
                let h = even(f64::from(edl.output.height));
                (even(f64::from(h) * r), h)
            }
            None => {
                let h = even(f64::from(edl.output.height).min(src_h));
                (even(f64::from(h) * src_w / src_h), h)
            }
        };
        // FitBlur shows the whole source frame (source aspect) full-width over
        // a blurred fill; the camera then works on source-aspect windows.
        let layout = match edl.output.layout {
            OutputLayout::Crop => Layout::Crop,
            OutputLayout::FitBlur => Layout::FitBlur {
                fg_width: out_w,
                fg_height: even(f64::from(out_w) * src_h / src_w).min(out_h),
            },
            OutputLayout::FitBlurTall => Layout::FitBlur {
                fg_width: out_w,
                fg_height: even(f64::from(out_w) * 5.0 / 4.0).min(out_h),
            },
        };
        let (pic_w, pic_h) = match layout {
            Layout::Crop => (out_w, out_h),
            Layout::FitBlur {
                fg_width,
                fg_height,
            } => (fg_width, fg_height),
        };
        let framing = Framing {
            src_w,
            src_h,
            aspect: f64::from(pic_w) / f64::from(pic_h),
        };

        // 6. Segments + camera.
        let mut segments = Vec::with_capacity(n);
        let mut t_out = 0_i64;
        for (i, (v, a)) in video_ranges.iter().zip(&audio).enumerate() {
            let (preset, intensity) = edl.motion_for(i);
            let camera = solve_segment(&SegmentCamera {
                index,
                framing,
                preset,
                intensity,
                source_start_us: v.start_us,
                frames: u32::try_from(v.frames()).unwrap_or(u32::MAX),
                fps,
                tolerance_px: self.config.camera_tolerance_px,
            });
            if preset != MotionPreset::None {
                rep.adjustments.push(Adjustment::Camera {
                    clip: i,
                    preset,
                    keyframes: camera.keyframes.len(),
                    frames: camera.solved_frames,
                    punch_ins: camera.punch_ins,
                    max_zoom: (camera.max_zoom(framing.base_window().0) * 1_000.0).round()
                        / 1_000.0,
                });
                if preset == MotionPreset::SpringPunchIn && camera.punch_ins == 0 {
                    rep.diag(
                        Level::Info,
                        "punch_in_no_peaks",
                        Some(i),
                        "SpringPunchIn requested but no vocal excitement peaks fall in this clip"
                            .into(),
                    );
                }
                if camera.subject_inset && preset == MotionPreset::SmoothTrackSubject {
                    rep.diag(
                        Level::Warning,
                        "subject_is_inset",
                        Some(i),
                        format!(
                            "the detected subject is a small picture-in-picture inset (facecam over content);                              it was not tracked. For screen/gameplay/reaction footage use \"output\": {{\"layout\": \"FitBlur\"}}{}",
                            if edl.output.layout == OutputLayout::FitBlur { " (already active)" } else { "" }
                        ),
                    );
                }
                if preset == MotionPreset::SmoothTrackSubject && index.visual.subjects.is_empty() {
                    rep.diag(
                        Level::Info,
                        "track_no_subjects",
                        Some(i),
                        "SmoothTrackSubject requested but the index has no subject boxes; framing stays centred"
                            .into(),
                    );
                }
            }
            segments.push(ResolvedSegment {
                clip_index: i,
                video: *v,
                audio: *a,
                output_start_us: t_out,
                camera,
            });
            t_out += v.len_us();
        }
        let duration_us = t_out;

        // 7. Captions.
        let captions = edl.captions.map(|spec| {
            let mut words = Vec::new();
            let mut breaks = Vec::new();
            for (i, s) in segments.iter().enumerate() {
                let speech = UsRange::new(
                    s.video.start_us - off(i as isize - 1),
                    s.video.end_us - off(i as isize),
                );
                breaks.push(words.len());
                for w in &index.transcript {
                    let (ws, we) = (ms_to_us(w.start_ms), ms_to_us(w.end_ms));
                    let mid = (ws + we) / 2;
                    if mid < speech.start_us || mid >= speech.end_us {
                        continue;
                    }
                    let map = |t: i64| {
                        s.output_start_us
                            + (t.clamp(speech.start_us, speech.end_us) - s.video.start_us)
                    };
                    words.push(TimedWord {
                        text: w.word.trim().to_owned(),
                        start_us: map(ws),
                        end_us: map(we),
                    });
                }
            }
            // Monotonic, in-range word times.
            let mut prev_end = 0_i64;
            for w in &mut words {
                w.start_us = w.start_us.clamp(prev_end, duration_us);
                w.end_us = w.end_us.clamp(w.start_us, duration_us);
                prev_end = w.end_us;
            }
            words.retain(|w| w.end_us > w.start_us);
            let breaks: Vec<usize> = breaks.into_iter().filter(|&b| b > 0).collect();
            let mut blocks = chunk(
                &words,
                &breaks,
                &spec.grouping,
                i64::from(self.config.caption_linger_ms) * US_PER_MS,
            );
            for b in &mut blocks {
                b.end_us = b.end_us.min(duration_us);
            }
            blocks.retain(|b| b.end_us > b.start_us);
            ResolvedCaptions {
                style: spec.style,
                position: spec.position,
                blocks,
                headline: edl
                    .headline
                    .as_deref()
                    .map(str::trim)
                    .filter(|h| !h.is_empty())
                    .map(str::to_owned),
                headline_end_us: edl
                    .headline_duration_ms
                    .map(|d| ms_to_us(u64::from(d)).min(duration_us)),
                highlight_words: edl.highlight_words.clone(),
            }
        });

        // Editorial sanity notes.
        if let Some(best) = edl
            .clips
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.hook_score.total_cmp(&b.1.hook_score))
            && best.0 != 0
            && best.1.hook_score > edl.clips[0].hook_score + 0.15
        {
            rep.diag(
                Level::Info,
                "weak_opening",
                Some(best.0),
                format!(
                    "clip {} has hook_score {:.2} but the edit opens with {:.2}; consider leading with it",
                    best.0, best.1.hook_score, edl.clips[0].hook_score
                ),
            );
        }
        for (i, a) in video_ranges.iter().enumerate() {
            for (j, b) in video_ranges.iter().enumerate().skip(i + 1) {
                if UsRange::new(a.start_us, a.end_us).overlaps(UsRange::new(b.start_us, b.end_us)) {
                    rep.diag(
                        Level::Warning,
                        "duplicate_source",
                        Some(j),
                        format!("clip {j} reuses source material from clip {i}"),
                    );
                }
            }
        }
        let edge_tol = ms_to_us(u64::from(self.config.word_edge_tolerance_ms));
        for (i, (v, a)) in video_ranges.iter().zip(&audio).enumerate() {
            let speech_in = a.start_us
                + if i > 0 {
                    splices[i - 1].crossfade_us / 2
                } else {
                    0
                };
            let speech_out = a.end_us - splices.get(i).map_or(0, |s| s.crossfade_us / 2);
            for (edge, t) in [(Edge::In, speech_in), (Edge::Out, speech_out)] {
                if let Some(w) = index.word_containing(t, edge_tol) {
                    rep.diag(
                        Level::Warning,
                        "cut_inside_word",
                        Some(i),
                        format!(
                            "{edge:?}-point at {} ms falls inside the word {:?}; widen the clip or move the cut to a pause",
                            us_to_ms(t),
                            w.word
                        ),
                    );
                }
            }
            rep.cuts.push(ResolvedCut {
                clip: i,
                requested_ms: (edl.clips[i].start_ms, edl.clips[i].end_ms),
                video_ms: (us_to_ms(v.start_us), us_to_ms(v.end_us)),
                audio_ms: (us_to_ms(speech_in), us_to_ms(speech_out)),
                frames: (v.start_frame, v.end_frame),
                output_at_ms: us_to_ms(segments[i].output_start_us),
            });
        }

        let timeline = ResolvedTimeline {
            version: TIMELINE_VERSION,
            source: TimelineSource {
                path: index.source.path.clone(),
                width: video.width,
                height: video.height,
                fps,
                duration_us: dur_us,
                has_audio,
            },
            output: OutputFormat {
                aspect: edl.output.aspect,
                width: out_w,
                height: out_h,
                fps,
                layout,
                grade: edl.output.grade,
                audio_mix: edl.output.audio_mix,
                opening_pop: edl.output.opening_pop,
            },
            segments,
            splices,
            captions,
            duration_us,
        };
        let violations = timeline.verify();
        if !violations.is_empty() {
            return Err(CoreError::Invariant(violations));
        }
        rep.output_duration_ms = us_to_ms(duration_us);
        rep.output = format!("{out_w}x{out_h}@{:.3}", fps.as_f64());
        Ok(Resolution {
            timeline,
            report: rep,
        })
    }

    fn resolve_edge(
        &self,
        index: &MediaIndex,
        fps: FrameRate,
        clip: usize,
        edge: Edge,
        t: i64,
        rep: &mut SolveReport,
    ) -> Cut {
        let mut cut = Cut { us: t, frame: None };
        if let Some((to, kind, db)) = self.breath_snap(index, t) {
            if to != t {
                rep.adjustments.push(Adjustment::BreathSnap {
                    clip,
                    edge,
                    from_ms: us_to_ms(t),
                    to_ms: us_to_ms(to),
                    pause: kind,
                    valley_dbfs: db,
                });
            }
            cut.us = to;
        } else if let Some(to) = self.word_gap_snap(index, t) {
            rep.adjustments.push(Adjustment::WordGapSnap {
                clip,
                edge,
                from_ms: us_to_ms(t),
                to_ms: us_to_ms(to),
            });
            cut.us = to;
        } else if index
            .word_containing(t, ms_to_us(u64::from(self.config.word_edge_tolerance_ms)))
            .is_some()
            || (index.transcript.is_empty() && !index.audio.pauses.is_empty())
        {
            // Only worth reporting when the cut is genuinely mid-word (or there
            // is no transcript to say otherwise); a word-boundary cut is fine.
            rep.diag(
                Level::Info,
                "no_valley",
                Some(clip),
                format!(
                    "no breath/silence valley within ±{} ms of the {edge:?}-point at {} ms",
                    self.config.snap_tolerance_ms,
                    us_to_ms(t)
                ),
            );
        }

        // Scene-collision guard.
        let guard = ms_to_us(u64::from(self.config.scene_guard_ms));
        let nearest = index
            .visual
            .shots
            .iter()
            .map(|s| (s, fps.frame_to_us(s.frame as i64)))
            .filter(|(_, su)| (su - cut.us).abs() <= guard)
            .min_by_key(|(_, su)| (su - cut.us).abs());
        if let Some((shot, su)) = nearest {
            let region = UsRange::new(su.min(cut.us), su.max(cut.us));
            // A 1-4 frame flash is a worse defect than trimming the padded edge
            // of an ASR word, so overlaps within the word-edge tolerance are allowed.
            let edge_tol = ms_to_us(u64::from(self.config.word_edge_tolerance_ms));
            let safe = region.len_us() < fps.frame_duration_us()
                || if index.transcript.is_empty() {
                    index.inside_pause(region) || region.len_us() <= edge_tol
                } else {
                    index.speech_overlap_us(region) <= edge_tol
                };
            if safe {
                if region.len_us() > 0 {
                    rep.adjustments.push(Adjustment::SceneSnap {
                        clip,
                        edge,
                        from_ms: us_to_ms(cut.us),
                        to_ms: shot.t_ms,
                    });
                }
                cut = Cut {
                    us: su,
                    frame: Some(shot.frame as i64),
                };
            } else {
                rep.diag(
                    Level::Warning,
                    "scene_collision",
                    Some(clip),
                    format!(
                        "{edge:?}-point at {} ms is {} ms from a shot change at {} ms, but moving it would cut speech; a brief flash may be visible",
                        us_to_ms(cut.us),
                        us_to_ms((su - cut.us).abs()),
                        shot.t_ms
                    ),
                );
            }
        }
        cut
    }

    /// Best pause-based cut within tolerance: `(time, kind, valley dBFS)`.
    fn breath_snap(&self, index: &MediaIndex, t: i64) -> Option<(i64, PauseKind, f32)> {
        let tol = ms_to_us(u64::from(self.config.snap_tolerance_ms));
        let lo_w = (t - tol).max(0);
        let hi_w = (t + tol).min(index.duration_us());
        let guard = ms_to_us(u64::from(self.config.word_guard_ms));
        let mut best: Option<(i64, i64, PauseKind, f32)> = None;
        for p in &index.audio.pauses {
            let r = p.range_us();
            if r.end_us < lo_w || r.start_us > hi_w {
                continue;
            }
            let (lo, hi) = (r.start_us.max(lo_w), r.end_us.min(hi_w));
            let (c, penalty) = match p.kind {
                PauseKind::Silence => (ms_to_us(p.valley_ms).clamp(lo, hi), 0),
                // Keep or drop a breath whole: cut at its leading edge.
                PauseKind::Breath => {
                    if r.start_us < lo_w || r.start_us > hi_w {
                        continue;
                    }
                    (
                        r.start_us,
                        ms_to_us(u64::from(self.config.breath_penalty_ms)),
                    )
                }
            };
            if !index.transcript.is_empty() && index.words_overlap(UsRange::new(c, c), guard) {
                continue;
            }
            let score = (c - t).abs() + penalty;
            if best.is_none_or(|b| score < b.0) {
                best = Some((score, c, p.kind, p.min_dbfs));
            }
        }
        best.map(|(_, c, k, db)| (c, k, db))
    }

    /// When `t` falls inside a word, moves it to that word's nearer edge (or
    /// into the adjacent inter-word gap), then refines to the quietest 10 ms
    /// hop within ±40 ms: fast speech has no pauses, but it always has word
    /// boundaries, and the true boundary is where the energy dips.
    fn word_gap_snap(&self, index: &MediaIndex, t: i64) -> Option<i64> {
        let w = index.word_containing(t, 0)?;
        let (s, e) = (ms_to_us(w.start_ms), ms_to_us(w.end_ms));
        let max = ms_to_us(u64::from(self.config.word_snap_max_ms));
        let words = &index.transcript;
        let i = words.iter().position(|x| std::ptr::eq(x, w))?;
        // Edge, or the middle of a real gap next to that edge.
        let before = match i.checked_sub(1).and_then(|j| words.get(j)) {
            Some(p) if ms_to_us(p.end_ms) < s => (ms_to_us(p.end_ms) + s) / 2,
            _ => s,
        };
        let after = match words.get(i + 1) {
            Some(n) if ms_to_us(n.start_ms) > e => (e + ms_to_us(n.start_ms)) / 2,
            _ => e,
        };
        let c = if (t - before).abs() <= (after - t).abs() {
            before
        } else {
            after
        };
        if (c - t).abs() > max {
            return None;
        }
        const REFINE_US: i64 = 40_000;
        Some(index.quietest_in(c - REFINE_US, c + REFINE_US).unwrap_or(c))
    }

    /// Nearest frame, preferring the neighbour that does not clip a word.
    fn quantize(&self, index: &MediaIndex, fps: FrameRate, cut: Cut) -> i64 {
        if let Some(f) = cut.frame {
            return f;
        }
        let near = fps.nearest_frame(cut.us);
        let near_us = fps.frame_to_us(near);
        let crosses = |f_us: i64| {
            !index.transcript.is_empty()
                && index.words_overlap(UsRange::new(f_us.min(cut.us), f_us.max(cut.us)), 0)
        };
        if !crosses(near_us) {
            return near;
        }
        let other = if near_us > cut.us { near - 1 } else { near + 1 };
        if crosses(fps.frame_to_us(other)) {
            near
        } else {
            other.max(0)
        }
    }

    /// Largest feasible J/L offset (µs, signed) for the splice after clip `k`.
    fn find_jl_offset(
        &self,
        index: &MediaIndex,
        v: &[FrameRange],
        k: usize,
        kind: TransitionKind,
        target_ms: u32,
        prev_offset: i64,
    ) -> Option<i64> {
        let lo = *crate::edl::JL_OFFSET_RANGE_MS.start();
        let target = target_ms.clamp(lo, *crate::edl::JL_OFFSET_RANGE_MS.end());
        let guard = ms_to_us(u64::from(self.config.word_guard_ms));
        let min_audio = ms_to_us(u64::from(self.config.min_audio_ms));
        let step = self.config.jl_step_ms.max(1) as usize;
        let (a, b) = (v[k], v[k + 1]);
        let mut candidates: Vec<u32> = (lo..=target).rev().step_by(step).collect();
        if candidates.last() != Some(&lo) {
            candidates.push(lo);
        }
        candidates.into_iter().find_map(|o_ms| {
            let o = ms_to_us(u64::from(o_ms));
            let (removed, added, signed) = match kind {
                TransitionKind::AutoJCut => (
                    UsRange::new(a.end_us - o, a.end_us),
                    UsRange::new(b.start_us - o, b.start_us),
                    o,
                ),
                TransitionKind::AutoLCut => (
                    UsRange::new(b.start_us, b.start_us + o),
                    UsRange::new(a.end_us, a.end_us + o),
                    -o,
                ),
                TransitionKind::Hard => return None,
            };
            let a_len = (a.end_us - signed) - (a.start_us - prev_offset);
            let b_len = b.end_us - (b.start_us - signed);
            (a_len >= min_audio
                && b_len >= min_audio
                && index.audio_range_is_safe(removed, guard)
                && index.audio_range_is_safe(added, guard))
            .then_some(signed)
        })
    }

    /// Picture-side J/L: keeps the audio splice where breath snapping put it
    /// and moves the *video* splice instead, so no audio is removed or
    /// inserted. A J-cut holds clip `k`'s picture past its out point (its
    /// post-roll is shown under the next clip's first words); an L-cut starts
    /// the next clip's picture early (its pre-roll is shown under the end of
    /// clip `k`). Only those extra frames must be speech-free (a closed mouth)
    /// and inside a single shot, so this works when speech is continuous on
    /// the other side. Returns the signed offset and the two new ranges.
    #[allow(clippy::too_many_arguments)]
    fn find_picture_shift(
        &self,
        index: &MediaIndex,
        fps: FrameRate,
        v: &[FrameRange],
        k: usize,
        kind: TransitionKind,
        target_ms: u32,
        (min_frames, total_frames): (i64, i64),
    ) -> Option<(i64, FrameRange, FrameRange)> {
        let lo = *crate::edl::JL_OFFSET_RANGE_MS.start();
        let target = target_ms.clamp(lo, *crate::edl::JL_OFFSET_RANGE_MS.end());
        let guard = ms_to_us(u64::from(self.config.word_guard_ms));
        let (a, b) = (v[k], v[k + 1]);
        let mut lo_frames = fps.nearest_frame(ms_to_us(u64::from(lo))).max(1);
        while fps.frame_to_us(lo_frames) < ms_to_us(u64::from(lo)) {
            lo_frames += 1;
        }
        let hi_frames = fps.floor_frame(ms_to_us(u64::from(target)));
        // A shot boundary at frame `f` inside `frames` would flash a new shot.
        let crosses_shot = |frames: std::ops::Range<i64>| {
            index
                .visual
                .shots
                .iter()
                .any(|s| frames.contains(&(s.frame as i64)))
        };
        (lo_frames..=hi_frames).rev().find_map(|d| match kind {
            TransitionKind::AutoJCut => {
                let na = FrameRange::new(a.start_frame, a.end_frame + d, fps);
                let nb = FrameRange::new(b.start_frame + d, b.end_frame, fps);
                (na.end_frame <= total_frames
                    && nb.frames() >= min_frames
                    && !crosses_shot(a.end_frame..na.end_frame)
                    && index.audio_range_is_safe(UsRange::new(a.end_us, na.end_us), guard))
                .then(|| (na.end_us - a.end_us, na, nb))
            }
            TransitionKind::AutoLCut => {
                let na = FrameRange::new(a.start_frame, a.end_frame - d, fps);
                let nb = FrameRange::new(b.start_frame - d, b.end_frame, fps);
                (nb.start_frame >= 0
                    && na.frames() >= min_frames
                    && !crosses_shot(nb.start_frame + 1..b.start_frame + 1)
                    && index.audio_range_is_safe(UsRange::new(nb.start_us, b.start_us), guard))
                .then(|| (-(a.end_us - na.end_us), na, nb))
            }
            TransitionKind::Hard => None,
        })
    }
}
