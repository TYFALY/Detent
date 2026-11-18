//! Virtual camera: turns a semantic [`MotionPreset`] into per-frame crop
//! windows, then compresses them into a minimal piecewise-linear path.
//!
//! Pipeline per clip:
//! 1. Build *target* trajectories for centre (`cx`, `cy`, normalised) and
//!    zoom (`z >= 1`) from subject boxes and vocal excitement peaks.
//! 2. Clamp centre targets to the region where a `z = 1` window fits. Since
//!    a spring from rest stays inside the hull of its targets (see
//!    [`crate::spring`]) and `z >= 1` only widens the feasible region, the
//!    solved window is inside the frame by construction.
//! 3. Integrate each channel with an exact critically damped spring.
//! 4. Compress with Ramer–Douglas–Peucker (L∞ error ≤ `tolerance_px`), so the
//!    renderer emits a handful of linear segments instead of one keyframe per
//!    frame.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::edl::{Intensity, MotionPreset};
use crate::index::MediaIndex;
use crate::spring::CriticalSpring;
use crate::time::{FrameRate, US_PER_MS};

/// A crop window in source pixels at a segment-local output time.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CropKeyframe {
    /// Segment-local time in microseconds.
    pub t_us: i64,
    /// Left edge (source px, sub-pixel).
    pub x: f64,
    /// Top edge (source px).
    pub y: f64,
    /// Width (source px).
    pub w: f64,
    /// Height (source px).
    pub h: f64,
}

/// Solved camera path for one segment. Values between keyframes are linearly
/// interpolated; after the last keyframe the last value is held.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CameraPath {
    /// Preset that produced the path.
    pub preset: MotionPreset,
    /// Compressed keyframes (at least one).
    pub keyframes: Vec<CropKeyframe>,
    /// Number of per-frame samples before compression.
    pub solved_frames: u32,
    /// Number of punch-in events triggered (SpringPunchIn only).
    pub punch_ins: u32,
    /// The detected subject in this segment is a picture-in-picture inset
    /// (e.g. a facecam over screen content), so it was not tracked.
    #[serde(default)]
    pub subject_inset: bool,
}

impl CameraPath {
    /// True when the path is a single static window.
    pub fn is_static(&self) -> bool {
        self.keyframes.len() <= 1
    }

    /// Largest zoom factor reached, relative to `base_w`.
    pub fn max_zoom(&self, base_w: f64) -> f64 {
        self.keyframes
            .iter()
            .map(|k| base_w / k.w.max(1e-9))
            .fold(1.0, f64::max)
    }

    /// Window at segment-local time `t_us` (linear interpolation).
    pub fn sample(&self, t_us: i64) -> Option<CropKeyframe> {
        let ks = &self.keyframes;
        let first = *ks.first()?;
        if t_us <= first.t_us {
            return Some(first);
        }
        let i = ks.partition_point(|k| k.t_us <= t_us);
        let a = ks[i - 1];
        let Some(&b) = ks.get(i) else {
            return Some(a);
        };
        let k = (t_us - a.t_us) as f64 / (b.t_us - a.t_us).max(1) as f64;
        let lerp = |p: f64, q: f64| p + (q - p) * k;
        Some(CropKeyframe {
            t_us,
            x: lerp(a.x, b.x),
            y: lerp(a.y, b.y),
            w: lerp(a.w, b.w),
            h: lerp(a.h, b.h),
        })
    }
}

/// Tunables derived from a preset and intensity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraParams {
    /// Spring settle time for centre tracking (s). `None` = static centre.
    pub track_settle_s: Option<f64>,
    /// Spring settle time for zoom (s).
    pub zoom_settle_s: f64,
    /// Baseline zoom.
    pub base_zoom: f64,
    /// Zoom while punched in.
    pub punch_zoom: f64,
    /// How long a punch-in is held after the triggering peak (ms).
    pub punch_hold_ms: u64,
    /// Zoom reached at the end of the clip for `SlowCreep`.
    pub creep_zoom: f64,
}

impl CameraParams {
    /// Parameters for a preset / intensity pair.
    pub fn for_preset(preset: MotionPreset, intensity: Intensity) -> Self {
        let k = match intensity {
            Intensity::Subtle => 0.5,
            Intensity::Medium => 1.0,
            Intensity::Strong => 1.6,
        };
        let base = Self {
            track_settle_s: None,
            zoom_settle_s: 0.6,
            base_zoom: 1.0,
            punch_zoom: 1.0,
            punch_hold_ms: 0,
            creep_zoom: 1.0,
        };
        match preset {
            MotionPreset::None => base,
            MotionPreset::SmoothTrackSubject => Self {
                track_settle_s: Some(1.6 / k),
                base_zoom: 1.0 + 0.04 * k,
                ..base
            },
            MotionPreset::SpringPunchIn => Self {
                track_settle_s: Some(1.2),
                zoom_settle_s: 0.28,
                punch_zoom: 1.0
                    + match intensity {
                        Intensity::Subtle => 0.06,
                        Intensity::Medium => 0.10,
                        Intensity::Strong => 0.15,
                    },
                punch_hold_ms: 1_100,
                ..base
            },
            MotionPreset::SlowCreep => Self {
                track_settle_s: Some(2.5),
                zoom_settle_s: 1.0,
                creep_zoom: 1.0 + 0.08 * k,
                ..base
            },
        }
    }
}

/// Geometry of the output window relative to the source frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Framing {
    /// Source width (px).
    pub src_w: f64,
    /// Source height (px).
    pub src_h: f64,
    /// Output aspect ratio (w / h).
    pub aspect: f64,
}

impl Framing {
    /// Largest window with the output aspect that fits the source (`z = 1`).
    pub fn base_window(&self) -> (f64, f64) {
        if self.src_w / self.src_h > self.aspect {
            (self.src_h * self.aspect, self.src_h)
        } else {
            (self.src_w, self.src_w / self.aspect)
        }
    }

    /// Window for normalised centre `(cx, cy)` and zoom `z`, clamped in-frame.
    pub fn window(&self, cx: f64, cy: f64, z: f64) -> (f64, f64, f64, f64) {
        let (bw, bh) = self.base_window();
        let z = z.max(1.0);
        let (w, h) = (bw / z, bh / z);
        let x = (cx * self.src_w - w / 2.0).clamp(0.0, (self.src_w - w).max(0.0));
        let y = (cy * self.src_h - h / 2.0).clamp(0.0, (self.src_h - h).max(0.0));
        (x, y, w, h)
    }

    /// Clamps a normalised centre so that a `z = 1` window fits.
    pub fn clamp_center(&self, cx: f64, cy: f64) -> (f64, f64) {
        let (bw, bh) = self.base_window();
        let hx = bw / 2.0 / self.src_w;
        let hy = bh / 2.0 / self.src_h;
        (cx.clamp(hx, 1.0 - hx), cy.clamp(hy, 1.0 - hy))
    }
}

/// Inputs for solving one segment.
#[derive(Debug, Clone, Copy)]
pub struct SegmentCamera<'a> {
    /// Media telemetry (subjects, peaks).
    pub index: &'a MediaIndex,
    /// Frame geometry.
    pub framing: Framing,
    /// Preset.
    pub preset: MotionPreset,
    /// Intensity.
    pub intensity: Intensity,
    /// Source time of the segment's first frame (µs).
    pub source_start_us: i64,
    /// Number of output frames in the segment.
    pub frames: u32,
    /// Frame rate.
    pub fps: FrameRate,
    /// Max L∞ error of the compressed path (px).
    pub tolerance_px: f64,
}

/// Solves and compresses the camera path for one segment.
pub fn solve_segment(seg: &SegmentCamera<'_>) -> CameraPath {
    let n = seg.frames.max(1) as usize;
    let p = CameraParams::for_preset(seg.preset, seg.intensity);
    let dt = 1.0 / seg.fps.as_f64();
    let fr = seg.framing;
    let t_ms =
        |k: usize| (seg.source_start_us + seg.fps.frame_to_us(k as i64)) as f64 / US_PER_MS as f64;

    // A tiny "subject" is a facecam inset: tracking it would frame the corner
    // of the screen and discard the content, so hold the centre instead.
    let (seg_start_ms, seg_end_ms) = (t_ms(0), t_ms(n));
    let in_segment: Vec<&crate::index::SubjectBox> = seg
        .index
        .visual
        .subjects
        .iter()
        .filter(|b| (b.t_ms as f64) >= seg_start_ms && (b.t_ms as f64) < seg_end_ms)
        .collect();
    let subject_inset = crate::index::looks_like_inset(&in_segment);

    // Centre targets.
    let center_at = |k: usize| {
        let (cx, cy) = if subject_inset {
            (0.5, 0.5)
        } else {
            seg.index.subject_center_at(t_ms(k)).unwrap_or((0.5, 0.5))
        };
        fr.clamp_center(cx, cy)
    };
    let (cx_t, cy_t): (Vec<f64>, Vec<f64>) = match p.track_settle_s {
        Some(_) => (0..n).map(center_at).unzip(),
        None => {
            let c = center_at(0);
            (vec![c.0; n], vec![c.1; n])
        }
    };

    // Zoom targets.
    let mut punch_ins = 0_u32;
    let z_t: Vec<f64> = match seg.preset {
        MotionPreset::SpringPunchIn => {
            let start_ms = seg.source_start_us as f64 / US_PER_MS as f64;
            let end_ms = t_ms(n);
            // Trigger slightly before the peak so the move lands on the emphasis.
            let lead_ms = 80.0;
            // Only the file's strongest emphasis earns a punch-in: fast,
            // animated speech produces a "peak" every half second.
            let threshold = peak_threshold(&seg.index.audio.peaks, PUNCH_PEAK_PERCENTILE);
            let mut windows: Vec<(f64, f64)> = Vec::new();
            for pk in seg
                .index
                .audio
                .peaks
                .iter()
                .filter(|pk| pk.score >= threshold)
            {
                let t = pk.t_ms as f64 - lead_ms;
                if t < start_ms || t >= end_ms {
                    continue;
                }
                let until = t + p.punch_hold_ms as f64;
                match windows.last_mut() {
                    // Continuing emphasis extends the hold, up to a cap.
                    Some(last) if t <= last.1 => {
                        last.1 = until.min(last.0 + PUNCH_MAX_MS);
                    }
                    // Too soon after the last punch-in: skip, avoid a "breathing" camera.
                    Some(last) if t < last.1 + PUNCH_COOLDOWN_MS => {}
                    _ => windows.push((t, until)),
                }
            }
            punch_ins = windows.len() as u32;
            (0..n)
                .map(|k| {
                    let t = t_ms(k);
                    if windows.iter().any(|&(a, b)| t >= a && t < b) {
                        p.punch_zoom
                    } else {
                        p.base_zoom
                    }
                })
                .collect()
        }
        MotionPreset::SlowCreep => (0..n)
            .map(|k| {
                let u = k as f64 / (n.max(2) - 1) as f64;
                p.base_zoom + (p.creep_zoom - p.base_zoom) * u
            })
            .collect(),
        MotionPreset::None | MotionPreset::SmoothTrackSubject => vec![p.base_zoom; n],
    };

    let track = CriticalSpring::with_settle_time(p.track_settle_s.unwrap_or(1.0));
    let zoom = CriticalSpring::with_settle_time(p.zoom_settle_s);
    let cx = track.simulate(&cx_t, dt);
    let cy = track.simulate(&cy_t, dt);
    let z = zoom.simulate(&z_t, dt);

    let frames: Vec<CropKeyframe> = (0..n)
        .map(|k| {
            let (x, y, w, h) = fr.window(cx[k], cy[k], z[k]);
            CropKeyframe {
                t_us: seg.fps.frame_to_us(k as i64),
                x,
                y,
                w,
                h,
            }
        })
        .collect();

    CameraPath {
        preset: seg.preset,
        keyframes: compress(&frames, seg.tolerance_px),
        solved_frames: n as u32,
        punch_ins,
        subject_inset,
    }
}

/// Peaks must reach this percentile of the file's peak scores to punch in.
pub const PUNCH_PEAK_PERCENTILE: f64 = 0.85;
/// Minimum gap between the end of one punch-in and the start of the next (ms).
pub const PUNCH_COOLDOWN_MS: f64 = 1_500.0;
/// Longest a single punch-in may be held (ms).
pub const PUNCH_MAX_MS: f64 = 3_000.0;

fn peak_threshold(peaks: &[crate::index::ExcitementPeak], q: f64) -> f32 {
    if peaks.len() < 8 {
        // Too few peaks for statistics: all of them count.
        return f32::NEG_INFINITY;
    }
    let mut s: Vec<f32> = peaks.iter().map(|p| p.score).collect();
    s.sort_by(f32::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
}

/// Ramer–Douglas–Peucker over time-parameterised keyframes using the L∞
/// distance across `x, y, w, h`. Iterative (explicit stack), O(n log n) typical.
/// A constant path collapses to a single keyframe.
pub fn compress(frames: &[CropKeyframe], tolerance_px: f64) -> Vec<CropKeyframe> {
    let n = frames.len();
    if n <= 2 {
        return dedupe_constant(frames.to_vec());
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0_usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (fa, fb) = (frames[a], frames[b]);
        let span = (fb.t_us - fa.t_us).max(1) as f64;
        let mut worst = (0.0_f64, 0_usize);
        for (i, f) in frames.iter().enumerate().take(b).skip(a + 1) {
            let k = (f.t_us - fa.t_us) as f64 / span;
            let err = [
                (f.x, fa.x, fb.x),
                (f.y, fa.y, fb.y),
                (f.w, fa.w, fb.w),
                (f.h, fa.h, fb.h),
            ]
            .iter()
            .map(|&(v, p, q)| (v - (p + (q - p) * k)).abs())
            .fold(0.0, f64::max);
            if err > worst.0 {
                worst = (err, i);
            }
        }
        if worst.0 > tolerance_px {
            keep[worst.1] = true;
            stack.push((a, worst.1));
            stack.push((worst.1, b));
        }
    }
    let kept = frames
        .iter()
        .zip(&keep)
        .filter_map(|(f, &k)| k.then_some(*f))
        .collect();
    dedupe_constant(kept)
}

/// Drops trailing keyframes identical to their predecessor (holds are implicit).
/// Uses a numerical epsilon, not the RDP tolerance, so the error bound holds.
fn dedupe_constant(mut ks: Vec<CropKeyframe>) -> Vec<CropKeyframe> {
    const TOL: f64 = 1e-9;
    let tol = TOL;
    let same = |a: &CropKeyframe, b: &CropKeyframe| {
        (a.x - b.x).abs() <= tol
            && (a.y - b.y).abs() <= tol
            && (a.w - b.w).abs() <= tol
            && (a.h - b.h).abs() <= tol
    };
    while ks.len() >= 2 && same(&ks[ks.len() - 1], &ks[ks.len() - 2]) {
        ks.pop();
    }
    ks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{
        AudioTelemetry, ExcitementPeak, INDEX_VERSION, SourceInfo, SubjectBox, SubjectSource,
        VideoStreamInfo, VisualTelemetry,
    };
    use proptest::prelude::*;

    fn index_with(subjects: Vec<SubjectBox>, peaks: Vec<ExcitementPeak>) -> MediaIndex {
        MediaIndex {
            version: INDEX_VERSION,
            source: SourceInfo {
                path: "x.mp4".into(),
                duration_ms: 60_000,
                video: Some(VideoStreamInfo {
                    width: 1920,
                    height: 1080,
                    fps: FrameRate::new(30, 1).unwrap(),
                    codec: "h264".into(),
                }),
                audio: None,
            },
            transcript: vec![],
            audio: AudioTelemetry {
                peaks,
                ..AudioTelemetry::default()
            },
            visual: VisualTelemetry {
                subjects,
                ..VisualTelemetry::default()
            },
        }
    }

    fn subject(t_ms: u64, cx: f32) -> SubjectBox {
        SubjectBox {
            t_ms,
            x: cx - 0.1,
            y: 0.3,
            w: 0.2,
            h: 0.3,
            confidence: 0.9,
            source: SubjectSource::Saliency,
        }
    }

    const FRAMING: Framing = Framing {
        src_w: 1920.0,
        src_h: 1080.0,
        aspect: 9.0 / 16.0,
    };

    fn seg<'a>(idx: &'a MediaIndex, preset: MotionPreset, frames: u32) -> SegmentCamera<'a> {
        SegmentCamera {
            index: idx,
            framing: FRAMING,
            preset,
            intensity: Intensity::Medium,
            source_start_us: 0,
            frames,
            fps: FrameRate::new(30, 1).unwrap(),
            tolerance_px: 0.25,
        }
    }

    #[test]
    fn base_window_is_full_height_for_vertical() {
        let (w, h) = FRAMING.base_window();
        assert!((w - 607.5).abs() < 1e-9 && (h - 1080.0).abs() < 1e-9);
    }

    #[test]
    fn static_preset_collapses_to_one_keyframe() {
        let idx = index_with(vec![], vec![]);
        let path = solve_segment(&seg(&idx, MotionPreset::None, 300));
        assert_eq!(path.keyframes.len(), 1);
        let k = path.keyframes[0];
        assert!((k.x - (960.0 - 303.75)).abs() < 1e-9);
    }

    #[test]
    fn tracking_follows_subject_smoothly() {
        let idx = index_with(vec![subject(0, 0.3), subject(2_000, 0.7)], vec![]);
        let path = solve_segment(&seg(&idx, MotionPreset::SmoothTrackSubject, 180));
        assert!(path.keyframes.len() > 2);
        let start = path.sample(0).unwrap();
        let end = path.sample(5_999_999).unwrap();
        assert!(end.x > start.x + 200.0, "{start:?} -> {end:?}");
        // Compressed path still has bounded error vs. a dense re-solve.
        assert!(path.keyframes.len() < 90);
    }

    #[test]
    fn small_faces_moving_around_a_wide_shot_are_tracked() {
        // Regression (real event interview): three people in a wide shot,
        // faces 6-8% of the frame wide. They are not a facecam.
        let face = |t_ms, x| SubjectBox {
            t_ms,
            x,
            y: 0.28,
            w: 0.065,
            h: 0.18,
            confidence: 0.99,
            source: SubjectSource::Face,
        };
        let boxes: Vec<SubjectBox> = (0..30)
            .map(|k| face(k * 200, if k < 15 { 0.25 } else { 0.8 }))
            .collect();
        let idx = index_with(boxes, vec![]);
        let path = solve_segment(&seg(&idx, MotionPreset::SmoothTrackSubject, 180));
        assert!(!path.subject_inset);
        let refs: Vec<&SubjectBox> = idx.visual.subjects.iter().collect();
        assert!(!crate::index::looks_like_inset(&refs));
    }

    #[test]
    fn facecam_inset_is_not_tracked() {
        // Regression (real reaction video): UltraFace found the 5%-wide
        // facecam in the top-left corner; tracking it framed the corner and
        // cropped away the content being discussed.
        let inset = |t_ms| SubjectBox {
            t_ms,
            x: 0.02,
            y: 0.03,
            w: 0.055,
            h: 0.1,
            confidence: 0.9,
            source: SubjectSource::Onnx,
        };
        let idx = index_with((0..30).map(|k| inset(k * 200)).collect(), vec![]);
        let path = solve_segment(&seg(&idx, MotionPreset::SmoothTrackSubject, 180));
        assert!(path.subject_inset);
        assert_eq!(path.keyframes.len(), 1, "camera holds still");
        let k = path.keyframes[0];
        assert!((k.x + k.w / 2.0 - 960.0).abs() < 1e-6, "centred: {k:?}");
    }

    #[test]
    fn punch_ins_need_strong_peaks_and_cooldown() {
        // Regression (real commentary): 1.6 peaks/s made SpringPunchIn zoom
        // almost continuously. Only top-15% peaks fire, with a cooldown.
        let mut peaks: Vec<ExcitementPeak> = (0..40)
            .map(|k| ExcitementPeak {
                t_ms: 100 + k * 150,
                score: 1.6,
            })
            .collect();
        for (t, s) in [(1_000, 4.0), (1_600, 4.0), (2_900, 4.0), (5_000, 4.0)] {
            peaks.push(ExcitementPeak { t_ms: t, score: s });
        }
        peaks.sort_by_key(|p| p.t_ms);
        let idx = index_with(vec![], peaks);
        let path = solve_segment(&seg(&idx, MotionPreset::SpringPunchIn, 210));
        // 1000 opens a window (to 2020, extended by 1600 to 2620); 2900 is
        // inside the 1.5 s cooldown; 5000 opens the second window.
        assert_eq!(path.punch_ins, 2, "{path:?}");
    }

    #[test]
    fn punch_in_triggers_on_peaks_and_returns() {
        let peaks = vec![ExcitementPeak {
            t_ms: 1_000,
            score: 3.0,
        }];
        let idx = index_with(vec![], peaks);
        let path = solve_segment(&seg(&idx, MotionPreset::SpringPunchIn, 120));
        assert_eq!(path.punch_ins, 1);
        let (bw, _) = FRAMING.base_window();
        let peak_zoom = path.max_zoom(bw);
        assert!(peak_zoom > 1.09 && peak_zoom <= 1.12 + 1e-9, "{peak_zoom}");
        let before = path.sample(800_000).unwrap();
        assert!((bw / before.w - 1.0).abs() < 1e-3);
        let after = path.sample(3_900_000).unwrap();
        assert!(bw / after.w < 1.01, "zoom eases back out");
    }

    #[test]
    fn creep_zooms_monotonically() {
        let idx = index_with(vec![], vec![]);
        let path = solve_segment(&seg(&idx, MotionPreset::SlowCreep, 150));
        let ws: Vec<f64> = (0..150)
            .map(|k| path.sample(k * 33_333).unwrap().w)
            .collect();
        for w in ws.windows(2) {
            assert!(w[1] <= w[0] + 0.25 + 1e-9);
        }
        assert!(ws[149] < ws[0] - 20.0);
    }

    proptest! {
        /// Crop windows stay inside the source and keep the output aspect for
        /// arbitrary subject trajectories, peaks and presets.
        #[test]
        fn windows_are_bounded_and_aspect_correct(
            xs in prop::collection::vec(-0.5f32..1.5, 1..12),
            peak_ms in prop::collection::vec(0u64..6_000, 0..8),
            preset_ix in 0usize..4,
            src_w in 320u32..4000,
            src_h in 240u32..4000,
            aspect_ix in 0usize..3,
        ) {
            let preset = [MotionPreset::None, MotionPreset::SpringPunchIn,
                MotionPreset::SmoothTrackSubject, MotionPreset::SlowCreep][preset_ix];
            let aspect = [9.0 / 16.0, 1.0, 16.0 / 9.0][aspect_ix];
            let subjects = xs.iter().enumerate()
                .map(|(i, &x)| subject(i as u64 * 500, x)).collect();
            let peaks = peak_ms.iter().map(|&t| ExcitementPeak { t_ms: t, score: 2.0 }).collect();
            let idx = index_with(subjects, peaks);
            let framing = Framing { src_w: f64::from(src_w), src_h: f64::from(src_h), aspect };
            let s = SegmentCamera { framing, ..seg(&idx, preset, 180) };
            let path = solve_segment(&s);
            prop_assert!(!path.keyframes.is_empty());
            for w in path.keyframes.windows(2) {
                prop_assert!(w[1].t_us > w[0].t_us);
            }
            for k in 0..180 {
                let c = path.sample(k * 33_333).unwrap();
                let eps = 1e-6 * framing.src_w.max(framing.src_h);
                prop_assert!(c.x >= -eps && c.y >= -eps);
                prop_assert!(c.x + c.w <= framing.src_w + eps);
                prop_assert!(c.y + c.h <= framing.src_h + eps);
                prop_assert!(c.w > 0.0 && c.h > 0.0);
                prop_assert!((c.w / c.h - aspect).abs() < 1e-6 * aspect.max(1.0) * 10.0);
            }
        }

        /// RDP never exceeds its tolerance on the original samples.
        #[test]
        fn compression_error_is_bounded(
            ys in prop::collection::vec(0.0f64..500.0, 2..200),
            tol in 0.05f64..5.0,
        ) {
            let frames: Vec<CropKeyframe> = ys.iter().enumerate().map(|(i, &y)| CropKeyframe {
                t_us: i as i64 * 33_333, x: y * 0.5, y, w: 100.0, h: 100.0,
            }).collect();
            let path = CameraPath {
                preset: MotionPreset::None,
                keyframes: compress(&frames, tol),
                solved_frames: frames.len() as u32,
                punch_ins: 0,
                subject_inset: false,
            };
            for f in &frames {
                let s = path.sample(f.t_us).unwrap();
                prop_assert!((s.x - f.x).abs() <= tol + 1e-9);
                prop_assert!((s.y - f.y).abs() <= tol + 1e-9);
            }
        }
    }
}
