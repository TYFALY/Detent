//! Shot-boundary detection from colour-histogram deltas.
//!
//! Each frame is summarised by a 3 x 16-bin RGB histogram. The distance
//! between consecutive frames is the mean per-channel total-variation
//! distance (in `[0, 1]`).
//!
//! * **Hard cuts**: the distance exceeds both an absolute floor and
//!   `mean + sigma * std` of the recent distances, which adapts to handheld
//!   footage and noisy sources without per-file tuning.
//! * **Gradual transitions** (dissolves, fades through black or white) use
//!   twin comparison: a run of frames whose deltas all sit well above the
//!   quiet baseline, bracketed by stable picture, lasting 4 frames to ~2 s,
//!   whose *accumulated* change (first frame vs last) is as large as a cut.
//!   The boundary is placed mid-transition. A fade out and back in becomes
//!   one boundary, centred on the hold between them. Slow drift (exposure,
//!   lighting) never qualifies, because it raises the baseline it is
//!   measured against.

use std::collections::VecDeque;

use detent_core::index::ShotBoundary;
use detent_core::time::FrameRate;

use crate::source::RgbFrame;

/// Detector configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShotConfig {
    /// Histogram bins per channel.
    pub bins: usize,
    /// Rolling window (frames) for the adaptive threshold.
    pub window: usize,
    /// Standard deviations above the rolling mean.
    pub sigma: f32,
    /// Absolute minimum distance for a cut.
    pub min_distance: f32,
    /// Minimum shot length in frames (suppresses flashes / double triggers).
    pub min_shot_frames: u64,
    /// Detect dissolves and fades as well as hard cuts.
    pub gradual: bool,
    /// Per-frame delta floor for a frame to belong to a transition.
    pub gradual_low: f32,
    /// A transition frame's delta must also exceed this multiple of the
    /// quiet baseline (mean delta outside transitions).
    pub gradual_baseline_factor: f32,
    /// Minimum accumulated histogram distance (first vs last frame).
    pub gradual_min_distance: f32,
    /// Minimum mean absolute change of an 8 x 8 colour grid (first vs last
    /// frame, in `[0, 1]`). Rejects histogram-bin artefacts of slow drift on
    /// flat pictures, which move the histogram but barely change the image.
    pub gradual_min_pixel: f32,
    /// Transition length bounds in frames.
    pub gradual_min_frames: u64,
    /// See [`ShotConfig::gradual_min_frames`].
    pub gradual_max_frames: u64,
}

impl Default for ShotConfig {
    fn default() -> Self {
        Self {
            bins: 16,
            window: 24,
            sigma: 4.0,
            min_distance: 0.30,
            min_shot_frames: 6,
            gradual: true,
            gradual_low: 0.025,
            gradual_baseline_factor: 3.0,
            gradual_min_distance: 0.35,
            gradual_min_pixel: 0.06,
            gradual_min_frames: 4,
            gradual_max_frames: 60,
        }
    }
}

/// Streaming shot detector.
#[derive(Debug, Clone)]
pub struct ShotDetector {
    cfg: ShotConfig,
    prev: Vec<f32>,
    cur: Vec<f32>,
    prev_grid: Vec<f32>,
    cur_grid: Vec<f32>,
    has_prev: bool,
    recent: VecDeque<f32>,
    frame: u64,
    last_cut: u64,
    cuts: Vec<(u64, f32)>,
    /// Deltas outside transitions (the baseline for gradual detection).
    quiet: VecDeque<f32>,
    run: Option<Run>,
    /// End frame and `cuts` index of the last gradual boundary.
    last_gradual: Option<(u64, usize)>,
    /// A hard-cut candidate awaiting the next frame.
    pending: Option<Pending>,
}

#[derive(Debug, Clone)]
struct Pending {
    frame: u64,
    score: f32,
    d: f32,
    before: Vec<f32>,
    before_grid: Vec<f32>,
    hist: Vec<f32>,
    grid: Vec<f32>,
}

/// A candidate gradual transition in progress.
#[derive(Debug, Clone)]
struct Run {
    /// First frame that changed.
    start: u64,
    /// Histogram and colour grid just before the run.
    before: Vec<f32>,
    before_grid: Vec<f32>,
    /// Last frame above the threshold, and its histogram.
    last: u64,
    last_hist: Vec<f32>,
    last_grid: Vec<f32>,
    /// Consecutive calm frames since `last`.
    calm: u64,
    /// Deltas inside the run.
    deltas: Vec<f32>,
}

impl ShotDetector {
    /// Creates a detector.
    pub fn new(cfg: ShotConfig) -> Self {
        let bins = cfg.bins.clamp(2, 256);
        Self {
            cfg: ShotConfig { bins, ..cfg },
            prev: vec![0.0; bins * 3],
            cur: vec![0.0; bins * 3],
            prev_grid: vec![0.0; GRID * GRID * 3],
            cur_grid: vec![0.0; GRID * GRID * 3],
            has_prev: false,
            recent: VecDeque::with_capacity(cfg.window + 1),
            frame: 0,
            last_cut: 0,
            cuts: Vec::new(),
            quiet: VecDeque::with_capacity(cfg.window + 1),
            run: None,
            last_gradual: None,
            pending: None,
        }
    }

    /// Feeds the next frame.
    pub fn push(&mut self, frame: &RgbFrame) {
        histogram(frame, self.cfg.bins, &mut self.cur);
        grid(frame, &mut self.cur_grid);
        if self.has_prev {
            let d = distance(&self.prev, &self.cur);
            // A cut is a lone spike: if the next delta stays high, or the
            // frame before was already changing, the jump is part of a fade
            // or dissolve instead.
            if let Some(p) = self.pending.take() {
                let in_run = self.run.as_ref().is_some_and(|r| r.last + 1 == p.frame);
                if d < 0.5 * p.d && !in_run {
                    self.confirm(p.frame, p.score);
                } else {
                    self.absorb(p);
                }
            }
            let (mean, std) = mean_std(&self.recent);
            let adaptive = mean + self.cfg.sigma * std;
            let warmed = self.recent.len() >= 3;
            if d >= self.cfg.min_distance
                && (!warmed || d > adaptive)
                && self.frame - self.last_cut >= self.cfg.min_shot_frames
            {
                let score = d / adaptive.max(self.cfg.min_distance);
                if self.cfg.gradual {
                    self.pending = Some(Pending {
                        frame: self.frame,
                        score,
                        d,
                        before: self.prev.clone(),
                        before_grid: self.prev_grid.clone(),
                        hist: self.cur.clone(),
                        grid: self.cur_grid.clone(),
                    });
                } else {
                    self.confirm(self.frame, score);
                }
            } else {
                self.recent.push_back(d);
                if self.recent.len() > self.cfg.window {
                    self.recent.pop_front();
                }
                if self.cfg.gradual {
                    self.gradual(d);
                }
            }
        }
        std::mem::swap(&mut self.prev, &mut self.cur);
        std::mem::swap(&mut self.prev_grid, &mut self.cur_grid);
        self.has_prev = true;
        self.frame += 1;
    }

    fn confirm(&mut self, frame: u64, score: f32) {
        self.cuts.push((frame, score));
        self.last_cut = frame;
        // A cut is not representative of in-shot motion.
        self.recent.clear();
        self.run = None;
        self.last_gradual = None;
    }

    /// Folds a rejected cut candidate into a gradual run.
    fn absorb(&mut self, p: Pending) {
        match &mut self.run {
            Some(r) => {
                r.last = p.frame;
                r.last_hist = p.hist;
                r.last_grid = p.grid;
                r.calm = 0;
                r.deltas.push(p.d);
            }
            None => {
                self.run = Some(Run {
                    start: p.frame,
                    before: p.before,
                    before_grid: p.before_grid,
                    last: p.frame,
                    last_hist: p.hist,
                    last_grid: p.grid,
                    calm: 0,
                    deltas: vec![p.d],
                });
            }
        }
    }

    fn gradual(&mut self, d: f32) {
        let base = mean_std(&self.quiet).0;
        let low = self
            .cfg
            .gradual_low
            .max(self.cfg.gradual_baseline_factor * base);
        let warmed = self.quiet.len() >= 3;
        let active = warmed && d >= low;
        match (&mut self.run, active) {
            (None, true) => {
                self.run = Some(Run {
                    start: self.frame,
                    before: self.prev.clone(),
                    before_grid: self.prev_grid.clone(),
                    last: self.frame,
                    last_hist: self.cur.clone(),
                    last_grid: self.cur_grid.clone(),
                    calm: 0,
                    deltas: vec![d],
                });
            }
            (Some(r), true) => {
                r.last = self.frame;
                r.last_hist.copy_from_slice(&self.cur);
                r.last_grid.copy_from_slice(&self.cur_grid);
                r.calm = 0;
                r.deltas.push(d);
                if r.last - r.start + 1 > self.cfg.gradual_max_frames {
                    // Too slow for a transition: sustained motion or drift.
                    // It becomes the new baseline, so it cannot re-trigger.
                    let deltas = std::mem::take(&mut r.deltas);
                    self.run = None;
                    for x in deltas {
                        self.quiet.push_back(x);
                        if self.quiet.len() > self.cfg.window {
                            self.quiet.pop_front();
                        }
                    }
                    return;
                }
            }
            (Some(r), false) => {
                r.calm += 1;
                // Two calm frames close the run.
                if r.calm >= 2
                    && let Some(r) = self.run.take()
                {
                    self.close(r);
                }
            }
            (None, false) => {}
        }
        if self.run.is_none() && !active {
            self.quiet.push_back(d);
            if self.quiet.len() > self.cfg.window {
                self.quiet.pop_front();
            }
        }
    }

    fn close(&mut self, r: Run) {
        let len = r.last - r.start + 1;
        let change = distance(&r.before, &r.last_hist);
        let pixel = mean_abs_diff(&r.before_grid, &r.last_grid);
        if len < self.cfg.gradual_min_frames
            || change < self.cfg.gradual_min_distance
            || pixel < self.cfg.gradual_min_pixel
        {
            return;
        }
        let score = ((change / self.cfg.gradual_min_distance) * 100.0).round() / 100.0;
        // Fade out, short hold, fade in: one boundary, centred on the hold.
        if let Some((end, idx)) = self.last_gradual
            && r.start.saturating_sub(end) <= self.cfg.window as u64
            && idx + 1 == self.cuts.len()
        {
            let mid = (end + r.start).div_ceil(2);
            let prev = self.cuts[idx].1;
            self.cuts[idx] = (mid, prev.max(score));
            self.last_gradual = Some((r.last, idx));
            self.last_cut = self.last_cut.max(mid);
            return;
        }
        let mid = (r.start + r.last).div_ceil(2);
        if mid.saturating_sub(self.last_cut) < self.cfg.min_shot_frames && !self.cuts.is_empty() {
            return;
        }
        self.cuts.push((mid, score));
        self.last_gradual = Some((r.last, self.cuts.len() - 1));
        self.last_cut = mid;
    }

    /// Frames consumed.
    pub fn frames(&self) -> u64 {
        self.frame
    }

    /// Returns detected boundaries with timestamps derived from `fps`.
    pub fn finish(mut self, fps: FrameRate) -> Vec<ShotBoundary> {
        if let Some(p) = self.pending.take() {
            self.confirm(p.frame, p.score);
        }
        self.cuts
            .into_iter()
            .map(|(frame, score)| ShotBoundary {
                t_ms: detent_core::time::us_to_ms(fps.frame_to_us(frame as i64)),
                frame,
                score: (score * 100.0).round() / 100.0,
            })
            .collect()
    }
}

fn histogram(frame: &RgbFrame, bins: usize, out: &mut [f32]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    let shift = 256 / bins;
    for px in frame.data.as_chunks::<3>().0 {
        for (c, &v) in px.iter().enumerate() {
            let b = (usize::from(v) / shift).min(bins - 1);
            out[c * bins + b] += 1.0;
        }
    }
    let n = (frame.data.len() / 3).max(1) as f32;
    out.iter_mut().for_each(|v| *v /= n);
}

/// Cells per side of the colour grid used to confirm gradual transitions.
const GRID: usize = 8;

/// Mean colour of each cell of a `GRID x GRID` grid, in `[0, 1]`.
fn grid(frame: &RgbFrame, out: &mut [f32]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    let (w, h) = (frame.width as usize, frame.height as usize);
    if w == 0 || h == 0 || frame.data.len() < w * h * 3 {
        return;
    }
    // Cell column boundaries, so the inner loop is plain summation.
    let xs: [usize; GRID + 1] = std::array::from_fn(|c| c * w / GRID);
    let mut sums = [[0_u32; 3]; GRID * GRID];
    let mut rows = [0_usize; GRID];
    for (y, row) in frame.data.chunks_exact(w * 3).take(h).enumerate() {
        let cy = (y * GRID / h).min(GRID - 1);
        rows[cy] += 1;
        for cx in 0..GRID {
            let cell = &mut sums[cy * GRID + cx];
            for px in row[xs[cx] * 3..xs[cx + 1] * 3].as_chunks::<3>().0 {
                cell[0] += u32::from(px[0]);
                cell[1] += u32::from(px[1]);
                cell[2] += u32::from(px[2]);
            }
        }
    }
    for (cy, &rows) in rows.iter().enumerate() {
        for cx in 0..GRID {
            let n = (rows * (xs[cx + 1] - xs[cx])).max(1) as f32 * 255.0;
            let cell = cy * GRID + cx;
            for c in 0..3 {
                out[cell * 3 + c] = sums[cell][c] as f32 / n;
            }
        }
    }
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len().max(1) as f32
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    let tv: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    // Three channels, each L1 distance in [0, 2] → mean total variation in [0, 1].
    tv / 6.0
}

fn mean_std(xs: &VecDeque<f32>) -> (f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noisy(rgb: [u8; 3], seed: u32) -> RgbFrame {
        textured(rgb, seed, 9)
    }

    /// A frame with per-pixel noise of `amp` levels, like real texture.
    fn textured(rgb: [u8; 3], seed: u32, amp: u32) -> RgbFrame {
        let mut f = RgbFrame::solid(32, 18, rgb);
        let mut s = seed.max(1);
        for v in &mut f.data {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *v = v.saturating_add((s % amp) as u8);
        }
        f
    }

    /// Pixel-wise mix of two frames, `a` weight `1 - w`.
    fn mix(a: &RgbFrame, b: &RgbFrame, w: f32) -> RgbFrame {
        let mut f = a.clone();
        for (o, (x, y)) in f.data.iter_mut().zip(a.data.iter().zip(&b.data)) {
            *o = (f32::from(*x) * (1.0 - w) + f32::from(*y) * w).round() as u8;
        }
        f
    }

    fn hard_only() -> ShotConfig {
        ShotConfig {
            gradual: false,
            ..ShotConfig::default()
        }
    }

    #[test]
    fn detects_a_dissolve_mid_transition() {
        let mut d = ShotDetector::new(ShotConfig::default());
        let a = textured([30, 60, 120], 7, 40);
        let b = textured([200, 150, 40], 99, 40);
        for i in 0..40 {
            d.push(&textured([30, 60, 120], 1 + i, 40));
        }
        // 20-frame cross-dissolve.
        for k in 1..=20 {
            d.push(&mix(&a, &b, k as f32 / 21.0));
        }
        for i in 0..40 {
            d.push(&textured([200, 150, 40], 500 + i, 40));
        }
        let shots = d.finish(FrameRate::new(25, 1).unwrap());
        assert_eq!(shots.len(), 1, "{shots:?}");
        assert!((45..=56).contains(&shots[0].frame), "{shots:?}");

        // The hard-cut detector alone misses it.
        let mut h = ShotDetector::new(hard_only());
        for i in 0..40 {
            h.push(&textured([30, 60, 120], 1 + i, 40));
        }
        for k in 1..=20 {
            h.push(&mix(&a, &b, k as f32 / 21.0));
        }
        assert!(h.finish(FrameRate::new(25, 1).unwrap()).is_empty());
    }

    #[test]
    fn fade_through_black_is_one_boundary() {
        let mut d = ShotDetector::new(ShotConfig::default());
        let a = textured([60, 140, 90], 3, 40);
        let b = textured([180, 80, 160], 4, 40);
        let black = RgbFrame::solid(32, 18, [0, 0, 0]);
        for i in 0..40 {
            d.push(&textured([60, 140, 90], 10 + i, 40));
        }
        for k in 1..=12 {
            d.push(&mix(&a, &black, k as f32 / 12.0));
        }
        for _ in 0..6 {
            d.push(&black);
        }
        for k in 1..=12 {
            d.push(&mix(&black, &b, k as f32 / 12.0));
        }
        for i in 0..40 {
            d.push(&textured([180, 80, 160], 300 + i, 40));
        }
        let shots = d.finish(FrameRate::new(25, 1).unwrap());
        assert_eq!(shots.len(), 1, "{shots:?}");
        // Centred on the black hold (frames 52..58).
        assert!((50..=60).contains(&shots[0].frame), "{shots:?}");
    }

    #[test]
    fn slow_pan_like_change_is_not_a_transition() {
        // 150 frames of steady change, far longer than any dissolve.
        let mut d = ShotDetector::new(ShotConfig::default());
        let a = textured([30, 60, 120], 7, 40);
        let b = textured([200, 150, 40], 99, 40);
        for i in 0..30 {
            d.push(&textured([30, 60, 120], 1 + i, 40));
        }
        for k in 1..=150 {
            d.push(&mix(&a, &b, k as f32 / 151.0));
        }
        for i in 0..30 {
            d.push(&textured([200, 150, 40], 700 + i, 40));
        }
        assert!(d.finish(FrameRate::new(25, 1).unwrap()).is_empty());
    }

    #[test]
    fn detects_hard_cuts_only() {
        let mut d = ShotDetector::new(ShotConfig::default());
        for i in 0..40 {
            d.push(&noisy([40, 60, 80], i));
        }
        for i in 0..40 {
            d.push(&noisy([220, 180, 30], 100 + i));
        }
        for i in 0..40 {
            d.push(&noisy([10, 200, 90], 200 + i));
        }
        let shots = d.finish(FrameRate::new(25, 1).unwrap());
        let frames: Vec<u64> = shots.iter().map(|s| s.frame).collect();
        assert_eq!(frames, vec![40, 80]);
        assert_eq!(shots[0].t_ms, 1_600);
    }

    #[test]
    fn ignores_gradual_brightness_drift() {
        let mut d = ShotDetector::new(ShotConfig::default());
        for i in 0..200_u32 {
            let v = (i / 2) as u8;
            d.push(&noisy([v, v, v], i));
        }
        let shots = d.finish(FrameRate::new(30, 1).unwrap());
        assert!(shots.is_empty(), "{shots:?}");
    }

    #[test]
    fn enforces_min_shot_length() {
        let mut d = ShotDetector::new(ShotConfig::default());
        for i in 0..20 {
            d.push(&noisy([0, 0, 0], i));
        }
        // Two-frame white flash.
        d.push(&noisy([250, 250, 250], 1));
        d.push(&noisy([250, 250, 250], 2));
        for i in 0..20 {
            d.push(&noisy([0, 0, 0], 50 + i));
        }
        let shots = d.finish(FrameRate::new(30, 1).unwrap());
        assert_eq!(shots.len(), 1, "{shots:?}");
    }
}
