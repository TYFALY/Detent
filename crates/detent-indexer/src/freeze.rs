//! Frozen-picture detection and per-second brightness.
//!
//! Stream lag and dropped frames show up as runs of *repeated* frames while
//! the footage around them moves: the picture stops, the clock does not. A
//! frame counts as repeated when it is all but bit-identical to the previous
//! analysis frame. A run of repeats becomes a [`FreezeInterval`] only when it
//! lasts at least `min_ms` and the frames just before *and* just after it
//! are clearly moving (median frame difference above `motion_threshold`),
//! so a genuinely still shot, a slide or a paused screen recording is never
//! reported as lag.

use detent_core::index::FreezeInterval;
use detent_core::time::{FrameRate, us_to_ms};

use crate::source::RgbFrame;

/// Freeze detector tunables (analysis frames are ~160 px wide RGB).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreezeConfig {
    /// Mean absolute byte difference below which a frame may be a repeat.
    pub max_mean_diff: f32,
    /// A byte counts as changed when it differs by more than this.
    pub changed_level: u8,
    /// A repeat may have at most this fraction of changed bytes.
    pub max_changed_frac: f32,
    /// Shortest reported freeze (ms).
    pub min_ms: u64,
    /// Median frame difference that counts as "moving" around a freeze.
    pub motion_threshold: f32,
    /// Frames examined on each side of a freeze for motion.
    pub context_frames: usize,
    /// Repeat runs separated by at most this many new frames are one freeze
    /// (a stuttering stream).
    pub stutter_frames: usize,
}

impl Default for FreezeConfig {
    fn default() -> Self {
        Self {
            max_mean_diff: 0.35,
            changed_level: 10,
            max_changed_frac: 0.001,
            min_ms: 150,
            motion_threshold: 1.5,
            context_frames: 8,
            stutter_frames: 2,
        }
    }
}

/// Streaming detector: feed every analysis frame in order.
#[derive(Debug, Clone)]
pub struct FreezeDetector {
    cfg: FreezeConfig,
    prev: Option<Vec<u8>>,
    /// Per frame: mean absolute difference to the previous frame.
    diffs: Vec<f32>,
    /// Per frame: repeated (all but identical to the previous one).
    repeat: Vec<bool>,
    /// Per second: (luma sum, frames).
    luma: Vec<(f64, u32)>,
}

impl FreezeDetector {
    /// Creates a detector.
    pub fn new(cfg: FreezeConfig) -> Self {
        Self {
            cfg,
            prev: None,
            diffs: Vec::new(),
            repeat: Vec::new(),
            luma: Vec::new(),
        }
    }

    /// Observes the next frame, shown at `t_ms`.
    pub fn push(&mut self, frame: &RgbFrame, t_ms: u64) {
        let data = &frame.data;
        let mut luma = 0_u64;
        for px in data.as_chunks::<3>().0 {
            luma += 299 * u64::from(px[0]) + 587 * u64::from(px[1]) + 114 * u64::from(px[2]);
        }
        let px = (data.len() / 3).max(1) as f64;
        let sec = (t_ms / 1_000) as usize;
        if self.luma.len() <= sec {
            self.luma.resize(sec + 1, (0.0, 0));
        }
        self.luma[sec].0 += luma as f64 / (px * 255_000.0);
        self.luma[sec].1 += 1;

        let (diff, repeat) = match &self.prev {
            Some(p) if p.len() == data.len() && !data.is_empty() => {
                let mut sum = 0_u64;
                let mut changed = 0_usize;
                for (a, b) in p.iter().zip(data) {
                    let d = a.abs_diff(*b);
                    sum += u64::from(d);
                    changed += usize::from(d > self.cfg.changed_level);
                }
                let mean = sum as f32 / data.len() as f32;
                let frac = changed as f32 / data.len() as f32;
                (
                    mean,
                    mean <= self.cfg.max_mean_diff && frac <= self.cfg.max_changed_frac,
                )
            }
            _ => (f32::INFINITY, false),
        };
        self.diffs.push(diff);
        self.repeat.push(repeat);
        match &mut self.prev {
            Some(p) if p.len() == data.len() => p.copy_from_slice(data),
            _ => self.prev = Some(data.clone()),
        }
    }

    /// Detected freezes (frame-accurate at `fps`) and the mean luma of every
    /// second of video.
    pub fn finish(self, fps: FrameRate) -> (Vec<FreezeInterval>, Vec<f32>) {
        let n = self.repeat.len();
        // Runs of repeated frames `[start, end)`; a lagging stream often
        // stutters (freeze, one or two new frames, freeze), so runs that
        // close together form one freeze.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut k = 0;
        while k < n {
            if !self.repeat[k] {
                k += 1;
                continue;
            }
            let start = k;
            while k < n && self.repeat[k] {
                k += 1;
            }
            match runs.last_mut() {
                Some(r) if start - r.1 <= self.cfg.stutter_frames => r.1 = k,
                _ => runs.push((start, k)),
            }
        }
        let mut out = Vec::new();
        for (start, end) in runs {
            // Frames start..end repeat (or barely move); frame `end` moves again.
            let (s_us, e_us) = (fps.frame_to_us(start as i64), fps.frame_to_us(end as i64));
            if us_to_ms(e_us - s_us) < self.cfg.min_ms || end >= n {
                continue;
            }
            let c = self.cfg.context_frames;
            let before = median(&self.diffs[start.saturating_sub(c)..start]);
            let after = median(&self.diffs[end..(end + c).min(n)]);
            let moving = |m: Option<f32>| m.is_some_and(|m| m >= self.cfg.motion_threshold);
            if moving(before) && moving(after) {
                out.push(FreezeInterval {
                    start_ms: us_to_ms(s_us),
                    end_ms: us_to_ms(e_us),
                });
            }
        }
        let luma = self
            .luma
            .into_iter()
            .map(|(s, c)| {
                if c == 0 {
                    0.0
                } else {
                    ((s / f64::from(c)) * 1_000.0).round() as f32 / 1_000.0
                }
            })
            .collect();
        (out, luma)
    }
}

/// Median of the finite values (needs at least 3).
fn median(v: &[f32]) -> Option<f32> {
    let mut x: Vec<f32> = v.iter().copied().filter(|d| d.is_finite()).collect();
    if x.len() < 3 {
        return None;
    }
    x.sort_by(f32::total_cmp);
    Some(x[x.len() / 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame whose content depends on `k` (moving texture).
    fn moving(k: u32) -> RgbFrame {
        let mut f = RgbFrame::new(32, 18);
        for (i, b) in f.data.iter_mut().enumerate() {
            *b = ((i as u32 * 7 + k * 31) % 251) as u8;
        }
        f
    }

    fn run(frames: &[RgbFrame]) -> (Vec<FreezeInterval>, Vec<f32>) {
        let fps = FrameRate::new(30, 1).unwrap();
        let mut d = FreezeDetector::new(FreezeConfig::default());
        for (k, f) in frames.iter().enumerate() {
            d.push(f, us_to_ms(fps.frame_to_us(k as i64)));
        }
        d.finish(fps)
    }

    #[test]
    fn finds_a_lag_freeze_in_moving_footage() {
        // Moving 0-29, frame 29 repeated for 30..45 (0.5 s), moving again.
        let mut frames: Vec<RgbFrame> = (0..30).map(moving).collect();
        frames.extend((30..45).map(|_| moving(29)));
        frames.extend((45..90).map(moving));
        let (f, luma) = run(&frames);
        assert_eq!(
            f,
            vec![FreezeInterval {
                start_ms: 1_000,
                end_ms: 1_500
            }]
        );
        assert_eq!(luma.len(), 3);
        assert!(luma.iter().all(|l| (0.0..=1.0).contains(l)));
    }

    #[test]
    fn a_stutter_is_one_freeze() {
        // Freeze 0.3 s, one new frame, freeze 1 s: one freeze 1.0-2.333 s.
        let mut frames: Vec<RgbFrame> = (0..30).map(moving).collect();
        frames.extend((30..39).map(|_| moving(29)));
        frames.push(moving(500));
        frames.extend((40..70).map(|_| moving(500)));
        frames.extend((70..110).map(moving));
        let (f, _) = run(&frames);
        assert_eq!(
            f,
            vec![FreezeInterval {
                start_ms: 1_000,
                end_ms: 2_333
            }]
        );
    }

    #[test]
    fn ignores_short_repeats_and_still_shots() {
        // A 3-frame repeat (100 ms) is too short.
        let mut frames: Vec<RgbFrame> = (0..30).map(moving).collect();
        frames.extend((0..3).map(|_| moving(29)));
        frames.extend((33..60).map(moving));
        assert!(run(&frames).0.is_empty());
        // A still shot that cuts to another still shot is not lag.
        let mut still: Vec<RgbFrame> = (0..30)
            .map(|_| RgbFrame::solid(32, 18, [40, 40, 40]))
            .collect();
        still.extend((0..30).map(|_| RgbFrame::solid(32, 18, [200, 30, 30])));
        assert!(run(&still).0.is_empty());
    }

    #[test]
    fn luma_tracks_brightness() {
        let dark: Vec<RgbFrame> = (0..30)
            .map(|_| RgbFrame::solid(16, 9, [20, 20, 20]))
            .collect();
        let (_, l) = run(&dark);
        assert!((l[0] - 20.0 / 255.0).abs() < 0.002, "{l:?}");
    }
}
