//! Hard-cut detection from colour-histogram deltas with an adaptive threshold.
//!
//! Each frame is summarised by a 3 x 16-bin RGB histogram. The distance
//! between consecutive frames is the mean per-channel total-variation
//! distance (in `[0, 1]`). A cut is declared when the distance exceeds both
//! an absolute floor and `mean + sigma * std` of the recent distances, which
//! adapts to handheld footage and noisy sources without per-file tuning.

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
}

impl Default for ShotConfig {
    fn default() -> Self {
        Self {
            bins: 16,
            window: 24,
            sigma: 4.0,
            min_distance: 0.30,
            min_shot_frames: 6,
        }
    }
}

/// Streaming shot detector.
#[derive(Debug, Clone)]
pub struct ShotDetector {
    cfg: ShotConfig,
    prev: Vec<f32>,
    cur: Vec<f32>,
    has_prev: bool,
    recent: VecDeque<f32>,
    frame: u64,
    last_cut: u64,
    cuts: Vec<(u64, f32)>,
}

impl ShotDetector {
    /// Creates a detector.
    pub fn new(cfg: ShotConfig) -> Self {
        let bins = cfg.bins.clamp(2, 256);
        Self {
            cfg: ShotConfig { bins, ..cfg },
            prev: vec![0.0; bins * 3],
            cur: vec![0.0; bins * 3],
            has_prev: false,
            recent: VecDeque::with_capacity(cfg.window + 1),
            frame: 0,
            last_cut: 0,
            cuts: Vec::new(),
        }
    }

    /// Feeds the next frame.
    pub fn push(&mut self, frame: &RgbFrame) {
        histogram(frame, self.cfg.bins, &mut self.cur);
        if self.has_prev {
            let d = distance(&self.prev, &self.cur);
            let (mean, std) = mean_std(&self.recent);
            let adaptive = mean + self.cfg.sigma * std;
            let warmed = self.recent.len() >= 3;
            if d >= self.cfg.min_distance
                && (!warmed || d > adaptive)
                && self.frame - self.last_cut >= self.cfg.min_shot_frames
            {
                let score = d / adaptive.max(self.cfg.min_distance);
                self.cuts.push((self.frame, score));
                self.last_cut = self.frame;
                // A cut is not representative of in-shot motion.
                self.recent.clear();
            } else {
                self.recent.push_back(d);
                if self.recent.len() > self.cfg.window {
                    self.recent.pop_front();
                }
            }
        }
        std::mem::swap(&mut self.prev, &mut self.cur);
        self.has_prev = true;
        self.frame += 1;
    }

    /// Frames consumed.
    pub fn frames(&self) -> u64 {
        self.frame
    }

    /// Returns detected boundaries with timestamps derived from `fps`.
    pub fn finish(self, fps: FrameRate) -> Vec<ShotBoundary> {
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
        let mut f = RgbFrame::solid(32, 18, rgb);
        let mut s = seed.max(1);
        for v in &mut f.data {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *v = v.saturating_add((s % 9) as u8);
        }
        f
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
        assert!(d.finish(FrameRate::new(30, 1).unwrap()).is_empty());
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
