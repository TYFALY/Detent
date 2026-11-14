//! Subject localisation.
//!
//! [`SubjectDetector`] is the extension point; the default implementation is
//! [`SaliencyTracker`], a dependency-free heuristic that combines temporal
//! motion energy, a YCbCr skin-tone prior and a weak centre bias into a
//! weighted centroid. It is not a face detector, but for talking-head and
//! single-subject footage it gives the camera solver a stable target with no
//! ML runtime. Enable the `onnx` feature for a learned detector.

use detent_core::index::{SubjectBox, SubjectSource};

use crate::error::Result;
use crate::source::RgbFrame;

/// Produces subject boxes from analysis frames.
pub trait SubjectDetector: Send {
    /// Observes a frame at `t_ms`; may emit a box.
    fn observe(&mut self, frame: &RgbFrame, t_ms: u64) -> Result<Option<SubjectBox>>;
}

/// Motion + skin-tone + centre-bias saliency centroid.
#[derive(Debug, Clone)]
pub struct SaliencyTracker {
    interval_ms: u64,
    next_emit_ms: u64,
    prev_luma: Vec<f32>,
    motion: Vec<f32>,
}

impl SaliencyTracker {
    /// Emits at most one box every `interval_ms`.
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms: interval_ms.max(1),
            next_emit_ms: 0,
            prev_luma: Vec::new(),
            motion: Vec::new(),
        }
    }
}

fn is_skin(r: f32, g: f32, b: f32) -> bool {
    let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
    (77.0..=127.0).contains(&cb) && (133.0..=173.0).contains(&cr)
}

impl SubjectDetector for SaliencyTracker {
    fn observe(&mut self, frame: &RgbFrame, t_ms: u64) -> Result<Option<SubjectBox>> {
        let (w, h) = (frame.width as usize, frame.height as usize);
        let n = w * h;
        if n == 0 {
            return Ok(None);
        }
        if self.prev_luma.len() != n {
            self.prev_luma = vec![0.0; n];
            self.motion = vec![0.0; n];
        }
        for (i, px) in frame.data.as_chunks::<3>().0.iter().enumerate().take(n) {
            let y = 0.299 * f32::from(px[0]) + 0.587 * f32::from(px[1]) + 0.114 * f32::from(px[2]);
            self.motion[i] += (y - self.prev_luma[i]).abs() / 255.0;
            self.prev_luma[i] = y;
        }
        if t_ms < self.next_emit_ms {
            return Ok(None);
        }
        self.next_emit_ms = t_ms + self.interval_ms;

        let motion_total: f32 = self.motion.iter().sum::<f32>().max(1e-6);
        let (mut sw, mut sx, mut sy, mut sxx, mut syy) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
        for (i, px) in frame.data.as_chunks::<3>().0.iter().enumerate().take(n) {
            let (x, y) = ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
            let (nx, ny) = (x / w as f64 - 0.5, y / h as f64 - 0.5);
            let center = (-(nx * nx + ny * ny) / 0.08).exp();
            let skin = if is_skin(f32::from(px[0]), f32::from(px[1]), f32::from(px[2])) {
                1.0
            } else {
                0.0
            };
            let motion = f64::from(self.motion[i] / motion_total) * n as f64;
            let wgt = motion.min(8.0) * 1.0 + skin * 0.6 + center * 0.25;
            sw += wgt;
            sx += wgt * x;
            sy += wgt * y;
            sxx += wgt * x * x;
            syy += wgt * y * y;
        }
        self.motion.iter_mut().for_each(|m| *m = 0.0);
        if sw <= 1e-9 {
            return Ok(None);
        }
        let (mx, my) = (sx / sw, sy / sw);
        let vx = (sxx / sw - mx * mx).max(0.0).sqrt();
        let vy = (syy / sw - my * my).max(0.0).sqrt();
        let bw = (2.0 * vx / w as f64).clamp(0.1, 0.8);
        let bh = (2.0 * vy / h as f64).clamp(0.1, 0.8);
        let cx = mx / w as f64;
        let cy = my / h as f64;
        // Concentrated weight → high confidence.
        let spread = (bw * bh).sqrt();
        Ok(Some(SubjectBox {
            t_ms,
            x: (cx - bw / 2.0).clamp(0.0, 1.0 - bw) as f32,
            y: (cy - bh / 2.0).clamp(0.0, 1.0 - bh) as f32,
            w: bw as f32,
            h: bh as f32,
            confidence: ((1.0 - spread).clamp(0.0, 1.0) * 100.0).round() as f32 / 100.0,
            source: SubjectSource::Saliency,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grey background with a moving skin-coloured square.
    fn frame_with_square(cx: usize, w: usize, h: usize) -> RgbFrame {
        let mut f = RgbFrame::solid(w as u32, h as u32, [90, 90, 90]);
        for y in h / 3..h / 3 + 12 {
            for x in cx.saturating_sub(6)..(cx + 6).min(w) {
                let i = (y * w + x) * 3;
                f.data[i..i + 3].copy_from_slice(&[224, 172, 140]);
            }
        }
        f
    }

    #[test]
    fn follows_a_moving_subject() {
        let mut t = SaliencyTracker::new(100);
        let mut boxes = Vec::new();
        for k in 0..60_u64 {
            let cx = 20 + (k as usize) * 2;
            if let Some(b) = t.observe(&frame_with_square(cx, 160, 90), k * 33).unwrap() {
                boxes.push(b);
            }
        }
        assert!(boxes.len() >= 15);
        let first = boxes[2].center().0;
        let last = boxes.last().unwrap().center().0;
        assert!(last > first + 0.3, "{first} -> {last}");
        for b in &boxes {
            assert!(b.x >= 0.0 && b.y >= 0.0 && b.x + b.w <= 1.0001 && b.y + b.h <= 1.0001);
        }
    }

    #[test]
    fn skin_classifier() {
        assert!(is_skin(224.0, 172.0, 140.0));
        assert!(!is_skin(30.0, 90.0, 220.0));
    }
}
