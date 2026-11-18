//! Subject localisation.
//!
//! [`SubjectDetector`] is the extension point; the default implementation is
//! [`SaliencyTracker`], a dependency-free heuristic that combines temporal
//! motion energy and a YCbCr skin-tone prior into a saliency map, splits the
//! map into connected blobs on a coarse grid, and follows one blob with
//! hysteresis. With several people in frame it therefore locks onto one of
//! them (preferring the one who moves, i.e. talks or gestures, and staying
//! with them until another is clearly more active) instead of averaging to
//! the empty space between them. It is not a face detector: on its own it
//! can follow a gesturing hand. Builds with the default `faces` feature use
//! [`crate::ultraface::FaceTracker`], which only falls back to this tracker
//! when no face is in shot.

use detent_core::index::{SubjectBox, SubjectSource};

use crate::error::Result;
use crate::source::RgbFrame;

/// Produces subject boxes from analysis frames.
pub trait SubjectDetector: Send {
    /// Observes a frame at `t_ms`; may emit a box.
    fn observe(&mut self, frame: &RgbFrame, t_ms: u64) -> Result<Option<SubjectBox>>;

    /// After the last frame: stretches `[start_ms, end_ms]` in which a face
    /// detector looked and found no face. `None` for detectors that do not
    /// look for faces.
    fn faceless_ms(&mut self) -> Option<Vec<(u64, u64)>> {
        None
    }

    /// After the last frame: stretches `[start_ms, end_ms]` in which no
    /// visible mouth moved (no face, or every face's mouth still). `None`
    /// for detectors that do not look at faces.
    fn lips_still_ms(&mut self) -> Option<Vec<(u64, u64)>> {
        None
    }
}

/// Grid columns used for blob segmentation (rows follow the aspect ratio).
const GRID_COLS: usize = 32;
/// Cells at or above this fraction of the strongest cell belong to a blob.
const BLOB_THRESHOLD: f64 = 0.3;
/// Blobs below this share of the total saliency are ignored.
const MIN_BLOB_SHARE: f64 = 0.05;
/// Score bonus for the blob that holds the locked subject, so the tracker
/// only switches to a clearly stronger subject.
const STICKINESS: f64 = 1.75;

/// Motion + skin-tone saliency, segmented into blobs, one blob followed.
#[derive(Debug, Clone)]
pub struct SaliencyTracker {
    interval_ms: u64,
    next_emit_ms: u64,
    prev_luma: Vec<f32>,
    motion: Vec<f32>,
    /// Centre (normalised) of the blob followed last time.
    locked: Option<(f64, f64)>,
}

impl SaliencyTracker {
    /// Emits at most one box every `interval_ms`.
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms: interval_ms.max(1),
            next_emit_ms: 0,
            prev_luma: Vec::new(),
            motion: Vec::new(),
            locked: None,
        }
    }

    /// Makes the tracker prefer the blob around `(x, y)` (normalised), e.g.
    /// the last place a face detector saw the subject.
    pub fn lock_at(&mut self, x: f64, y: f64) {
        self.locked = Some((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)));
    }
}

/// One connected region of salient grid cells (normalised coordinates).
#[derive(Debug, Clone, Copy)]
struct Blob {
    mass: f64,
    cx: f64,
    cy: f64,
    sx: f64,
    sy: f64,
    /// Bounding box of the blob's cells: x0, y0, x1, y1.
    bounds: (f64, f64, f64, f64),
}

impl Blob {
    fn contains(&self, (x, y): (f64, f64)) -> bool {
        let (x0, y0, x1, y1) = self.bounds;
        (x0..=x1).contains(&x) && (y0..=y1).contains(&y)
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
        // A new size (or the first frame) primes the luma buffer without
        // counting motion; otherwise the whole frame would look like motion.
        let primed = self.prev_luma.len() == n;
        if !primed {
            self.prev_luma = vec![0.0; n];
            self.motion = vec![0.0; n];
        }
        for (i, px) in frame.data.as_chunks::<3>().0.iter().enumerate().take(n) {
            let y = 0.299 * f32::from(px[0]) + 0.587 * f32::from(px[1]) + 0.114 * f32::from(px[2]);
            if primed {
                self.motion[i] += (y - self.prev_luma[i]).abs() / 255.0;
            }
            self.prev_luma[i] = y;
        }
        if t_ms < self.next_emit_ms {
            return Ok(None);
        }
        self.next_emit_ms = t_ms + self.interval_ms;

        // Saliency per grid cell: mean of (motion share + skin prior).
        let motion_total: f32 = self.motion.iter().sum::<f32>().max(1e-6);
        let cols = GRID_COLS.min(w).max(1);
        let rows = ((cols as f64 * h as f64 / w as f64).round() as usize).clamp(1, h);
        let mut grid = vec![0.0_f64; cols * rows];
        let mut count = vec![0_u32; cols * rows];
        for (i, px) in frame.data.as_chunks::<3>().0.iter().enumerate().take(n) {
            let (x, y) = (i % w, i / w);
            let skin = if is_skin(f32::from(px[0]), f32::from(px[1]), f32::from(px[2])) {
                1.0
            } else {
                0.0
            };
            let motion = f64::from(self.motion[i] / motion_total) * n as f64;
            let c = (y * rows / h) * cols + x * cols / w;
            grid[c] += motion.min(8.0) + skin * 0.6;
            count[c] += 1;
        }
        self.motion.iter_mut().for_each(|m| *m = 0.0);
        for (g, &c) in grid.iter_mut().zip(&count) {
            *g /= f64::from(c.max(1));
        }

        let blobs = blobs(&grid, cols, rows);
        let total: f64 = blobs.iter().map(|b| b.mass).sum();
        let locked = self.locked;
        let score = |b: &Blob| {
            let (nx, ny) = (b.cx - 0.5, b.cy - 0.5);
            let center = 0.75 + 0.25 * (-(nx * nx + ny * ny) / 0.08).exp();
            let sticky = if locked.is_some_and(|p| b.contains(p)) {
                STICKINESS
            } else {
                1.0
            };
            b.mass * center * sticky
        };
        let Some(best) = blobs
            .iter()
            .filter(|b| b.mass >= MIN_BLOB_SHARE * total)
            .max_by(|a, b| score(a).total_cmp(&score(b)))
            .copied()
        else {
            return Ok(None);
        };
        self.locked = Some((best.cx, best.cy));

        // Box spans ±1 std of the blob plus one cell of margin.
        let bw = (2.0 * best.sx + 1.0 / cols as f64).clamp(0.1, 0.8);
        let bh = (2.0 * best.sy + 1.0 / rows as f64).clamp(0.1, 0.8);
        // Confidence: the share of the frame's saliency this blob holds.
        let confidence = (best.mass / total.max(1e-9)).clamp(0.0, 1.0);
        Ok(Some(SubjectBox {
            t_ms,
            x: (best.cx - bw / 2.0).clamp(0.0, 1.0 - bw) as f32,
            y: (best.cy - bh / 2.0).clamp(0.0, 1.0 - bh) as f32,
            w: bw as f32,
            h: bh as f32,
            confidence: (confidence * 100.0).round() as f32 / 100.0,
            source: SubjectSource::Saliency,
        }))
    }
}

/// Connected (8-neighbour) components of cells at or above
/// [`BLOB_THRESHOLD`] times the strongest cell, with weighted moments.
fn blobs(grid: &[f64], cols: usize, rows: usize) -> Vec<Blob> {
    let peak = grid.iter().copied().fold(0.0_f64, f64::max);
    if peak <= 1e-9 {
        return Vec::new();
    }
    let thr = peak * BLOB_THRESHOLD;
    let (cw, ch) = (1.0 / cols as f64, 1.0 / rows as f64);
    let mut seen = vec![false; grid.len()];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..grid.len() {
        if seen[start] || grid[start] < thr {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut m, mut sx, mut sy, mut sxx, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut bounds = (1.0_f64, 1.0_f64, 0.0_f64, 0.0_f64);
        while let Some(c) = stack.pop() {
            let (x, y) = (c % cols, c / cols);
            let v = grid[c];
            let (fx, fy) = ((x as f64 + 0.5) * cw, (y as f64 + 0.5) * ch);
            m += v;
            sx += v * fx;
            sy += v * fy;
            sxx += v * fx * fx;
            syy += v * fy * fy;
            bounds = (
                bounds.0.min(x as f64 * cw),
                bounds.1.min(y as f64 * ch),
                bounds.2.max((x + 1) as f64 * cw),
                bounds.3.max((y + 1) as f64 * ch),
            );
            for ny in y.saturating_sub(1)..=(y + 1).min(rows - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(cols - 1) {
                    let nc = ny * cols + nx;
                    if !seen[nc] && grid[nc] >= thr {
                        seen[nc] = true;
                        stack.push(nc);
                    }
                }
            }
        }
        let (cx, cy) = (sx / m, sy / m);
        out.push(Blob {
            mass: m,
            cx,
            cy,
            sx: (sxx / m - cx * cx).max(0.0).sqrt(),
            sy: (syy / m - cy * cy).max(0.0).sqrt(),
            bounds,
        });
    }
    out
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

    /// Two skin-coloured squares at x = 0.2 and 0.8; the one with index
    /// `active` bobs up and down ("talks") when `move_active` is set.
    fn two_people(k: u64, active: usize, move_active: bool) -> RgbFrame {
        let (w, h) = (160_usize, 90_usize);
        let mut f = RgbFrame::solid(w as u32, h as u32, [90, 90, 90]);
        for (i, cx) in [32_usize, 128].into_iter().enumerate() {
            let bob = if move_active && i == active {
                (k % 4) as usize * 2
            } else {
                0
            };
            for y in 30 + bob..48 + bob {
                for x in cx - 9..cx + 9 {
                    let p = (y * w + x) * 3;
                    f.data[p..p + 3].copy_from_slice(&[224, 172, 140]);
                }
            }
        }
        f
    }

    fn track(frames: impl Iterator<Item = RgbFrame>) -> Vec<SubjectBox> {
        let mut t = SaliencyTracker::new(100);
        frames
            .enumerate()
            .filter_map(|(k, f)| t.observe(&f, k as u64 * 33).unwrap())
            .collect()
    }

    #[test]
    fn follows_a_moving_subject() {
        let boxes = track((0..60).map(|k| frame_with_square(20 + k * 2, 160, 90)));
        assert!(boxes.len() >= 15);
        let first = boxes[2].center().0;
        let last = boxes.last().unwrap().center().0;
        assert!(last > first + 0.3, "{first} -> {last}");
        for b in &boxes {
            assert!(b.x >= 0.0 && b.y >= 0.0 && b.x + b.w <= 1.0001 && b.y + b.h <= 1.0001);
        }
    }

    #[test]
    fn locks_onto_one_of_two_people_instead_of_the_gap() {
        // Both still: a global centroid would sit at x = 0.5 (empty wall).
        let boxes = track((0..40).map(|k| two_people(k, 0, false)));
        assert!(!boxes.is_empty());
        for b in &boxes {
            let cx = b.center().0;
            assert!(!(0.3..=0.7).contains(&cx), "box centred on the gap: {cx}");
        }
        // And it does not flip-flop between them.
        let sides: Vec<bool> = boxes.iter().map(|b| b.center().0 < 0.5).collect();
        assert!(sides.windows(2).all(|w| w[0] == w[1]), "{sides:?}");
    }

    #[test]
    fn follows_the_person_who_moves_and_switches_once() {
        // The right person talks first, then the left one takes over.
        let boxes = track((0..90_u64).map(|k| two_people(k, usize::from(k < 45), true)));
        let xs: Vec<f64> = boxes.iter().map(|b| b.center().0).collect();
        assert!(xs[3] > 0.7, "should start on the right speaker: {xs:?}");
        assert!(
            *xs.last().unwrap() < 0.3,
            "should end on the left speaker: {xs:?}"
        );
        // The very first box precedes any motion evidence; count after it.
        let switches = xs[1..]
            .windows(2)
            .filter(|w| (w[0] < 0.5) != (w[1] < 0.5))
            .count();
        assert_eq!(switches, 1, "{xs:?}");
    }

    #[test]
    fn skin_classifier() {
        assert!(is_skin(224.0, 172.0, 140.0));
        assert!(!is_skin(30.0, 90.0, 220.0));
    }
}
