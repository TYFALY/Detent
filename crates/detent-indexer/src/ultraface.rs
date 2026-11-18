//! UltraFace pre- and post-processing, shared by the built-in face detector
//! ([`crate::faces`], pure Rust) and the ONNX Runtime adapter
//! ([`crate::onnx`]).
//!
//! The model (`version-RFB-320.onnx`, 1.2 MB) takes `1x3x240x320` RGB
//! normalised as `(px - 127) / 128` and outputs `scores [1,N,2]` and decoded
//! `boxes [1,N,4]` (normalised `x1,y1,x2,y2`).
//!
//! * Frames are **letterboxed** into the 4:3 input, so 16:9 and 9:16 footage
//!   is not distorted, and boxes are mapped back to frame coordinates.
//! * Overlapping candidates are merged with greedy **NMS** (IoU 0.3).
//! * With several people in shot, one **primary subject** is chosen by
//!   `confidence x sqrt(area)` with hysteresis ([`FaceTracker`]): the previous
//!   subject is kept unless another face is clearly more prominent, so the
//!   camera does not hop between faces in a crowd.

use detent_core::index::{SubjectBox, SubjectSource};
use image::{Rgb, RgbImage, imageops};

use crate::error::Result;
use crate::saliency::{SaliencyTracker, SubjectDetector};
use crate::source::RgbFrame;

/// Model input width.
pub const IN_W: u32 = 320;
/// Model input height.
pub const IN_H: u32 = 240;
/// A new face must be this much more prominent to steal the subject.
const SWITCH_RATIO: f32 = 1.5;
/// After the last face was seen, the tracker holds its framing this long
/// (a head turned to profile, a hand in front of the face) before it falls
/// back to motion saliency.
pub const FACE_HOLD_MS: u64 = 1500;

/// A detected face in normalised frame coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    /// Left.
    pub x: f32,
    /// Top.
    pub y: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
    /// Detector score.
    pub score: f32,
}

impl Face {
    fn area(&self) -> f32 {
        self.w * self.h
    }

    fn prominence(&self) -> f32 {
        self.score * self.area().sqrt()
    }

    fn iou(&self, o: &Face) -> f32 {
        let ix = (self.x + self.w).min(o.x + o.w) - self.x.max(o.x);
        let iy = (self.y + self.h).min(o.y + o.h) - self.y.max(o.y);
        if ix <= 0.0 || iy <= 0.0 {
            return 0.0;
        }
        let inter = ix * iy;
        inter / (self.area() + o.area() - inter).max(1e-9)
    }
}

/// Greedy non-maximum suppression (sorted by score, descending).
pub fn nms(mut faces: Vec<Face>, iou: f32) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Face> = Vec::new();
    for f in faces {
        if keep.iter().all(|k| k.iou(&f) < iou) {
            keep.push(f);
        }
    }
    keep
}

/// Picks the primary subject with hysteresis toward `previous`.
pub fn pick_primary(faces: &[Face], previous: Option<&Face>) -> Option<Face> {
    let best = faces
        .iter()
        .copied()
        .max_by(|a, b| a.prominence().total_cmp(&b.prominence()))?;
    let Some(prev) = previous else {
        return Some(best);
    };
    let same = faces
        .iter()
        .copied()
        .filter(|f| f.iou(prev) > 0.2)
        .max_by(|a, b| a.iou(prev).total_cmp(&b.iou(prev)));
    match same {
        Some(s) if best.prominence() < s.prominence() * SWITCH_RATIO => Some(s),
        _ => Some(best),
    }
}

/// Where the frame sits inside the padded model input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Letterbox {
    rw: u32,
    rh: u32,
    ox: u32,
    oy: u32,
}

/// Letterboxes `img` into the model input and writes the planar,
/// normalised tensor data into `input` (`3 * IN_W * IN_H` floats).
pub fn preprocess(img: &RgbImage, input: &mut [f32]) -> Letterbox {
    let (w, h) = (img.width().max(1), img.height().max(1));
    let s = (IN_W as f32 / w as f32).min(IN_H as f32 / h as f32);
    let (rw, rh) = (
        ((w as f32 * s).round() as u32).clamp(1, IN_W),
        ((h as f32 * s).round() as u32).clamp(1, IN_H),
    );
    let (ox, oy) = ((IN_W - rw) / 2, (IN_H - rh) / 2);
    let resized = imageops::resize(img, rw, rh, imageops::FilterType::Triangle);
    let mut canvas = RgbImage::from_pixel(IN_W, IN_H, Rgb([127, 127, 127]));
    imageops::replace(&mut canvas, &resized, i64::from(ox), i64::from(oy));
    let plane = (IN_W * IN_H) as usize;
    for (i, px) in canvas.pixels().enumerate() {
        for c in 0..3 {
            input[c * plane + i] = (f32::from(px[c]) - 127.0) / 128.0;
        }
    }
    Letterbox { rw, rh, ox, oy }
}

/// Turns raw model outputs into faces in frame coordinates (after NMS).
pub fn decode(scores: &[f32], boxes: &[f32], lb: Letterbox, threshold: f32) -> Vec<Face> {
    let (fx, fy) = (IN_W as f32 / lb.rw as f32, IN_H as f32 / lb.rh as f32);
    let (px, py) = (lb.ox as f32 / IN_W as f32, lb.oy as f32 / IN_H as f32);
    let faces = scores
        .as_chunks::<2>()
        .0
        .iter()
        .zip(boxes.as_chunks::<4>().0)
        .filter(|(s, _)| s[1] >= threshold)
        .filter_map(|(s, b)| {
            let x1 = ((b[0] - px) * fx).clamp(0.0, 1.0);
            let y1 = ((b[1] - py) * fy).clamp(0.0, 1.0);
            let x2 = ((b[2] - px) * fx).clamp(0.0, 1.0);
            let y2 = ((b[3] - py) * fy).clamp(0.0, 1.0);
            (x2 > x1 && y2 > y1).then_some(Face {
                x: x1,
                y: y1,
                w: x2 - x1,
                h: y2 - y1,
                score: s[1],
            })
        })
        .collect();
    nms(faces, 0.3)
}

/// Faces shorter than this (pixels in the analysis frame) are too small to
/// judge the mouth; they count as possibly talking.
pub const MIN_MOUTH_FACE_PX: f32 = 24.0;
/// Mean grey-level change of the (aligned) mouth region between scans that
/// counts as a moving mouth...
pub const MOUTH_MIN_DIFF: f32 = 8.0;
/// ...when it is also this much larger than the change around the eyes
/// (which cancels head motion and lighting changes).
pub const MOUTH_TO_UPPER: f32 = 1.4;
/// A face whose mouth moved counts as talking this long before and after
/// (one 200 ms scan often catches a talking mouth mid-shape).
pub const TALK_HOLD_MS: u64 = 600;

/// Grey-level frame used to compare mouths between scans.
#[derive(Debug, Clone)]
pub struct Gray {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

impl Gray {
    /// Converts an RGB image.
    pub fn new(img: &RgbImage) -> Self {
        let px = img
            .pixels()
            .map(|p| {
                ((299 * u32::from(p[0]) + 587 * u32::from(p[1]) + 114 * u32::from(p[2])) / 1000)
                    as u8
            })
            .collect();
        Self {
            w: img.width(),
            h: img.height(),
            px,
        }
    }

    /// Mean absolute difference to `o` (read `shift` pixels away) over a
    /// sub-rectangle of `face` (fractions of the box), or `None` when the
    /// region is empty.
    fn region_diff(
        &self,
        o: &Gray,
        face: &Face,
        (fx0, fy0, fx1, fy1): (f32, f32, f32, f32),
        (dx, dy): (i32, i32),
    ) -> Option<f32> {
        if self.w != o.w || self.h != o.h {
            return None;
        }
        let (w, h) = (self.w as f32, self.h as f32);
        let x0 = ((face.x + face.w * fx0) * w).max(0.0) as u32;
        let x1 = (((face.x + face.w * fx1) * w) as u32).min(self.w);
        let y0 = ((face.y + face.h * fy0) * h).max(0.0) as u32;
        let y1 = (((face.y + face.h * fy1) * h) as u32).min(self.h);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let mut sum = 0_u64;
        let mut n = 0_u32;
        for y in y0..y1 {
            let oy = y as i32 + dy;
            if oy < 0 || oy >= self.h as i32 {
                continue;
            }
            for x in x0..x1 {
                let ox = x as i32 + dx;
                if ox < 0 || ox >= self.w as i32 {
                    continue;
                }
                let a = self.px[(y * self.w + x) as usize];
                let b = o.px[(oy as u32 * self.w + ox as u32) as usize];
                sum += u64::from(a.abs_diff(b));
                n += 1;
            }
        }
        (n > 0).then(|| sum as f32 / n as f32)
    }
}

/// Mouth and eye-region change of `face` between two scans:
/// `(mouth, upper)` mean grey-level differences, or `None` when the face is
/// too small to tell.
pub fn mouth_activity(prev: &Gray, cur: &Gray, face: &Face) -> Option<(f32, f32)> {
    if face.h * (cur.h as f32) < MIN_MOUTH_FACE_PX {
        return None;
    }
    // Align the face first (camera shake, head motion): the shift that best
    // matches the eyes and nose, so what still differs at the mouth moved
    // inside the face.
    const EYES: (f32, f32, f32, f32) = (0.15, 0.15, 0.85, 0.6);
    const R: i32 = 4;
    let mut best = (f32::INFINITY, (0, 0));
    for dy in -R..=R {
        for dx in -R..=R {
            if let Some(d) = cur.region_diff(prev, face, EYES, (dx, dy))
                && d < best.0
            {
                best = (d, (dx, dy));
            }
        }
    }
    let upper = best.0;
    if !upper.is_finite() {
        return None;
    }
    let mouth = cur.region_diff(prev, face, (0.25, 0.65, 0.75, 0.95), best.1)?;
    Some((mouth, upper))
}

/// Whether a face's mouth moved between scans (unknown counts as moving).
pub fn mouth_moving(activity: Option<(f32, f32)>) -> bool {
    activity.is_none_or(|(m, u)| m >= MOUTH_MIN_DIFF && m >= MOUTH_TO_UPPER * u)
}

/// `[first, last]` scan times of runs where no scan within `hold_ms` saw a
/// moving mouth.
pub fn still_runs(scans: &[(u64, bool)], hold_ms: u64) -> Vec<(u64, u64)> {
    let talk: Vec<u64> = scans.iter().filter(|s| s.1).map(|s| s.0).collect();
    let near_talk = |t: u64| {
        let i = talk.partition_point(|&x| x + hold_ms < t);
        talk.get(i).is_some_and(|&x| x <= t + hold_ms)
    };
    let mut out: Vec<(u64, u64)> = Vec::new();
    let mut open: Option<(u64, u64)> = None;
    for &(t, _) in scans {
        if near_talk(t) {
            if let Some(r) = open.take()
                && r.1 > r.0
            {
                out.push(r);
            }
        } else {
            open.get_or_insert((t, t)).1 = t;
        }
    }
    if let Some(r) = open
        && r.1 > r.0
    {
        out.push(r);
    }
    out
}

/// Something that finds faces in an image.
pub trait FaceBackend: Send {
    /// All faces (after NMS) in `img`.
    fn detect(&mut self, img: &RgbImage) -> Result<Vec<Face>>;
}

/// Subject tracking driven by a face detector.
///
/// Every `interval_ms` the backend looks for faces and one primary face is
/// followed with hysteresis. Hands, arms and moving objects never produce a
/// face, so they cannot pull the camera. When no face has been seen for
/// [`FACE_HOLD_MS`] (screen recordings, gameplay, product shots, someone
/// facing away), the motion-saliency tracker takes over, seeded at the last
/// face so it keeps following the same person.
pub struct FaceTracker {
    backend: Box<dyn FaceBackend>,
    saliency: SaliencyTracker,
    interval_ms: u64,
    next_ms: u64,
    previous: Option<Face>,
    /// A new face seen once; it becomes the subject only if the next scan
    /// sees it again (one-scan false positives never move the camera).
    pending: Option<Face>,
    last_face_ms: Option<u64>,
    source: SubjectSource,
    /// Closed face-free stretches, and the open one (first, last empty scan).
    faceless: Vec<(u64, u64)>,
    open_faceless: Option<(u64, u64)>,
    /// Per scan: time and whether any visible mouth moved.
    scans: Vec<(u64, bool)>,
    prev_gray: Option<Gray>,
}

impl std::fmt::Debug for FaceTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaceTracker")
            .field("interval_ms", &self.interval_ms)
            .field("previous", &self.previous)
            .finish_non_exhaustive()
    }
}

impl FaceTracker {
    /// Emits at most one box every `interval_ms`, tagged with `source`.
    pub fn new(backend: Box<dyn FaceBackend>, interval_ms: u64, source: SubjectSource) -> Self {
        let interval_ms = interval_ms.max(1);
        Self {
            backend,
            saliency: SaliencyTracker::new(interval_ms),
            interval_ms,
            next_ms: 0,
            previous: None,
            pending: None,
            last_face_ms: None,
            source,
            faceless: Vec::new(),
            open_faceless: None,
            scans: Vec::new(),
            prev_gray: None,
        }
    }

    /// Runs the backend directly on one image.
    pub fn detect(&mut self, img: &RgbImage) -> Result<Vec<Face>> {
        self.backend.detect(img)
    }
}

impl SubjectDetector for FaceTracker {
    fn observe(&mut self, frame: &RgbFrame, t_ms: u64) -> Result<Option<SubjectBox>> {
        // The saliency tracker sees every frame so its motion map is ready
        // whenever it has to take over.
        let fallback = self.saliency.observe(frame, t_ms)?;
        if t_ms < self.next_ms {
            return Ok(None);
        }
        self.next_ms = t_ms + self.interval_ms;
        let Some(img) = RgbImage::from_raw(frame.width, frame.height, frame.data.clone()) else {
            return Ok(None);
        };
        let faces = self.backend.detect(&img)?;
        let gray = Gray::new(&img);
        // A frame identical to the last scan is a frozen picture (lag):
        // still mouths there say nothing about who is talking.
        let frozen = self.prev_gray.as_ref().is_some_and(|p| p.px == gray.px);
        let talking = frozen
            || faces.iter().any(|f| {
                mouth_moving(
                    self.prev_gray
                        .as_ref()
                        .and_then(|p| mouth_activity(p, &gray, f)),
                )
            });
        self.scans.push((t_ms, talking));
        self.prev_gray = Some(gray);
        // Face-free stretches run from the first to the last scan that saw
        // no face; between a face and an empty scan nothing is assumed.
        if faces.is_empty() {
            let run = self.open_faceless.get_or_insert((t_ms, t_ms));
            run.1 = t_ms;
        } else if let Some(run) = self.open_faceless.take()
            && run.1 > run.0
        {
            self.faceless.push(run);
        }
        let candidate = pick_primary(&faces, self.previous.as_ref());
        let confirmed = candidate.filter(|c| {
            let same = |p: &Option<Face>, iou: f32| p.as_ref().is_some_and(|p| c.iou(p) > iou);
            same(&self.previous, 0.2) || same(&self.pending, 0.3)
        });
        self.pending = if confirmed.is_none() { candidate } else { None };
        if let Some(f) = confirmed {
            self.previous = Some(f);
            self.last_face_ms = Some(t_ms);
            self.saliency
                .lock_at(f64::from(f.x + f.w / 2.0), f64::from(f.y + f.h / 2.0));
            return Ok(Some(SubjectBox {
                t_ms,
                x: f.x,
                y: f.y,
                w: f.w,
                h: f.h,
                confidence: f.score,
                source: self.source,
            }));
        }
        if self.pending.is_some()
            || self
                .last_face_ms
                .is_some_and(|last| t_ms.saturating_sub(last) < FACE_HOLD_MS)
        {
            // Hold: no box means the camera keeps its current framing.
            return Ok(None);
        }
        self.previous = None;
        Ok(fallback)
    }

    fn lips_still_ms(&mut self) -> Option<Vec<(u64, u64)>> {
        Some(still_runs(&self.scans, TALK_HOLD_MS))
    }

    fn faceless_ms(&mut self) -> Option<Vec<(u64, u64)>> {
        let mut out = std::mem::take(&mut self.faceless);
        if let Some(run) = self.open_faceless.take()
            && run.1 > run.0
        {
            out.push(run);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x: f32, w: f32, score: f32) -> Face {
        Face {
            x,
            y: 0.2,
            w,
            h: w,
            score,
        }
    }

    #[test]
    fn nms_merges_duplicates() {
        let f = vec![
            face(0.1, 0.2, 0.9),
            face(0.11, 0.2, 0.8),
            face(0.6, 0.2, 0.7),
        ];
        let kept = nms(f, 0.3);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].score, 0.9);
    }

    #[test]
    fn primary_subject_has_hysteresis() {
        let a = face(0.1, 0.20, 0.9);
        let b = face(0.6, 0.22, 0.9); // slightly bigger, not 1.5x more prominent
        assert_eq!(pick_primary(&[a, b], None), Some(b));
        assert_eq!(pick_primary(&[a, b], Some(&a)), Some(a), "stick with a");
        let c = face(0.6, 0.6, 0.95); // much bigger face appears
        assert_eq!(pick_primary(&[a, c], Some(&a)), Some(c));
        assert_eq!(pick_primary(&[], Some(&a)), None);
    }

    /// Scripted backend: a face at x = 0.15 for the first `face_until`
    /// calls, none afterwards.
    struct Scripted {
        calls: usize,
        face_until: usize,
    }

    impl FaceBackend for Scripted {
        fn detect(&mut self, _: &RgbImage) -> Result<Vec<Face>> {
            self.calls += 1;
            Ok(if self.calls <= self.face_until {
                vec![face(0.1, 0.1, 0.9)]
            } else {
                Vec::new()
            })
        }
    }

    /// A person on the left (still) and a big waving hand on the right.
    fn hand_frame(k: u64) -> RgbFrame {
        let (w, h) = (160_usize, 90_usize);
        let mut f = RgbFrame::solid(w as u32, h as u32, [90, 90, 90]);
        let wave = (k % 4) as usize * 4;
        let mut paint = |x0: usize, y0: usize, s: usize| {
            for y in y0..(y0 + s).min(h) {
                for x in x0..(x0 + s).min(w) {
                    let p = (y * w + x) * 3;
                    f.data[p..p + 3].copy_from_slice(&[224, 172, 140]);
                }
            }
        };
        paint(16, 18, 16);
        paint(110 + wave, 40, 24);
        f
    }

    fn run(face_until: usize, frames: u64) -> Vec<SubjectBox> {
        let backend = Scripted {
            calls: 0,
            face_until,
        };
        let mut t = FaceTracker::new(Box::new(backend), 100, SubjectSource::Face);
        (0..frames)
            .filter_map(|k| t.observe(&hand_frame(k), k * 33).unwrap())
            .collect()
    }

    #[test]
    fn a_waving_hand_never_steals_the_face() {
        let boxes = run(usize::MAX, 90);
        assert!(boxes.len() >= 20, "{}", boxes.len());
        for b in &boxes {
            assert_eq!(b.source, SubjectSource::Face);
            assert!(b.center().0 < 0.3, "camera left the face: {b:?}");
        }
    }

    #[test]
    fn a_face_seen_in_one_scan_only_is_ignored() {
        let boxes = run(1, 60);
        assert!(boxes.iter().all(|b| b.source == SubjectSource::Saliency));
        // The first sighting waits for confirmation; the second confirms.
        let boxes = run(2, 60);
        let faces: Vec<_> = boxes
            .iter()
            .filter(|b| b.source == SubjectSource::Face)
            .collect();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].t_ms, 132);
    }

    #[test]
    fn holds_through_a_short_miss_then_falls_back_to_saliency() {
        // Faces for 1 s (10 detections; the first one only arms), then none.
        let boxes = run(10, 120);
        let faces = boxes.iter().filter(|b| b.source == SubjectSource::Face);
        assert_eq!(faces.count(), 9);
        let fallback: Vec<_> = boxes
            .iter()
            .filter(|b| b.source == SubjectSource::Saliency)
            .collect();
        assert!(!fallback.is_empty(), "never fell back");
        // Nothing during the hold window right after the last face.
        let last_face = boxes
            .iter()
            .filter(|b| b.source == SubjectSource::Face)
            .map(|b| b.t_ms)
            .max()
            .unwrap();
        assert!(fallback[0].t_ms >= last_face + FACE_HOLD_MS);
    }

    #[test]
    fn still_runs_keep_clear_of_talking_scans() {
        let scans: Vec<(u64, bool)> = (0..20_u64).map(|k| (k * 200, k == 8)).collect();
        // Talking at 1600 ms blocks 1000..=2200.
        assert_eq!(still_runs(&scans, 600), vec![(0, 800), (2_400, 3_800)]);
        assert!(still_runs(&[(0, true), (200, true)], 600).is_empty());
    }

    /// A face-sized grey patch whose lower part ("mouth") changes per call.
    fn talking_face(k: u64, mouth_moves: bool) -> RgbImage {
        let mut img = RgbImage::from_pixel(320, 180, Rgb([90, 90, 90]));
        for y in 40..100 {
            for x in 100..150 {
                let v = if y >= 80 && mouth_moves && k % 2 == 1 {
                    20
                } else {
                    200
                };
                let eye = if (52..58).contains(&y) && (x % 20) < 6 {
                    30
                } else {
                    v.max(150)
                };
                let px = if y >= 80 { v } else { eye };
                img.put_pixel(x, y, Rgb([px, px, px]));
            }
        }
        img
    }

    #[test]
    fn mouth_activity_tells_talking_from_listening() {
        let face = Face {
            x: 100.0 / 320.0,
            y: 40.0 / 180.0,
            w: 50.0 / 320.0,
            h: 60.0 / 180.0,
            score: 0.9,
        };
        let a = Gray::new(&talking_face(0, true));
        let b = Gray::new(&talking_face(1, true));
        assert!(mouth_moving(mouth_activity(&a, &b, &face)));
        let c = Gray::new(&talking_face(1, false));
        assert!(!mouth_moving(mouth_activity(&a, &c, &face)));
        // Too small to judge: counts as talking.
        let tiny = Face { h: 0.05, ..face };
        assert!(mouth_moving(mouth_activity(&a, &c, &tiny)));
    }

    #[test]
    fn records_where_no_face_was_seen() {
        let backend = Scripted {
            calls: 0,
            face_until: 5,
        };
        let mut t = FaceTracker::new(Box::new(backend), 100, SubjectSource::Face);
        for k in 0..60_u64 {
            t.observe(&hand_frame(k), k * 33).unwrap();
        }
        // Scans every 132 ms; faces in the first five (0..528), none after.
        assert_eq!(t.faceless_ms(), Some(vec![(660, 1_848)]));
    }
}
