//! ONNX Runtime face detection (feature `onnx`).
//!
//! Targets the UltraFace family (`version-RFB-320.onnx`, 1.2 MB): input
//! `1x3x240x320` RGB normalised as `(px - 127) / 128`, outputs `scores
//! [1,N,2]` and decoded `boxes [1,N,4]` (normalised `x1,y1,x2,y2`).
//!
//! * Frames are **letterboxed** into the 4:3 input, so 16:9 and 9:16 footage
//!   is not distorted, and boxes are mapped back to frame coordinates.
//! * Overlapping candidates are merged with greedy **NMS** (IoU 0.3).
//! * With several people in shot, one **primary subject** is chosen by
//!   `confidence x sqrt(area)` with hysteresis: the previous subject is kept
//!   unless another face is clearly more prominent, so the camera does not
//!   hop between faces in a crowd.
//!
//! ONNX Runtime is loaded at run time (`ORT_DYLIB_PATH`), so no native
//! libraries are needed to build detent.

use std::path::Path;

use detent_core::index::{SubjectBox, SubjectSource};
use image::{Rgb, RgbImage, imageops};
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{IndexError, Result};
use crate::saliency::SubjectDetector;
use crate::source::RgbFrame;

const IN_W: u32 = 320;
const IN_H: u32 = 240;
/// A new face must be this much more prominent to steal the subject.
const SWITCH_RATIO: f32 = 1.5;

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

/// UltraFace detector.
pub struct OnnxFaceDetector {
    session: Session,
    threshold: f32,
    interval_ms: u64,
    next_ms: u64,
    previous: Option<Face>,
    input: Vec<f32>,
}

impl std::fmt::Debug for OnnxFaceDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxFaceDetector")
            .field("threshold", &self.threshold)
            .field("interval_ms", &self.interval_ms)
            .finish_non_exhaustive()
    }
}

fn ort_err(e: impl std::fmt::Display) -> IndexError {
    IndexError::Detector(e.to_string())
}

impl OnnxFaceDetector {
    /// Loads a model. Emits at most one subject box per `interval_ms`.
    pub fn new(model: &Path, threshold: f32, interval_ms: u64) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort_err)?
            .commit_from_file(model)
            .map_err(ort_err)?;
        Ok(Self {
            session,
            threshold,
            interval_ms: interval_ms.max(1),
            next_ms: 0,
            previous: None,
            input: vec![0.0; (3 * IN_W * IN_H) as usize],
        })
    }

    /// Detects all faces (after NMS) in an RGB image.
    pub fn detect(&mut self, img: &RgbImage) -> Result<Vec<Face>> {
        let (w, h) = (img.width().max(1), img.height().max(1));
        // Letterbox: fit inside 320x240 keeping aspect, pad with grey.
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
                self.input[c * plane + i] = (f32::from(px[c]) - 127.0) / 128.0;
            }
        }
        let tensor = Tensor::from_array((
            [1_usize, 3, IN_H as usize, IN_W as usize],
            self.input.clone(),
        ))
        .map_err(ort_err)?;
        let outputs = self.session.run(ort::inputs![tensor]).map_err(ort_err)?;
        let (scores, boxes) = match (outputs.get("scores"), outputs.get("boxes")) {
            (Some(s), Some(b)) => (s, b),
            _ if outputs.len() >= 2 => (&outputs[0], &outputs[1]),
            _ => {
                return Err(IndexError::Detector(
                    "model must output `scores` [1,N,2] and `boxes` [1,N,4]".into(),
                ));
            }
        };
        let (_, scores) = scores.try_extract_tensor::<f32>().map_err(ort_err)?;
        let (_, boxes) = boxes.try_extract_tensor::<f32>().map_err(ort_err)?;
        // Undo the letterbox: model coords → frame coords in [0, 1].
        let (fx, fy) = (IN_W as f32 / rw as f32, IN_H as f32 / rh as f32);
        let (px, py) = (ox as f32 / IN_W as f32, oy as f32 / IN_H as f32);
        let faces = scores
            .as_chunks::<2>()
            .0
            .iter()
            .zip(boxes.as_chunks::<4>().0)
            .filter(|(s, _)| s[1] >= self.threshold)
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
        Ok(nms(faces, 0.3))
    }
}

impl SubjectDetector for OnnxFaceDetector {
    fn observe(&mut self, frame: &RgbFrame, t_ms: u64) -> Result<Option<SubjectBox>> {
        if t_ms < self.next_ms {
            return Ok(None);
        }
        self.next_ms = t_ms + self.interval_ms;
        let Some(img) = RgbImage::from_raw(frame.width, frame.height, frame.data.clone()) else {
            return Ok(None);
        };
        let faces = self.detect(&img)?;
        let primary = pick_primary(&faces, self.previous.as_ref());
        self.previous = primary;
        Ok(primary.map(|f| SubjectBox {
            t_ms,
            x: f.x,
            y: f.y,
            w: f.w,
            h: f.h,
            confidence: f.score,
            source: SubjectSource::Onnx,
        }))
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
}
