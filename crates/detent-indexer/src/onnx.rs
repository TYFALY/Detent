//! ONNX Runtime face detection (feature `onnx`).
//!
//! Targets the UltraFace family (`version-RFB-320.onnx`, 1.2 MB): input
//! `1x3x240x320` normalised as `(px - 127) / 128`, outputs `scores [1,N,2]`
//! and `boxes [1,N,4]` (normalised `x1,y1,x2,y2`). The highest-scoring face
//! above the threshold becomes the subject box. Detection runs at most once
//! per `interval_ms`; the camera spring smooths between samples.

use std::path::Path;

use detent_core::index::{SubjectBox, SubjectSource};
use image::{RgbImage, imageops};
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{IndexError, Result};
use crate::saliency::SubjectDetector;
use crate::source::RgbFrame;

const IN_W: u32 = 320;
const IN_H: u32 = 240;

/// UltraFace detector.
pub struct OnnxFaceDetector {
    session: Session,
    threshold: f32,
    interval_ms: u64,
    next_ms: u64,
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
    /// Loads a model file.
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
            input: vec![0.0; (3 * IN_W * IN_H) as usize],
        })
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
        let img = imageops::resize(&img, IN_W, IN_H, imageops::FilterType::Triangle);
        let plane = (IN_W * IN_H) as usize;
        for (i, px) in img.pixels().enumerate() {
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
        if outputs.len() < 2 {
            return Err(IndexError::Detector(
                "expected `scores` and `boxes` outputs".into(),
            ));
        }
        let (_, scores) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        let (_, boxes) = outputs[1].try_extract_tensor::<f32>().map_err(ort_err)?;
        let best = scores
            .as_chunks::<2>()
            .0
            .iter()
            .zip(boxes.as_chunks::<4>().0)
            .map(|(s, b)| (s[1], b))
            .filter(|(s, _)| *s >= self.threshold)
            .max_by(|a, b| a.0.total_cmp(&b.0));
        Ok(best.map(|(score, b)| {
            let x1 = b[0].clamp(0.0, 1.0);
            let y1 = b[1].clamp(0.0, 1.0);
            let x2 = b[2].clamp(x1, 1.0);
            let y2 = b[3].clamp(y1, 1.0);
            SubjectBox {
                t_ms,
                x: x1,
                y: y1,
                w: x2 - x1,
                h: y2 - y1,
                confidence: score,
                source: SubjectSource::Onnx,
            }
        }))
    }
}
