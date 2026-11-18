//! ONNX Runtime face detection (feature `onnx`).
//!
//! Runs an UltraFace model file (see [`crate::ultraface`] for the format and
//! the tracking) through ONNX Runtime, e.g. to use a GPU execution provider.
//! The default build already detects faces in pure Rust ([`crate::faces`]).
//!
//! ONNX Runtime is loaded at run time (`ORT_DYLIB_PATH`), so no native
//! libraries are needed to build detent.

use std::path::Path;

use detent_core::index::{SubjectBox, SubjectSource};
use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{IndexError, Result};
use crate::saliency::SubjectDetector;
use crate::source::RgbFrame;
use crate::ultraface::{self, Face, FaceBackend, FaceTracker, IN_H, IN_W};

pub use crate::ultraface::{nms, pick_primary};

/// UltraFace run by ONNX Runtime.
struct OrtFaces {
    session: Session,
    threshold: f32,
    input: Vec<f32>,
}

fn ort_err(e: impl std::fmt::Display) -> IndexError {
    IndexError::Detector(e.to_string())
}

impl FaceBackend for OrtFaces {
    fn detect(&mut self, img: &RgbImage) -> Result<Vec<Face>> {
        let lb = ultraface::preprocess(img, &mut self.input);
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
        Ok(ultraface::decode(scores, boxes, lb, self.threshold))
    }
}

/// UltraFace through ONNX Runtime, with the same tracking (hysteresis, hold,
/// saliency fallback) as the built-in detector.
#[derive(Debug)]
pub struct OnnxFaceDetector {
    tracker: FaceTracker,
}

impl OnnxFaceDetector {
    /// Loads a model. Emits at most one subject box per `interval_ms`.
    pub fn new(model: &Path, threshold: f32, interval_ms: u64) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort_err)?
            .commit_from_file(model)
            .map_err(ort_err)?;
        let backend = OrtFaces {
            session,
            threshold,
            input: vec![0.0; (3 * IN_W * IN_H) as usize],
        };
        Ok(Self {
            tracker: FaceTracker::new(Box::new(backend), interval_ms, SubjectSource::Onnx),
        })
    }

    /// Detects all faces (after NMS) in an RGB image.
    pub fn detect(&mut self, img: &RgbImage) -> Result<Vec<Face>> {
        self.tracker.detect(img)
    }
}

impl SubjectDetector for OnnxFaceDetector {
    fn observe(&mut self, frame: &RgbFrame, t_ms: u64) -> Result<Option<SubjectBox>> {
        self.tracker.observe(frame, t_ms)
    }

    fn faceless_ms(&mut self) -> Option<Vec<(u64, u64)>> {
        self.tracker.faceless_ms()
    }

    fn lips_still_ms(&mut self) -> Option<Vec<(u64, u64)>> {
        self.tracker.lips_still_ms()
    }
}
