//! Built-in face detection (feature `faces`, on by default).
//!
//! Runs the UltraFace RFB-320 model (MIT licensed, 1.2 MB, embedded in the
//! binary) with [`tract`](https://github.com/sonos/tract), a pure-Rust
//! inference engine, so real face detection needs no native runtime and no
//! download. Pre- and post-processing live in [`crate::ultraface`].

use std::io::Cursor;

use detent_core::index::SubjectSource;
use image::RgbImage;
use tract_onnx::prelude::*;

use crate::error::{IndexError, Result};
use crate::ultraface::{self, Face, FaceBackend, FaceTracker, IN_H, IN_W};

/// The embedded model (`version-RFB-320.onnx` from
/// Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB, MIT).
pub const MODEL: &[u8] = include_bytes!("../models/ultraface-rfb-320.onnx");

/// Default score threshold.
pub const DEFAULT_THRESHOLD: f32 = 0.7;

type Plan = std::sync::Arc<TypedRunnableModel>;

/// UltraFace run by tract.
pub struct TractFaces {
    plan: Plan,
    threshold: f32,
    input: Vec<f32>,
}

impl std::fmt::Debug for TractFaces {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TractFaces")
            .field("threshold", &self.threshold)
            .finish_non_exhaustive()
    }
}

fn err(e: impl std::fmt::Display) -> IndexError {
    IndexError::Detector(e.to_string())
}

impl TractFaces {
    /// Loads the embedded model.
    pub fn new(threshold: f32) -> Result<Self> {
        let plan = tract_onnx::onnx()
            .model_for_read(&mut Cursor::new(MODEL))
            .map_err(err)?
            .with_input_fact(0, f32::fact([1, 3, IN_H as usize, IN_W as usize]).into())
            .map_err(err)?
            .into_optimized()
            .map_err(err)?
            .into_runnable()
            .map_err(err)?;
        Ok(Self {
            plan,
            threshold,
            input: vec![0.0; (3 * IN_W * IN_H) as usize],
        })
    }
}

impl FaceBackend for TractFaces {
    fn detect(&mut self, img: &RgbImage) -> Result<Vec<Face>> {
        let lb = ultraface::preprocess(img, &mut self.input);
        let tensor =
            Tensor::from_shape(&[1, 3, IN_H as usize, IN_W as usize], &self.input).map_err(err)?;
        let outputs = self.plan.run(tvec!(tensor.into())).map_err(err)?;
        // Scores are [1,N,2], boxes [1,N,4]; tell them apart by shape.
        let mut scores = None;
        let mut boxes = None;
        for o in &outputs {
            let slot = match o.shape().last() {
                Some(2) => &mut scores,
                Some(4) => &mut boxes,
                _ => continue,
            };
            let view = o.try_as_plain_ram().map_err(err)?;
            *slot = Some(view.as_slice::<f32>().map_err(err)?.to_vec());
        }
        let (Some(scores), Some(boxes)) = (scores, boxes) else {
            return Err(IndexError::Detector(
                "UltraFace must output scores [1,N,2] and boxes [1,N,4]".into(),
            ));
        };
        Ok(ultraface::decode(&scores, &boxes, lb, self.threshold))
    }
}

/// The default subject detector: the built-in face detector with saliency
/// fallback, one box per `interval_ms`.
pub fn tracker(interval_ms: u64) -> Result<FaceTracker> {
    Ok(FaceTracker::new(
        Box::new(TractFaces::new(DEFAULT_THRESHOLD)?),
        interval_ms,
        SubjectSource::Face,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn model_loads_and_finds_nothing_in_a_flat_frame() {
        let mut d = TractFaces::new(DEFAULT_THRESHOLD).unwrap();
        let img = RgbImage::from_pixel(320, 180, Rgb([90, 90, 90]));
        assert!(d.detect(&img).unwrap().is_empty());
    }
}
