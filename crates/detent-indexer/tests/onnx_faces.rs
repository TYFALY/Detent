//! Runs the UltraFace adapter against a real model and real photos.
//!
//! Requires `--features onnx` plus:
//! * `ORT_DYLIB_PATH` → `onnxruntime.dll` / `libonnxruntime.so` (1.24+)
//! * `DETENT_ONNX_MODEL` → `version-RFB-320.onnx`
//! * `DETENT_ONNX_IMAGES` → a directory of `.jpg` photos with people
//!
//! Skips with a message when any of them is missing.

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use detent_indexer::RgbFrame;
use detent_indexer::onnx::OnnxFaceDetector;
use detent_indexer::saliency::SubjectDetector;

fn env(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

#[test]
fn detects_and_tracks_faces_in_real_photos() {
    let (Some(_), Some(model), Some(dir)) = (
        env("ORT_DYLIB_PATH"),
        env("DETENT_ONNX_MODEL"),
        env("DETENT_ONNX_IMAGES"),
    ) else {
        eprintln!("skipping: set ORT_DYLIB_PATH, DETENT_ONNX_MODEL and DETENT_ONNX_IMAGES");
        return;
    };
    let mut det = OnnxFaceDetector::new(&model, 0.7, 1).unwrap();
    let mut photos: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("jpg")))
        .collect();
    photos.sort();
    assert!(!photos.is_empty(), "no .jpg files in {}", dir.display());

    for (i, p) in photos.iter().enumerate() {
        let img = image::open(p).unwrap().to_rgb8();
        let faces = det.detect(&img).unwrap();
        eprintln!("{}: {} faces", p.display(), faces.len());
        assert!(!faces.is_empty(), "no face found in {}", p.display());
        for f in &faces {
            assert!(f.x >= 0.0 && f.y >= 0.0 && f.x + f.w <= 1.0 + 1e-4 && f.y + f.h <= 1.0 + 1e-4);
            assert!(f.w > 0.0 && f.h > 0.0 && (0.7..=1.0).contains(&f.score));
            // Faces are roughly square; letterboxing must not distort them.
            let aspect = (f.w * img.width() as f32) / (f.h * img.height() as f32);
            assert!((0.45..=1.6).contains(&aspect), "distorted box {f:?}");
        }

        // Feed the same photo as a 1 s "video": the primary subject must be stable.
        let frame = RgbFrame {
            width: img.width(),
            height: img.height(),
            data: img.into_raw(),
        };
        let mut boxes = Vec::new();
        for k in 0..10_u64 {
            if let Some(b) = det.observe(&frame, 10_000 * i as u64 + k * 100).unwrap() {
                boxes.push(b);
            }
        }
        assert_eq!(boxes.len(), 10);
        assert!(boxes.windows(2).all(|w| (w[0].x - w[1].x).abs() < 1e-6));
    }
}
