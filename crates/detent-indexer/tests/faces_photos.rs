//! Runs the built-in (pure-Rust) face detector against real photos.
//!
//! Set `DETENT_FACE_IMAGES` to a directory of `.jpg` photos with people;
//! skips with a message otherwise.

#![cfg(feature = "faces")]

use std::path::PathBuf;

use detent_core::index::SubjectSource;
use detent_indexer::RgbFrame;
use detent_indexer::faces::{self, TractFaces};
use detent_indexer::saliency::SubjectDetector;
use detent_indexer::ultraface::FaceBackend;

#[test]
fn detects_and_tracks_faces_in_real_photos() {
    let Some(dir) = std::env::var_os("DETENT_FACE_IMAGES").map(PathBuf::from) else {
        eprintln!("skipping: set DETENT_FACE_IMAGES");
        return;
    };
    let mut det = TractFaces::new(faces::DEFAULT_THRESHOLD).unwrap();
    let mut photos: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("jpg")))
        .collect();
    photos.sort();
    assert!(!photos.is_empty(), "no .jpg files in {}", dir.display());

    for (i, p) in photos.iter().enumerate() {
        let img = image::open(p).unwrap().to_rgb8();
        let found = det.detect(&img).unwrap();
        eprintln!("{}: {} faces", p.display(), found.len());
        assert!(!found.is_empty(), "no face found in {}", p.display());
        for f in &found {
            assert!(f.x >= 0.0 && f.y >= 0.0 && f.x + f.w <= 1.0 + 1e-4 && f.y + f.h <= 1.0 + 1e-4);
            assert!(f.w > 0.0 && f.h > 0.0 && (0.7..=1.0).contains(&f.score));
            let aspect = (f.w * img.width() as f32) / (f.h * img.height() as f32);
            assert!((0.45..=1.6).contains(&aspect), "distorted box {f:?}");
        }

        // The same photo as a 1 s "video": after the first sighting confirms
        // it, the primary subject must be a face and must not move.
        let frame = RgbFrame {
            width: img.width(),
            height: img.height(),
            data: img.into_raw(),
        };
        let mut tracker = faces::tracker(1).unwrap();
        let boxes: Vec<_> = (0..10_u64)
            .filter_map(|k| {
                tracker
                    .observe(&frame, 10_000 * i as u64 + k * 100)
                    .unwrap()
            })
            .collect();
        let boxes: Vec<_> = boxes
            .into_iter()
            .filter(|b| b.source == SubjectSource::Face)
            .collect();
        assert_eq!(boxes.len(), 9);
        assert!(boxes.windows(2).all(|w| (w[0].x - w[1].x).abs() < 1e-6));
    }
}
