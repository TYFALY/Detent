//! End-to-end indexing of a synthetic file generated with FFmpeg's lavfi.
//! Skipped (with a message) when FFmpeg is not installed.

#![cfg(feature = "ffmpeg-cli")]

use std::path::{Path, PathBuf};
use std::process::Command;

use detent_core::index::PauseKind;
use detent_indexer::ffmpeg::FfmpegTools;
use detent_indexer::{
    IndexOptions, NoTranscript, SaliencyTracker, contact_sheet_range, index_file,
};

/// 6 s, 320x180 @ 25 fps: warm colour for 3 s then cool colour (one hard cut);
/// 220 Hz tone with a silent gap 1.5-2.2 s.
fn make_fixture(dir: &Path) -> Option<PathBuf> {
    let tools = FfmpegTools::from_env();
    if !tools.available() {
        eprintln!("skipping: ffmpeg/ffprobe not found");
        return None;
    }
    let out = dir.join("fixture.mp4");
    let status = Command::new(&tools.ffmpeg)
        .args(["-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i"])
        .arg("color=c=0xC83C28:s=320x180:r=25:d=3[a];color=c=0x1E50D2:s=320x180:r=25:d=3[b];[a][b]concat=n=2:v=1:a=0,noise=alls=8:allf=t")
        .args(["-f", "lavfi", "-i"])
        .arg("aevalsrc='if(between(t,1.5,2.2),0.0005*random(0),0.3*sin(2*PI*220*t))':s=48000:d=6")
        .args(["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
        .arg(&out)
        .status()
        .ok()?;
    status.success().then_some(out)
}

#[test]
fn indexes_real_file() {
    let dir = tempfile::tempdir().unwrap();
    let Some(path) = make_fixture(dir.path()) else {
        return;
    };
    let tools = FfmpegTools::from_env();
    let opts = IndexOptions {
        contact_sheet_path: Some(dir.path().join("grid.jpg")),
        ..IndexOptions::default()
    };
    let out = index_file(
        &tools,
        &path,
        &mut NoTranscript,
        &mut SaliencyTracker::new(200),
        &opts,
    )
    .unwrap();
    let idx = &out.index;
    assert!(
        idx.source.duration_ms.abs_diff(6_000) <= 60,
        "{}",
        idx.source.duration_ms
    );
    let v = idx.source.video.as_ref().unwrap();
    assert_eq!((v.width, v.height, v.fps.num, v.fps.den), (320, 180, 25, 1));

    // Exactly one hard cut, at frame 75 (3.0 s).
    assert_eq!(idx.visual.shots.len(), 1, "{:?}", idx.visual.shots);
    assert_eq!(idx.visual.shots[0].frame, 75);

    // The silent gap is found within ~40 ms (AAC priming may shift slightly).
    let gap = idx
        .audio
        .pauses
        .iter()
        .find(|p| p.kind == PauseKind::Silence && p.end_ms - p.start_ms > 500)
        .unwrap_or_else(|| panic!("{:?}", idx.audio.pauses));
    assert!(
        gap.start_ms.abs_diff(1_500) <= 50 && gap.end_ms.abs_diff(2_200) <= 50,
        "{gap:?}"
    );

    // ~100 hops per second of audio.
    assert!(idx.audio.rms_dbfs.len().abs_diff(600) <= 6);
    assert_eq!(out.stats.video_frames, 150);
    let sheet = idx.visual.contact_sheet.as_ref().unwrap();
    assert!(Path::new(&sheet.path).exists());
    assert_eq!(sheet.tile_ms.len(), 16);
    assert!(out.stats.realtime_factor > 0.0);
}

#[test]
fn renders_range_contact_sheet() {
    let dir = tempfile::tempdir().unwrap();
    let Some(path) = make_fixture(dir.path()) else {
        return;
    };
    let tools = FfmpegTools::from_env();
    let out = dir.path().join("range.jpg");
    let sheet = contact_sheet_range(&tools, &path, 2_000, 4_000, (3, 2), 96, &out).unwrap();
    assert_eq!(sheet.tile_ms.len(), 6);
    assert!(
        sheet.tile_ms.iter().all(|&t| (2_000..=4_000).contains(&t)),
        "{:?}",
        sheet.tile_ms
    );
    let img = image::open(&out).unwrap();
    assert_eq!(img.width(), 3 * 96);
}

/// A real FFmpeg cross-dissolve (0.8 s from 2.6 s) and a fade through
/// black (0.8 s from 4.8 s) are each found once, mid-transition.
#[test]
fn finds_dissolves_and_fades() {
    let tools = FfmpegTools::from_env();
    if !tools.available() {
        eprintln!("skipping: ffmpeg/ffprobe not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("dissolve.mp4");
    let graph = "testsrc2=s=320x180:r=25:d=3.4[a];\
mandelbrot=s=320x180:r=25,trim=duration=3,setpts=PTS-STARTPTS[b];\
smptehdbars=s=320x180:r=25:d=2[c];\
[a][b]xfade=transition=fade:duration=0.8:offset=2.6[ab];\
[ab][c]xfade=transition=fadeblack:duration=0.8:offset=4.8[v]";
    let ok = Command::new(&tools.ffmpeg)
        .args([
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-filter_complex",
            graph,
            "-map",
            "[v]",
        ])
        .args([
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&out)
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        eprintln!("skipping: lavfi sources unavailable");
        return;
    }
    let out = index_file(
        &tools,
        &out,
        &mut NoTranscript,
        &mut SaliencyTracker::new(200),
        &IndexOptions::default(),
    )
    .unwrap();
    let ms: Vec<u64> = out.index.visual.shots.iter().map(|s| s.t_ms).collect();
    assert_eq!(ms.len(), 2, "{:?}", out.index.visual.shots);
    // Transition centres: 3.0 s and 5.2 s.
    assert!(ms[0].abs_diff(3_000) <= 280, "{ms:?}");
    assert!(ms[1].abs_diff(5_200) <= 280, "{ms:?}");
}
