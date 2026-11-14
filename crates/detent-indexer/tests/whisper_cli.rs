//! Real ASR through `whisper-cli`, including acoustic re-timing.
//!
//! Runs only when `DETENT_WHISPER_MODEL` (a ggml model) is set, `whisper-cli`
//! is reachable (PATH or `DETENT_WHISPER_CLI`), and FFmpeg has the `flite`
//! speech synthesiser. Otherwise it prints why and passes.

#![cfg(feature = "ffmpeg-cli")]

use std::path::{Path, PathBuf};
use std::process::Command;

use detent_indexer::ffmpeg::FfmpegTools;
use detent_indexer::{IndexOptions, SaliencyTracker, WhisperCli, WhisperCliConfig, index_file};

fn speech_fixture(dir: &Path) -> Option<PathBuf> {
    let tools = FfmpegTools::from_env();
    let out = dir.join("speech.mp4");
    // Two sentences separated by 1.2 s of silence, over a test pattern.
    let ok = Command::new(&tools.ffmpeg)
        .args(["-nostdin", "-v", "error", "-y"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "flite=text='The quick brown fox jumps over the lazy dog.'",
        ])
        .args(["-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono:d=1.2"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "flite=text='Every cut lands in the silence.'",
        ])
        .args(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=10"])
        .args(["-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]"])
        .args(["-map", "3:v", "-map", "[a]", "-shortest", "-c:v", "libx264"])
        .args(["-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac"])
        .arg(&out)
        .status()
        .is_ok_and(|s| s.success());
    ok.then_some(out)
}

#[test]
fn transcribes_and_retimes_real_speech() {
    let Some(model) = std::env::var_os("DETENT_WHISPER_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: DETENT_WHISPER_MODEL not set");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let Some(media) = speech_fixture(dir.path()) else {
        eprintln!("skipping: ffmpeg with libflite not available");
        return;
    };
    let mut asr = WhisperCli::new(WhisperCliConfig::new(model)).unwrap();
    let out = index_file(
        &FfmpegTools::from_env(),
        &media,
        &mut asr,
        &mut SaliencyTracker::new(200),
        &IndexOptions::default(),
    )
    .unwrap();
    let words: Vec<String> = out
        .index
        .transcript
        .iter()
        .map(|w| w.word.to_lowercase())
        .collect();
    let text = words.join(" ");
    for needle in ["quick", "fox", "lazy", "silence"] {
        assert!(text.contains(needle), "missing {needle:?} in {text:?}");
    }
    // After re-timing no word may cover the 1.2 s gap between the sentences.
    let gap = out
        .index
        .audio
        .pauses
        .iter()
        .find(|p| p.end_ms - p.start_ms >= 900)
        .unwrap_or_else(|| panic!("gap not detected: {:?}", out.index.audio.pauses));
    for w in &out.index.transcript {
        assert!(
            w.end_ms <= gap.start_ms + 30 || w.start_ms + 30 >= gap.end_ms,
            "{w:?} overlaps the pause {gap:?}"
        );
    }
}
