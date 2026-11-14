//! Container / stream metadata via `ffprobe -print_format json`.

use std::path::Path;
use std::process::{Command, Stdio};

use detent_core::index::{AudioStreamInfo, SourceInfo, VideoStreamInfo};
use detent_core::time::FrameRate;
use serde_json::Value;

use crate::error::{IndexError, Result};
use crate::ffmpeg::FfmpegTools;

/// Runs `ffprobe` on `path`.
pub fn probe(tools: &FfmpegTools, path: &Path) -> Result<SourceInfo> {
    let out = Command::new(&tools.ffprobe)
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .map_err(|source| IndexError::ToolNotFound {
            tool: "ffprobe",
            path: tools.ffprobe.display().to_string(),
            source,
        })?;
    if !out.status.success() {
        return Err(IndexError::ToolFailed {
            tool: "ffprobe",
            status: out.status.to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_owned(),
        });
    }
    let json: Value = serde_json::from_slice(&out.stdout).map_err(|source| IndexError::Json {
        what: "ffprobe output".into(),
        source,
    })?;
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    parse_probe(&json, &display_path(&abs))
}

/// Strips the Windows verbatim prefix so paths stay readable and FFmpeg-friendly.
pub fn display_path(p: &Path) -> String {
    let s = p.display().to_string();
    s.strip_prefix(r"\\?\").map(str::to_owned).unwrap_or(s)
}

/// Parses `ffprobe` JSON (separated for testing).
pub fn parse_probe(json: &Value, path: &str) -> Result<SourceInfo> {
    let streams = json
        .get("streams")
        .and_then(Value::as_array)
        .ok_or_else(|| IndexError::Probe("missing `streams`".into()))?;
    let is_attached_pic = |s: &Value| {
        s.pointer("/disposition/attached_pic")
            .and_then(Value::as_i64)
            .is_some_and(|v| v == 1)
    };
    let video = streams
        .iter()
        .find(|s| {
            s.get("codec_type").and_then(Value::as_str) == Some("video") && !is_attached_pic(s)
        })
        .map(parse_video)
        .transpose()?;
    let audio = streams
        .iter()
        .find(|s| s.get("codec_type").and_then(Value::as_str) == Some("audio"))
        .map(parse_audio)
        .transpose()?;
    if video.is_none() && audio.is_none() {
        return Err(IndexError::NoStreams(path.to_owned()));
    }
    let seconds = json
        .pointer("/format/duration")
        .and_then(as_f64)
        .or_else(|| {
            streams
                .iter()
                .filter_map(|s| s.get("duration").and_then(as_f64))
                .reduce(f64::max)
        })
        .ok_or_else(|| IndexError::Probe("no duration".into()))?;
    Ok(SourceInfo {
        path: path.to_owned(),
        duration_ms: (seconds * 1_000.0).round().max(0.0) as u64,
        video,
        audio,
    })
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn parse_rate(s: &str) -> Option<FrameRate> {
    let (n, d) = s.split_once('/').unwrap_or((s, "1"));
    let (n, d): (u64, u64) = (n.trim().parse().ok()?, d.trim().parse().ok()?);
    if n == 0 || d == 0 {
        return None;
    }
    let g = gcd(n, d);
    FrameRate::new(u32::try_from(n / g).ok()?, u32::try_from(d / g).ok()?).ok()
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn parse_video(s: &Value) -> Result<VideoStreamInfo> {
    let dim = |k: &str| {
        s.get(k)
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .filter(|&v| v > 0)
            .ok_or_else(|| IndexError::Probe(format!("video stream missing `{k}`")))
    };
    let (mut w, mut h) = (dim("width")?, dim("height")?);
    let rotation = s
        .get("side_data_list")
        .and_then(Value::as_array)
        .and_then(|l| l.iter().find_map(|d| d.get("rotation").and_then(as_f64)))
        .or_else(|| s.pointer("/tags/rotate").and_then(as_f64))
        .unwrap_or(0.0);
    if (rotation.abs() - 90.0).abs() < 1.0 || (rotation.abs() - 270.0).abs() < 1.0 {
        std::mem::swap(&mut w, &mut h);
    }
    let fps = ["avg_frame_rate", "r_frame_rate"]
        .iter()
        .find_map(|k| s.get(*k).and_then(Value::as_str).and_then(parse_rate))
        .ok_or_else(|| IndexError::Probe("video stream has no usable frame rate".into()))?;
    Ok(VideoStreamInfo {
        width: w,
        height: h,
        fps,
        codec: s
            .get("codec_name")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned(),
    })
}

fn parse_audio(s: &Value) -> Result<AudioStreamInfo> {
    Ok(AudioStreamInfo {
        sample_rate: s
            .get("sample_rate")
            .and_then(as_f64)
            .map(|v| v as u32)
            .ok_or_else(|| IndexError::Probe("audio stream missing sample_rate".into()))?,
        channels: s
            .get("channels")
            .and_then(Value::as_u64)
            .map_or(0, |c| c as u32),
        codec: s
            .get("codec_name")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_typical_mp4() {
        let j = json!({
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
                 "avg_frame_rate": "30000/1001", "r_frame_rate": "30000/1001"},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2}
            ],
            "format": {"duration": "12.345"}
        });
        let s = parse_probe(&j, "/a.mp4").unwrap();
        assert_eq!(s.duration_ms, 12_345);
        let v = s.video.unwrap();
        assert_eq!(
            (v.width, v.height, v.fps.num, v.fps.den),
            (1920, 1080, 30_000, 1_001)
        );
        assert_eq!(s.audio.unwrap().sample_rate, 48_000);
    }

    #[test]
    fn handles_rotation_cover_art_and_bad_rates() {
        let j = json!({
            "streams": [
                {"codec_type": "video", "codec_name": "mjpeg", "width": 600, "height": 600,
                 "avg_frame_rate": "0/0", "disposition": {"attached_pic": 1}},
                {"codec_type": "video", "codec_name": "hevc", "width": 1920, "height": 1080,
                 "avg_frame_rate": "0/0", "r_frame_rate": "60/1",
                 "side_data_list": [{"rotation": -90}]}
            ],
            "format": {"duration": "1.0"}
        });
        let v = parse_probe(&j, "x").unwrap().video.unwrap();
        assert_eq!((v.width, v.height, v.fps.num), (1080, 1920, 60));
    }

    #[test]
    fn rejects_streamless_files() {
        let j = json!({"streams": [{"codec_type": "data"}], "format": {"duration": "1"}});
        assert!(matches!(
            parse_probe(&j, "x"),
            Err(IndexError::NoStreams(_))
        ));
    }

    #[test]
    fn reduces_rates() {
        assert_eq!(parse_rate("50/2").unwrap(), FrameRate::new(25, 1).unwrap());
        assert!(parse_rate("0/0").is_none());
    }
}
