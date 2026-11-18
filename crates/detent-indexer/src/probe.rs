//! Container / stream metadata via `ffprobe -print_format json`.

use std::path::Path;
use std::process::{Command, Stdio};

use detent_core::index::{AudioStreamInfo, FreezeInterval, SourceInfo, VideoStreamInfo};
use detent_core::time::{FrameRate, us_to_ms};
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
    let rate = |k: &str| s.get(k).and_then(Value::as_str).and_then(parse_rate);
    // A stream capture that lagged has timestamp gaps, which pull the
    // *average* rate below the nominal one (30 fps reads as 27.67). Use the
    // nominal rate when it is a little above the average; a far higher one
    // (field rate of interlaced video, 1000 fps timebases) is not a frame rate.
    let fps = match (rate("avg_frame_rate"), rate("r_frame_rate")) {
        (Some(avg), Some(r))
            if r.as_f64() > avg.as_f64() * 1.005
                && r.as_f64() <= avg.as_f64() * 1.5
                && r.as_f64() <= 120.0 =>
        {
            Some(r)
        }
        (Some(avg), _) => Some(avg),
        (None, r) => r,
    }
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

/// Stretches where the video stream has no frames at all (`[start, end)` in
/// ms from the first frame, on the `fps` grid): a lagging stream capture
/// stores nothing while the stream stalls. Reads packet timestamps only (no
/// decoding). See [`timestamp_gaps`] for what counts as a gap.
pub fn frame_gaps(tools: &FfmpegTools, path: &Path, fps: FrameRate) -> Result<Vec<FreezeInterval>> {
    let out = Command::new(&tools.ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
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
    let pts: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().trim_end_matches(',').parse().ok())
        .collect();
    Ok(timestamp_gaps(pts, fps, 150))
}

/// Frame-timestamp gaps of at least `min_ms` and three frame periods.
///
/// Files that store frames only when the picture changes (some screen
/// recorders) have gaps whenever the screen is still; when gaps cover more
/// than half of the file its timestamps say nothing about lag, so none are
/// reported. (A badly lagging stream capture can lose a quarter of its
/// frames, so the cut-off sits well above that.)
pub fn timestamp_gaps(mut pts: Vec<f64>, fps: FrameRate, min_ms: u64) -> Vec<FreezeInterval> {
    pts.retain(|t| t.is_finite());
    pts.sort_by(f64::total_cmp);
    pts.dedup();
    let (Some(&first), Some(&last)) = (pts.first(), pts.last()) else {
        return Vec::new();
    };
    let rate = fps.as_f64();
    let period = 1.0 / rate;
    let min_gap = (3.0 * period).max(min_ms as f64 / 1_000.0);
    let mut out = Vec::new();
    let mut covered = 0.0;
    for w in pts.windows(2) {
        if w[1] - w[0] < min_gap {
            continue;
        }
        // The frame at w[0] is held; repeats start one frame later.
        let a = ((w[0] - first) * rate).round() as i64 + 1;
        let b = ((w[1] - first) * rate).round() as i64;
        if b <= a {
            continue;
        }
        covered += w[1] - w[0];
        out.push(FreezeInterval {
            start_ms: us_to_ms(fps.frame_to_us(a)),
            end_ms: us_to_ms(fps.frame_to_us(b)),
            still: false,
        });
    }
    if covered > 0.5 * (last - first).max(period) {
        return Vec::new();
    }
    out
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
    fn lagged_captures_keep_their_nominal_rate() {
        let rate = |avg: &str, r: &str| {
            let j = json!({
                "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920,
                             "height": 1080, "avg_frame_rate": avg, "r_frame_rate": r}],
                "format": {"duration": "30"}
            });
            let v = parse_probe(&j, "x").unwrap().video.unwrap();
            (v.fps.num, v.fps.den)
        };
        assert_eq!(rate("83/3", "30/1"), (30, 1), "gaps lower the average");
        assert_eq!(
            rate("30000/1001", "60000/1001"),
            (30_000, 1_001),
            "field rate"
        );
        assert_eq!(rate("25/1", "1000/1"), (25, 1), "timebase, not a rate");
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

    #[test]
    fn timestamp_gaps_find_a_stall_and_ignore_change_only_captures() {
        let fps = FrameRate::new(30, 1).unwrap();
        // 30 fps from 5.0 s, nothing between 6.0 and 6.5 s, then 30 fps again.
        let mut pts: Vec<f64> = (0..=30).map(|k| 5.0 + f64::from(k) / 30.0).collect();
        pts.extend((0..60).map(|k| 6.5 + f64::from(k) / 30.0));
        let gaps = timestamp_gaps(pts.clone(), fps, 150);
        assert_eq!(
            gaps,
            vec![FreezeInterval {
                start_ms: 1_033,
                end_ms: 1_500,
                still: false
            }]
        );
        // A 100 ms hiccup is too short.
        let mut short: Vec<f64> = (0..=30).map(|k| f64::from(k) / 30.0).collect();
        short.extend((0..30).map(|k| 1.1 + f64::from(k) / 30.0));
        assert!(timestamp_gaps(short, fps, 150).is_empty());
        // Frames only when something changes: gaps everywhere, no lag.
        let sparse: Vec<f64> = (0..20).map(|k| f64::from(k) * 0.5).collect();
        assert!(timestamp_gaps(sparse, fps, 150).is_empty());
    }
}
