//! Executes renders and reports throughput.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use detent_core::ResolvedTimeline;
use detent_core::captions::to_ass;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::encoder::{EncoderChoice, FfmpegInfo, Quality, Tools};
use crate::error::{RenderError, Result, io_err};
use crate::graph::compile;

/// Render mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum RenderMode {
    /// Full filtergraph render (camera, captions, crossfades).
    #[default]
    Full,
    /// Fast preview: keyframe-aligned stream copy when the timeline needs no
    /// reframing or captions; otherwise a draft-quality re-encode.
    Preview,
}

/// Render options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RenderOptions {
    /// Quality preset.
    pub quality: Quality,
    /// Encoder policy.
    pub encoder: EncoderChoice,
    /// Mode.
    pub mode: RenderMode,
}

/// Live progress sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
    /// Frames written.
    pub frames: u64,
    /// Output position (µs).
    pub out_time_us: i64,
    /// Fraction complete in `[0, 1]`.
    pub fraction: f64,
}

/// Throughput metrics for a finished render.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RenderMetrics {
    /// `filtergraph`, `stream_copy` or `draft_reencode`.
    pub mode: String,
    /// Encoder used (`null` for stream copy).
    pub encoder: Option<String>,
    /// Output file.
    pub output_path: String,
    /// Output bytes.
    pub output_bytes: u64,
    /// Output duration (ms).
    pub output_duration_ms: u64,
    /// Frames written.
    pub frames: u64,
    /// Wall-clock time (ms).
    pub wall_ms: u64,
    /// Encoded frames per wall-clock second.
    pub encode_fps: f64,
    /// Output seconds per wall-clock second.
    pub realtime_factor: f64,
    /// Total extra lead-in caused by snapping to keyframes (stream copy only).
    pub keyframe_drift_ms: Option<u64>,
}

/// Executes timelines with FFmpeg.
#[derive(Debug)]
pub struct Renderer {
    info: FfmpegInfo,
}

impl Renderer {
    /// Probes FFmpeg capabilities.
    pub fn new(tools: Tools) -> Result<Self> {
        Ok(Self {
            info: FfmpegInfo::probe(tools)?,
        })
    }

    /// Capabilities.
    pub fn info(&self) -> &FfmpegInfo {
        &self.info
    }

    /// Renders `t` to `output`.
    pub fn render(
        &self,
        t: &ResolvedTimeline,
        output: &Path,
        opts: &RenderOptions,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<RenderMetrics> {
        let violations = t.verify();
        if !violations.is_empty() {
            return Err(RenderError::InvalidTimeline(violations));
        }
        let output = absolute(output)?;
        if opts.mode == RenderMode::Preview {
            if copy_eligible(t) {
                return self.stream_copy(t, &output, on_progress);
            }
            let draft = RenderOptions {
                quality: Quality::Draft,
                ..*opts
            };
            let mut m = self.full(t, &output, &draft, on_progress)?;
            m.mode = "draft_reencode".into();
            return Ok(m);
        }
        self.full(t, &output, opts, on_progress)
    }

    fn full(
        &self,
        t: &ResolvedTimeline,
        output: &Path,
        opts: &RenderOptions,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<RenderMetrics> {
        let encoder = self.info.select(opts.encoder)?;
        let work = tempfile::Builder::new()
            .prefix("detent-render-")
            .tempdir()
            .map_err(io_err(Path::new("temp dir")))?;
        let ass = match &t.captions {
            Some(c) if !c.blocks.is_empty() => {
                let doc = to_ass(
                    &c.blocks,
                    c.style,
                    c.position,
                    t.output.width,
                    t.output.height,
                );
                let p = work.path().join("captions.ass");
                std::fs::write(&p, doc).map_err(io_err(&p))?;
                Some("captions.ass")
            }
            _ => None,
        };
        let graph = compile(t, encoder, ass);
        let graph_path = work.path().join("graph.txt");
        std::fs::write(&graph_path, &graph.filter).map_err(io_err(&graph_path))?;

        let mut cmd = Command::new(&self.info.tools.ffmpeg);
        cmd.current_dir(work.path())
            .args(["-nostdin", "-hide_banner", "-v", "error", "-y"])
            .args(encoder.global_args());
        for input in &graph.inputs {
            cmd.arg("-ss")
                .arg(format!("{:.6}", input.seek_s))
                .arg("-t")
                .arg(format!("{:.6}", input.duration_s))
                .arg("-i")
                .arg(&t.source.path);
        }
        if self.info.supports_option_files() {
            cmd.args(["-/filter_complex", "graph.txt"]);
        } else {
            cmd.args(["-filter_complex_script", "graph.txt"]);
        }
        cmd.args(["-map", "[vout]"]);
        if graph.has_audio {
            cmd.args([
                "-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            ]);
        }
        cmd.args(encoder.codec_args(opts.quality))
            .arg("-r")
            .arg(format!("{}/{}", t.output.fps.num, t.output.fps.den))
            .args(["-fps_mode", "cfr", "-movflags", "+faststart"]);
        let mut m = run_with_progress(cmd, output, t.duration_us, on_progress)?;
        // Defence in depth: FFmpeg can silently drop frames (e.g. a filter
        // expression failing at runtime). A short file is an error, not a result.
        let expected: i64 = t.segments.iter().map(|s| s.video.frames()).sum();
        if m.frames != expected as u64 {
            return Err(RenderError::FrameCountMismatch {
                expected: expected as u64,
                written: m.frames,
            });
        }
        m.mode = "filtergraph".into();
        m.encoder = Some(encoder.ffmpeg_name().into());
        Ok(m)
    }

    fn stream_copy(
        &self,
        t: &ResolvedTimeline,
        output: &Path,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<RenderMetrics> {
        let keys = keyframes(&self.info.tools, &t.source.path)?;
        let work = tempfile::Builder::new()
            .prefix("detent-preview-")
            .tempdir()
            .map_err(io_err(Path::new("temp dir")))?;
        let src = t.source.path.replace('\\', "/").replace('\'', r"'\''");
        let mut list = String::from("ffconcat version 1.0\n");
        let mut drift_us = 0_i64;
        for seg in &t.segments {
            let start = seg.video.start_us;
            let key = keys
                .iter()
                .copied()
                .take_while(|&k| k <= start + 1)
                .last()
                .unwrap_or(0);
            drift_us += start - key;
            list.push_str(&format!(
                "file '{src}'\ninpoint {:.6}\noutpoint {:.6}\n",
                key as f64 / 1e6,
                seg.video.end_us as f64 / 1e6
            ));
        }
        let list_path = work.path().join("list.ffconcat");
        std::fs::write(&list_path, list).map_err(io_err(&list_path))?;
        let mut cmd = Command::new(&self.info.tools.ffmpeg);
        cmd.current_dir(work.path())
            .args(["-nostdin", "-hide_banner", "-v", "error", "-y"])
            .args(["-f", "concat", "-safe", "0", "-i", "list.ffconcat"])
            .args(["-map", "0:v:0", "-map", "0:a:0?", "-c", "copy"])
            .args(["-avoid_negative_ts", "make_zero", "-movflags", "+faststart"]);
        let mut m = run_with_progress(cmd, output, t.duration_us + drift_us, on_progress)?;
        m.mode = "stream_copy".into();
        m.keyframe_drift_ms = Some(detent_core::time::us_to_ms(drift_us));
        Ok(m)
    }
}

/// True when the timeline can be previewed without decoding.
pub fn copy_eligible(t: &ResolvedTimeline) -> bool {
    !t.needs_reframe() && t.captions.as_ref().is_none_or(|c| c.blocks.is_empty())
}

fn absolute(p: &Path) -> Result<PathBuf> {
    if p.is_absolute() {
        return Ok(p.to_path_buf());
    }
    let cwd = std::env::current_dir().map_err(io_err(p))?;
    Ok(cwd.join(p))
}

/// Keyframe presentation times (µs) of the first video stream.
fn keyframes(tools: &Tools, path: &str) -> Result<Vec<i64>> {
    let out = Command::new(&tools.ffprobe)
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .args(["packet=pts_time,flags", "-of", "csv=p=0"])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .map_err(|source| RenderError::ToolNotFound {
            tool: "ffprobe",
            path: tools.ffprobe.display().to_string(),
            source,
        })?;
    if !out.status.success() {
        return Err(RenderError::ToolFailed {
            tool: "ffprobe",
            status: out.status.to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_owned(),
        });
    }
    let mut keys: Vec<i64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let (pts, flags) = l.split_once(',')?;
            flags
                .contains('K')
                .then(|| pts.trim().parse::<f64>().ok())
                .flatten()
                .map(|s| (s * 1e6).round() as i64)
        })
        .collect();
    keys.sort_unstable();
    Ok(keys)
}

/// Runs FFmpeg with `-progress pipe:1`, streaming progress to the callback.
fn run_with_progress(
    mut cmd: Command,
    output: &Path,
    expected_us: i64,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<RenderMetrics> {
    cmd.args(["-progress", "pipe:1", "-nostats"])
        .arg(output)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    tracing::debug!(?cmd, "spawning ffmpeg");
    let started = Instant::now();
    let mut child = cmd.spawn().map_err(|source| RenderError::ToolNotFound {
        tool: "ffmpeg",
        path: "ffmpeg".into(),
        source,
    })?;
    let stderr = child.stderr.take().map(|mut e| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = e.read_to_end(&mut buf);
            let s = String::from_utf8_lossy(&buf);
            let tail: String = s
                .chars()
                .rev()
                .take(4_000)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            tail.trim().to_owned()
        })
    });
    let mut frames = 0_u64;
    if let Some(out) = child.stdout.take() {
        let mut out_time_us = 0_i64;
        for line in BufReader::new(out)
            .lines()
            .map_while(std::result::Result::ok)
        {
            if let Some(v) = line.strip_prefix("frame=") {
                frames = v.trim().parse().unwrap_or(frames);
            } else if let Some(v) = line.strip_prefix("out_time_us=") {
                out_time_us = v.trim().parse().unwrap_or(out_time_us);
            } else if line.starts_with("progress=") {
                on_progress(Progress {
                    frames,
                    out_time_us,
                    fraction: (out_time_us as f64 / expected_us.max(1) as f64).clamp(0.0, 1.0),
                });
            }
        }
    }
    let status = child.wait().map_err(io_err(output))?;
    let stderr = stderr.and_then(|h| h.join().ok()).unwrap_or_default();
    if !status.success() {
        return Err(RenderError::ToolFailed {
            tool: "ffmpeg",
            status: status.to_string(),
            stderr,
        });
    }
    let wall = started.elapsed().as_secs_f64();
    let bytes = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    let out_s = expected_us as f64 / 1e6;
    Ok(RenderMetrics {
        mode: String::new(),
        encoder: None,
        output_path: output.display().to_string(),
        output_bytes: bytes,
        output_duration_ms: detent_core::time::us_to_ms(expected_us),
        frames,
        wall_ms: (wall * 1_000.0).round() as u64,
        encode_fps: if wall > 0.0 {
            frames as f64 / wall
        } else {
            0.0
        },
        realtime_factor: if wall > 0.0 { out_s / wall } else { 0.0 },
        keyframe_drift_ms: None,
    })
}
