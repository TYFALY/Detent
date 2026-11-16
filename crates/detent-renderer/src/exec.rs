//! Executes renders and reports throughput.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use detent_core::ResolvedTimeline;
use detent_core::captions::{AssExtras, to_ass_with};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::encoder::{Encoder, EncoderChoice, FfmpegInfo, Quality, Tools};
use crate::error::{RenderError, Result, io_err};
use crate::graph::{
    BATCH_SEGMENTS, ExtraInput, InputSpec, batches, compile, compile_batch, compile_join,
};

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
    /// Segments per FFmpeg process (`None`: [`BATCH_SEGMENTS`]). Longer
    /// timelines are rendered in lossless batches and joined, so very long
    /// EDLs never open hundreds of inputs at once.
    pub batch_segments: Option<usize>,
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
                let doc = to_ass_with(
                    &c.blocks,
                    c.style,
                    c.position,
                    t.output.width,
                    t.output.height,
                    &AssExtras {
                        headline: c.headline.as_deref(),
                        highlight_words: &c.highlight_words,
                        duration_us: t.duration_us,
                        headline_end_us: c.headline_end_us,
                        word_colors: &c.word_colors,
                        chapters: &c.chapters,
                    },
                );
                let p = work.path().join("captions.ass");
                std::fs::write(&p, doc).map_err(io_err(&p))?;
                Some("captions.ass")
            }
            _ => None,
        };
        // Inserts and music are read from the caller's working directory,
        // but FFmpeg runs in the temp dir: resolve them up front (and fail
        // early when one is missing).
        let mut extra_files: Vec<&str> = t.inserts.iter().map(|i| i.path.as_str()).collect();
        extra_files.extend(
            t.sound
                .as_ref()
                .and_then(|s| s.music.as_ref())
                .map(|m| m.path.as_str()),
        );
        let resolved = extra_files
            .into_iter()
            .map(|f| {
                let p = absolute(Path::new(f))?;
                std::fs::metadata(&p).map_err(io_err(&p))?;
                Ok((f.to_owned(), p))
            })
            .collect::<Result<Vec<(String, PathBuf)>>>()?;
        let n = t.segments.len();
        let batch = opts.batch_segments.unwrap_or(BATCH_SEGMENTS).max(1);
        let started = Instant::now();
        let expected: i64 = t.segments.iter().map(|s| s.video.frames()).sum();

        let mut m = if n <= batch {
            let graph = compile(t, encoder, ass);
            let mut cmd = self.command(work.path(), "graph.txt", &graph.filter)?;
            cmd.args(encoder.global_args());
            for input in &graph.inputs {
                source_input(&mut cmd, input, &t.source.path);
            }
            extra_inputs(&mut cmd, &graph.extra, &resolved, t);
            self.encode_args(&mut cmd, "graph.txt", graph.has_audio, t, encoder, opts);
            let mut m = run_with_progress(cmd, output, t.duration_us, on_progress)?;
            m.mode = "filtergraph".into();
            m
        } else {
            // Long EDLs: a bounded number of seeked inputs per process.
            // Each batch is rendered losslessly (FFV1 + float PCM in NUT,
            // which keeps exact timestamps), then one join pass applies the
            // cross-batch crossfades, grade, captions and the final encode.
            let ranges = batches(n, batch);
            let total_us = t.duration_us.max(1);
            let mut done_us = 0_i64;
            for (k, r) in ranges.iter().enumerate() {
                let graph = compile_batch(t, r.clone());
                let script = format!("batch{k}.txt");
                let mut cmd = self.command(work.path(), &script, &graph.filter)?;
                for input in &graph.inputs {
                    source_input(&mut cmd, input, &t.source.path);
                }
                self.filter_arg(&mut cmd, &script);
                cmd.args(["-map", "[vout]", "-c:v", "ffv1", "-level", "3"]);
                if graph.has_audio {
                    cmd.args(["-map", "[aout]", "-c:a", "pcm_f32le"]);
                }
                cmd.args(["-fps_mode", "passthrough", "-f", "nut"]);
                let part_us: i64 = t.segments[r.clone()].iter().map(|s| s.video.len_us()).sum();
                let part_frames: i64 = t.segments[r.clone()].iter().map(|s| s.video.frames()).sum();
                let path = work.path().join(format!("batch{k}.nut"));
                let base = done_us;
                let pm = run_with_progress(cmd, &path, part_us, &mut |p| {
                    let at = base + p.out_time_us;
                    on_progress(Progress {
                        frames: p.frames,
                        out_time_us: at,
                        fraction: (0.5 * at as f64 / total_us as f64).clamp(0.0, 0.5),
                    });
                })?;
                if pm.frames != part_frames as u64 {
                    return Err(RenderError::FrameCountMismatch {
                        expected: part_frames as u64,
                        written: pm.frames,
                    });
                }
                done_us += part_us;
            }
            let graph = compile_join(t, &ranges, encoder, ass);
            let mut cmd = self.command(work.path(), "join.txt", &graph.filter)?;
            cmd.args(encoder.global_args());
            for k in 0..ranges.len() {
                cmd.arg("-i").arg(format!("batch{k}.nut"));
            }
            extra_inputs(&mut cmd, &graph.extra, &resolved, t);
            self.encode_args(&mut cmd, "join.txt", graph.has_audio, t, encoder, opts);
            let mut m = run_with_progress(cmd, output, t.duration_us, &mut |p| {
                on_progress(Progress {
                    fraction: 0.5 + 0.5 * p.fraction,
                    ..p
                });
            })?;
            let wall = started.elapsed().as_secs_f64();
            m.wall_ms = (wall * 1_000.0).round() as u64;
            if wall > 0.0 {
                m.encode_fps = m.frames as f64 / wall;
                m.realtime_factor = t.duration_us as f64 / 1e6 / wall;
            }
            m.mode = "filtergraph_batched".into();
            m
        };
        // Defence in depth: FFmpeg can silently drop frames (e.g. a filter
        // expression failing at runtime). A short file is an error, not a result.
        if m.frames != expected as u64 {
            return Err(RenderError::FrameCountMismatch {
                expected: expected as u64,
                written: m.frames,
            });
        }
        m.encoder = Some(encoder.ffmpeg_name().into());
        Ok(m)
    }

    /// An FFmpeg command in `dir` with `filter` written to `script`.
    fn command(&self, dir: &Path, script: &str, filter: &str) -> Result<Command> {
        let p = dir.join(script);
        std::fs::write(&p, filter).map_err(io_err(&p))?;
        let mut cmd = Command::new(&self.info.tools.ffmpeg);
        cmd.current_dir(dir)
            .args(["-nostdin", "-hide_banner", "-v", "error", "-y"]);
        Ok(cmd)
    }

    fn filter_arg(&self, cmd: &mut Command, script: &str) {
        if self.info.supports_option_files() {
            cmd.arg("-/filter_complex").arg(script);
        } else {
            cmd.arg("-filter_complex_script").arg(script);
        }
    }

    /// Filter script, stream maps and the final encode settings.
    fn encode_args(
        &self,
        cmd: &mut Command,
        script: &str,
        has_audio: bool,
        t: &ResolvedTimeline,
        encoder: Encoder,
        opts: &RenderOptions,
    ) {
        self.filter_arg(cmd, script);
        cmd.args(["-map", "[vout]"]);
        if has_audio {
            cmd.args([
                "-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            ]);
        }
        cmd.args(encoder.codec_args(opts.quality))
            .arg("-r")
            .arg(format!("{}/{}", t.output.fps.num, t.output.fps.den))
            .args(["-fps_mode", "cfr", "-movflags", "+faststart"]);
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
    let o = &t.output;
    let finishing = o.opening_pop
        || o.grade != detent_core::edl::ColorGrade::None
        || o.audio_mix != detent_core::edl::AudioMix::Source
        || t.splices.iter().any(|s| s.flash)
        || !t.inserts.is_empty()
        || t.sound.is_some();
    !finishing && !t.needs_reframe() && t.captions.as_ref().is_none_or(|c| c.blocks.is_empty())
}

fn source_input(cmd: &mut Command, input: &InputSpec, src: &str) {
    cmd.arg("-ss")
        .arg(format!("{:.6}", input.seek_s))
        .arg("-t")
        .arg(format!("{:.6}", input.duration_s))
        .arg("-i")
        .arg(src);
}

/// Insert and music inputs, in the graph's order: stills loop at the output
/// frame rate, videos and music loop from their start.
fn extra_inputs(
    cmd: &mut Command,
    extra: &[ExtraInput],
    resolved: &[(String, PathBuf)],
    t: &ResolvedTimeline,
) {
    let fps = t.output.fps;
    for e in extra {
        let path = resolved
            .iter()
            .find(|(f, _)| f == e.path())
            .map_or_else(|| PathBuf::from(e.path()), |(_, p)| p.clone());
        let len_s = match e {
            ExtraInput::Still { len_s, .. } => {
                cmd.args(["-loop", "1", "-framerate"])
                    .arg(format!("{}/{}", fps.num, fps.den));
                len_s
            }
            ExtraInput::Looped { len_s, .. } => {
                cmd.args(["-stream_loop", "-1"]);
                len_s
            }
        };
        cmd.arg("-t").arg(format!("{len_s:.6}")).arg("-i").arg(path);
    }
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
