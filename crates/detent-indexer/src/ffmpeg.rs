//! Streaming decoders backed by an `ffmpeg` child process.
//!
//! Raw PCM / RGB is read from the child's stdout in fixed-size chunks, so
//! memory use is independent of media length. stderr is drained on a
//! separate thread (keeping only a bounded tail) so a chatty decoder can never
//! deadlock the pipe.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::thread::JoinHandle;

use crate::error::{IndexError, Result};
use crate::source::{FrameSource, PcmSource, RgbFrame};

/// Bytes of stderr retained for error messages.
const STDERR_TAIL: usize = 4 * 1024;

/// Locations of the FFmpeg binaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegTools {
    /// `ffmpeg` executable.
    pub ffmpeg: PathBuf,
    /// `ffprobe` executable.
    pub ffprobe: PathBuf,
}

impl Default for FfmpegTools {
    fn default() -> Self {
        Self::from_env()
    }
}

impl FfmpegTools {
    /// Uses `DETENT_FFMPEG` / `DETENT_FFPROBE` if set, otherwise `PATH`.
    pub fn from_env() -> Self {
        let get = |var: &str, default: &str| {
            std::env::var_os(var)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(default))
        };
        Self {
            ffmpeg: get("DETENT_FFMPEG", "ffmpeg"),
            ffprobe: get("DETENT_FFPROBE", "ffprobe"),
        }
    }

    /// Returns the first line of `ffmpeg -version`.
    pub fn version(&self) -> Result<String> {
        let out = Command::new(&self.ffmpeg)
            .arg("-version")
            .stdin(Stdio::null())
            .output()
            .map_err(|source| IndexError::ToolNotFound {
                tool: "ffmpeg",
                path: self.ffmpeg.display().to_string(),
                source,
            })?;
        Ok(String::from_utf8_lossy(&out.stdout)
            .lines()
            .next()
            .unwrap_or_default()
            .to_owned())
    }

    /// True when both binaries can be executed.
    pub fn available(&self) -> bool {
        let ok = |p: &Path| {
            Command::new(p)
                .arg("-version")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        };
        ok(&self.ffmpeg) && ok(&self.ffprobe)
    }
}

/// A running decoder process with a drained stderr.
#[derive(Debug)]
struct ChildStream {
    child: Option<Child>,
    stdout: ChildStdout,
    stderr: Option<JoinHandle<String>>,
    tool: &'static str,
}

impl ChildStream {
    fn spawn(mut cmd: Command, tool: &'static str, path: &Path) -> Result<Self> {
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|source| IndexError::ToolNotFound {
            tool,
            path: path.display().to_string(),
            source,
        })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| IndexError::Pipe(std::io::Error::other("stdout not captured")))?;
        let stderr = child.stderr.take().map(|mut err| {
            std::thread::spawn(move || {
                let mut tail: Vec<u8> = Vec::new();
                let mut buf = [0_u8; 1024];
                while let Ok(n) = err.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    tail.extend_from_slice(&buf[..n]);
                    if tail.len() > STDERR_TAIL {
                        tail.drain(..tail.len() - STDERR_TAIL);
                    }
                }
                String::from_utf8_lossy(&tail).trim().to_owned()
            })
        });
        Ok(Self {
            child: Some(child),
            stdout,
            stderr,
            tool,
        })
    }

    /// Reads until `buf` is full or EOF; returns bytes read.
    fn fill(&mut self, buf: &mut [u8]) -> Result<usize> {
        let mut filled = 0;
        while filled < buf.len() {
            match self.stdout.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(IndexError::Pipe(e)),
            }
        }
        Ok(filled)
    }

    fn finish(mut self) -> Result<()> {
        let Some(mut child) = self.child.take() else {
            return Ok(());
        };
        let status = child.wait().map_err(IndexError::Pipe)?;
        let stderr = self
            .stderr
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or_default();
        if status.success() {
            Ok(())
        } else {
            Err(IndexError::ToolFailed {
                tool: self.tool,
                status: status.to_string(),
                stderr,
            })
        }
    }
}

impl Drop for ChildStream {
    fn drop(&mut self) {
        // Abandoned early (error path): don't leave a zombie decoder running.
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Optional time range for decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeRange {
    /// Seek position (ms).
    pub start_ms: u64,
    /// Duration (ms); `None` = to the end.
    pub duration_ms: Option<u64>,
}

fn base_command(tools: &FfmpegTools, path: &Path, range: DecodeRange) -> Command {
    let mut cmd = Command::new(&tools.ffmpeg);
    cmd.args(["-nostdin", "-hide_banner", "-v", "error"]);
    if range.start_ms > 0 {
        cmd.arg("-ss")
            .arg(format!("{:.3}", range.start_ms as f64 / 1e3));
    }
    if let Some(d) = range.duration_ms {
        cmd.arg("-t").arg(format!("{:.3}", d as f64 / 1e3));
    }
    cmd.arg("-i").arg(path);
    cmd
}

/// Mono `f32` PCM decoded and resampled by FFmpeg.
#[derive(Debug)]
pub struct FfmpegPcm {
    stream: ChildStream,
    bytes: Vec<u8>,
    bytes_read: u64,
}

impl FfmpegPcm {
    /// Starts decoding the first audio stream of `path` at `sample_rate` Hz.
    pub fn spawn(tools: &FfmpegTools, path: &Path, sample_rate: u32) -> Result<Self> {
        let mut cmd = base_command(tools, path, DecodeRange::default());
        cmd.args(["-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar"])
            .arg(sample_rate.to_string())
            .args(["-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]);
        Ok(Self {
            stream: ChildStream::spawn(cmd, "ffmpeg", &tools.ffmpeg)?,
            bytes: Vec::new(),
            bytes_read: 0,
        })
    }

    /// Total decoded bytes so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }
}

impl PcmSource for FfmpegPcm {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        self.bytes.resize(out.len() * 4, 0);
        let n = self.stream.fill(&mut self.bytes)?;
        self.bytes_read += n as u64;
        let samples = n / 4;
        for (dst, b) in out
            .iter_mut()
            .zip(self.bytes[..samples * 4].as_chunks::<4>().0)
        {
            *dst = f32::from_le_bytes(*b);
        }
        Ok(samples)
    }

    fn close(self) -> Result<()> {
        self.stream.finish()
    }
}

/// Downscaled RGB24 frames decoded by FFmpeg (one per source frame).
#[derive(Debug)]
pub struct FfmpegFrames {
    stream: ChildStream,
    width: u32,
    height: u32,
    bytes_read: u64,
}

impl FfmpegFrames {
    /// Starts decoding the first video stream of `path`, scaled to `width x height`.
    pub fn spawn(
        tools: &FfmpegTools,
        path: &Path,
        width: u32,
        height: u32,
        range: DecodeRange,
        sample_fps: Option<f64>,
    ) -> Result<Self> {
        let mut cmd = base_command(tools, path, range);
        let mut vf = String::new();
        if let Some(fps) = sample_fps {
            vf.push_str(&format!("fps={fps:.6},"));
        }
        vf.push_str(&format!("scale={width}:{height}:flags=area,format=rgb24"));
        cmd.args(["-map", "0:v:0", "-an", "-sn", "-dn", "-vf"])
            .arg(vf)
            .args(["-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]);
        Ok(Self {
            stream: ChildStream::spawn(cmd, "ffmpeg", &tools.ffmpeg)?,
            width,
            height,
            bytes_read: 0,
        })
    }

    /// Total decoded bytes so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }
}

impl FrameSource for FfmpegFrames {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn next_frame(&mut self, frame: &mut RgbFrame) -> Result<bool> {
        let len = RgbFrame::byte_len(self.width, self.height);
        frame.width = self.width;
        frame.height = self.height;
        frame.data.resize(len, 0);
        let n = self.stream.fill(&mut frame.data)?;
        self.bytes_read += n as u64;
        // A truncated trailing frame is treated as EOF.
        Ok(n == len)
    }

    fn close(self) -> Result<()> {
        self.stream.finish()
    }
}
