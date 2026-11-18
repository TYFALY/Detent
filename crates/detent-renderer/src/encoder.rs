//! FFmpeg capability discovery and hardware encoder selection.
//!
//! An encoder appearing in `ffmpeg -encoders` only means it was compiled in.
//! Each hardware candidate is therefore validated with a tiny real encode
//! (`color` source → `-f null`) before it is chosen; the result is cached.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{RenderError, Result};

/// FFmpeg binary locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tools {
    /// `ffmpeg`.
    pub ffmpeg: PathBuf,
    /// `ffprobe`.
    pub ffprobe: PathBuf,
}

impl Default for Tools {
    fn default() -> Self {
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
}

/// H.264 encoders detent knows how to drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Encoder {
    /// NVIDIA NVENC.
    H264Nvenc,
    /// Apple VideoToolbox.
    H264Videotoolbox,
    /// VA-API (Intel / AMD on Linux).
    H264Vaapi,
    /// Software x264.
    Libx264,
}

impl Encoder {
    /// FFmpeg encoder name.
    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            Self::H264Nvenc => "h264_nvenc",
            Self::H264Videotoolbox => "h264_videotoolbox",
            Self::H264Vaapi => "h264_vaapi",
            Self::Libx264 => "libx264",
        }
    }

    /// Hardware candidates in preference order for this OS.
    pub fn hardware_candidates() -> &'static [Encoder] {
        if cfg!(target_os = "macos") {
            &[Self::H264Videotoolbox]
        } else if cfg!(target_os = "linux") {
            &[Self::H264Nvenc, Self::H264Vaapi]
        } else {
            &[Self::H264Nvenc]
        }
    }

    /// Global args that must precede inputs.
    pub fn global_args(self) -> Vec<String> {
        match self {
            Self::H264Vaapi => vec!["-vaapi_device".into(), vaapi_device()],
            _ => vec![],
        }
    }

    /// Filter appended to the video chain (hardware upload).
    pub fn upload_filter(self) -> &'static str {
        match self {
            Self::H264Vaapi => "format=nv12,hwupload",
            _ => "format=yuv420p",
        }
    }

    /// Codec / rate-control args for a quality preset.
    pub fn codec_args(self, q: Quality) -> Vec<String> {
        let pick = |d: &str, s: &str, h: &str| -> String {
            match q {
                Quality::Draft => d,
                Quality::Standard => s,
                Quality::High => h,
            }
            .to_owned()
        };
        let mut a: Vec<String> = vec!["-c:v".into(), self.ffmpeg_name().into()];
        match self {
            Self::Libx264 => a.extend([
                "-preset".into(),
                pick("veryfast", "medium", "slow"),
                "-crf".into(),
                pick("26", "20", "17"),
                "-profile:v".into(),
                "high".into(),
            ]),
            Self::H264Nvenc => a.extend([
                "-preset".into(),
                pick("p2", "p5", "p7"),
                "-tune".into(),
                "hq".into(),
                "-rc".into(),
                "vbr".into(),
                "-cq".into(),
                pick("28", "21", "18"),
                "-b:v".into(),
                "0".into(),
                "-profile:v".into(),
                "high".into(),
            ]),
            Self::H264Videotoolbox => a.extend([
                "-q:v".into(),
                pick("45", "60", "72"),
                "-profile:v".into(),
                "high".into(),
            ]),
            Self::H264Vaapi => a.extend(["-qp".into(), pick("28", "22", "18")]),
        }
        a
    }

    /// Average-bitrate encode at `kbps` kbit/s (peaks capped at 1.25x over a
    /// two-second buffer), for renders with a file-size limit.
    pub fn bitrate_args(self, q: Quality, kbps: u32) -> Vec<String> {
        let rate = |k: u32| format!("{k}k");
        let mut a: Vec<String> = vec!["-c:v".into(), self.ffmpeg_name().into()];
        match self {
            Self::Libx264 => a.extend([
                "-preset".into(),
                match q {
                    Quality::Draft => "veryfast",
                    Quality::Standard => "medium",
                    Quality::High => "slow",
                }
                .into(),
                "-profile:v".into(),
                "high".into(),
            ]),
            Self::H264Nvenc => a.extend([
                "-preset".into(),
                "p5".into(),
                "-rc".into(),
                "vbr".into(),
                "-profile:v".into(),
                "high".into(),
            ]),
            Self::H264Videotoolbox => a.extend(["-profile:v".into(), "high".into()]),
            Self::H264Vaapi => a.extend(["-rc_mode".into(), "VBR".into()]),
        }
        a.extend([
            "-b:v".into(),
            rate(kbps),
            "-maxrate".into(),
            rate(kbps + kbps / 4),
            "-bufsize".into(),
            rate(kbps * 2),
        ]);
        a
    }
}

fn vaapi_device() -> String {
    std::env::var("DETENT_VAAPI_DEVICE").unwrap_or_else(|_| "/dev/dri/renderD128".into())
}

/// Encoding quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    /// Fast, larger artefacts.
    Draft,
    /// Balanced (default).
    #[default]
    Standard,
    /// Slow, visually transparent.
    High,
}

/// Encoder selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncoderChoice {
    /// First working hardware encoder, else libx264.
    #[default]
    Auto,
    /// Exactly this encoder (validated).
    Force(Encoder),
}

/// Discovered FFmpeg capabilities.
#[derive(Debug)]
pub struct FfmpegInfo {
    /// Binaries.
    pub tools: Tools,
    /// Major version, `None` for git snapshots (treated as newest).
    pub major_version: Option<u32>,
    /// Compiled-in encoder names.
    pub encoders: Vec<String>,
    working: OnceLock<Vec<Encoder>>,
}

impl FfmpegInfo {
    /// Queries `ffmpeg -version` and `ffmpeg -encoders`.
    pub fn probe(tools: Tools) -> Result<Self> {
        let run = |args: &[&str]| -> Result<String> {
            let out = Command::new(&tools.ffmpeg)
                .args(args)
                .stdin(Stdio::null())
                .output()
                .map_err(|source| RenderError::ToolNotFound {
                    tool: "ffmpeg",
                    path: tools.ffmpeg.display().to_string(),
                    source,
                })?;
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        };
        let version = run(&["-hide_banner", "-version"])?;
        let encoders = run(&["-hide_banner", "-encoders"])?;
        Ok(Self {
            major_version: parse_major(&version),
            encoders: parse_encoders(&encoders),
            tools,
            working: OnceLock::new(),
        })
    }

    /// Whether `-/filter_complex <file>` is supported (FFmpeg >= 7).
    pub fn supports_option_files(&self) -> bool {
        self.major_version.is_none_or(|v| v >= 7)
    }

    fn compiled(&self, e: Encoder) -> bool {
        self.encoders.iter().any(|n| n == e.ffmpeg_name())
    }

    /// Runs a one-frame test encode.
    pub fn test_encode(&self, e: Encoder) -> bool {
        if !self.compiled(e) {
            return false;
        }
        let mut cmd = Command::new(&self.tools.ffmpeg);
        cmd.args(["-nostdin", "-hide_banner", "-v", "error"])
            .args(e.global_args())
            .args([
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:r=30:d=0.1",
                "-vf",
            ])
            .arg(e.upload_filter())
            .args(e.codec_args(Quality::Draft))
            .args(["-frames:v", "2", "-f", "null", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        cmd.status().is_ok_and(|s| s.success())
    }

    /// Working encoders, hardware first (probed once).
    pub fn working_encoders(&self) -> &[Encoder] {
        self.working.get_or_init(|| {
            let mut v: Vec<Encoder> = Encoder::hardware_candidates()
                .iter()
                .copied()
                .filter(|&e| self.test_encode(e))
                .collect();
            if self.compiled(Encoder::Libx264) {
                v.push(Encoder::Libx264);
            }
            v
        })
    }

    /// Resolves an [`EncoderChoice`].
    pub fn select(&self, choice: EncoderChoice) -> Result<Encoder> {
        match choice {
            EncoderChoice::Auto => self.working_encoders().first().copied().ok_or_else(|| {
                RenderError::NoEncoder(
                    Encoder::hardware_candidates()
                        .iter()
                        .chain([&Encoder::Libx264])
                        .map(|e| e.ffmpeg_name())
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            }),
            EncoderChoice::Force(e) => {
                if e == Encoder::Libx264 && self.compiled(e) || self.test_encode(e) {
                    Ok(e)
                } else {
                    Err(RenderError::NoEncoder(e.ffmpeg_name().into()))
                }
            }
        }
    }
}

/// `"ffmpeg version 6.1.1-3ubuntu5 ..."` → 6, `"n7.1"` → 7, git snapshot → None.
pub fn parse_major(version_output: &str) -> Option<u32> {
    let v = version_output
        .lines()
        .next()?
        .split_whitespace()
        .nth(2)?
        .trim_start_matches('n');
    let major = v.split(['.', '-']).next()?;
    if major.len() >= 4 {
        // Date-stamped git build such as 2026-09-10-git-...
        return None;
    }
    major.parse().ok()
}

/// Encoder names from `ffmpeg -encoders` (video encoders only).
pub fn parse_encoders(out: &str) -> Vec<String> {
    out.lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let flags = it.next()?;
            let name = it.next()?;
            (flags.len() == 6 && flags.starts_with('V') && name != "=").then(|| name.to_owned())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_versions() {
        assert_eq!(
            parse_major("ffmpeg version 6.1.1-3ubuntu5 Copyright"),
            Some(6)
        );
        assert_eq!(parse_major("ffmpeg version n7.1 Copyright"), Some(7));
        assert_eq!(
            parse_major("ffmpeg version 2026-09-10-git-fd7c73d01e-full_build-www.gyan.dev"),
            None
        );
        assert_eq!(parse_major(""), None);
    }

    #[test]
    fn parses_encoder_list() {
        let out = "Encoders:\n V..... = Video\n ------\n V....D libx264              libx264 H.264\n V....D h264_nvenc           NVIDIA NVENC\n A....D aac                  AAC\n";
        assert_eq!(parse_encoders(out), vec!["libx264", "h264_nvenc"]);
    }

    #[test]
    fn quality_maps_to_rate_control() {
        let a = Encoder::Libx264.codec_args(Quality::High).join(" ");
        assert!(a.contains("-crf 17") && a.contains("-preset slow"));
        let n = Encoder::H264Nvenc.codec_args(Quality::Draft).join(" ");
        assert!(n.contains("-cq 28") && n.contains("-preset p2"));
        assert_eq!(Encoder::H264Vaapi.upload_filter(), "format=nv12,hwupload");
    }
}
