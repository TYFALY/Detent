//! Speech recognition by shelling out to a prebuilt `whisper-cli`
//! (whisper.cpp). No C++ toolchain is needed to build detent, so this is the
//! portable default on every OS, including Windows/MinGW where `whisper-rs`
//! does not compile.
//!
//! PCM is streamed to a temporary 16-bit WAV on disk while indexing (RAM
//! stays bounded), then `whisper-cli` runs once with `-ml 1 -sow` (one segment
//! per word) and `-ojf` (full JSON with token probabilities).

use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use detent_core::index::WordToken;
use serde::Deserialize;

use crate::error::{IndexError, Result};
use crate::transcript::Transcriber;

/// Sample rate whisper.cpp expects.
pub const RATE: u32 = 16_000;

/// Configuration for [`WhisperCli`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperCliConfig {
    /// `whisper-cli` executable (`DETENT_WHISPER_CLI` or `whisper-cli` on PATH).
    pub binary: PathBuf,
    /// ggml model, e.g. `ggml-base.en.bin`.
    pub model: PathBuf,
    /// Spoken language (`en`, `fr`, …); `None` = auto-detect.
    pub language: Option<String>,
    /// CPU threads.
    pub threads: usize,
}

impl WhisperCliConfig {
    /// Uses `DETENT_WHISPER_CLI` when set, otherwise `whisper-cli` on PATH.
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            binary: std::env::var_os("DETENT_WHISPER_CLI")
                .map_or_else(|| PathBuf::from("whisper-cli"), PathBuf::from),
            model: model.into(),
            language: None,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get().min(8)),
        }
    }
}

/// Streaming transcriber backed by an external `whisper-cli` process.
#[derive(Debug)]
pub struct WhisperCli {
    cfg: WhisperCliConfig,
    dir: tempfile::TempDir,
    wav: Option<BufWriter<std::fs::File>>,
    samples: u64,
}

impl WhisperCli {
    /// Validates paths and opens the temporary WAV.
    pub fn new(cfg: WhisperCliConfig) -> Result<Self> {
        if !cfg.model.is_file() {
            return Err(IndexError::Transcriber(format!(
                "whisper model not found: {}",
                cfg.model.display()
            )));
        }
        let dir = tempfile::Builder::new()
            .prefix("detent-asr-")
            .tempdir()
            .map_err(|source| IndexError::Io {
                path: "temp dir".into(),
                source,
            })?;
        let path = dir.path().join("audio.wav");
        let file = std::fs::File::create(&path).map_err(|source| IndexError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut wav = BufWriter::with_capacity(1 << 16, file);
        write_wav_header(&mut wav, 0).map_err(IndexError::Pipe)?;
        Ok(Self {
            cfg,
            dir,
            wav: Some(wav),
            samples: 0,
        })
    }

    fn run(&self, wav: &Path) -> Result<Vec<WordToken>> {
        let prefix = self.dir.path().join("out");
        let mut cmd = Command::new(&self.cfg.binary);
        cmd.arg("-m")
            .arg(&self.cfg.model)
            .arg("-f")
            .arg(wav)
            .args(["-ml", "1", "-sow", "-oj", "-ojf", "-np", "-t"])
            .arg(self.cfg.threads.to_string())
            .arg("-l")
            .arg(
                self.cfg
                    .language
                    .as_deref()
                    .unwrap_or_else(|| default_language(&self.cfg.model)),
            )
            .arg("-of")
            .arg(&prefix)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let out = cmd.output().map_err(|source| IndexError::ToolNotFound {
            tool: "whisper-cli",
            path: self.cfg.binary.display().to_string(),
            source,
        })?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let tail: String = stderr.lines().rev().take(8).collect::<Vec<_>>().join(" | ");
            return Err(IndexError::ToolFailed {
                tool: "whisper-cli",
                status: out.status.to_string(),
                stderr: tail,
            });
        }
        let json_path = prefix.with_extension("json");
        let bytes = std::fs::read(&json_path).map_err(|source| IndexError::Io {
            path: json_path.display().to_string(),
            source,
        })?;
        parse_whisper_json(&bytes)
    }
}

impl Transcriber for WhisperCli {
    fn push(&mut self, samples: &[f32]) -> Result<()> {
        if let Some(w) = self.wav.as_mut() {
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
                w.write_all(&v.to_le_bytes()).map_err(IndexError::Pipe)?;
            }
            self.samples += samples.len() as u64;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<WordToken>> {
        let Some(mut w) = self.wav.take() else {
            return Ok(Vec::new());
        };
        // whisper.cpp needs at least ~1 s of audio.
        if self.samples < u64::from(RATE) {
            return Ok(Vec::new());
        }
        let data_bytes = u32::try_from(self.samples * 2).map_err(|_| {
            IndexError::Transcriber("audio longer than a WAV file can hold (~37 h)".into())
        })?;
        write_wav_header(&mut w, data_bytes).map_err(IndexError::Pipe)?;
        w.flush().map_err(IndexError::Pipe)?;
        drop(w);
        self.run(&self.dir.path().join("audio.wav"))
    }

    fn needs_alignment(&self) -> bool {
        true
    }
}

/// English-only models (`*.en.bin`) reject auto-detection; others auto-detect.
fn default_language(model: &Path) -> &'static str {
    let name = model
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    if name.contains(".en.") || name.ends_with(".en") {
        "en"
    } else {
        "auto"
    }
}

/// Writes (or rewrites, by seeking to 0) a mono 16-bit PCM WAV header.
fn write_wav_header<W: Write + Seek>(w: &mut W, data_bytes: u32) -> std::io::Result<()> {
    let end = w.stream_position()?;
    w.seek(SeekFrom::Start(0))?;
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + data_bytes).to_le_bytes())?;
    w.write_all(b"WAVEfmt ")?;
    w.write_all(&16_u32.to_le_bytes())?; // PCM chunk size
    w.write_all(&1_u16.to_le_bytes())?; // PCM
    w.write_all(&1_u16.to_le_bytes())?; // mono
    w.write_all(&RATE.to_le_bytes())?;
    w.write_all(&(RATE * 2).to_le_bytes())?; // byte rate
    w.write_all(&2_u16.to_le_bytes())?; // block align
    w.write_all(&16_u16.to_le_bytes())?; // bits per sample
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;
    if end > 44 {
        w.seek(SeekFrom::Start(end))?;
    }
    Ok(())
}

#[derive(Deserialize)]
struct WFile {
    transcription: Vec<WSegment>,
}

#[derive(Deserialize)]
struct WOffsets {
    from: u64,
    to: u64,
}

#[derive(Deserialize)]
struct WToken {
    text: String,
    #[serde(default)]
    p: Option<f32>,
}

#[derive(Deserialize)]
struct WSegment {
    offsets: WOffsets,
    text: String,
    #[serde(default)]
    tokens: Vec<WToken>,
}

/// Parses `whisper-cli -ojf` output into words.
pub fn parse_whisper_json(bytes: &[u8]) -> Result<Vec<WordToken>> {
    // whisper.cpp may emit invalid UTF-8 when a multi-byte character is split
    // across tokens; decode lossily rather than failing the whole index.
    let text = String::from_utf8_lossy(bytes);
    let file: WFile = serde_json::from_str(&text).map_err(|source| IndexError::Json {
        what: "whisper-cli output".into(),
        source,
    })?;
    let mut words: Vec<WordToken> = file
        .transcription
        .into_iter()
        .filter_map(|s| {
            let word = s.text.trim().to_owned();
            // Skip empty segments and non-speech annotations like "[MUSIC]".
            if word.is_empty() || (word.starts_with('[') && word.ends_with(']')) {
                return None;
            }
            let probs: Vec<f32> = s
                .tokens
                .iter()
                .filter(|t| !t.text.starts_with("[_"))
                .filter_map(|t| t.p)
                .collect();
            let confidence = if probs.is_empty() {
                1.0
            } else {
                probs.iter().sum::<f32>() / probs.len() as f32
            };
            Some(WordToken {
                word,
                start_ms: s.offsets.from,
                end_ms: s.offsets.to.max(s.offsets.from),
                confidence: confidence.clamp(0.0, 1.0),
            })
        })
        .collect();
    words.sort_by_key(|w| w.start_ms);
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_word_level_json() {
        let json = br#"{"result":{"language":"en"},"transcription":[
            {"timestamps":{"from":"00:00:00,000","to":"00:00:00,000"},"offsets":{"from":0,"to":0},"text":""},
            {"offsets":{"from":320,"to":610},"text":" Rust","tokens":[
                {"text":"[_BEG_]","p":0.9},{"text":" Rust","p":0.8}]},
            {"offsets":{"from":610,"to":900},"text":" [MUSIC]"},
            {"offsets":{"from":0,"to":300},"text":" Hello,","tokens":[{"text":" Hello","p":0.6},{"text":",","p":1.0}]}
        ]}"#;
        let w = parse_whisper_json(json).unwrap();
        assert_eq!(w.len(), 2);
        assert_eq!(
            (w[0].word.as_str(), w[0].start_ms, w[0].end_ms),
            ("Hello,", 0, 300)
        );
        assert!((w[0].confidence - 0.8).abs() < 1e-6);
        assert!(
            (w[1].confidence - 0.8).abs() < 1e-6,
            "special tokens are ignored"
        );
    }

    #[test]
    fn wav_header_is_rewritten_in_place() {
        let mut c = std::io::Cursor::new(Vec::new());
        write_wav_header(&mut c, 0).unwrap();
        c.write_all(&[1, 0, 2, 0]).unwrap();
        write_wav_header(&mut c, 4).unwrap();
        let b = c.into_inner();
        assert_eq!(b.len(), 48);
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes([b[4], b[5], b[6], b[7]]), 40);
        assert_eq!(u32::from_le_bytes([b[40], b[41], b[42], b[43]]), 4);
        assert_eq!(&b[44..], &[1, 0, 2, 0]);
    }

    #[test]
    fn picks_language_from_model_name() {
        assert_eq!(default_language(Path::new("m/ggml-base.en.bin")), "en");
        assert_eq!(default_language(Path::new("ggml-large-v3.bin")), "auto");
    }

    #[test]
    fn missing_model_is_a_clear_error() {
        let err = WhisperCli::new(WhisperCliConfig::new("does/not/exist.bin")).unwrap_err();
        assert!(err.to_string().contains("model not found"), "{err}");
    }
}
