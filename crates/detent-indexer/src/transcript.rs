//! Speech-to-text adapters.
//!
//! [`Transcriber`] is a streaming interface: PCM is pushed as it is decoded
//! and recognisers process bounded windows, so a two-hour file never sits in
//! memory. [`PauseChunker`] splits the stream at the quietest point near a
//! window boundary so words are not cut in half between windows.

use std::path::Path;

use detent_core::index::WordToken;
use serde::Deserialize;

use crate::error::{IndexError, Result};

/// Streaming speech recogniser over 16 kHz mono audio.
pub trait Transcriber: Send {
    /// Pushes the next block of samples.
    fn push(&mut self, samples: &[f32]) -> Result<()>;
    /// Flushes and returns all words sorted by start time.
    fn finish(&mut self) -> Result<Vec<WordToken>>;
    /// Whether word times should be re-timed against acoustic pauses
    /// (true for raw whisper output, false for force-aligned imports).
    fn needs_alignment(&self) -> bool {
        false
    }
}

/// Produces no words (indexing without ASR).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoTranscript;

impl Transcriber for NoTranscript {
    fn push(&mut self, _samples: &[f32]) -> Result<()> {
        Ok(())
    }
    fn finish(&mut self) -> Result<Vec<WordToken>> {
        Ok(Vec::new())
    }
}

/// Returns a fixed word list (tests, or words imported from another tool).
#[derive(Debug, Clone, Default)]
pub struct StaticTranscript(pub Vec<WordToken>);

impl Transcriber for StaticTranscript {
    fn push(&mut self, _samples: &[f32]) -> Result<()> {
        Ok(())
    }
    fn finish(&mut self) -> Result<Vec<WordToken>> {
        Ok(std::mem::take(&mut self.0))
    }
}

#[derive(Deserialize)]
struct XWord {
    word: String,
    start: f64,
    end: f64,
    #[serde(default, alias = "probability")]
    score: Option<f32>,
}

#[derive(Deserialize)]
struct XSegment {
    #[serde(default)]
    words: Vec<XWord>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TranscriptFile {
    Native(Vec<WordToken>),
    Segments { segments: Vec<XSegment> },
}

/// Loads word timings produced elsewhere. Accepts detent's own
/// `[{word,start_ms,end_ms,confidence}]` or the WhisperX / faster-whisper
/// `{"segments":[{"words":[{word,start,end,score|probability}]}]}` layout.
pub fn load_transcript_json(path: &Path) -> Result<Vec<WordToken>> {
    let text = std::fs::read_to_string(path).map_err(|source| IndexError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse_transcript_json(&text)
}

/// Parses a transcript document (see [`load_transcript_json`]).
pub fn parse_transcript_json(text: &str) -> Result<Vec<WordToken>> {
    let file: TranscriptFile = serde_json::from_str(text).map_err(|source| IndexError::Json {
        what: "transcript".into(),
        source,
    })?;
    let mut words = match file {
        TranscriptFile::Native(w) => w,
        TranscriptFile::Segments { segments } => segments
            .into_iter()
            .flat_map(|s| s.words)
            .filter(|w| w.end >= w.start && w.start >= 0.0)
            .map(|w| WordToken {
                word: w.word.trim().to_owned(),
                start_ms: (w.start * 1_000.0).round() as u64,
                end_ms: (w.end * 1_000.0).round() as u64,
                confidence: w.score.unwrap_or(1.0).clamp(0.0, 1.0),
            })
            .collect(),
    };
    words.retain(|w| !w.word.is_empty() && w.end_ms >= w.start_ms);
    words.sort_by_key(|w| w.start_ms);
    Ok(words)
}

/// Splits a sample stream into windows of at most `max_len` samples, cutting
/// at the quietest 10 ms hop in the last `search` samples of each window.
#[derive(Debug, Clone)]
pub struct PauseChunker {
    buf: Vec<f32>,
    offset: u64,
    max_len: usize,
    search: usize,
    hop: usize,
}

impl PauseChunker {
    /// `sample_rate` Hz, windows up to `max_s` seconds, cut search over the
    /// final `search_s` seconds.
    pub fn new(sample_rate: u32, max_s: f32, search_s: f32) -> Self {
        let sr = sample_rate as f32;
        let max_len = (max_s * sr) as usize;
        Self {
            buf: Vec::with_capacity(max_len),
            offset: 0,
            max_len: max_len.max(1),
            search: ((search_s * sr) as usize).clamp(1, max_len.max(1)),
            hop: ((sr / 100.0) as usize).max(1),
        }
    }

    /// Pushes samples; returns completed `(offset_samples, window)` chunks.
    pub fn push(&mut self, mut samples: &[f32]) -> Vec<(u64, Vec<f32>)> {
        let mut out = Vec::new();
        while !samples.is_empty() {
            let take = (self.max_len - self.buf.len()).min(samples.len());
            self.buf.extend_from_slice(&samples[..take]);
            samples = &samples[take..];
            if self.buf.len() >= self.max_len {
                let cut = self.quietest_cut();
                let rest = self.buf.split_off(cut);
                let chunk = std::mem::replace(&mut self.buf, rest);
                out.push((self.offset, chunk));
                self.offset += cut as u64;
            }
        }
        out
    }

    /// Returns the remaining partial window, if any.
    pub fn flush(&mut self) -> Option<(u64, Vec<f32>)> {
        if self.buf.is_empty() {
            return None;
        }
        let chunk = std::mem::take(&mut self.buf);
        let off = self.offset;
        self.offset += chunk.len() as u64;
        Some((off, chunk))
    }

    fn quietest_cut(&self) -> usize {
        let n = self.buf.len();
        let start = n - self.search.min(n);
        let mut best = (f32::INFINITY, n);
        let mut i = start;
        while i + self.hop <= n {
            let e: f32 = self.buf[i..i + self.hop].iter().map(|x| x * x).sum();
            if e < best.0 {
                best = (e, i + self.hop / 2);
            }
            i += self.hop;
        }
        best.1.clamp(1, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_native_and_whisperx_layouts() {
        let native = r#"[{"word":"b","start_ms":500,"end_ms":700,"confidence":0.9},
                         {"word":"a","start_ms":100,"end_ms":300,"confidence":0.8}]"#;
        let w = parse_transcript_json(native).unwrap();
        assert_eq!(w[0].word, "a");
        let wx = r#"{"segments":[{"words":[{"word":" hi","start":0.12,"end":0.4,"score":0.97},
                    {"word":"there","start":0.45,"end":0.8,"probability":0.5}]}]}"#;
        let w = parse_transcript_json(wx).unwrap();
        assert_eq!(w.len(), 2);
        assert_eq!(
            (w[0].word.as_str(), w[0].start_ms, w[1].confidence),
            ("hi", 120, 0.5)
        );
        assert!(parse_transcript_json("{\"nope\":1}").is_err());
    }

    #[test]
    fn chunker_cuts_in_silence_and_preserves_every_sample() {
        let sr = 1_000;
        // 1 s of tone with a silent gap at 0.8-0.85 s, repeated.
        let sig: Vec<f32> = (0..5_000)
            .map(|i| {
                let t = i % 1_000;
                if (800..850).contains(&t) {
                    0.0
                } else {
                    ((i as f32) * 0.3).sin()
                }
            })
            .collect();
        let mut c = PauseChunker::new(sr, 1.0, 0.4);
        let mut chunks = Vec::new();
        for block in sig.chunks(333) {
            chunks.extend(c.push(block));
        }
        chunks.extend(c.flush());
        let total: usize = chunks.iter().map(|(_, c)| c.len()).sum();
        assert_eq!(total, sig.len());
        let mut expected_off = 0;
        for (off, chunk) in &chunks {
            assert_eq!(*off, expected_off);
            expected_off += chunk.len() as u64;
            assert!(chunk.len() <= 1_000);
        }
        // First cut lands inside the first silent gap.
        assert!(
            (800..=850).contains(&chunks[0].1.len()),
            "{}",
            chunks[0].1.len()
        );
    }
}
