//! whisper.cpp speech recognition (feature `whisper`).
//!
//! Audio is streamed through a [`PauseChunker`] into ≤ 28 s windows cut at
//! the quietest point, so memory stays bounded and words are not split
//! across windows. whisper.cpp runs with `max_len = 1` + `split_on_word`,
//! which yields one segment per word with token-level timestamps.

use std::path::Path;

use detent_core::index::WordToken;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::error::{IndexError, Result};
use crate::transcript::{PauseChunker, Transcriber};

/// Sample rate whisper.cpp expects.
pub const WHISPER_RATE: u32 = 16_000;

/// Streaming whisper.cpp transcriber.
pub struct WhisperTranscriber {
    // Keeps the model alive for `state`.
    _ctx: WhisperContext,
    state: WhisperState,
    chunker: PauseChunker,
    language: Option<String>,
    threads: i32,
    words: Vec<WordToken>,
}

impl std::fmt::Debug for WhisperTranscriber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperTranscriber")
            .field("language", &self.language)
            .field("words", &self.words.len())
            .finish_non_exhaustive()
    }
}

impl WhisperTranscriber {
    /// Loads a ggml model (e.g. `ggml-base.en.bin`). `language = None` auto-detects.
    pub fn new(model: &Path, language: Option<String>) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(model, WhisperContextParameters::default())
            .map_err(|e| IndexError::Transcriber(format!("loading {}: {e}", model.display())))?;
        let state = ctx
            .create_state()
            .map_err(|e| IndexError::Transcriber(e.to_string()))?;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8)) as i32;
        Ok(Self {
            _ctx: ctx,
            state,
            chunker: PauseChunker::new(WHISPER_RATE, 28.0, 6.0),
            language,
            threads,
            words: Vec::new(),
        })
    }

    fn run(&mut self, offset_samples: u64, pcm: &[f32]) -> Result<()> {
        let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        p.set_n_threads(self.threads);
        p.set_token_timestamps(true);
        p.set_split_on_word(true);
        p.set_max_len(1);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_special(false);
        p.set_print_timestamps(false);
        p.set_language(self.language.as_deref());
        self.state
            .full(p, pcm)
            .map_err(|e| IndexError::Transcriber(e.to_string()))?;
        let offset_ms = offset_samples * 1_000 / u64::from(WHISPER_RATE);
        for seg in self.state.as_iter() {
            let text = seg
                .to_str_lossy()
                .map_err(|e| IndexError::Transcriber(e.to_string()))?;
            let text = text.trim();
            if text.is_empty() || text.starts_with('[') {
                continue;
            }
            let n = seg.n_tokens();
            let probs: Vec<f32> = (0..n)
                .filter_map(|i| seg.get_token(i))
                .map(|t| t.token_probability())
                .collect();
            let confidence = if probs.is_empty() {
                0.0
            } else {
                probs.iter().sum::<f32>() / probs.len() as f32
            };
            // Segment timestamps are in centiseconds.
            let start = offset_ms + seg.start_timestamp().max(0) as u64 * 10;
            let end = offset_ms + seg.end_timestamp().max(0) as u64 * 10;
            self.words.push(WordToken {
                word: text.to_owned(),
                start_ms: start,
                end_ms: end.max(start),
                confidence,
            });
        }
        Ok(())
    }
}

impl Transcriber for WhisperTranscriber {
    fn push(&mut self, samples: &[f32]) -> Result<()> {
        for (off, chunk) in self.chunker.push(samples) {
            self.run(off, &chunk)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<WordToken>> {
        if let Some((off, chunk)) = self.chunker.flush() {
            // whisper.cpp needs at least ~1 s of audio.
            if chunk.len() >= WHISPER_RATE as usize {
                self.run(off, &chunk)?;
            }
        }
        let mut words = std::mem::take(&mut self.words);
        words.sort_by_key(|w| w.start_ms);
        Ok(words)
    }
}
