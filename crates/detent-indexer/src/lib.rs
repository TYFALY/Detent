//! # detent-indexer
//!
//! Single-pass, bounded-memory multi-modal media indexing.
//!
//! * Audio: 16 kHz mono PCM streamed through [`detent_core::dsp`] (loudness,
//!   ZCR, spectral flux, pauses, excitement peaks) and an optional
//!   [`Transcriber`].
//! * Video: downscaled RGB frames streamed through a histogram
//!   [`ShotDetector`], a [`SubjectDetector`] and a [`ContactSheetBuilder`].
//!
//! Decoding sits behind the [`PcmSource`] / [`FrameSource`] traits. The
//! default `ffmpeg-cli` feature implements them by streaming raw samples out
//! of an `ffmpeg` child process; `whisper` and `onnx` add native ML backends.

#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![warn(missing_docs)]

pub mod contact;
pub mod error;
#[cfg(feature = "ffmpeg-cli")]
pub mod ffmpeg;
pub mod pipeline;
#[cfg(feature = "ffmpeg-cli")]
pub mod probe;
pub mod saliency;
pub mod shots;
pub mod source;
pub mod transcript;

#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "whisper")]
pub mod whisper;

pub use contact::{ContactSheetBuilder, GridSpec};
pub use error::{IndexError, Result};
pub use pipeline::{IndexOptions, IndexStats, Indexed, analyze};
#[cfg(feature = "ffmpeg-cli")]
pub use pipeline::{contact_sheet_range, index_file};
pub use saliency::{SaliencyTracker, SubjectDetector};
pub use shots::{ShotConfig, ShotDetector};
pub use source::{FrameSource, PcmSource, RgbFrame};
pub use transcript::{NoTranscript, StaticTranscript, Transcriber, load_transcript_json};
