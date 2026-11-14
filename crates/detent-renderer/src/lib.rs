//! # detent-renderer
//!
//! Deterministic execution of a [`detent_core::ResolvedTimeline`]:
//!
//! * [`graph`] compiles segments, camera paths, micro-crossfades and captions
//!   into an FFmpeg filtergraph (pure function, unit-tested).
//! * [`encoder`] discovers working hardware encoders (`h264_nvenc`,
//!   `h264_videotoolbox`, `h264_vaapi`) with a real test encode and falls
//!   back to `libx264`.
//! * [`exec`] runs FFmpeg, streams progress and reports throughput, with a
//!   keyframe-aligned stream-copy preview path.

#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![warn(missing_docs)]

pub mod encoder;
pub mod error;
pub mod exec;
pub mod expr;
pub mod graph;

pub use encoder::{Encoder, EncoderChoice, FfmpegInfo, Quality, Tools};
pub use error::{RenderError, Result};
pub use exec::{Progress, RenderMetrics, RenderMode, RenderOptions, Renderer, copy_eligible};
