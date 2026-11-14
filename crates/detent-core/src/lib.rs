//! # detent-core
//!
//! Domain model and deterministic solver for **detent**, a local-first media
//! indexing and Semantic EDL engine.
//!
//! * [`index`] — `MediaIndex`, the compact multi-modal telemetry of a source.
//! * [`edl`] — `SemanticEdl`, the strict schema an LLM is allowed to emit.
//! * [`solver`] — `EdlSolver`, which turns intent into a frame-exact timeline.
//! * [`dsp`] — streaming loudness / ZCR / spectral-flux analysis.
//! * [`spring`], [`camera`] — critically damped virtual camera.
//! * [`captions`] — syntactic caption chunking and `.ass` output.
//! * [`timeline`] — `ResolvedTimeline` and its invariant checker.
//!
//! The crate is pure Rust, has no I/O and never panics on untrusted input.

#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![warn(missing_docs)]

pub mod align;
pub mod camera;
pub mod captions;
pub mod dsp;
pub mod edl;
pub mod error;
pub mod index;
pub mod solver;
pub mod spring;
pub mod testkit;
pub mod time;
pub mod timeline;

pub use edl::SemanticEdl;
pub use error::{CoreError, Result, ValidationIssue};
pub use index::MediaIndex;
pub use solver::{EdlSolver, Resolution, SolveReport, SolverConfig};
pub use timeline::ResolvedTimeline;
