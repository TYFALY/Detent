//! Structured error types for `detent-core`.

use serde::Serialize;
use thiserror::Error;

/// A single, addressable problem found while validating untrusted input
/// (typically a Semantic EDL emitted by an LLM).
///
/// `path` is a JSON pointer into the offending document so a model can fix
/// every issue in one round-trip instead of playing whack-a-mole.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValidationIssue {
    /// JSON pointer, e.g. `/clips/2/end_ms`.
    pub path: String,
    /// Human/LLM readable explanation including the accepted range.
    pub message: String,
}

impl ValidationIssue {
    pub(crate) fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

/// Errors produced by `detent-core`.
#[derive(Debug, Error)]
pub enum CoreError {
    /// The input document is not valid JSON or does not match the schema.
    #[error("malformed JSON for {what}: {source}")]
    Json {
        /// Which document failed to parse.
        what: &'static str,
        /// Underlying serde error (carries line/column).
        #[source]
        source: serde_json::Error,
    },

    /// The document parsed but violates semantic constraints.
    #[error("{} validation issue(s): {}", .0.len(), join_issues(.0))]
    Validation(Vec<ValidationIssue>),

    /// A frame rate with a zero numerator or denominator.
    #[error("invalid frame rate {num}/{den}")]
    InvalidFrameRate {
        /// Numerator.
        num: u32,
        /// Denominator.
        den: u32,
    },

    /// The media index lacks a stream the operation requires.
    #[error("media index has no {0} stream")]
    MissingStream(&'static str),

    /// The solver produced a timeline that violates its own invariants.
    /// This indicates a bug and is surfaced instead of rendering garbage.
    #[error("solver invariant violated: {}", .0.join("; "))]
    Invariant(Vec<String>),

    /// Audio analysis was configured with impossible parameters.
    #[error("invalid DSP configuration: {0}")]
    InvalidDspConfig(String),
}

fn join_issues(issues: &[ValidationIssue]) -> String {
    issues
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

/// Convenience alias.
pub type Result<T, E = CoreError> = std::result::Result<T, E>;
