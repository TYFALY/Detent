//! Renderer errors.

use thiserror::Error;

/// Errors produced while rendering.
#[derive(Debug, Error)]
pub enum RenderError {
    /// A binary could not be launched.
    #[error(
        "could not run `{tool}` ({path}): {source}. Install FFmpeg or set DETENT_FFMPEG / DETENT_FFPROBE"
    )]
    ToolNotFound {
        /// Tool name.
        tool: &'static str,
        /// Path tried.
        path: String,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// FFmpeg exited unsuccessfully.
    #[error("`{tool}` failed ({status}): {stderr}")]
    ToolFailed {
        /// Tool name.
        tool: &'static str,
        /// Exit status.
        status: String,
        /// Tail of stderr.
        stderr: String,
    },

    /// The timeline violates solver invariants (tampered or corrupt file).
    #[error("invalid timeline: {}", .0.join("; "))]
    InvalidTimeline(Vec<String>),

    /// No usable H.264 encoder was found.
    #[error("no usable H.264 encoder (tried: {0})")]
    NoEncoder(String),

    /// The requested mode cannot express this timeline.
    #[error("stream-copy preview is impossible: {0}")]
    CopyUnsupported(String),

    /// The encoded file does not contain exactly the solved frames.
    #[error("render wrote {written} frames but the timeline has {expected}")]
    FrameCountMismatch {
        /// Frames in the timeline.
        expected: u64,
        /// Frames FFmpeg reported writing.
        written: u64,
    },

    /// The output stayed above the size limit even at a reduced bitrate.
    #[error(
        "output is {bytes} bytes, over the {max}-byte limit even after re-encoding at a lower bitrate; shorten the edit"
    )]
    TooLarge {
        /// Final size.
        bytes: u64,
        /// Limit.
        max: u64,
    },

    /// Filesystem error.
    #[error("I/O error on {path}: {source}")]
    Io {
        /// Path.
        path: String,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
}

/// Convenience alias.
pub type Result<T, E = RenderError> = std::result::Result<T, E>;

pub(crate) fn io_err(path: &std::path::Path) -> impl FnOnce(std::io::Error) -> RenderError + '_ {
    move |source| RenderError::Io {
        path: path.display().to_string(),
        source,
    }
}
