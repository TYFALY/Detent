//! Indexer errors.

use thiserror::Error;

/// Errors produced while indexing media.
#[derive(Debug, Error)]
pub enum IndexError {
    /// `ffmpeg` / `ffprobe` could not be launched.
    #[error(
        "could not run `{tool}` ({path}): {source}. Install FFmpeg or set DETENT_FFMPEG / DETENT_FFPROBE"
    )]
    ToolNotFound {
        /// Tool name.
        tool: &'static str,
        /// Path that was tried.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A child process exited unsuccessfully.
    #[error("`{tool}` failed ({status}): {stderr}")]
    ToolFailed {
        /// Tool name.
        tool: &'static str,
        /// Exit status description.
        status: String,
        /// Tail of stderr.
        stderr: String,
    },

    /// `ffprobe` output could not be understood.
    #[error("could not interpret ffprobe output: {0}")]
    Probe(String),

    /// The file has neither a video nor an audio stream.
    #[error("{0} has no audio or video stream")]
    NoStreams(String),

    /// Filesystem error.
    #[error("I/O error on {path}: {source}")]
    Io {
        /// Path involved.
        path: String,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// Error reading a decoder pipe.
    #[error("decoder pipe error: {0}")]
    Pipe(#[source] std::io::Error),

    /// Image encoding failed.
    #[error("image encoding failed: {0}")]
    Image(#[from] image::ImageError),

    /// JSON (de)serialisation failed.
    #[error("invalid JSON in {what}: {source}")]
    Json {
        /// Document.
        what: String,
        /// Underlying error.
        #[source]
        source: serde_json::Error,
    },

    /// Speech recognition failed.
    #[error("transcription failed: {0}")]
    Transcriber(String),

    /// Subject detection failed.
    #[error("subject detection failed: {0}")]
    Detector(String),

    /// Error from the core crate.
    #[error(transparent)]
    Core(#[from] detent_core::CoreError),

    /// A worker thread panicked (should be impossible; reported, not propagated).
    #[error("internal worker panicked: {0}")]
    WorkerPanic(&'static str),
}

/// Convenience alias.
pub type Result<T, E = IndexError> = std::result::Result<T, E>;
