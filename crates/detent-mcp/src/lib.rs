//! # detent-mcp
//!
//! A Model Context Protocol server (stdio) that lets an LLM act as a creative
//! director: it reads compact telemetry, emits a Semantic EDL, and receives
//! deterministic diagnostics — it never touches pixels or FFmpeg syntax.
//!
//! Tools:
//! * `index_media` — index a file; returns a bounded digest + contact sheet.
//! * `inspect_keyframes` — contact sheet for a time range (image content).
//! * `solve_and_validate_edl` — run the solver; returns exact cuts and every
//!   automatic adjustment, or JSON-pointer validation issues.
//! * `render_timeline` — render a solved timeline; returns throughput.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine as _;
use detent_core::index::{DigestLimits, IndexDigest};
use detent_core::{CoreError, EdlSolver, MediaIndex, ResolvedTimeline, SemanticEdl, SolveReport};
use detent_indexer::ffmpeg::FfmpegTools;
use detent_indexer::{
    IndexOptions, IndexStats, NoTranscript, SaliencyTracker, StaticTranscript, SubjectDetector,
    Transcriber, WhisperCli, WhisperCliConfig,
};
use detent_renderer::{
    Encoder, EncoderChoice, Quality, RenderMetrics, RenderMode, RenderOptions, Renderer, Tools,
};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ServerCapabilities, ServerConfig};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Workflow guidance sent to the client at initialisation.
pub const INSTRUCTIONS: &str = "\
detent turns long talking-head / screen recordings into tight, human-feeling edits.
Workflow:
1. index_media(path) → read `digest.phrases` (what is said, with ms timings), \
`digest.cut_windows` (pauses where cuts sound natural), `digest.excitement_ms` (vocal emphasis) \
and `digest.shots_ms`. Look at the contact sheet image to judge framing. If `digest.framing_hint` is `PipFacecam` (a small facecam over screen/game content), set `output.layout` to `FitBlur` so the content is not cropped away; use `Crop` for talking heads.
2. Optionally inspect_keyframes(index_path, start_ms, end_ms) to verify a moment visually.
3. solve_and_validate_edl(index_path, edl): choose clips by meaning (hook first), express camera \
and audio intent only via presets. Cut points may be approximate: the solver snaps them into breath \
valleys (±120 ms), onto nearby shot changes, applies J/L-cuts and micro-crossfades, and reports \
every adjustment. If it returns validation issues, fix ALL of them (each has a JSON pointer) and retry.
For short-form vertical edits: open on the boldest line (no preamble), set `tighten` to cut dead air, \
a 3-5 word `headline` with `headline_duration_ms: 3000`, `captions` with 2-4 words per block, \
`output.opening_pop`, `output.grade: MobilePop` and `output.audio_mix: VoiceForward`. Keep most splices \
clean cuts; use `video_transitions` `Flash` only for a shock moment or beat drop.
4. render_timeline(timeline_path, output_path) — use mode=preview first for a fast check.
Never invent pixel coordinates or FFmpeg filters; the schema does not accept them.";

/// Server configuration.
#[derive(Debug, Clone, Default)]
pub struct ServerSettings {
    /// FFmpeg binaries used by the indexer.
    pub ffmpeg: Option<FfmpegTools>,
    /// FFmpeg binaries used by the renderer.
    pub render_tools: Option<Tools>,
}

struct State {
    ffmpeg: FfmpegTools,
    render_tools: Tools,
    renderer: tokio::sync::OnceCell<Arc<Renderer>>,
}

/// The MCP server.
#[derive(Clone)]
pub struct DetentServer {
    state: Arc<State>,
    tool_router: ToolRouter<Self>,
}

impl std::fmt::Debug for DetentServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetentServer").finish_non_exhaustive()
    }
}

/// `index_media` options.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IndexMediaOptions {
    /// Word-level transcript to attach: detent's `[{word,start_ms,end_ms,confidence}]` or
    /// WhisperX/faster-whisper `{"segments":[{"words":[...]}]}` JSON. Strongly recommended:
    /// without words the solver can only use acoustic pauses.
    #[serde(default)]
    pub transcript_json: Option<String>,
    /// Transcribe locally with this whisper.cpp ggml model (e.g. `ggml-base.en.bin`).
    /// Defaults to the server's `DETENT_WHISPER_MODEL`. Ignored when `transcript_json` is set.
    #[serde(default)]
    pub whisper_model: Option<String>,
    /// Spoken language for whisper (`en`, `fr`, …); auto-detected when omitted.
    #[serde(default)]
    pub language: Option<String>,
    /// Directory for index.json / grid.jpg (default: `<media>.detent/` next to the file).
    #[serde(default)]
    pub out_dir: Option<String>,
    /// Attach the contact sheet image to the response (default true).
    #[serde(default = "yes")]
    pub include_image: bool,
}

fn yes() -> bool {
    true
}

impl Default for IndexMediaOptions {
    fn default() -> Self {
        Self {
            transcript_json: None,
            whisper_model: None,
            language: None,
            out_dir: None,
            include_image: true,
        }
    }
}

/// `index_media` parameters.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IndexMediaParams {
    /// Absolute path of the media file.
    pub path: String,
    /// Options.
    #[serde(default)]
    pub options: IndexMediaOptions,
}

/// `index_media` result.
#[derive(Debug, Clone, Serialize)]
pub struct IndexMediaResult {
    /// Full index on disk (pass to the other tools).
    pub index_path: String,
    /// Contact sheet on disk.
    pub contact_sheet_path: Option<String>,
    /// Bounded LLM-facing summary.
    pub digest: IndexDigest,
    /// Indexing throughput.
    pub stats: IndexStats,
}

/// `inspect_keyframes` parameters.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InspectParams {
    /// Path returned by `index_media`.
    pub index_path: String,
    /// Range start (ms).
    pub start_ms: u64,
    /// Range end (ms).
    pub end_ms: u64,
    /// Grid columns (1-6, default 4).
    #[serde(default)]
    #[schemars(range(min = 1, max = 6))]
    pub columns: Option<u32>,
    /// Grid rows (1-6, default 3).
    #[serde(default)]
    #[schemars(range(min = 1, max = 6))]
    pub rows: Option<u32>,
}

/// `solve_and_validate_edl` parameters.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SolveParams {
    /// Path returned by `index_media`.
    pub index_path: String,
    /// The Semantic EDL (editorial intent only).
    pub edl: SemanticEdl,
}

/// `solve_and_validate_edl` result.
#[derive(Debug, Clone, Serialize)]
pub struct SolveResult {
    /// Resolved timeline on disk (pass to `render_timeline`).
    pub timeline_path: String,
    /// Exact cuts, automatic adjustments and diagnostics.
    pub report: SolveReport,
}

/// `render_timeline` parameters.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RenderParams {
    /// Path returned by `solve_and_validate_edl`.
    pub resolved_timeline_path: String,
    /// Output `.mp4` path.
    pub output_path: String,
    /// Quality preset (default standard).
    #[serde(default)]
    pub quality: Quality,
    /// `full` (default) or `preview` (fast stream copy when possible).
    #[serde(default)]
    pub mode: RenderMode,
    /// Force an encoder (default: best working hardware encoder, else libx264).
    #[serde(default)]
    pub encoder: Option<Encoder>,
}

fn tool_error(msg: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(msg.into())])
}

fn json_block<T: Serialize>(v: &T) -> ContentBlock {
    ContentBlock::text(
        serde_json::to_string_pretty(v).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}")),
    )
}

fn image_block(path: &Path) -> Option<ContentBlock> {
    let bytes = std::fs::read(path).ok()?;
    Some(ContentBlock::image(
        base64::engine::general_purpose::STANDARD.encode(bytes),
        "image/jpeg",
    ))
}

fn read_index(path: &str) -> Result<MediaIndex, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    MediaIndex::from_json(&text).map_err(|e| e.to_string())
}

/// How to obtain word timings while indexing.
#[derive(Debug, Clone, Default)]
pub struct AsrChoice {
    /// Pre-computed transcript JSON (wins over everything else).
    pub transcript_json: Option<PathBuf>,
    /// whisper.cpp model; falls back to `DETENT_WHISPER_MODEL`.
    pub whisper_model: Option<PathBuf>,
    /// `whisper-cli` binary; falls back to `DETENT_WHISPER_CLI`, then PATH.
    pub whisper_cli: Option<PathBuf>,
    /// Spoken language; auto-detected when `None` (English for `*.en` models).
    pub language: Option<String>,
}

/// Picks a transcriber: transcript JSON → whisper model (native `whisper-rs`
/// when compiled with the `whisper` feature, otherwise the `whisper-cli`
/// subprocess) → no transcript.
pub fn make_transcriber(asr: AsrChoice) -> anyhow::Result<Box<dyn Transcriber>> {
    if let Some(t) = asr.transcript_json {
        return Ok(Box::new(StaticTranscript(
            detent_indexer::load_transcript_json(&t)?,
        )));
    }
    let model = asr
        .whisper_model
        .or_else(|| std::env::var_os("DETENT_WHISPER_MODEL").map(PathBuf::from));
    let Some(model) = model else {
        return Ok(Box::new(NoTranscript));
    };
    #[cfg(feature = "whisper")]
    {
        let _ = asr.whisper_cli;
        Ok(Box::new(detent_indexer::whisper::WhisperTranscriber::new(
            &model,
            asr.language,
        )?))
    }
    #[cfg(not(feature = "whisper"))]
    {
        let mut cfg = WhisperCliConfig::new(model);
        if let Some(bin) = asr.whisper_cli {
            cfg.binary = bin;
        }
        cfg.language = asr.language;
        Ok(Box::new(WhisperCli::new(cfg)?))
    }
}

/// Default sidecar directory `<media>.detent/`.
pub fn default_out_dir(media: &Path) -> PathBuf {
    let mut name = media.file_name().unwrap_or_default().to_os_string();
    name.push(".detent");
    media.with_file_name(name)
}

/// Subject detector + analysis frame width. With the `onnx` feature and
/// `DETENT_FACE_MODEL` set (plus `ORT_DYLIB_PATH` for ONNX Runtime), uses the
/// UltraFace detector on 320 px frames; otherwise the saliency heuristic on
/// 160 px frames.
pub fn make_detector() -> anyhow::Result<(Box<dyn SubjectDetector>, u32)> {
    #[cfg(feature = "onnx")]
    if let Some(model) = std::env::var_os("DETENT_FACE_MODEL") {
        let det = detent_indexer::onnx::OnnxFaceDetector::new(Path::new(&model), 0.7, 200)?;
        return Ok((Box::new(det), 320));
    }
    Ok((Box::new(SaliencyTracker::new(200)), 160))
}

/// Blocking implementation of `index_media`, shared with the CLI.
pub fn index_media_blocking(
    tools: &FfmpegTools,
    path: &Path,
    out_dir: &Path,
    transcriber: &mut dyn Transcriber,
) -> anyhow::Result<IndexMediaResult> {
    std::fs::create_dir_all(out_dir)?;
    let (mut detector, analysis_width) = make_detector()?;
    let opts = IndexOptions {
        contact_sheet_path: Some(out_dir.join("grid.jpg")),
        analysis_width,
        ..IndexOptions::default()
    };
    let indexed = detent_indexer::index_file(tools, path, transcriber, detector.as_mut(), &opts)?;
    let index_path = out_dir.join("index.json");
    std::fs::write(&index_path, serde_json::to_vec(&indexed.index)?)?;
    Ok(IndexMediaResult {
        index_path: index_path.display().to_string(),
        contact_sheet_path: indexed
            .index
            .visual
            .contact_sheet
            .as_ref()
            .map(|c| c.path.clone()),
        digest: indexed.index.digest(DigestLimits::default()),
        stats: indexed.stats,
    })
}

#[tool_router(router = tool_router)]
impl DetentServer {
    /// Creates a server.
    pub fn new(settings: ServerSettings) -> Self {
        Self {
            state: Arc::new(State {
                ffmpeg: settings.ffmpeg.unwrap_or_default(),
                render_tools: settings.render_tools.unwrap_or_default(),
                renderer: tokio::sync::OnceCell::new(),
            }),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "index_media",
        description = "Index a media file in one bounded-memory pass: audio loudness/breath/silence \
valleys, vocal excitement peaks, shot boundaries, subject boxes and a timestamped keyframe contact \
sheet. Returns a compact digest (phrases with ms timings, natural cut windows, emphasis peaks, shots) \
plus the index_path needed by the other tools."
    )]
    async fn index_media(&self, Parameters(p): Parameters<IndexMediaParams>) -> CallToolResult {
        let state = self.state.clone();
        let include_image = p.options.include_image;
        let res = tokio::task::spawn_blocking(move || -> anyhow::Result<IndexMediaResult> {
            let path = PathBuf::from(&p.path);
            let out_dir = p
                .options
                .out_dir
                .map(PathBuf::from)
                .unwrap_or_else(|| default_out_dir(&path));
            let mut transcriber = make_transcriber(AsrChoice {
                transcript_json: p.options.transcript_json.map(PathBuf::from),
                whisper_model: p.options.whisper_model.map(PathBuf::from),
                whisper_cli: None,
                language: p.options.language,
            })?;
            index_media_blocking(&state.ffmpeg, &path, &out_dir, transcriber.as_mut())
        })
        .await;
        match res {
            Ok(Ok(r)) => {
                let mut content = vec![json_block(&r)];
                if include_image
                    && let Some(img) = r
                        .contact_sheet_path
                        .as_deref()
                        .map(Path::new)
                        .and_then(image_block)
                {
                    content.push(img);
                }
                CallToolResult::success(content)
            }
            Ok(Err(e)) => tool_error(format!("index_media failed: {e:#}")),
            Err(e) => tool_error(format!("index_media worker failed: {e}")),
        }
    }

    #[tool(
        name = "inspect_keyframes",
        description = "Render a low-resolution timestamped contact sheet for [start_ms, end_ms) of an \
indexed file, to visually verify framing, gestures or scene content before committing to a cut."
    )]
    async fn inspect_keyframes(&self, Parameters(p): Parameters<InspectParams>) -> CallToolResult {
        if p.end_ms <= p.start_ms {
            return tool_error("end_ms must be greater than start_ms");
        }
        let state = self.state.clone();
        let res = tokio::task::spawn_blocking(move || -> anyhow::Result<(String, Vec<u64>)> {
            let idx = read_index(&p.index_path).map_err(anyhow::Error::msg)?;
            let dir = Path::new(&p.index_path)
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default();
            let out = dir.join(format!("inspect_{}_{}.jpg", p.start_ms, p.end_ms));
            let grid = (
                p.columns.unwrap_or(4).clamp(1, 6),
                p.rows.unwrap_or(3).clamp(1, 6),
            );
            let sheet = detent_indexer::contact_sheet_range(
                &state.ffmpeg,
                Path::new(&idx.source.path),
                p.start_ms,
                p.end_ms,
                grid,
                192,
                &out,
            )?;
            Ok((sheet.path, sheet.tile_ms))
        })
        .await;
        match res {
            Ok(Ok((path, tiles))) => {
                let mut content = vec![json_block(&serde_json::json!({
                    "contact_sheet_path": path,
                    "tile_ms": tiles,
                    "layout": "row-major; each tile is labelled with its timestamp",
                }))];
                if let Some(img) = image_block(Path::new(&path)) {
                    content.push(img);
                }
                CallToolResult::success(content)
            }
            Ok(Err(e)) => tool_error(format!("inspect_keyframes failed: {e:#}")),
            Err(e) => tool_error(format!("inspect_keyframes worker failed: {e}")),
        }
    }

    #[tool(
        name = "solve_and_validate_edl",
        description = "Validate a Semantic EDL against the index and deterministically resolve it: \
breath/silence snapping (±120 ms), scene-collision guard (150 ms), J/L-cut injection (150-300 ms), \
15-25 ms equal-power crossfades, critically damped spring camera paths and syntactic captions. \
Returns the frame-exact cut list, every automatic adjustment and diagnostics, plus timeline_path. \
On invalid input returns ALL issues as JSON pointers."
    )]
    async fn solve_and_validate_edl(
        &self,
        Parameters(p): Parameters<SolveParams>,
    ) -> CallToolResult {
        let res = tokio::task::spawn_blocking(move || -> Result<SolveResult, CallToolResult> {
            let idx = read_index(&p.index_path).map_err(tool_error)?;
            match EdlSolver::default().solve(&p.edl, &idx) {
                Ok(r) => {
                    let path = Path::new(&p.index_path).with_file_name("timeline.json");
                    let json =
                        serde_json::to_vec(&r.timeline).map_err(|e| tool_error(e.to_string()))?;
                    std::fs::write(&path, json)
                        .map_err(|e| tool_error(format!("writing {}: {e}", path.display())))?;
                    Ok(SolveResult {
                        timeline_path: path.display().to_string(),
                        report: r.report,
                    })
                }
                Err(CoreError::Validation(issues)) => Err(CallToolResult::error(vec![json_block(
                    &serde_json::json!({ "valid": false, "issues": issues }),
                )])),
                Err(e) => Err(tool_error(e.to_string())),
            }
        })
        .await;
        match res {
            Ok(Ok(r)) => CallToolResult::success(vec![json_block(&r)]),
            Ok(Err(e)) => e,
            Err(e) => tool_error(format!("solver worker failed: {e}")),
        }
    }

    #[tool(
        name = "render_timeline",
        description = "Render a solved timeline to MP4 with the best working encoder (NVENC, \
VideoToolbox, VA-API, else libx264). mode=preview uses keyframe-aligned stream copy when no \
reframing or captions are needed. Returns throughput metrics."
    )]
    async fn render_timeline(&self, Parameters(p): Parameters<RenderParams>) -> CallToolResult {
        let renderer = match self
            .state
            .renderer
            .get_or_try_init(|| {
                let tools = self.state.render_tools.clone();
                async move {
                    tokio::task::spawn_blocking(move || Renderer::new(tools).map(Arc::new))
                        .await
                        .map_err(|e| e.to_string())?
                        .map_err(|e| e.to_string())
                }
            })
            .await
        {
            Ok(r) => r.clone(),
            Err(e) => return tool_error(format!("renderer unavailable: {e}")),
        };
        let res = tokio::task::spawn_blocking(move || -> anyhow::Result<RenderMetrics> {
            let text = std::fs::read_to_string(&p.resolved_timeline_path)?;
            let t = ResolvedTimeline::from_json(&text)?;
            let opts = RenderOptions {
                quality: p.quality,
                encoder: p.encoder.map_or(EncoderChoice::Auto, EncoderChoice::Force),
                mode: p.mode,
            };
            Ok(renderer.render(&t, Path::new(&p.output_path), &opts, &mut |_| {})?)
        })
        .await;
        match res {
            Ok(Ok(m)) => CallToolResult::success(vec![json_block(&m)]),
            Ok(Err(e)) => tool_error(format!("render_timeline failed: {e:#}")),
            Err(e) => tool_error(format!("render worker failed: {e}")),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for DetentServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(INSTRUCTIONS)
    }
}

/// Serves MCP over stdio until the client disconnects.
pub async fn serve_stdio(settings: ServerSettings) -> anyhow::Result<()> {
    let service = DetentServer::new(settings)
        .serve(rmcp::transport::stdio())
        .await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_schemas_are_strict_and_complete() {
        let server = DetentServer::new(ServerSettings::default());
        let tools = server.tool_router.list_all();
        let names: Vec<_> = tools.iter().map(|t| t.name.to_string()).collect();
        for n in [
            "index_media",
            "inspect_keyframes",
            "solve_and_validate_edl",
            "render_timeline",
        ] {
            assert!(names.contains(&n.to_owned()), "missing {n}: {names:?}");
        }
        let solve = tools
            .iter()
            .find(|t| t.name == "solve_and_validate_edl")
            .unwrap();
        let schema = serde_json::to_string(&solve.input_schema).unwrap();
        assert!(schema.contains("SpringPunchIn"));
        assert!(schema.contains("AutoJCut"));
        assert!(schema.contains("\"additionalProperties\":false"));
    }

    #[test]
    fn default_out_dir_is_sidecar() {
        let p = default_out_dir(Path::new("/m/talk.mp4"));
        assert_eq!(p, Path::new("/m/talk.mp4.detent"));
    }
}
