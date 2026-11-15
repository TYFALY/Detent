//! `detent` — index, solve, render, serve MCP, benchmark.

mod bench;

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use detent_core::solver::Level;
use detent_core::{EdlSolver, MediaIndex, ResolvedTimeline, SemanticEdl};
use detent_indexer::ffmpeg::FfmpegTools;
use detent_renderer::{
    Encoder, EncoderChoice, Quality, RenderMode, RenderOptions, Renderer, Tools,
};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "detent",
    version,
    about = "Local-first media indexing, deterministic Semantic EDL solving and rendering",
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Index a media file → `<out>/index.json` + `<out>/grid.jpg`.
    Index {
        /// Media file.
        input: PathBuf,
        /// Output directory (default: `<input>.detent/`).
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Word-level transcript JSON (detent or WhisperX layout).
        #[arg(long, conflicts_with = "whisper_model")]
        transcript: Option<PathBuf>,
        /// whisper.cpp ggml model, e.g. `ggml-base.en.bin` (default: `DETENT_WHISPER_MODEL`).
        /// Runs through `whisper-cli` unless built with `--features whisper`.
        #[arg(long)]
        whisper_model: Option<PathBuf>,
        /// `whisper-cli` executable (default: `DETENT_WHISPER_CLI` or PATH).
        #[arg(long)]
        whisper_cli: Option<PathBuf>,
        /// Language for whisper (e.g. `en`); auto-detect when omitted.
        #[arg(long)]
        language: Option<String>,
        /// Print the LLM digest as JSON instead of a summary.
        #[arg(long)]
        digest: bool,
    },
    /// Validate + solve a Semantic EDL → resolved timeline JSON.
    Solve {
        /// `index.json` from `detent index`.
        index: PathBuf,
        /// Semantic EDL JSON.
        edl: PathBuf,
        /// Output timeline path (default: `timeline.json` next to the index).
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Also write the full solve report here.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Render a resolved timeline.
    Render {
        /// `timeline.json` from `detent solve`.
        timeline: PathBuf,
        /// Output `.mp4`.
        #[arg(short, long)]
        out: PathBuf,
        /// Quality preset.
        #[arg(long, value_enum, default_value_t = QualityArg::Standard)]
        quality: QualityArg,
        /// Encoder (default: best working hardware encoder).
        #[arg(long, value_enum, default_value_t = EncoderArg::Auto)]
        encoder: EncoderArg,
        /// Fast preview (stream copy when no reframing/captions).
        #[arg(long)]
        preview: bool,
    },
    /// Serve the MCP tools over stdio (for Claude Code / Claude Desktop).
    Mcp,
    /// Print a JSON Schema.
    Schema {
        /// Which document.
        #[arg(value_enum, default_value_t = SchemaArg::Edl)]
        which: SchemaArg,
    },
    /// List FFmpeg encoders that actually work on this machine.
    Encoders,
    /// Run the reproducible benchmark suite and print a Markdown table.
    Bench {
        /// Also index + render this real file.
        #[arg(long)]
        input: Option<PathBuf>,
        /// Seconds of synthetic audio for the DSP benchmark.
        #[arg(long, default_value_t = 600.0)]
        seconds: f32,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum QualityArg {
    Draft,
    Standard,
    High,
}

#[derive(Clone, Copy, ValueEnum)]
enum EncoderArg {
    Auto,
    Nvenc,
    Videotoolbox,
    Vaapi,
    X264,
}

#[derive(Clone, Copy, ValueEnum)]
enum SchemaArg {
    Edl,
    Index,
    Timeline,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();
    match cli.cmd {
        Cmd::Index {
            input,
            out,
            transcript,
            whisper_model,
            whisper_cli,
            language,
            digest,
        } => index(
            &input,
            out,
            detent_mcp::AsrChoice {
                transcript_json: transcript,
                whisper_model,
                whisper_cli,
                language,
            },
            digest,
        ),
        Cmd::Solve {
            index,
            edl,
            out,
            report,
        } => solve(&index, &edl, out, report),
        Cmd::Render {
            timeline,
            out,
            quality,
            encoder,
            preview,
        } => render(&timeline, &out, quality, encoder, preview),
        Cmd::Mcp => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(detent_mcp::serve_stdio(
                detent_mcp::ServerSettings::default(),
            )),
        Cmd::Schema { which } => {
            let s = match which {
                SchemaArg::Edl => SemanticEdl::json_schema(),
                SchemaArg::Index => schemars::schema_for!(MediaIndex),
                SchemaArg::Timeline => schemars::schema_for!(ResolvedTimeline),
            };
            println!("{}", serde_json::to_string_pretty(&s)?);
            Ok(())
        }
        Cmd::Encoders => {
            let info = detent_renderer::FfmpegInfo::probe(Tools::default())?;
            println!(
                "ffmpeg major version: {}",
                info.major_version
                    .map_or("git snapshot".to_owned(), |v| v.to_string())
            );
            for e in info.working_encoders() {
                println!("  ✓ {}", e.ffmpeg_name());
            }
            Ok(())
        }
        Cmd::Bench { input, seconds } => bench::run(input.as_deref(), seconds),
    }
}

fn index(
    input: &Path,
    out: Option<PathBuf>,
    asr: detent_mcp::AsrChoice,
    digest: bool,
) -> Result<()> {
    let out_dir = out.unwrap_or_else(|| detent_mcp::default_out_dir(input));
    let mut t = detent_mcp::make_transcriber(asr)?;
    let r = detent_mcp::index_media_blocking(&FfmpegTools::from_env(), input, &out_dir, t.as_mut())
        .with_context(|| format!("indexing {}", input.display()))?;
    if digest {
        println!("{}", serde_json::to_string_pretty(&r.digest)?);
        return Ok(());
    }
    let d = &r.digest;
    eprintln!(
        "indexed {:.1}s of media in {:.0} ms ({:.1}x realtime)",
        r.stats.media_ms as f64 / 1e3,
        r.stats.wall_ms,
        r.stats.realtime_factor
    );
    eprintln!(
        "  video {} | {} phrases | {} cut windows | {} shots | {} excitement peaks",
        d.video.as_deref().unwrap_or("none"),
        d.phrases.len(),
        d.cut_windows.len(),
        d.shots_ms.len(),
        d.excitement_ms.len()
    );
    println!("{}", r.index_path);
    if let Some(c) = &r.contact_sheet_path {
        eprintln!("  contact sheet: {c}");
    }
    Ok(())
}

fn solve(index: &Path, edl: &Path, out: Option<PathBuf>, report: Option<PathBuf>) -> Result<()> {
    let idx = MediaIndex::from_json(&std::fs::read_to_string(index)?)
        .with_context(|| format!("reading {}", index.display()))?;
    let edl = SemanticEdl::from_json(&std::fs::read_to_string(edl)?)?;
    let res = match EdlSolver::default().solve(&edl, &idx) {
        Ok(r) => r,
        Err(detent_core::CoreError::Validation(issues)) => {
            for i in &issues {
                eprintln!("  ✗ {i}");
            }
            bail!("EDL rejected with {} issue(s)", issues.len());
        }
        Err(e) => return Err(e.into()),
    };
    let out = out.unwrap_or_else(|| index.with_file_name("timeline.json"));
    std::fs::write(&out, serde_json::to_vec_pretty(&res.timeline)?)?;
    if let Some(p) = report {
        std::fs::write(p, serde_json::to_vec_pretty(&res.report)?)?;
    }
    let r = &res.report;
    eprintln!(
        "solved {} clip(s) → {} ({:.2}s), {} adjustment(s)",
        r.cuts.len(),
        r.output,
        r.output_duration_ms as f64 / 1e3,
        r.adjustments.len()
    );
    for c in &r.cuts {
        eprintln!(
            "  clip {}: requested {}-{} ms → video {}-{} ms (frames {}-{}), at {} ms",
            c.clip,
            c.requested_ms.0,
            c.requested_ms.1,
            c.video_ms.0,
            c.video_ms.1,
            c.frames.0,
            c.frames.1,
            c.output_at_ms
        );
    }
    for d in &r.diagnostics {
        let tag = if d.level == Level::Warning {
            "warn"
        } else {
            "info"
        };
        eprintln!("  [{tag}] {}: {}", d.code, d.message);
    }
    println!("{}", out.display());
    Ok(())
}

fn render(
    timeline: &Path,
    out: &Path,
    quality: QualityArg,
    encoder: EncoderArg,
    preview: bool,
) -> Result<()> {
    let t = ResolvedTimeline::from_json(&std::fs::read_to_string(timeline)?)?;
    let renderer = Renderer::new(Tools::default())?;
    let opts = RenderOptions {
        quality: match quality {
            QualityArg::Draft => Quality::Draft,
            QualityArg::Standard => Quality::Standard,
            QualityArg::High => Quality::High,
        },
        encoder: match encoder {
            EncoderArg::Auto => EncoderChoice::Auto,
            EncoderArg::Nvenc => EncoderChoice::Force(Encoder::H264Nvenc),
            EncoderArg::Videotoolbox => EncoderChoice::Force(Encoder::H264Videotoolbox),
            EncoderArg::Vaapi => EncoderChoice::Force(Encoder::H264Vaapi),
            EncoderArg::X264 => EncoderChoice::Force(Encoder::Libx264),
        },
        mode: if preview {
            RenderMode::Preview
        } else {
            RenderMode::Full
        },
        batch_segments: None,
    };
    let mut stderr = std::io::stderr();
    let m = renderer.render(&t, out, &opts, &mut |p| {
        let _ = write!(
            stderr,
            "\r  rendering {:>5.1}%  frame {}",
            p.fraction * 100.0,
            p.frames
        );
        let _ = stderr.flush();
    })?;
    eprintln!();
    eprintln!(
        "{} via {} → {} ({:.1} MB) in {} ms: {:.0} fps, {:.1}x realtime",
        m.mode,
        m.encoder.as_deref().unwrap_or("stream copy"),
        m.output_path,
        m.output_bytes as f64 / 1e6,
        m.wall_ms,
        m.encode_fps,
        m.realtime_factor
    );
    Ok(())
}
