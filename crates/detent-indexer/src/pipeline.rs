//! The indexing pipeline: audio and video are decoded and analysed
//! concurrently on two scoped threads, each in a single streaming pass with
//! fixed-size buffers.

use std::time::Instant;

use detent_core::dsp::{AudioAnalyzer, DspConfig, PauseConfig};
use detent_core::index::ContactSheet;
use detent_core::index::{
    AudioTelemetry, INDEX_VERSION, MAX_WORD_MS, MediaIndex, ShotBoundary, SourceInfo, SubjectBox,
    VisualTelemetry, WordToken,
};
use detent_core::time::us_to_ms;

use crate::contact::{ContactSheetBuilder, GridSpec};
use crate::error::{IndexError, Result};
use crate::freeze::{FreezeConfig, FreezeDetector};
use crate::saliency::SubjectDetector;
use crate::shots::{ShotConfig, ShotDetector};
use crate::source::{FrameSource, PcmSource, RgbFrame};
use crate::transcript::Transcriber;

/// PCM read block (samples). 16 384 samples = ~1 s at 16 kHz, 64 KiB.
const PCM_BLOCK: usize = 16_384;

/// Indexing options.
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Audio analysis.
    pub dsp: DspConfig,
    /// Pause / peak detection.
    pub pauses: PauseConfig,
    /// Shot detection.
    pub shots: ShotConfig,
    /// Frozen-picture detection.
    pub freezes: FreezeConfig,
    /// Width of analysis frames (height follows the aspect ratio).
    pub analysis_width: u32,
    /// Contact sheet grid `(columns, rows)`, written to `contact_sheet_path`.
    pub contact_grid: Option<(u32, u32)>,
    /// Where to write the contact sheet.
    pub contact_sheet_path: Option<std::path::PathBuf>,
    /// Re-time raw ASR words against acoustic pauses (see `detent_core::align`).
    pub align_words: bool,
    /// Stretches with no frames in the source (from its timestamps);
    /// `index_file` fills this in from the file itself.
    pub frame_gaps: Vec<detent_core::index::FreezeInterval>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            dsp: DspConfig::default(),
            pauses: PauseConfig::default(),
            shots: ShotConfig::default(),
            freezes: FreezeConfig::default(),
            analysis_width: 160,
            contact_grid: Some((4, 4)),
            contact_sheet_path: None,
            align_words: true,
            frame_gaps: Vec::new(),
        }
    }
}

/// Throughput numbers for one indexing run.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize)]
pub struct IndexStats {
    /// Wall-clock time (ms).
    pub wall_ms: f64,
    /// Media duration (ms).
    pub media_ms: u64,
    /// PCM samples analysed.
    pub audio_samples: u64,
    /// Video frames analysed.
    pub video_frames: u64,
    /// Media seconds processed per wall-clock second.
    pub realtime_factor: f64,
}

/// Result of indexing.
#[derive(Debug, Clone)]
pub struct Indexed {
    /// The index.
    pub index: MediaIndex,
    /// Throughput.
    pub stats: IndexStats,
}

/// Analysis frame size for a source, preserving aspect (even dimensions).
pub fn analysis_dims(width: u32, src_w: u32, src_h: u32) -> (u32, u32) {
    let w = (width.max(16) / 2) * 2;
    let h = ((f64::from(w) * f64::from(src_h) / f64::from(src_w.max(1)) / 2.0).round() as u32 * 2)
        .clamp(2, 4 * w);
    (w, h)
}

struct AudioOut {
    telemetry: AudioTelemetry,
    words: Vec<WordToken>,
    samples: u64,
}

struct VideoOut {
    freezes: Vec<detent_core::index::FreezeInterval>,
    luma_per_s: Vec<f32>,
    faceless_ms: Option<Vec<(u64, u64)>>,
    lips_still_ms: Option<Vec<(u64, u64)>>,
    shots: Vec<ShotBoundary>,
    subjects: Vec<SubjectBox>,
    sheet: Option<ContactSheet>,
    frames: u64,
}

/// Runs the analysis over arbitrary decoders.
pub fn analyze<P, F>(
    source: SourceInfo,
    pcm: Option<P>,
    frames: Option<F>,
    transcriber: &mut dyn Transcriber,
    detector: &mut dyn SubjectDetector,
    opts: &IndexOptions,
) -> Result<Indexed>
where
    P: PcmSource + Send,
    F: FrameSource + Send,
{
    let started = Instant::now();
    let duration_ms = source.duration_ms;
    let fps = source.video.as_ref().map(|v| v.fps);

    let (audio, video) = std::thread::scope(|s| {
        let audio = s.spawn(|| -> Result<Option<AudioOut>> {
            let Some(mut pcm) = pcm else {
                return Ok(None);
            };
            let mut analyzer = AudioAnalyzer::new(opts.dsp)?;
            let mut buf = vec![0.0_f32; PCM_BLOCK];
            loop {
                let n = pcm.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                if cfg!(feature = "dsp") {
                    analyzer.push(&buf[..n]);
                }
                transcriber.push(&buf[..n])?;
            }
            pcm.close()?;
            let samples = analyzer.samples_consumed();
            let telemetry = detent_core::dsp::analyze(analyzer.finish(), &opts.pauses);
            let mut words = transcriber.finish()?;
            if opts.align_words && transcriber.needs_alignment() {
                words = detent_core::align::retime_words(
                    &words,
                    &telemetry.pauses,
                    duration_ms,
                    &detent_core::align::AlignConfig::default(),
                );
            }
            Ok(Some(AudioOut {
                telemetry,
                words,
                samples,
            }))
        });
        let video = s.spawn(|| -> Result<Option<VideoOut>> {
            let (Some(mut src), Some(fps)) = (frames, fps) else {
                return Ok(None);
            };
            let (w, h) = (src.width(), src.height());
            let mut shots = ShotDetector::new(opts.shots);
            let mut freezes = FreezeDetector::new(opts.freezes);
            let mut sheet = match (opts.contact_grid, &opts.contact_sheet_path) {
                (Some((columns, rows)), Some(_)) => Some(ContactSheetBuilder::new(
                    GridSpec {
                        columns,
                        rows,
                        tile_width: w,
                        tile_height: h,
                    },
                    0,
                    duration_ms,
                )),
                _ => None,
            };
            let mut subjects = Vec::new();
            let mut frame = RgbFrame::new(w, h);
            let mut k: u64 = 0;
            while src.next_frame(&mut frame)? {
                let t_ms = us_to_ms(fps.frame_to_us(k as i64));
                shots.push(&frame);
                freezes.push(&frame, t_ms);
                if let Some(b) = detector.observe(&frame, t_ms)? {
                    subjects.push(b);
                }
                if let Some(sheet) = sheet.as_mut() {
                    sheet.offer(&frame, t_ms);
                }
                k += 1;
            }
            src.close()?;
            let faceless_ms = detector.faceless_ms();
            let lips_still_ms = detector.lips_still_ms();
            let sheet = match (sheet, &opts.contact_sheet_path) {
                (Some(b), Some(p)) => Some(b.finish(p)?),
                _ => None,
            };
            let (freezes, luma_per_s) = freezes.finish(fps, &opts.frame_gaps);
            Ok(Some(VideoOut {
                freezes,
                luma_per_s,
                faceless_ms,
                lips_still_ms,
                shots: shots.finish(fps),
                subjects,
                sheet,
                frames: k,
            }))
        });
        (
            audio.join().map_err(|_| IndexError::WorkerPanic("audio")),
            video.join().map_err(|_| IndexError::WorkerPanic("video")),
        )
    });
    let audio = audio??;
    let video = video??;

    let (telemetry, mut words, samples) = match audio {
        Some(a) => (a.telemetry, a.words, a.samples),
        None => (AudioTelemetry::default(), Vec::new(), 0),
    };
    sanitize_words(&mut words, duration_ms);
    let (visual, frames) = match video {
        Some(v) => (
            VisualTelemetry {
                shots: v.shots,
                subjects: v.subjects,
                contact_sheet: v.sheet,
                freezes: v.freezes,
                luma_per_s: v.luma_per_s,
                faceless_ms: v.faceless_ms,
                lips_still_ms: v.lips_still_ms,
            },
            v.frames,
        ),
        None => (VisualTelemetry::default(), 0),
    };
    let index = MediaIndex {
        version: INDEX_VERSION,
        source,
        transcript: words,
        audio: telemetry,
        visual,
    };
    index.validate()?;
    let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
    Ok(Indexed {
        stats: IndexStats {
            wall_ms,
            media_ms: duration_ms,
            audio_samples: samples,
            video_frames: frames,
            realtime_factor: if wall_ms > 0.0 {
                duration_ms as f64 / wall_ms
            } else {
                0.0
            },
        },
        index,
    })
}

/// Sorts, clamps and drops implausible ASR tokens so the index validates.
fn sanitize_words(words: &mut Vec<WordToken>, duration_ms: u64) {
    for w in words.iter_mut() {
        w.end_ms = w.end_ms.min(duration_ms);
        w.start_ms = w.start_ms.min(w.end_ms);
        w.confidence = w.confidence.clamp(0.0, 1.0);
    }
    words.retain(|w| !w.word.trim().is_empty() && w.end_ms - w.start_ms <= MAX_WORD_MS);
    words.sort_by_key(|w| w.start_ms);
}

#[cfg(feature = "ffmpeg-cli")]
mod file {
    use std::path::Path;

    use super::*;
    use crate::ffmpeg::{DecodeRange, FfmpegFrames, FfmpegPcm, FfmpegTools};

    /// Probes and indexes a media file with FFmpeg decoders.
    pub fn index_file(
        tools: &FfmpegTools,
        path: &Path,
        transcriber: &mut dyn Transcriber,
        detector: &mut dyn SubjectDetector,
        opts: &IndexOptions,
    ) -> Result<Indexed> {
        let source = crate::probe::probe(tools, path)?;
        let mut opts = opts.clone();
        if let Some(v) = &source.video {
            opts.frame_gaps = crate::probe::frame_gaps(tools, path, v.fps).unwrap_or_else(|e| {
                tracing::warn!("could not read frame timestamps: {e}");
                Vec::new()
            });
        }
        let opts = &opts;
        let pcm = match source.audio {
            Some(_) => Some(FfmpegPcm::spawn(tools, path, opts.dsp.sample_rate)?),
            None => None,
        };
        let frames = match &source.video {
            Some(v) => {
                let (w, h) = analysis_dims(opts.analysis_width, v.width, v.height);
                // Resample to the nominal rate: variable-frame-rate files
                // (stream captures) leave timestamp gaps when the stream
                // lags; filling them with repeats keeps frame k at k/fps and
                // makes the lag visible to the freeze detector.
                Some(FfmpegFrames::spawn(
                    tools,
                    path,
                    w,
                    h,
                    DecodeRange::default(),
                    Some(v.fps.as_f64()),
                )?)
            }
            None => None,
        };
        analyze(source, pcm, frames, transcriber, detector, opts)
    }

    /// Renders a contact sheet for `[start_ms, end_ms)` (used for targeted
    /// visual verification). Decoding stops as soon as the grid is full.
    pub fn contact_sheet_range(
        tools: &FfmpegTools,
        path: &Path,
        start_ms: u64,
        end_ms: u64,
        grid: (u32, u32),
        tile_width: u32,
        out: &Path,
    ) -> Result<ContactSheet> {
        let source = crate::probe::probe(tools, path)?;
        let v = source
            .video
            .ok_or_else(|| IndexError::Probe("no video stream".into()))?;
        let end_ms = end_ms.min(source.duration_ms).max(start_ms + 1);
        let (w, h) = analysis_dims(tile_width, v.width, v.height);
        let tiles = u64::from(grid.0 * grid.1).max(1);
        let span_ms = end_ms - start_ms;
        // Sample a little denser than one frame per tile.
        let sample_fps = (2.0 * tiles as f64 * 1_000.0 / span_ms as f64).min(v.fps.as_f64());
        let mut src = FfmpegFrames::spawn(
            tools,
            path,
            w,
            h,
            DecodeRange {
                start_ms,
                duration_ms: Some(span_ms),
            },
            Some(sample_fps),
        )?;
        let mut sheet = ContactSheetBuilder::new(
            GridSpec {
                columns: grid.0,
                rows: grid.1,
                tile_width: w,
                tile_height: h,
            },
            start_ms,
            end_ms,
        );
        let mut frame = RgbFrame::new(w, h);
        let mut k = 0_u64;
        while !sheet.is_complete() && src.next_frame(&mut frame)? {
            let t = start_ms + (k as f64 * 1_000.0 / sample_fps).round() as u64;
            sheet.offer(&frame, t);
            k += 1;
        }
        if sheet.is_complete() {
            drop(src); // kills the decoder early
        } else {
            src.close()?;
        }
        sheet.finish(out)
    }
}

#[cfg(feature = "ffmpeg-cli")]
pub use file::{contact_sheet_range, index_file};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::saliency::SaliencyTracker;
    use crate::source::{VecFrames, VecPcm};
    use crate::transcript::{NoTranscript, StaticTranscript};
    use detent_core::index::{AudioStreamInfo, VideoStreamInfo};
    use detent_core::time::FrameRate;

    fn source(duration_ms: u64) -> SourceInfo {
        SourceInfo {
            path: "synthetic".into(),
            duration_ms,
            video: Some(VideoStreamInfo {
                width: 320,
                height: 180,
                fps: FrameRate::new(25, 1).unwrap(),
                codec: "raw".into(),
            }),
            audio: Some(AudioStreamInfo {
                sample_rate: 16_000,
                channels: 1,
                codec: "pcm".into(),
            }),
        }
    }

    #[test]
    fn analyzes_synthetic_av_concurrently() {
        // 4 s: tone 0-1.5 s, silence 1.5-2.2 s, tone to 4 s. Scene change at frame 50 (2 s).
        let pcm: Vec<f32> = (0..64_000)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                if (1.5..2.2).contains(&t) {
                    0.0003 * ((i * 7919 % 13) as f32 - 6.0) / 6.0
                } else {
                    0.3 * (std::f32::consts::TAU * 180.0 * t).sin()
                }
            })
            .collect();
        let frames: Vec<RgbFrame> = (0..100)
            .map(|k| {
                if k < 50 {
                    RgbFrame::solid(32, 18, [20, 40, 200])
                } else {
                    RgbFrame::solid(32, 18, [230, 200, 20])
                }
            })
            .collect();
        let words = vec![WordToken {
            word: "hello".into(),
            start_ms: 100,
            end_ms: 400,
            confidence: 0.9,
        }];
        let dir = tempfile::tempdir().unwrap();
        let opts = IndexOptions {
            contact_sheet_path: Some(dir.path().join("grid.jpg")),
            contact_grid: Some((2, 2)),
            ..IndexOptions::default()
        };
        let out = analyze(
            source(4_000),
            Some(VecPcm::new(pcm)),
            Some(VecFrames::new(32, 18, frames)),
            &mut StaticTranscript(words),
            &mut SaliencyTracker::new(200),
            &opts,
        )
        .unwrap();
        let idx = out.index;
        assert_eq!(idx.audio.rms_dbfs.len(), 400);
        assert!(
            idx.audio
                .pauses
                .iter()
                .any(|p| p.start_ms.abs_diff(1_500) < 40 && p.end_ms.abs_diff(2_200) < 40),
            "{:?}",
            idx.audio.pauses
        );
        assert_eq!(idx.visual.shots.len(), 1);
        assert_eq!(idx.visual.shots[0].frame, 50);
        assert_eq!(idx.transcript.len(), 1);
        assert!(!idx.visual.subjects.is_empty());
        assert_eq!(idx.visual.contact_sheet.as_ref().unwrap().tile_ms.len(), 4);
        assert_eq!(out.stats.video_frames, 100);
        assert_eq!(out.stats.audio_samples, 64_000);
    }

    #[test]
    fn audio_only_and_video_only() {
        let mut s = source(1_000);
        s.video = None;
        let out = analyze::<VecPcm, VecFrames>(
            s,
            Some(VecPcm::new(vec![0.1; 16_000])),
            None,
            &mut NoTranscript,
            &mut SaliencyTracker::new(200),
            &IndexOptions::default(),
        )
        .unwrap();
        assert!(out.index.visual.shots.is_empty());
        assert_eq!(out.index.audio.rms_dbfs.len(), 100);
    }

    #[test]
    fn sanitizes_transcript() {
        let mut w = vec![
            WordToken {
                word: "late".into(),
                start_ms: 900,
                end_ms: 2_000,
                confidence: 2.0,
            },
            WordToken {
                word: " ".into(),
                start_ms: 0,
                end_ms: 10,
                confidence: 1.0,
            },
            WordToken {
                word: "early".into(),
                start_ms: 100,
                end_ms: 200,
                confidence: 0.5,
            },
        ];
        sanitize_words(&mut w, 1_000);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].word, "early");
        assert_eq!((w[1].end_ms, w[1].confidence), (1_000, 1.0));
    }

    #[test]
    fn analysis_dims_preserve_aspect() {
        assert_eq!(analysis_dims(160, 1920, 1080), (160, 90));
        assert_eq!(analysis_dims(160, 1080, 1920), (160, 284));
    }
}
