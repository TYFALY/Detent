//! Compiles a [`ResolvedTimeline`] into FFmpeg inputs + a filtergraph.
//!
//! Memory stays bounded regardless of edit order: every segment gets its own
//! seeked input (`-ss … -t … -i src`) instead of `split`-ing one decoded
//! stream (which would queue raw frames for out-of-order segments). Video is
//! cut with `trim=end_frame` (exact frame counts), audio with
//! `atrim`+`apad` (exact sample counts), so the A/V invariant proved by the
//! solver survives into the container.

use detent_core::ResolvedTimeline;
use detent_core::camera::CameraPath;
use detent_core::edl::{AudioMix, ColorGrade, InsertPlacement, LogoPlacement, is_video_insert};
use detent_core::timeline::SfxKind;
use std::ops::Range;

use detent_core::timeline::{Layout, Splice};

use crate::encoder::Encoder;
use crate::expr::Hinges;

/// One `-i` input.
#[derive(Debug, Clone, PartialEq)]
pub struct InputSpec {
    /// Which timeline source to read (0 = `source`, k = `extra_sources[k-1]`).
    pub source: usize,
    /// Seek position (s), placed before `-i` (fast + accurate).
    pub seek_s: f64,
    /// Read duration (s).
    pub duration_s: f64,
}

/// `atempo` chain for `speed` (each stage must stay within 0.5..=2).
pub fn atempo(speed: f64) -> String {
    if speed < 0.5 {
        format!("atempo=0.5,atempo={:.6}", speed / 0.5)
    } else if speed > 2.0 {
        format!("atempo=2,atempo={:.6}", speed / 2.0)
    } else {
        format!("atempo={speed:.6}")
    }
}

/// Inputs and filtergraph for one render.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledGraph {
    /// Inputs in `-i` order (all the same source file).
    pub inputs: Vec<InputSpec>,
    /// `-filter_complex` body.
    pub filter: String,
    /// Whether `[aout]` exists.
    pub has_audio: bool,
    /// Insert and music files, added after `inputs` (or after the batch
    /// files of a join), in this order.
    pub extra: Vec<ExtraInput>,
}

/// How a segment's virtual camera maps to filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraKind {
    /// Same raster, no crop.
    Identity,
    /// One fixed crop window (or just a resize).
    Static,
    /// Moving window of constant size (`crop` with time expressions).
    Pan,
    /// Changing window size (`scale` with `eval=frame`, then `crop`).
    Zoom,
}

/// Classifies a camera path.
pub fn camera_kind(path: &CameraPath, src: (u32, u32), out: (u32, u32)) -> CameraKind {
    let (sw, sh) = (f64::from(src.0), f64::from(src.1));
    let Some(first) = path.keyframes.first() else {
        return CameraKind::Identity;
    };
    if path.is_static() {
        let full = first.x.abs() < 0.5
            && first.y.abs() < 0.5
            && (first.w - sw).abs() < 0.5
            && (first.h - sh).abs() < 0.5;
        return if full && src == out {
            CameraKind::Identity
        } else {
            CameraKind::Static
        };
    }
    let w_const = path
        .keyframes
        .iter()
        .all(|k| (k.w - first.w).abs() < 0.5 && (k.h - first.h).abs() < 0.5);
    if w_const {
        CameraKind::Pan
    } else {
        CameraKind::Zoom
    }
}

fn even_round(x: f64) -> u32 {
    ((x / 2.0).round().max(1.0) as u32) * 2
}

fn hinge(path: &CameraPath, f: impl Fn(&detent_core::camera::CropKeyframe) -> f64) -> Hinges {
    let pts: Vec<(f64, f64)> = path
        .keyframes
        .iter()
        .map(|k| (k.t_us as f64 / 1e6, f(k)))
        .collect();
    Hinges::from_points(&pts)
}

/// Camera filter chain for one segment (without leading/trailing commas).
pub fn camera_filter(path: &CameraPath, src: (u32, u32), out: (u32, u32)) -> Option<String> {
    let (sw, sh) = (f64::from(src.0), f64::from(src.1));
    let (ow, oh) = out;
    let first = path.keyframes.first()?;
    match camera_kind(path, src, out) {
        CameraKind::Identity => None,
        CameraKind::Static => {
            let w = even_round(first.w).min(src.0);
            let h = even_round(first.h).min(src.1);
            let x = first.x.round().clamp(0.0, sw - f64::from(w));
            let y = first.y.round().clamp(0.0, sh - f64::from(h));
            Some(format!(
                "crop={w}:{h}:{x}:{y}:exact=1,scale={ow}:{oh}:flags=lanczos"
            ))
        }
        CameraKind::Pan => {
            let w = even_round(first.w).min(src.0);
            let h = even_round(first.h).min(src.1);
            let x = hinge(path, |k| k.x).to_expr("t");
            let y = hinge(path, |k| k.y).to_expr("t");
            Some(format!(
                "crop=w={w}:h={h}:x='min(max(0,{x}),{mx})':y='min(max(0,{y}),{my})':exact=1,scale={ow}:{oh}:flags=lanczos",
                mx = src.0 - w,
                my = src.1 - h,
            ))
        }
        CameraKind::Zoom => {
            // Scale the whole frame so the crop window becomes exactly ow x oh,
            // then crop a fixed-size window. s(t) = ow / w(t).
            let w = hinge(path, |k| k.w).to_expr("t");
            let x = hinge(path, |k| k.x / sw).to_expr("t");
            let y = hinge(path, |k| k.y / sh).to_expr("t");
            Some(format!(
                "scale=w='2*ceil({sw}*{ow}/(2*({w})))':h='2*ceil({sh}*{ow}/(2*({w})))':eval=frame:flags=bicubic,\
crop=w={ow}:h={oh}:x='min(max(0,({x})*iw),iw-{ow})':y='min(max(0,({y})*ih),ih-{oh})':exact=1"
            ))
        }
    }
}

/// Frames of white at the start of a `Flash` splice.
pub const FLASH_FRAMES: u32 = 4;
/// Opening scale-in: final zoom and duration (s).
pub const OPENING_POP: (f64, f64) = (1.08, 0.4);

/// Per-segment finishing filters on the composed picture (leading comma
/// included, empty when there is nothing to do).
fn segment_post(t: &ResolvedTimeline, i: usize) -> String {
    let mut post = String::new();
    if i == 0 && t.output.opening_pop {
        // Smoothstep scale 1 → 1.08 over 0.4 s, centre crop back to the raster.
        let (w, h) = (t.output.width, t.output.height);
        let (z, d) = OPENING_POP;
        let e = format!(
            "(1+{dz}*(3*pow(min(t/{d},1),2)-2*pow(min(t/{d},1),3)))",
            dz = z - 1.0
        );
        post.push_str(&format!(
            ",scale=w='2*ceil({w}*{e}/2)':h='2*ceil({h}*{e}/2)':eval=frame:flags=bicubic,crop={w}:{h}:exact=1"
        ));
    }
    if i > 0 && t.splices.get(i - 1).is_some_and(|s| s.flash) {
        post.push_str(&format!(",fade=t=in:s=0:n={FLASH_FRAMES}:color=white"));
    }
    post
}

/// Segments per FFmpeg process before the renderer switches to batching.
///
/// Every segment opens its own seeked input (two with audio); past this many
/// the renderer renders batches to lossless intermediates and joins them.
pub const BATCH_SEGMENTS: usize = 32;

/// Builds the graph. `ass_file` is a path relative to FFmpeg's working dir.
pub fn compile(t: &ResolvedTimeline, encoder: Encoder, ass_file: Option<&str>) -> CompiledGraph {
    let mut inputs = Vec::new();
    let mut stmts = Vec::new();
    segments(t, 0..t.segments.len(), &mut inputs, &mut stmts);
    let extra = finish(t, encoder, ass_file, inputs.len(), &mut stmts);
    CompiledGraph {
        inputs,
        filter: stmts.join(";\n"),
        has_audio: t.has_audio(),
        extra,
    }
}

/// Graph for one batch of segments, ending in raw `[vout]` / `[aout]`
/// (no grade, captions, mix or upload: those run once in [`compile_join`]).
pub fn compile_batch(t: &ResolvedTimeline, range: Range<usize>) -> CompiledGraph {
    let mut inputs = Vec::new();
    let mut stmts = Vec::new();
    segments(t, range, &mut inputs, &mut stmts);
    stmts.push("[vcat]null[vout]".to_owned());
    if t.has_audio() {
        stmts.push("[achain]anull[aout]".to_owned());
    }
    CompiledGraph {
        inputs,
        filter: stmts.join(";\n"),
        has_audio: t.has_audio(),
        extra: Vec::new(),
    }
}

/// Joins batch files (input `k` is batch `k`, split at `ranges`) and applies
/// the finishing chain once. The audio crossfade across each batch boundary
/// is the same splice an unbatched render would use, so lengths are identical.
pub fn compile_join(
    t: &ResolvedTimeline,
    ranges: &[Range<usize>],
    encoder: Encoder,
    ass_file: Option<&str>,
) -> CompiledGraph {
    let k = ranges.len();
    let mut stmts = Vec::new();
    let pads: String = (0..k).map(|i| format!("[{i}:v:0]")).collect();
    stmts.push(format!("{pads}concat=n={k}:v=1:a=0[vcat]"));
    if t.has_audio() {
        let labels: Vec<String> = (0..k).map(|i| format!("{i}:a:0")).collect();
        let splices: Vec<&Splice> = ranges[..k.saturating_sub(1)]
            .iter()
            .map(|r| &t.splices[r.end - 1])
            .collect();
        audio_chain(&labels, &splices, "achain", "jx", &mut stmts);
    }
    let extra = finish(t, encoder, ass_file, k, &mut stmts);
    CompiledGraph {
        inputs: Vec::new(),
        filter: stmts.join(";\n"),
        has_audio: t.has_audio(),
        extra,
    }
}

/// Splits `n` segments into consecutive batches of at most `size`.
pub fn batches(n: usize, size: usize) -> Vec<Range<usize>> {
    let size = size.max(1);
    (0..n).step_by(size).map(|s| s..(s + size).min(n)).collect()
}

/// Per-segment picture and sound for `range`, ending in `[vcat]` / `[achain]`.
fn segments(
    t: &ResolvedTimeline,
    range: Range<usize>,
    inputs: &mut Vec<InputSpec>,
    stmts: &mut Vec<String>,
) {
    let raster = (t.output.width, t.output.height);
    // The camera fills the picture area: the whole raster, or the FitBlur foreground.
    let out = t.output.picture_size();
    let fit_blur = matches!(t.output.layout, Layout::FitBlur { .. });
    let out_fps = t.output.fps;
    let has_audio = t.has_audio();
    // Several sources (or silent ones) may differ in channel layout: bring
    // every segment to one format before the crossfades.
    let mixed = !t.extra_sources.is_empty() || !t.source.has_audio;
    let mut vlabels = Vec::new();
    let mut alabels = Vec::new();

    for i in range.clone() {
        let seg = &t.segments[i];
        let source = t.source_of(seg);
        let src = (source.width, source.height);
        let frame_s = source.fps.frame_duration_us() as f64 / 1e6;
        // Seek half a frame early so the first wanted frame is never dropped
        // by µs rounding; `trim=end_frame` then fixes the exact count.
        let v_in = inputs.len();
        let seek = (seg.video.start_us as f64 / 1e6 - frame_s / 2.0).max(0.0);
        inputs.push(InputSpec {
            source: seg.source,
            seek_s: seek,
            duration_s: seg.video.len_us() as f64 / 1e6 + 3.0 * frame_s,
        });
        let cam = camera_filter(&seg.camera, src, out)
            .map(|c| format!(",{c}"))
            .unwrap_or_default();
        let gamma = if seg.gamma == 1.0 {
            String::new()
        } else {
            format!(",eq=gamma={:.3}", seg.gamma)
        };
        // `fps` first: a variable-frame-rate source (stream capture) skips
        // timestamps when it lagged; constant-rate frames keep the frame
        // count (and so A/V sync) exact. A no-op on constant-rate files.
        let head = format!(
            "[{v_in}:v:0]fps={}/{},tpad=stop_mode=clone:stop=3,trim=end_frame={frames},setpts=PTS-STARTPTS{gamma}",
            source.fps.num,
            source.fps.den,
            frames = seg.video.frames(),
        );
        // Speed change and/or another frame rate: retime on the output
        // clock, then fix the exact output frame count.
        let conv = if seg.speed == 1.0 && source.fps == out_fps {
            String::new()
        } else {
            let retime = if seg.speed == 1.0 {
                String::new()
            } else {
                format!(",setpts=(PTS-STARTPTS)/{:.6}", seg.speed)
            };
            format!(
                "{retime},fps={}/{},tpad=stop_mode=clone:stop=4,trim=end_frame={},setpts=PTS-STARTPTS",
                out_fps.num,
                out_fps.den,
                t.output_frames(i)
            )
        };
        let post = segment_post(t, i);
        if fit_blur {
            // Blurred, darkened cover fill computed at quarter resolution
            // (cheap), with the camera picture centred on top.
            let (rw, rh) = raster;
            let cam = camera_filter(&seg.camera, src, out)
                .unwrap_or_else(|| format!("scale={}:{}", out.0, out.1));
            stmts.push(format!("{head},split=2[f{i}][b{i}]"));
            stmts.push(format!(
                "[b{i}]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},boxblur=8:2,eq=brightness=-0.12:saturation=0.85,scale={rw}:{rh}:flags=bilinear,setsar=1[g{i}]",
                bw = rw / 4 / 2 * 2,
                bh = rh / 4 / 2 * 2,
            ));
            stmts.push(format!("[f{i}]{cam},setsar=1[p{i}]"));
            stmts.push(format!(
                "[g{i}][p{i}]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1{conv}{post},setsar=1[v{i}]"
            ));
        } else {
            stmts.push(format!("{head}{cam}{conv}{post},setsar=1[v{i}]"));
        }
        vlabels.push(format!("v{i}"));
        if has_audio {
            let olen = seg.audio_output_len_us() as f64 / 1e6;
            let fmt = if mixed {
                ",aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
            } else {
                ""
            };
            if source.has_audio {
                let a_in = inputs.len();
                let len = seg.audio.len_us() as f64 / 1e6;
                inputs.push(InputSpec {
                    source: seg.source,
                    seek_s: seg.audio.start_us as f64 / 1e6,
                    duration_s: len + 0.1,
                });
                let tempo = if seg.speed == 1.0 {
                    String::new()
                } else {
                    format!(
                        ",{},atrim=duration={olen:.6},apad=whole_dur={olen:.6}",
                        atempo(seg.speed)
                    )
                };
                stmts.push(format!(
                    "[{a_in}:a:0]aresample=48000,atrim=duration={len:.6},apad=whole_dur={len:.6},asetpts=PTS-STARTPTS{tempo}{fmt}[a{i}]"
                ));
            } else {
                // A file with no sound track contributes exact-length silence.
                stmts.push(format!(
                    "anullsrc=r=48000:cl=stereo,atrim=duration={olen:.6},asetpts=PTS-STARTPTS{fmt}[a{i}]"
                ));
            }
            alabels.push(format!("a{i}"));
        }
    }

    // Picture: hard concat (cuts are already frame-exact).
    if vlabels.len() == 1 {
        stmts.push(format!("[{}]null[vcat]", vlabels[0]));
    } else {
        let pads: String = vlabels.iter().map(|l| format!("[{l}]")).collect();
        stmts.push(format!("{pads}concat=n={}:v=1:a=0[vcat]", vlabels.len()));
    }
    if has_audio {
        let splices: Vec<&Splice> = t.splices[range.start..range.end - 1].iter().collect();
        audio_chain(&alabels, &splices, "achain", "ax", stmts);
    }
}

/// Chain of equal-power micro-crossfades (or plain joins) ending in `[out]`.
fn audio_chain(
    labels: &[String],
    splices: &[&Splice],
    out: &str,
    tmp: &str,
    stmts: &mut Vec<String>,
) {
    if splices.is_empty() {
        stmts.push(format!("[{}]anull[{out}]", labels[0]));
        return;
    }
    let mut acc = labels[0].clone();
    for (k, sp) in splices.iter().enumerate() {
        let next = &labels[k + 1];
        let label = if k + 1 == splices.len() {
            out.to_owned()
        } else {
            format!("{tmp}{k}")
        };
        stmts.push(if sp.crossfade_us > 0 {
            format!(
                "[{acc}][{next}]acrossfade=d={:.6}:c1=qsin:c2=qsin[{label}]",
                sp.crossfade_us as f64 / 1e6
            )
        } else {
            format!("[{acc}][{next}]concat=n=2:v=0:a=1[{label}]")
        });
        acc = label;
    }
}

/// An input beyond the source segments: an insert or the music bed.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtraInput {
    /// A still image looped for `len_s` at the output frame rate.
    Still {
        /// File path as written in the EDL.
        path: String,
        /// Read length (s).
        len_s: f64,
    },
    /// A video (or music) file looped for `len_s`.
    Looped {
        /// File path as written in the EDL.
        path: String,
        /// Read length (s).
        len_s: f64,
    },
}

impl ExtraInput {
    /// File path.
    pub fn path(&self) -> &str {
        match self {
            Self::Still { path, .. } | Self::Looped { path, .. } => path,
        }
    }
}

/// Border colour of cards.
const CARD_BORDER: &str = "0xFFE600";
/// Seconds a card takes to pop to full size.
pub const CARD_POP_S: f64 = 0.16;
/// Fade-out length of a card (s).
pub const CARD_OUT_S: f64 = 0.10;
/// Card scale at the first frame of the pop.
const CARD_POP_FROM: f64 = 0.5;

/// Filters for insert `j` (input `input`), overlaid on `[base]`, ending in
/// `[out]`. The input's timestamps start at 0, so they are shifted to the
/// insert's start first and every time below is on the output clock.
fn insert(t: &ResolvedTimeline, j: usize, input: usize, base: &str, out: &str) -> Vec<String> {
    let ins = &t.inserts[j];
    let (w, h) = (t.output.width, t.output.height);
    let (s, e) = (ins.start_us as f64 / 1e6, ins.end_us as f64 / 1e6);
    let fps = t.output.fps;
    let head = format!(
        "[{input}:v:0]fps={}/{},setpts=PTS-STARTPTS+{s:.6}/TB",
        fps.num, fps.den
    );
    let enable = format!("enable='between(t,{s:.3},{e:.3})'");
    if !ins.placement.is_card() {
        // B-roll cutaway: cover the raster, hard cut in and out.
        return vec![
            format!(
                "{head},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1[ins{j}]"
            ),
            format!("[{base}][ins{j}]overlay=x=0:y=0:{enable}:eof_action=pass[{out}]"),
        ];
    }
    let (wf, hf) = (f64::from(w), f64::from(h));
    let b = (((hf * 0.006).round() as u32) / 2 * 2).max(2);
    let max_w = (wf * 0.80) as u32 / 2 * 2;
    let max_h = (hf * 0.32) as u32 / 2 * 2;
    let cy = hf
        * match ins.placement {
            InsertPlacement::Upper => 0.33,
            InsertPlacement::Lower => 0.60,
            _ => 0.48,
        };
    // Ease-out-back from 50% to 100% (peaking near 105%) over CARD_POP_S:
    // the card "pops" like the stickers in commentary Shorts.
    let p = format!("clip((t-{s:.6})/{CARD_POP_S},0,1)");
    let k = format!(
        "({CARD_POP_FROM}+{r}*(1+2.70158*pow({p}-1,3)+1.70158*pow({p}-1,2)))",
        r = 1.0 - CARD_POP_FROM
    );
    // Canvas with room for the overshoot, so the overlay size never changes.
    let canvas = |m: u32| ((f64::from(m + 2 * b) * 1.06).ceil() as u32).div_ceil(2) * 2;
    let out_st = (e - CARD_OUT_S).max(s);
    vec![
        format!(
            "{head},scale=w={max_w}:h={max_h}:force_original_aspect_ratio=decrease,\
scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2,pad=w=iw+{b2}:h=ih+{b2}:x={b}:y={b}:color={CARD_BORDER},\
format=rgba,scale=w='2*trunc(iw*{k}/2)':h='2*trunc(ih*{k}/2)':eval=frame,\
pad=w={cw}:h={ch}:x=(ow-iw)/2:y=(oh-ih)/2:color=black@0:eval=frame,\
fade=t=out:st={out_st:.3}:d={CARD_OUT_S}:alpha=1[ins{j}]",
            b2 = 2 * b,
            cw = canvas(max_w),
            ch = canvas(max_h),
        ),
        format!("[{base}][ins{j}]overlay=x=(W-w)/2:y={cy:.0}-h/2:{enable}:eof_action=pass[{out}]"),
    ]
}

/// Synthesised effect sources (mono, 48 kHz), each `asplit` to its uses.
fn sfx_source(kind: SfxKind) -> &'static str {
    match kind {
        // Pink noise through a wide band-pass, swelling for 250 ms into the
        // cut and dying away just after it.
        SfxKind::Whoosh => {
            "anoisesrc=d=0.42:c=pink:r=48000:a=0.9,bandpass=f=1600:width_type=o:w=2.5,\
afade=t=in:d=0.25:curve=exp,afade=t=out:st=0.25:d=0.17:curve=exp"
        }
        // A 2.4 kHz tick with a burst of noise, gone in 60 ms.
        SfxKind::Click => {
            "aevalsrc='0.9*sin(2*PI*2400*t)*exp(-t*90)+0.6*(random(0)-0.5)*exp(-t*160)':d=0.06:s=48000"
        }
        // A sine whose pitch falls from 1280 Hz to 380 Hz in ~100 ms.
        SfxKind::Pop => {
            "aevalsrc='0.8*sin(2*PI*(380*t+30*(1-exp(-30*t))))*exp(-t*22)':d=0.16:s=48000"
        }
    }
}

const MIX_FORMAT: &str = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo";

/// Voice processing, effects, ducked music and the final level, from
/// `[achain]` to `[aout]`. The music input (if any) is `music_input`.
fn audio_finish(t: &ResolvedTimeline, music_input: Option<usize>, stmts: &mut Vec<String>) {
    let forward = t.output.audio_mix == AudioMix::VoiceForward;
    let Some(sound) = t.sound.as_ref() else {
        stmts.push(if forward {
            // Rumble cut, 3:1 compression from -20 dBFS, then single-pass
            // loudness normalisation (which works at 192 kHz internally).
            "[achain]highpass=f=80,acompressor=threshold=0.1:ratio=3:attack=5:release=150:makeup=1.5,\
loudnorm=I=-14:TP=-3:LRA=9,aresample=48000[aout]"
                .to_owned()
        } else {
            "[achain]anull[aout]".to_owned()
        });
        return;
    };
    let voice = if forward {
        "highpass=f=80,acompressor=threshold=0.1:ratio=3:attack=5:release=150:makeup=1.5,"
    } else {
        ""
    };
    let music = music_input.zip(sound.music.as_ref());
    let duck = music.is_some_and(|(_, m)| m.duck);
    stmts.push(format!(
        "[achain]{voice}{MIX_FORMAT}{}",
        if duck {
            ",asplit=2[vox][voxsc]"
        } else {
            "[vox]"
        }
    ));
    let mut mix = vec!["vox".to_owned()];
    if let Some((input, m)) = music {
        let dur = t.duration_us as f64 / 1e6;
        let fade_out = (dur - 0.8).max(0.0);
        let tail = if duck { "[mus0]" } else { "[mus]" };
        stmts.push(format!(
            "[{input}:a:0]loudnorm=I=-14:TP=-2:LRA=11,{MIX_FORMAT},volume={:.1}dB,afade=t=in:d=0.5,afade=t=out:st={fade_out:.3}:d=0.8{tail}",
            m.volume_db
        ));
        if duck {
            // Dip ~8-10 dB under speech, recover over ~0.4 s in the gaps.
            stmts.push(
                "[mus0][voxsc]sidechaincompress=threshold=0.03:ratio=6:attack=15:release=400[mus]"
                    .to_owned(),
            );
        }
        mix.push("mus".to_owned());
    }
    for kind in [SfxKind::Whoosh, SfxKind::Click, SfxKind::Pop] {
        let uses: Vec<i64> = sound
            .sfx
            .iter()
            .filter(|e| e.kind == kind)
            .map(|e| (e.at_us - kind.lead_us()).max(0))
            .collect();
        if uses.is_empty() {
            continue;
        }
        let tag = format!("{kind:?}").to_lowercase();
        let pads: String = (0..uses.len()).map(|n| format!("[{tag}{n}]")).collect();
        stmts.push(format!(
            "{},{MIX_FORMAT},volume={:.1}dB,asplit={}{pads}",
            sfx_source(kind),
            sound.sfx_db,
            uses.len()
        ));
        for (n, at) in uses.iter().enumerate() {
            let ms = (*at as f64 / 1e3).round() as i64;
            stmts.push(format!("[{tag}{n}]adelay=delays={ms}:all=1[{tag}d{n}]"));
            mix.push(format!("{tag}d{n}"));
        }
    }
    let pads: String = mix.iter().map(|l| format!("[{l}]")).collect();
    let level = if forward {
        "loudnorm=I=-14:TP=-3:LRA=9,aresample=48000"
    } else {
        // Keep the source level, but never let the extra layers clip.
        "alimiter=limit=0.95:level=0"
    };
    stmts.push(if mix.len() == 1 {
        format!("[vox]{level}[aout]")
    } else {
        format!(
            "{pads}amix=inputs={}:normalize=0:duration=first:dropout_transition=0,{level}[aout]",
            mix.len()
        )
    });
}

/// Grade, inserts, captions, upload and the audio finish, from `[vcat]` /
/// `[achain]`. Extra inputs (inserts, then music) are numbered from
/// `extra_base`; returns them in input order.
fn finish(
    t: &ResolvedTimeline,
    encoder: Encoder,
    ass_file: Option<&str>,
    extra_base: usize,
    stmts: &mut Vec<String>,
) -> Vec<ExtraInput> {
    let subs = ass_file
        .map(|p| format!("ass=filename={p},"))
        .unwrap_or_default();
    let grade = match t.output.grade {
        ColorGrade::None => "",
        ColorGrade::MobilePop => "eq=contrast=1.08:saturation=1.05,unsharp=5:5:0.6:5:5:0.0,",
    };
    let mut extra = Vec::new();
    let mut base = "vcat".to_owned();
    let mut pre = grade;
    if !t.inserts.is_empty() {
        stmts.push(format!("[vcat]{grade}null[vg]"));
        base = "vg".to_owned();
        pre = "";
        for (j, ins) in t.inserts.iter().enumerate() {
            let out = format!("vi{j}");
            stmts.extend(insert(t, j, extra_base + extra.len(), &base, &out));
            base = out;
            let len_s = (ins.end_us - ins.start_us) as f64 / 1e6 + 0.1;
            let path = ins.path.clone();
            extra.push(if is_video_insert(&path) {
                ExtraInput::Looped { path, len_s }
            } else {
                ExtraInput::Still { path, len_s }
            });
        }
    }
    if t.logo.is_none() && !t.output.progress_bar {
        stmts.push(format!(
            "[{base}]{pre}{subs}{}[vout]",
            encoder.upload_filter()
        ));
    } else {
        let (w, h) = (t.output.width, t.output.height);
        let dur = t.duration_us as f64 / 1e6;
        stmts.push(format!("[{base}]{pre}{subs}null[vs]"));
        base = "vs".to_owned();
        if let Some(l) = &t.logo {
            let input = extra_base + extra.len();
            extra.push(ExtraInput::Still {
                path: l.path.clone(),
                len_s: dur + 0.1,
            });
            let (wf, hf) = (f64::from(w), f64::from(h));
            let (x, y) = match l.placement {
                LogoPlacement::TopCenter => ("(W-w)/2".to_owned(), format!("{:.0}", hf * 0.024)),
                LogoPlacement::TopLeft => {
                    (format!("{:.0}", wf * 0.04), format!("{:.0}", hf * 0.024))
                }
                LogoPlacement::TopRight => (
                    format!("W-w-{:.0}", wf * 0.04),
                    format!("{:.0}", hf * 0.024),
                ),
                LogoPlacement::BottomCenter => {
                    ("(W-w)/2".to_owned(), format!("H-h-{:.0}", hf * 0.14))
                }
            };
            stmts.push(format!(
                "[{input}:v:0]scale={}:-2:flags=lanczos,format=rgba[logo]",
                l.width
            ));
            stmts.push(format!(
                "[{base}][logo]overlay=x={x}:y={y}:eof_action=repeat:shortest=1[vl]"
            ));
            base = "vl".to_owned();
        }
        if t.output.progress_bar {
            // A yellow bar that grows from the left edge to full width.
            let bh = (((f64::from(h) * 0.0065).round() as u32) / 2 * 2).max(2);
            stmts.push(format!(
                "color=c=0xFFE600:s={w}x{bh}:r={}/{}:d={:.6}[pbar]",
                t.output.fps.num,
                t.output.fps.den,
                dur + 0.5
            ));
            stmts.push(format!(
                "[{base}][pbar]overlay=x='-w+w*t/{dur:.6}':y=H-h:shortest=1[vp]"
            ));
            base = "vp".to_owned();
        }
        stmts.push(format!("[{base}]{}[vout]", encoder.upload_filter()));
    }
    if t.has_audio() {
        let music = t.sound.as_ref().and_then(|s| s.music.as_ref()).map(|m| {
            extra.push(ExtraInput::Looped {
                path: m.path.clone(),
                len_s: t.duration_us as f64 / 1e6 + 0.1,
            });
            extra_base + extra.len() - 1
        });
        audio_finish(t, music, stmts);
    }
    extra
}

#[cfg(test)]
mod tests {
    use super::*;
    use detent_core::camera::{CameraPath, CropKeyframe};
    use detent_core::edl::{AspectPreset, MotionPreset};
    use detent_core::time::{FrameRate, UsRange};
    use detent_core::timeline::{
        FrameRange, OutputFormat, ResolvedSegment, Splice, TIMELINE_VERSION, TimelineSource,
    };

    fn kf(t_us: i64, x: f64, w: f64) -> CropKeyframe {
        CropKeyframe {
            t_us,
            x,
            y: 0.0,
            w,
            h: w * 16.0 / 9.0,
        }
    }

    fn path(keys: Vec<CropKeyframe>) -> CameraPath {
        CameraPath {
            preset: MotionPreset::SmoothTrackSubject,
            keyframes: keys,
            solved_frames: 30,
            punch_ins: 0,
            subject_inset: false,
        }
    }

    fn timeline() -> ResolvedTimeline {
        let fps = FrameRate::new(30, 1).unwrap();
        let seg = |i: usize, a: i64, b: i64, out: i64, cam: CameraPath| {
            let v = FrameRange::new(a, b, fps);
            ResolvedSegment {
                clip_index: i,
                video: v,
                audio: UsRange::new(v.start_us - 10_000, v.end_us + 10_000),
                output_start_us: out,
                camera: cam,
                source: 0,
                speed: 1.0,
                gamma: 1.0,
            }
        };
        ResolvedTimeline {
            version: TIMELINE_VERSION,
            source: TimelineSource {
                path: "in.mp4".into(),
                width: 1920,
                height: 1080,
                fps,
                duration_us: 60_000_000,
                has_audio: true,
            },
            output: OutputFormat {
                aspect: AspectPreset::Vertical9x16,
                width: 1080,
                height: 1920,
                fps,
                layout: Default::default(),
                grade: Default::default(),
                audio_mix: Default::default(),
                opening_pop: false,
                progress_bar: false,
            },
            segments: vec![
                seg(0, 30, 90, 0, path(vec![kf(0, 656.25, 607.5)])),
                seg(
                    1,
                    300,
                    360,
                    2_000_000,
                    path(vec![kf(0, 600.0, 607.5), kf(1_000_000, 700.0, 500.0)]),
                ),
            ],
            splices: vec![Splice {
                after_segment: 0,
                crossfade_us: 20_000,
                av_offset_us: 0,
                flash: false,
            }],
            captions: None,
            inserts: Vec::new(),
            sound: None,
            extra_sources: Vec::new(),
            texts: Vec::new(),
            logo: None,
            duration_us: 4_000_000,
        }
    }

    #[test]
    fn classifies_cameras() {
        let full = path(vec![CropKeyframe {
            t_us: 0,
            x: 0.0,
            y: 0.0,
            w: 1920.0,
            h: 1080.0,
        }]);
        assert_eq!(
            camera_kind(&full, (1920, 1080), (1920, 1080)),
            CameraKind::Identity
        );
        assert_eq!(
            camera_kind(&full, (1920, 1080), (1280, 720)),
            CameraKind::Static
        );
        let pan = path(vec![kf(0, 0.0, 607.5), kf(500_000, 100.0, 607.5)]);
        assert_eq!(
            camera_kind(&pan, (1920, 1080), (1080, 1920)),
            CameraKind::Pan
        );
        let zoom = path(vec![kf(0, 0.0, 607.5), kf(500_000, 10.0, 550.0)]);
        assert_eq!(
            camera_kind(&zoom, (1920, 1080), (1080, 1920)),
            CameraKind::Zoom
        );
    }

    #[test]
    fn compiles_expected_graph() {
        let g = compile(&timeline(), Encoder::Libx264, Some("captions.ass"));
        assert_eq!(g.inputs.len(), 4, "video + audio input per segment");
        assert!((g.inputs[0].seek_s - (1.0 - 1.0 / 60.0)).abs() < 1e-6);
        assert!((g.inputs[1].seek_s - 0.99).abs() < 1e-9);
        let f = &g.filter;
        assert!(f.contains("[0:v:0]fps=30/1,tpad=stop_mode=clone:stop=3,trim=end_frame=60,setpts=PTS-STARTPTS,crop=608:1080:656:0:exact=1,scale=1080:1920"), "{f}");
        assert!(
            f.contains("[1:a:0]aresample=48000,atrim=duration=2.020000,apad=whole_dur=2.020000"),
            "{f}"
        );
        assert!(
            f.contains("eval=frame"),
            "zoom segment uses per-frame scale: {f}"
        );
        assert!(f.contains("[v0][v1]concat=n=2:v=1:a=0[vcat]"));
        assert!(f.contains("ass=filename=captions.ass,format=yuv420p[vout]"));
        assert!(
            f.contains("[a0][a1]acrossfade=d=0.020000:c1=qsin:c2=qsin[achain]"),
            "{f}"
        );
        assert!(!f.trim_end().ends_with(';'));
    }

    #[test]
    fn inserts_overlay_before_captions() {
        use detent_core::timeline::ResolvedInsert;
        let mut t = timeline();
        t.output.grade = ColorGrade::MobilePop;
        t.inserts = vec![
            ResolvedInsert {
                path: "shot.png".into(),
                start_us: 500_000,
                end_us: 2_000_000,
                placement: InsertPlacement::Upper,
            },
            ResolvedInsert {
                path: "news.jpg".into(),
                start_us: 2_500_000,
                end_us: 3_500_000,
                placement: InsertPlacement::Lower,
            },
        ];
        t.inserts.push(ResolvedInsert {
            path: "broll.mp4".into(),
            start_us: 3_600_000,
            end_us: 4_000_000,
            placement: InsertPlacement::Fullscreen,
        });
        let g = compile(&t, Encoder::Libx264, Some("captions.ass"));
        let paths: Vec<&str> = g.extra.iter().map(ExtraInput::path).collect();
        assert_eq!(paths, ["shot.png", "news.jpg", "broll.mp4"]);
        assert!(
            matches!(g.extra[0], ExtraInput::Still { len_s, .. } if (len_s - 1.6).abs() < 1e-9)
        );
        assert!(matches!(g.extra[2], ExtraInput::Looped { .. }));
        let f = &g.filter;
        // Inserts follow the 4 segment inputs, shifted onto the output clock.
        assert!(
            f.contains("[4:v:0]fps=30/1,setpts=PTS-STARTPTS+0.500000/TB,scale=w=864:h=614:force_original_aspect_ratio=decrease"),
            "{f}"
        );
        assert!(
            f.contains("[5:v:0]fps=30/1,setpts=PTS-STARTPTS+2.500000/TB,"),
            "{f}"
        );
        assert!(
            f.contains("pad=w=iw+24:h=ih+24:x=12:y=12:color=0xFFE600"),
            "{f}"
        );
        // Cards pop from 50% with an overshoot on a fixed transparent canvas.
        assert!(
            f.contains(
                "scale=w='2*trunc(iw*(0.5+0.5*(1+2.70158*pow(clip((t-0.500000)/0.16,0,1)-1,3)"
            ),
            "{f}"
        );
        assert!(
            f.contains("pad=w=942:h=678:x=(ow-iw)/2:y=(oh-ih)/2:color=black@0:eval=frame"),
            "{f}"
        );
        assert!(f.contains("fade=t=out:st=1.900:d=0.1:alpha=1[ins0]"), "{f}");
        // Grade first, inserts in order, then captions on top.
        assert!(f.contains("[vcat]eq=contrast=1.08"), "{f}");
        assert!(
            f.contains("[vg][ins0]overlay=x=(W-w)/2:y=634-h/2:enable='between(t,0.500,2.000)'"),
            "{f}"
        );
        assert!(
            f.contains("[vi0][ins1]overlay=x=(W-w)/2:y=1152-h/2:"),
            "{f}"
        );
        // The cutaway covers the frame.
        assert!(
            f.contains("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"),
            "{f}"
        );
        assert!(
            f.contains("[vi1][ins2]overlay=x=0:y=0:enable='between(t,3.600,4.000)'"),
            "{f}"
        );
        assert!(
            f.contains("[vi2]ass=filename=captions.ass,format=yuv420p[vout]"),
            "{f}"
        );
        // A join numbers the inserts after its batch files.
        let j = compile_join(&t, &[0..1, 1..2], Encoder::Libx264, None);
        assert!(j.filter.contains("[2:v:0]fps="), "{}", j.filter);
        assert!(
            j.filter.contains("[vi2]format=yuv420p[vout]"),
            "{}",
            j.filter
        );
    }

    #[test]
    fn sound_design_mixes_effects_and_ducked_music() {
        use detent_core::edl::MusicSpec;
        use detent_core::timeline::{ResolvedSound, SfxEvent};
        let mut t = timeline();
        t.output.audio_mix = AudioMix::VoiceForward;
        t.sound = Some(ResolvedSound {
            sfx: vec![
                SfxEvent {
                    kind: SfxKind::Pop,
                    at_us: 100_000,
                },
                SfxEvent {
                    kind: SfxKind::Whoosh,
                    at_us: 2_000_000,
                },
                SfxEvent {
                    kind: SfxKind::Whoosh,
                    at_us: 3_000_000,
                },
            ],
            sfx_db: -12.0,
            music: Some(MusicSpec {
                path: "bed.mp3".into(),
                volume_db: -20.0,
                duck: true,
            }),
        });
        let g = compile(&t, Encoder::Libx264, None);
        assert!(matches!(&g.extra[..], [ExtraInput::Looped { path, .. }] if path == "bed.mp3"));
        let f = &g.filter;
        // Voice: processed, then split for the sidechain.
        assert!(f.contains("[achain]highpass=f=80,acompressor="), "{f}");
        assert!(f.contains(",asplit=2[vox][voxsc]"), "{f}");
        // Music (input 4) under the voice, ducked.
        assert!(
            f.contains("[4:a:0]loudnorm=I=-14:TP=-2:LRA=11,aformat="),
            "{f}"
        );
        assert!(
            f.contains("volume=-20.0dB,afade=t=in:d=0.5,afade=t=out:st=3.200:d=0.8[mus0]"),
            "{f}"
        );
        assert!(f.contains("[mus0][voxsc]sidechaincompress="), "{f}");
        // One whoosh source split twice, swelling 250 ms into each cut.
        assert!(f.contains("anoisesrc=d=0.42"), "{f}");
        assert!(
            f.contains("volume=-12.0dB,asplit=2[whoosh0][whoosh1]"),
            "{f}"
        );
        assert!(
            f.contains("[whoosh0]adelay=delays=1750:all=1[whooshd0]"),
            "{f}"
        );
        assert!(
            f.contains("[whoosh1]adelay=delays=2750:all=1[whooshd1]"),
            "{f}"
        );
        assert!(f.contains("[pop0]adelay=delays=100:all=1[popd0]"), "{f}");
        assert!(
            f.contains("[vox][mus][whooshd0][whooshd1][popd0]amix=inputs=5:normalize=0:duration=first:dropout_transition=0,loudnorm=I=-14"),
            "{f}"
        );
        // Without sound design the mix is unchanged.
        t.sound = None;
        let g = compile(&t, Encoder::Libx264, None);
        assert!(g.extra.is_empty());
        assert!(!g.filter.contains("amix"));
    }

    #[test]
    fn fit_blur_composites_picture_over_blurred_fill() {
        let mut t = timeline();
        t.output.layout = Layout::FitBlur {
            fg_width: 1080,
            fg_height: 608,
        };
        for s in &mut t.segments {
            s.camera = path(vec![CropKeyframe {
                t_us: 0,
                x: 0.0,
                y: 0.0,
                w: 1920.0,
                h: 1080.0,
            }]);
        }
        let f = compile(&t, Encoder::Libx264, None).filter;
        assert!(f.contains("split=2[f0][b0]"), "{f}");
        assert!(
            f.contains(
                "[b0]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur"
            ),
            "{f}"
        );
        assert!(f.contains("scale=1080:1920:flags=bilinear"), "{f}");
        assert!(
            f.contains("crop=1920:1080:0:0:exact=1,scale=1080:608"),
            "{f}"
        );
        assert!(f.contains("[g0][p0]overlay=x=(W-w)/2:y=(H-h)/2"), "{f}");
    }

    #[test]
    fn single_segment_without_audio() {
        let mut t = timeline();
        t.segments.truncate(1);
        t.splices.clear();
        t.source.has_audio = false;
        let g = compile(&t, Encoder::H264Vaapi, None);
        assert_eq!(g.inputs.len(), 1);
        assert!(g.filter.contains("[v0]null[vcat]"));
        assert!(g.filter.ends_with("[vcat]format=nv12,hwupload[vout]"));
        assert!(!g.has_audio);
    }

    #[test]
    fn speed_silent_sources_logo_and_progress_bar() {
        use detent_core::edl::LogoPlacement;
        use detent_core::timeline::{ResolvedLogo, TimelineSource};
        let mut t = timeline();
        // Segment 1 reads a silent 25 fps source at half speed.
        t.extra_sources = vec![TimelineSource {
            path: "broll.mp4".into(),
            width: 1080,
            height: 1920,
            fps: FrameRate::new(25, 1).unwrap(),
            duration_us: 60_000_000,
            has_audio: false,
        }];
        let s = &mut t.segments[1];
        s.source = 1;
        s.speed = 0.5;
        s.gamma = 1.25;
        s.video = FrameRange::new(250, 300, FrameRate::new(25, 1).unwrap());
        s.camera = path(vec![CropKeyframe {
            t_us: 0,
            x: 0.0,
            y: 0.0,
            w: 1080.0,
            h: 1920.0,
        }]);
        s.audio = UsRange::new(s.video.start_us - 5_000, s.video.end_us);
        t.duration_us = 2_000_000 + t.segments[1].output_len_us();
        t.logo = Some(ResolvedLogo {
            path: "logo.png".into(),
            placement: LogoPlacement::TopRight,
            width: 300,
        });
        t.output.progress_bar = true;
        let g = compile(&t, Encoder::Libx264, None);
        // Video input of segment 1 reads source 1; no audio input for it.
        assert_eq!(g.inputs.len(), 3);
        assert_eq!(g.inputs[2].source, 1);
        let f = &g.filter;
        assert!(f.contains("[2:v:0]fps=25/1,tpad"), "{f}");
        assert!(f.contains("eq=gamma=1.250"), "{f}");
        assert!(
            f.contains(&format!(
                ",setpts=(PTS-STARTPTS)/0.500000,fps=30/1,tpad=stop_mode=clone:stop=4,trim=end_frame={}",
                t.output_frames(1)
            )),
            "{f}"
        );
        assert_eq!(t.output_frames(1), 120);
        assert!(
            f.contains("anullsrc=r=48000:cl=stereo,atrim=duration=4.010000"),
            "{f}"
        );
        assert!(
            f.contains("aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a0]"),
            "{f}"
        );
        assert!(matches!(&g.extra[..], [ExtraInput::Still { path, .. }] if path == "logo.png"));
        assert!(
            f.contains("[3:v:0]scale=300:-2:flags=lanczos,format=rgba[logo]"),
            "{f}"
        );
        assert!(f.contains("[vs][logo]overlay=x=W-w-43:y=46"), "{f}");
        assert!(f.contains("color=c=0xFFE600:s=1080x12:r=30/1"), "{f}");
        assert!(
            f.contains("[vl][pbar]overlay=x='-w+w*t/6.000000':y=H-h:shortest=1[vp]"),
            "{f}"
        );
        assert!(f.contains("[vp]format=yuv420p[vout]"), "{f}");
        assert_eq!(atempo(0.25), "atempo=0.5,atempo=0.500000");
        assert_eq!(atempo(3.0), "atempo=2,atempo=1.500000");
        assert_eq!(atempo(1.5), "atempo=1.500000");
    }

    #[test]
    fn three_way_crossfade_chain() {
        let mut t = timeline();
        let mut s = t.segments[1].clone();
        s.clip_index = 2;
        t.segments.push(s);
        t.splices.push(Splice {
            after_segment: 1,
            crossfade_us: 0,
            av_offset_us: 0,
            flash: false,
        });
        let f = compile(&t, Encoder::Libx264, None).filter;
        assert!(f.contains("[a0][a1]acrossfade=d=0.020000:c1=qsin:c2=qsin[ax0]"));
        assert!(f.contains("[ax0][a2]concat=n=2:v=0:a=1[achain]"));
    }
}
