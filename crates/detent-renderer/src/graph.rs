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
use detent_core::edl::{AudioMix, ColorGrade};
use detent_core::timeline::Layout;

use crate::encoder::Encoder;
use crate::expr::Hinges;

/// One `-i` input.
#[derive(Debug, Clone, PartialEq)]
pub struct InputSpec {
    /// Seek position (s), placed before `-i` (fast + accurate).
    pub seek_s: f64,
    /// Read duration (s).
    pub duration_s: f64,
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

/// Builds the graph. `ass_file` is a path relative to FFmpeg's working dir.
pub fn compile(t: &ResolvedTimeline, encoder: Encoder, ass_file: Option<&str>) -> CompiledGraph {
    let src = (t.source.width, t.source.height);
    let raster = (t.output.width, t.output.height);
    // The camera fills the picture area: the whole raster, or the FitBlur foreground.
    let out = t.output.picture_size();
    let fit_blur = matches!(t.output.layout, Layout::FitBlur { .. });
    let fps = t.source.fps;
    let frame_s = fps.frame_duration_us() as f64 / 1e6;
    let mut inputs = Vec::new();
    let mut stmts: Vec<String> = Vec::new();
    let n = t.segments.len();

    for (i, seg) in t.segments.iter().enumerate() {
        // Seek half a frame early so the first wanted frame is never dropped
        // by µs rounding; `trim=end_frame` then fixes the exact count.
        let v_in = inputs.len();
        let seek = (seg.video.start_us as f64 / 1e6 - frame_s / 2.0).max(0.0);
        inputs.push(InputSpec {
            seek_s: seek,
            duration_s: seg.video.len_us() as f64 / 1e6 + 3.0 * frame_s,
        });
        let cam = camera_filter(&seg.camera, src, out)
            .map(|c| format!(",{c}"))
            .unwrap_or_default();
        let head = format!(
            "[{v_in}:v:0]tpad=stop_mode=clone:stop=3,trim=end_frame={frames},setpts=PTS-STARTPTS",
            frames = seg.video.frames(),
        );
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
                "[g{i}][p{i}]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1{post},setsar=1[v{i}]"
            ));
        } else {
            stmts.push(format!("{head}{cam}{post},setsar=1[v{i}]"));
        }
        if t.source.has_audio {
            let a_in = inputs.len();
            let len = seg.audio.len_us() as f64 / 1e6;
            inputs.push(InputSpec {
                seek_s: seg.audio.start_us as f64 / 1e6,
                duration_s: len + 0.1,
            });
            stmts.push(format!(
                "[{a_in}:a:0]aresample=48000,atrim=duration={len:.6},apad=whole_dur={len:.6},asetpts=PTS-STARTPTS[a{i}]"
            ));
        }
    }

    // Picture: hard concat (cuts are already frame-exact).
    if n == 1 {
        stmts.push("[v0]null[vcat]".to_owned());
    } else {
        let pads: String = (0..n).map(|i| format!("[v{i}]")).collect();
        stmts.push(format!("{pads}concat=n={n}:v=1:a=0[vcat]"));
    }
    let subs = ass_file
        .map(|p| format!("ass=filename={p},"))
        .unwrap_or_default();
    let grade = match t.output.grade {
        ColorGrade::None => "",
        ColorGrade::MobilePop => "eq=contrast=1.08:saturation=1.05,unsharp=5:5:0.6:5:5:0.0,",
    };
    stmts.push(format!(
        "[vcat]{grade}{subs}{}[vout]",
        encoder.upload_filter()
    ));

    // Sound: chain of equal-power micro-crossfades.
    if t.source.has_audio {
        let voice = t.output.audio_mix == AudioMix::VoiceForward;
        let final_label = if voice { "amix0" } else { "aout" };
        if t.splices.is_empty() {
            stmts.push(format!("[a0]anull[{final_label}]"));
        }
        let mut acc = "a0".to_owned();
        for (k, sp) in t.splices.iter().enumerate() {
            let next = format!("a{}", k + 1);
            let label = if k + 1 == t.splices.len() {
                final_label.to_owned()
            } else {
                format!("ax{k}")
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
        if voice {
            // Rumble cut, 3:1 compression from -20 dBFS, then single-pass
            // loudness normalisation (which works at 192 kHz internally).
            stmts.push(
                "[amix0]highpass=f=80,acompressor=threshold=0.1:ratio=3:attack=5:release=150:makeup=1.5,\
loudnorm=I=-14:TP=-3:LRA=9,aresample=48000[aout]"
                    .to_owned(),
            );
        }
    }
    let f = stmts.join(";\n");

    CompiledGraph {
        inputs,
        filter: f,
        has_audio: t.source.has_audio,
    }
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
        assert!(f.contains("[0:v:0]tpad=stop_mode=clone:stop=3,trim=end_frame=60,setpts=PTS-STARTPTS,crop=608:1080:656:0:exact=1,scale=1080:1920"), "{f}");
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
            f.contains("[a0][a1]acrossfade=d=0.020000:c1=qsin:c2=qsin[aout]"),
            "{f}"
        );
        assert!(!f.trim_end().ends_with(';'));
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
        assert!(f.contains("[ax0][a2]concat=n=2:v=0:a=1[aout]"));
    }
}
