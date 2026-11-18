//! The Semantic EDL: the *only* thing an LLM is allowed to emit.
//!
//! It expresses editorial intent (which ranges, in which order, with which
//! camera feel and audio transition) and nothing else. There are no pixel
//! coordinates, no FFmpeg expressions and no frame numbers: the deterministic
//! solver owns all of that. Unknown fields are rejected so schema drift or
//! hallucinated keys fail loudly at the boundary.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result, ValidationIssue};
use crate::index::MediaIndex;

/// Accepted J/L-cut offset range (ms).
pub const JL_OFFSET_RANGE_MS: std::ops::RangeInclusive<u32> = 150..=300;
/// Default J/L-cut offset (ms).
pub const JL_DEFAULT_OFFSET_MS: u32 = 220;
/// Shortest clip the solver will accept (ms).
pub const MIN_CLIP_MS: u64 = 300;

/// An edit decision list expressed as editorial intent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SemanticEdl {
    /// Source ranges in playback order. Ranges may be reordered relative to
    /// the source (e.g. a hook first) and may not overlap each other's output.
    pub clips: Vec<EdlClip>,
    /// Camera intent per clip. Clips without a cue get `None` (static framing).
    #[serde(default)]
    pub motion_cues: Vec<MotionCue>,
    /// Audio transition per splice. Splices without an entry are `Hard`.
    #[serde(default)]
    pub audio_transitions: Vec<AudioTransition>,
    /// Burned-in caption settings. Omit for no captions.
    #[serde(default)]
    pub captions: Option<CaptionSpec>,
    /// A short hook headline pinned to the top of the frame for the whole
    /// video (white box, black text), e.g. "Agust D's song in the new Warzone
    /// trailer". Requires `captions`. A newline forces a line break.
    #[serde(default)]
    #[schemars(length(max = 90))]
    pub headline: Option<String>,
    /// Caption words to emphasise (case-insensitive, punctuation ignored).
    /// `WordBounce` renders them green and slightly larger.
    #[serde(default)]
    pub highlight_words: Vec<String>,
    /// Per-word accent colours, the "colour-code the keywords" look (names
    /// red, places yellow, the punchline in blue...). Case-insensitive,
    /// punctuation ignored; overrides `highlight_words` for the same word.
    /// Used by `WordPunch`, `WordBounce` and `BoldPop`.
    #[serde(default)]
    pub word_colors: Vec<WordColor>,
    /// Persistent step/section labels ("1. Choose Distro", "2. Get the ISO")
    /// shown under the captions from the start of `clip` until the next
    /// chapter begins. Requires `captions`.
    #[serde(default)]
    pub chapters: Vec<Chapter>,
    /// B-roll over the A-roll: image/video cards (screenshots, headlines,
    /// product shots) that pop in with a coloured border, or full-frame
    /// cutaways while the voice keeps playing.
    #[serde(default)]
    pub inserts: Vec<Insert>,
    /// Sound design: whooshes on cuts, pops on cards, a ducked music bed.
    #[serde(default)]
    pub sound: Option<SoundSpec>,
    /// How long the headline stays on screen (ms from the start). Omit to pin
    /// it for the whole video; 3000 gives a classic 3-second hook banner.
    #[serde(default)]
    #[schemars(range(min = 1000, max = 60000))]
    pub headline_duration_ms: Option<u32>,
    /// Picture transition per splice. Splices without an entry are clean hard
    /// cuts (85-90% of cuts should stay that way).
    #[serde(default)]
    pub video_transitions: Vec<VideoTransition>,
    /// Automatic dead-air removal inside clips. Omit to keep clips as chosen.
    #[serde(default)]
    pub tighten: Option<TightenSpec>,
    /// On-screen text that does not come from speech: "show this text from
    /// here to here". Works without a transcript (music trailers, silent
    /// product footage), e.g. a line per clip.
    #[serde(default)]
    pub texts: Vec<TextOverlay>,
    /// Cut out frozen-picture stretches (dropped frames from stream lag)
    /// that the indexer found inside the chosen clips. On by default.
    #[serde(default = "default_true")]
    pub drop_freezes: bool,
    /// A logo or watermark image over the whole edit.
    #[serde(default)]
    pub logo: Option<Logo>,
    /// Output framing.
    #[serde(default)]
    pub output: OutputSpec,
}

/// Clips after [`SemanticEdl::split_clips`].
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    /// The expanded EDL (one clip per piece, `tighten` cleared).
    pub edl: SemanticEdl,
    /// Original clip index of every piece.
    pub origin: Vec<usize>,
    /// Dead air removed (ms).
    pub pause_ms: u64,
    /// Frozen picture removed (ms).
    pub freeze_ms: u64,
    /// Freezes removed or trimmed away.
    pub freezes: usize,
    /// Pauses removed.
    pub pauses: usize,
}

/// Slowest and fastest accepted clip speed.
pub const SPEED_RANGE: std::ops::RangeInclusive<f32> = 0.25..=4.0;

/// One source range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdlClip {
    /// Which indexed video this clip comes from: 0 is the main `index_path`,
    /// 1, 2, ... are the extra indexes in the order they were passed. Lets
    /// one edit combine a whole folder of footage.
    #[serde(default)]
    pub source: usize,
    /// Approximate in-point in source milliseconds. The solver snaps it into
    /// the nearest breath/silence valley within the snap tolerance (±120 ms).
    pub start_ms: u64,
    /// Approximate out-point in source milliseconds (exclusive).
    pub end_ms: u64,
    /// How strongly this clip works as an opening hook, 0.0-1.0.
    #[schemars(range(min = 0.0, max = 1.0))]
    pub hook_score: f32,
    /// One short sentence explaining the editorial choice.
    #[schemars(length(max = 280))]
    pub editorial_note: String,
    /// Playback speed, 0.25-4.0 (default 1). 0.5 is a half-speed slow-mo
    /// replay: repeat the payoff range as a second clip with `speed: 0.5`.
    /// Audio is time-stretched with its pitch kept.
    #[serde(default)]
    #[schemars(range(min = 0.25, max = 4.0))]
    pub speed: Option<f32>,
}

impl EdlClip {
    /// Effective speed (1.0 when unset).
    pub fn speed(&self) -> f64 {
        f64::from(self.speed.unwrap_or(1.0))
    }
}

/// On-screen text placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum TextPosition {
    /// Upper third, clear of the platform's top bar (the usual hook spot).
    #[default]
    Upper,
    /// Centre of the frame.
    Center,
    /// Lower third, above the platform's caption area.
    Lower,
}

/// Look of an on-screen text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum TextStyle {
    /// Heavy white type with a thick black stroke that pops in.
    #[default]
    Bold,
    /// Black type in a white box (like the headline).
    Boxed,
    /// Heavy yellow type with a black stroke.
    Yellow,
}

/// Text shown from a point in one clip to a later point, independent of
/// speech and captions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextOverlay {
    /// The text. A newline forces a line break.
    #[schemars(length(min = 1, max = 120))]
    pub text: String,
    /// Index into `clips`: the text appears during this clip.
    pub clip: usize,
    /// Delay from the clip's start in the output (ms).
    #[serde(default)]
    pub offset_ms: u32,
    /// Time on screen (ms). Omit to keep it up until the end of `end_clip`
    /// (or of `clip` when `end_clip` is omitted).
    #[serde(default)]
    #[schemars(range(min = 300, max = 60000))]
    pub duration_ms: Option<u32>,
    /// With no `duration_ms`, the clip whose end removes the text.
    #[serde(default)]
    pub end_clip: Option<usize>,
    /// Placement.
    #[serde(default)]
    pub position: TextPosition,
    /// Look.
    #[serde(default)]
    pub style: TextStyle,
}

/// Where the logo sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum LogoPlacement {
    /// Top centre.
    #[default]
    TopCenter,
    /// Top left corner.
    TopLeft,
    /// Top right corner.
    TopRight,
    /// Bottom centre, above the platform UI.
    BottomCenter,
}

/// A logo / watermark image over the whole edit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Logo {
    /// PNG (transparency kept), JPEG or WebP image.
    #[schemars(length(min = 1))]
    pub path: String,
    /// Placement preset.
    #[serde(default)]
    pub placement: LogoPlacement,
    /// Logo width as a percentage of the frame width (10-60, default 28).
    #[serde(default)]
    #[schemars(range(min = 10, max = 60))]
    pub width_pct: Option<u8>,
}

/// Brightness correction for dark footage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum Exposure {
    /// Leave the brightness alone.
    #[default]
    None,
    /// Lift the mid-tones of every clip (gamma 1.25), for dark game footage.
    Lift,
    /// Measure each clip and lift only the dark ones, more for darker clips.
    Auto,
}

/// Camera motion preset. Purely semantic; geometry is solved deterministically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum MotionPreset {
    /// Static framing centred on the subject at clip start.
    #[default]
    None,
    /// Snappy, overshoot-free zoom-in on vocal emphasis peaks, easing back out.
    SpringPunchIn,
    /// Keep the detected subject centred with a soft, critically damped follow.
    SmoothTrackSubject,
    /// Slow continuous push-in across the clip.
    SlowCreep,
}

/// Strength of a motion preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum Intensity {
    /// Barely perceptible.
    Subtle,
    /// Default.
    #[default]
    Medium,
    /// Pronounced.
    Strong,
}

/// Camera intent for one clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MotionCue {
    /// Index into `clips`.
    pub clip: usize,
    /// Preset to apply.
    pub preset: MotionPreset,
    /// Preset strength.
    #[serde(default)]
    pub intensity: Intensity,
}

/// Audio behaviour at a splice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum TransitionKind {
    /// Audio and video cut together (still receives a micro-crossfade).
    #[default]
    Hard,
    /// Next clip's audio starts *before* the picture cut (audio leads).
    AutoJCut,
    /// Previous clip's audio continues *past* the picture cut (audio lags).
    AutoLCut,
}

/// Audio transition at the splice between `after_clip` and `after_clip + 1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AudioTransition {
    /// Index of the clip *before* the splice.
    pub after_clip: usize,
    /// Transition kind.
    pub kind: TransitionKind,
    /// Target J/L offset in ms (150-300). The solver may shorten it (never
    /// below 150) or downgrade to `Hard` if surrounding audio is not safe.
    #[serde(default)]
    #[schemars(range(min = 150, max = 300))]
    pub offset_ms: Option<u32>,
}

/// Picture transition preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum VideoTransitionKind {
    /// Clean cut (the default for every splice).
    #[default]
    Cut,
    /// A 4-frame white flash into the next clip: kills, shock moments, beat drops.
    Flash,
}

/// Picture transition at the splice between `after_clip` and `after_clip + 1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VideoTransition {
    /// Index of the clip *before* the splice.
    pub after_clip: usize,
    /// Transition kind.
    pub kind: VideoTransitionKind,
}

/// Dead-air removal settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TightenSpec {
    /// Pauses (silence or breath) longer than this inside a clip are cut out,
    /// leaving a short natural gap. 200 ms is tight talking-head pacing.
    #[serde(default = "default_max_silence")]
    #[schemars(range(min = 150, max = 2000))]
    pub max_silence_ms: u32,
}

/// Default `max_silence_ms`.
pub const DEFAULT_MAX_SILENCE_MS: u32 = 200;

fn default_max_silence() -> u32 {
    DEFAULT_MAX_SILENCE_MS
}

impl Default for TightenSpec {
    fn default() -> Self {
        Self {
            max_silence_ms: DEFAULT_MAX_SILENCE_MS,
        }
    }
}

/// Colour finish applied to the whole output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum ColorGrade {
    /// Leave the source colours untouched.
    #[default]
    None,
    /// Crisp phone finish: a little more contrast (+8%), saturation (+5%,
    /// safe for skin tones) and light sharpening.
    MobilePop,
}

/// Audio finish applied to the whole output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum AudioMix {
    /// Keep the source levels.
    #[default]
    Source,
    /// Rumble cut, gentle compression and loudness normalisation to about
    /// -14 LUFS with peaks under -3 dBTP, so whispers and laughs stay even.
    VoiceForward,
}

/// Caption visual preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum CaptionStyle {
    /// Heavy white text, thick black outline, active line pops in.
    #[default]
    BoldPop,
    /// Small clean sans-serif with soft shadow.
    Minimal,
    /// Word-by-word highlight synced to speech.
    Karaoke,
    /// One word at a time in heavy yellow with a thick black stroke; each word
    /// lands with a small overshoot bounce.
    WordBounce,
    /// One word at a time in big white type with a heavy black stroke that
    /// slams in from 130%: the fast commentary-Short look. Pair it with
    /// `word_colors` to colour-code the keywords.
    WordPunch,
}

/// Accent colour for an emphasised caption word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum AccentColor {
    /// Danger, names, villains.
    Red,
    /// The default emphasis colour.
    Yellow,
    /// Money, wins, "yes".
    Green,
    /// Calm contrast, second names.
    Blue,
    /// Warm alternative to red.
    Orange,
    /// Playful accents.
    Pink,
}

impl AccentColor {
    /// ASS colour (`&HBBGGRR&`).
    pub fn ass(self) -> &'static str {
        match self {
            Self::Red => "&H3030FF&",
            Self::Yellow => "&H00E6FF&",
            Self::Green => "&H5CFF39&",
            Self::Blue => "&HFF9E2E&",
            Self::Orange => "&H1A8CFF&",
            Self::Pink => "&HC86BFF&",
        }
    }
}

/// One coloured caption word.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WordColor {
    /// The word as spoken (case-insensitive, punctuation ignored).
    #[schemars(length(min = 1, max = 40))]
    pub word: String,
    /// Its colour.
    pub color: AccentColor,
}

/// A persistent section label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Chapter {
    /// Index into `clips` where this chapter starts.
    pub clip: usize,
    /// Label text, e.g. "3. Make a Bootable USB".
    #[schemars(length(min = 1, max = 40))]
    pub text: String,
}

/// Where an insert sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum InsertPlacement {
    /// Card in the upper half, clear of the headline (the usual spot, above
    /// the face).
    #[default]
    Upper,
    /// Card in the centre of the frame (covers the speaker: short reveals).
    Center,
    /// Card in the lower half, above the captions.
    Lower,
    /// B-roll cutaway: the insert covers the whole frame (hard cut in and
    /// out, no border) while the A-roll audio keeps playing. Captions,
    /// chapters and the headline stay on top.
    Fullscreen,
}

impl InsertPlacement {
    /// True for a bordered card (not a full-frame cutaway).
    pub fn is_card(self) -> bool {
        self != Self::Fullscreen
    }
}

/// An image or video over the picture: a bordered card that pops in, or a
/// full-frame B-roll cutaway.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Insert {
    /// Image (PNG, JPEG, WebP) or video (MP4, MOV, WebM, MKV) file, absolute
    /// or relative to the working directory. Videos play from their start,
    /// muted, looping if shorter than `duration_ms`.
    #[schemars(length(min = 1))]
    pub path: String,
    /// Index into `clips`; the card appears during this clip.
    pub clip: usize,
    /// Delay from the clip's start in the output (ms).
    #[serde(default)]
    pub offset_ms: u32,
    /// Time on screen (ms).
    #[schemars(range(min = 400, max = 15000))]
    pub duration_ms: u32,
    /// Placement preset.
    #[serde(default)]
    pub placement: InsertPlacement,
}

/// Image extensions accepted for inserts.
pub const INSERT_EXTENSIONS: [&str; 4] = ["png", "jpg", "jpeg", "webp"];
/// Video extensions accepted for inserts.
pub const INSERT_VIDEO_EXTENSIONS: [&str; 4] = ["mp4", "mov", "webm", "mkv"];

/// True when `path` names a video insert (by extension).
pub fn is_video_insert(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| INSERT_VIDEO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Sound effect played on picture cuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum CutSfx {
    /// No effect.
    #[default]
    None,
    /// A short airy swoosh that peaks on the cut (synthesised, royalty free).
    Whoosh,
    /// A tight percussive click (synthesised).
    Click,
}

/// Sound effect played when a card pops in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum PopSfx {
    /// No effect.
    #[default]
    None,
    /// A bubbly "pop" with a fast downward pitch sweep (synthesised).
    Pop,
}

/// Background music under the voice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MusicSpec {
    /// Audio file you have the rights to use (loops if shorter than the edit).
    #[schemars(length(min = 1))]
    pub path: String,
    /// Bed level (dB) after the track is normalised to -14 LUFS, before
    /// ducking. -20 puts it about 20 dB under a -14 LUFS voice, like most
    /// commentary Shorts.
    #[serde(default = "default_music_db")]
    #[schemars(range(min = -40.0, max = -6.0))]
    pub volume_db: f32,
    /// Dip the music while someone talks (sidechain compression).
    #[serde(default = "default_true")]
    pub duck: bool,
}

fn default_music_db() -> f32 {
    -20.0
}

fn default_true() -> bool {
    true
}

/// Sound design layered over the A-roll audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct SoundSpec {
    /// Effect on every cut between two different clips (not on the jump cuts
    /// `tighten` makes inside a clip) and on every B-roll cutaway.
    #[serde(default)]
    pub cut_sfx: CutSfx,
    /// Effect when a card pops in.
    #[serde(default)]
    pub pop_sfx: PopSfx,
    /// Effect level relative to the voice (dB).
    #[serde(default = "default_sfx_db")]
    #[schemars(range(min = -30.0, max = 0.0))]
    pub sfx_db: f32,
    /// Background music bed.
    #[serde(default)]
    pub music: Option<MusicSpec>,
}

fn default_sfx_db() -> f32 {
    -12.0
}

/// Vertical caption placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum CaptionPosition {
    /// Lower third (safe area above platform UI).
    #[default]
    Lower,
    /// Centre of frame.
    Center,
    /// Upper third.
    Upper,
}

/// Syntactic grouping rules for caption blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CaptionGrouping {
    /// Minimum words per block (soft: pauses and sentence ends may force fewer).
    #[schemars(range(min = 1, max = 8))]
    pub min_words: u8,
    /// Maximum words per block.
    #[schemars(range(min = 1, max = 8))]
    pub max_words: u8,
    /// Maximum characters per block.
    #[schemars(range(min = 8, max = 64))]
    pub max_chars: u8,
    /// A silence at least this long (ms) always starts a new block.
    #[schemars(range(min = 100, max = 2000))]
    pub pause_break_ms: u32,
}

impl Default for CaptionGrouping {
    fn default() -> Self {
        Self {
            min_words: 3,
            max_words: 5,
            max_chars: 32,
            pause_break_ms: 300,
        }
    }
}

/// Caption settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct CaptionSpec {
    /// Visual preset.
    #[serde(default)]
    pub style: CaptionStyle,
    /// Placement.
    #[serde(default)]
    pub position: CaptionPosition,
    /// Grouping rules.
    #[serde(default)]
    pub grouping: CaptionGrouping,
}

/// Output aspect ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum AspectPreset {
    /// 9:16 vertical (Shorts / Reels / TikTok).
    #[default]
    Vertical9x16,
    /// 1:1 square.
    Square1x1,
    /// Keep the source aspect ratio (no reframing unless motion is requested).
    Source,
}

impl AspectPreset {
    /// Width / height ratio, or `None` for `Source`.
    pub fn ratio(self) -> Option<f64> {
        match self {
            Self::Vertical9x16 => Some(9.0 / 16.0),
            Self::Square1x1 => Some(1.0),
            Self::Source => None,
        }
    }
}

/// Output framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    /// Aspect ratio preset.
    #[serde(default)]
    pub aspect: AspectPreset,
    /// Output height in pixels (width is derived; both rounded to even).
    #[serde(default = "default_height")]
    #[schemars(range(min = 144, max = 4320))]
    pub height: u32,
    /// How the source fills the output frame.
    #[serde(default)]
    pub layout: OutputLayout,
    /// Colour finish.
    #[serde(default)]
    pub grade: ColorGrade,
    /// Audio finish.
    #[serde(default)]
    pub audio_mix: AudioMix,
    /// Open with motion: the first clip scales in from 100% to 108% over
    /// 0.4 s so the very first frame already feels alive.
    #[serde(default)]
    pub opening_pop: bool,
    /// Brightness correction for dark footage.
    #[serde(default)]
    pub exposure: Exposure,
    /// A thin bar along the bottom edge that fills as the video plays.
    #[serde(default)]
    pub progress_bar: bool,
}

/// How the source picture fills the output raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
pub enum OutputLayout {
    /// Crop a window with the output aspect (camera tracks within the source).
    /// Best for talking heads and vlogs.
    #[default]
    Crop,
    /// Show the whole source frame (letterboxed, full width) over a blurred,
    /// zoomed copy of itself. Best for screen recordings, gameplay and
    /// reaction videos with a facecam inset, where cropping would discard the
    /// content. Captions sit in the lower blurred band.
    FitBlur,
    /// Like `FitBlur`, but the foreground is a taller 4:5 window cropped from
    /// the source (the camera still tracks inside it), so the picture fills
    /// about 70% of a 9:16 frame instead of a thin strip. Best for cinematic
    /// trailers and gameplay where the action is near the centre.
    FitBlurTall,
}

fn default_height() -> u32 {
    1920
}

impl Default for OutputSpec {
    fn default() -> Self {
        Self {
            aspect: AspectPreset::default(),
            height: default_height(),
            layout: OutputLayout::default(),
            grade: ColorGrade::default(),
            audio_mix: AudioMix::default(),
            opening_pop: false,
            exposure: Exposure::default(),
            progress_bar: false,
        }
    }
}

impl SemanticEdl {
    /// Parses an EDL from JSON. Unknown fields are a hard error.
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|source| CoreError::Json {
            what: "SemanticEdl",
            source,
        })
    }

    /// The JSON Schema handed to the model.
    pub fn json_schema() -> schemars::Schema {
        schemars::schema_for!(SemanticEdl)
    }

    /// Effective motion preset for `clip`.
    pub fn motion_for(&self, clip: usize) -> (MotionPreset, Intensity) {
        self.motion_cues
            .iter()
            .find(|c| c.clip == clip)
            .map_or((MotionPreset::None, Intensity::Medium), |c| {
                (c.preset, c.intensity)
            })
    }

    /// Effective transition at the splice after `clip`.
    pub fn transition_after(&self, clip: usize) -> (TransitionKind, u32) {
        self.audio_transitions
            .iter()
            .find(|t| t.after_clip == clip)
            .map_or((TransitionKind::Hard, 0), |t| {
                (t.kind, t.offset_ms.unwrap_or(JL_DEFAULT_OFFSET_MS))
            })
    }

    /// Picture transition at the splice after `clip`.
    pub fn video_transition_after(&self, clip: usize) -> VideoTransitionKind {
        self.video_transitions
            .iter()
            .find(|t| t.after_clip == clip)
            .map_or(VideoTransitionKind::Cut, |t| t.kind)
    }

    /// Splits clips at pauses longer than `max_silence_ms` (dead-air removal).
    ///
    /// Returns the expanded EDL, the original clip index of every new clip,
    /// and the total milliseconds removed. Motion cues are copied onto every
    /// piece; audio and video transitions stay on the splice after the *last*
    /// piece. Pauses that overlap a transcript word are never cut, no piece is
    /// shorter than [`MIN_CLIP_MS`], and each cut keeps 60 ms of room tone on
    /// both sides so speech never sounds clipped.
    pub fn tightened(&self, index: &MediaIndex, max_silence_ms: u32) -> (Self, Vec<usize>, u64) {
        let s = self.split_clips(std::slice::from_ref(index), Some(max_silence_ms), false);
        (s.edl, s.origin, s.pause_ms)
    }

    /// Splits clips around the stretches that should not reach the output:
    /// pauses longer than `max_silence_ms` (dead-air removal, when given) and,
    /// with `drop_freezes`, frozen-picture stretches from each clip's own
    /// source index. A freeze at a clip's edge trims the clip instead. No
    /// piece is shorter than [`MIN_CLIP_MS`]; annotations follow the pieces
    /// as described for [`Self::tightened`].
    pub fn split_clips(
        &self,
        indexes: &[MediaIndex],
        max_silence_ms: Option<u32>,
        drop_freezes: bool,
    ) -> Split {
        const KEEP_MS: u64 = 60;
        let mut clips = Vec::new();
        let mut origin = Vec::new();
        let (mut pause_ms, mut freeze_ms, mut freezes, mut pauses) =
            (0_u64, 0_u64, 0_usize, 0_usize);
        for (i, c) in self.clips.iter().enumerate() {
            let Some(index) = indexes.get(c.source) else {
                clips.push(c.clone());
                origin.push(i);
                continue;
            };
            let (mut lo, mut hi) = (c.start_ms, c.end_ms);
            // (start, end, is_freeze) ranges to remove from inside the clip.
            let mut cuts: Vec<(u64, u64, bool)> = Vec::new();
            if drop_freezes {
                for f in index.visual.freezes.iter().filter(|f| !f.still) {
                    if f.end_ms <= lo || f.start_ms >= hi {
                        continue;
                    }
                    if f.start_ms <= lo {
                        if hi.saturating_sub(f.end_ms) >= MIN_CLIP_MS {
                            freeze_ms += f.end_ms - lo;
                            freezes += 1;
                            lo = f.end_ms;
                        }
                    } else if f.end_ms >= hi {
                        if f.start_ms.saturating_sub(lo) >= MIN_CLIP_MS {
                            freeze_ms += hi - f.start_ms;
                            freezes += 1;
                            hi = f.start_ms;
                        }
                    } else {
                        cuts.push((f.start_ms, f.end_ms, true));
                    }
                }
            }
            if let Some(max) = max_silence_ms {
                for p in &index.audio.pauses {
                    if p.end_ms.saturating_sub(p.start_ms) <= u64::from(max) {
                        continue;
                    }
                    let (ps, pe) = (p.start_ms + KEEP_MS, p.end_ms.saturating_sub(KEEP_MS));
                    let speech = index
                        .transcript
                        .iter()
                        .any(|w| w.start_ms < pe && w.end_ms > ps);
                    if pe > ps && !speech {
                        cuts.push((ps, pe, false));
                    }
                }
            }
            cuts.sort_by_key(|r| r.0);
            let mut merged: Vec<(u64, u64, bool)> = Vec::new();
            for r in cuts {
                match merged.last_mut() {
                    Some(m) if r.0 <= m.1 => {
                        m.1 = m.1.max(r.1);
                        m.2 |= r.2;
                    }
                    _ => merged.push(r),
                }
            }
            let mut start = lo;
            for (ps, pe, freeze) in merged {
                if ps < start + MIN_CLIP_MS || pe + MIN_CLIP_MS > hi {
                    continue;
                }
                clips.push(EdlClip {
                    start_ms: start,
                    end_ms: ps,
                    ..c.clone()
                });
                origin.push(i);
                if freeze {
                    freeze_ms += pe - ps;
                    freezes += 1;
                } else {
                    pause_ms += pe - ps;
                    pauses += 1;
                }
                start = pe;
            }
            clips.push(EdlClip {
                start_ms: start,
                end_ms: hi,
                ..c.clone()
            });
            origin.push(i);
        }
        let last_piece = |orig: usize| origin.iter().rposition(|&o| o == orig);
        let motion_cues = origin
            .iter()
            .enumerate()
            .filter_map(|(k, &o)| {
                self.motion_cues
                    .iter()
                    .find(|m| m.clip == o)
                    .map(|m| MotionCue {
                        clip: k,
                        ..m.clone()
                    })
            })
            .collect();
        let audio_transitions = self
            .audio_transitions
            .iter()
            .filter_map(|t| {
                last_piece(t.after_clip).map(|k| AudioTransition {
                    after_clip: k,
                    ..t.clone()
                })
            })
            .collect();
        let video_transitions = self
            .video_transitions
            .iter()
            .filter_map(|t| {
                last_piece(t.after_clip).map(|k| VideoTransition {
                    after_clip: k,
                    ..t.clone()
                })
            })
            .collect();
        // Chapters and inserts start with the first piece of their clip.
        let first_piece = |orig: usize| origin.iter().position(|&o| o == orig).unwrap_or(orig);
        let chapters = self
            .chapters
            .iter()
            .map(|c| Chapter {
                clip: first_piece(c.clip),
                ..c.clone()
            })
            .collect();
        let inserts = self
            .inserts
            .iter()
            .map(|c| Insert {
                clip: first_piece(c.clip),
                ..c.clone()
            })
            .collect();
        // Texts start with the first piece of their clip and, without an
        // explicit duration, end with the last piece of their end clip.
        let texts = self
            .texts
            .iter()
            .map(|t| TextOverlay {
                clip: first_piece(t.clip),
                end_clip: if t.duration_ms.is_none() {
                    last_piece(t.end_clip.unwrap_or(t.clip))
                } else {
                    t.end_clip.map(first_piece)
                },
                ..t.clone()
            })
            .collect();
        let edl = Self {
            clips,
            motion_cues,
            audio_transitions,
            video_transitions,
            chapters,
            inserts,
            texts,
            tighten: None,
            ..self.clone()
        };
        Split {
            edl,
            origin,
            pause_ms,
            freeze_ms,
            freezes,
            pauses,
        }
    }

    /// Semantic validation against a single media index (every clip uses
    /// source 0). Collects *every* issue.
    pub fn validate(&self, index: &MediaIndex) -> Result<()> {
        self.validate_sources(std::slice::from_ref(index))
    }

    /// Semantic validation against the indexes of every source, in `source`
    /// order. Collects *every* issue.
    pub fn validate_sources(&self, indexes: &[MediaIndex]) -> Result<()> {
        let mut issues = Vec::new();
        let n = self.clips.len();
        let used: Vec<&MediaIndex> = {
            let mut s: Vec<usize> = self.clips.iter().map(|c| c.source).collect();
            s.sort_unstable();
            s.dedup();
            s.into_iter().filter_map(|k| indexes.get(k)).collect()
        };
        if indexes.is_empty() {
            issues.push(ValidationIssue::new("/clips", "no source index given"));
        }
        if n == 0 {
            issues.push(ValidationIssue::new(
                "/clips",
                "at least one clip is required",
            ));
        }
        for (i, c) in self.clips.iter().enumerate() {
            let p = format!("/clips/{i}");
            let Some(index) = indexes.get(c.source) else {
                issues.push(ValidationIssue::new(
                    format!("{p}/source"),
                    format!(
                        "source {} out of range: {} index(es) given (0 is the main index, 1.. the extra ones)",
                        c.source,
                        indexes.len()
                    ),
                ));
                continue;
            };
            let dur = index.source.duration_ms;
            if let Some(s) = c.speed
                && (!SPEED_RANGE.contains(&s) || s.is_nan())
            {
                issues.push(ValidationIssue::new(
                    format!("{p}/speed"),
                    "speed must be within 0.25..=4.0",
                ));
            }
            if index.source.video.is_none() {
                issues.push(ValidationIssue::new(
                    format!("{p}/source"),
                    format!(
                        "source {} ({}) has no video stream; detent edits audio-visual media",
                        c.source, index.source.path
                    ),
                ));
            }
            if c.end_ms <= c.start_ms {
                issues.push(ValidationIssue::new(
                    format!("{p}/end_ms"),
                    format!(
                        "end_ms ({}) must be greater than start_ms ({})",
                        c.end_ms, c.start_ms
                    ),
                ));
            } else if c.end_ms - c.start_ms < MIN_CLIP_MS {
                issues.push(ValidationIssue::new(
                    p.clone(),
                    format!(
                        "clip is {} ms; minimum is {MIN_CLIP_MS} ms",
                        c.end_ms - c.start_ms
                    ),
                ));
            }
            if c.end_ms > dur {
                issues.push(ValidationIssue::new(
                    format!("{p}/end_ms"),
                    format!("end_ms ({}) exceeds media duration ({dur} ms)", c.end_ms),
                ));
            }
            if !(0.0..=1.0).contains(&c.hook_score) || c.hook_score.is_nan() {
                issues.push(ValidationIssue::new(
                    format!("{p}/hook_score"),
                    "hook_score must be within [0.0, 1.0]",
                ));
            }
            if c.editorial_note.chars().count() > 280 {
                issues.push(ValidationIssue::new(
                    format!("{p}/editorial_note"),
                    "editorial_note must be <= 280 characters",
                ));
            }
        }
        let mut seen = std::collections::HashSet::new();
        for (i, m) in self.motion_cues.iter().enumerate() {
            if m.clip >= n {
                issues.push(ValidationIssue::new(
                    format!("/motion_cues/{i}/clip"),
                    format!("clip index {} out of range (0..{n})", m.clip),
                ));
            }
            if !seen.insert(m.clip) {
                issues.push(ValidationIssue::new(
                    format!("/motion_cues/{i}"),
                    format!("duplicate motion cue for clip {}", m.clip),
                ));
            }
            let no_video = self
                .clips
                .get(m.clip)
                .and_then(|c| indexes.get(c.source))
                .is_some_and(|ix| ix.source.video.is_none());
            if m.preset != MotionPreset::None && no_video {
                issues.push(ValidationIssue::new(
                    format!("/motion_cues/{i}/preset"),
                    "motion requires a video stream",
                ));
            }
        }
        let mut seen = std::collections::HashSet::new();
        for (i, t) in self.audio_transitions.iter().enumerate() {
            let p = format!("/audio_transitions/{i}");
            if t.after_clip + 1 >= n {
                issues.push(ValidationIssue::new(
                    format!("{p}/after_clip"),
                    format!(
                        "after_clip must reference a splice (0..{})",
                        n.saturating_sub(1)
                    ),
                ));
            }
            if !seen.insert(t.after_clip) {
                issues.push(ValidationIssue::new(
                    p.clone(),
                    format!("duplicate transition after clip {}", t.after_clip),
                ));
            }
            if let Some(o) = t.offset_ms {
                if t.kind == TransitionKind::Hard {
                    issues.push(ValidationIssue::new(
                        format!("{p}/offset_ms"),
                        "offset_ms is only meaningful for AutoJCut / AutoLCut",
                    ));
                } else if !JL_OFFSET_RANGE_MS.contains(&o) {
                    issues.push(ValidationIssue::new(
                        format!("{p}/offset_ms"),
                        format!("offset_ms must be within 150..=300 (got {o})"),
                    ));
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        for (i, t) in self.video_transitions.iter().enumerate() {
            let p = format!("/video_transitions/{i}");
            if t.after_clip + 1 >= n {
                issues.push(ValidationIssue::new(
                    format!("{p}/after_clip"),
                    format!(
                        "after_clip must reference a splice (0..{})",
                        n.saturating_sub(1)
                    ),
                ));
            }
            if !seen.insert(t.after_clip) {
                issues.push(ValidationIssue::new(
                    p,
                    format!("duplicate video transition after clip {}", t.after_clip),
                ));
            }
        }
        if let Some(t) = &self.tighten
            && !(150..=2000).contains(&t.max_silence_ms)
        {
            issues.push(ValidationIssue::new(
                "/tighten/max_silence_ms",
                "max_silence_ms must be within 150..=2000",
            ));
        }
        if let Some(d) = self.headline_duration_ms {
            if !(1000..=60000).contains(&d) {
                issues.push(ValidationIssue::new(
                    "/headline_duration_ms",
                    "headline_duration_ms must be within 1000..=60000",
                ));
            }
            if self.headline.is_none() {
                issues.push(ValidationIssue::new(
                    "/headline_duration_ms",
                    "headline_duration_ms needs a headline",
                ));
            }
        }
        for (i, w) in self.word_colors.iter().enumerate() {
            let len = w.word.trim().chars().count();
            if len == 0 || len > 40 {
                issues.push(ValidationIssue::new(
                    format!("/word_colors/{i}/word"),
                    "word must be 1-40 characters",
                ));
            }
        }
        let mut last = None;
        for (i, c) in self.chapters.iter().enumerate() {
            let p = format!("/chapters/{i}");
            if c.clip >= n {
                issues.push(ValidationIssue::new(
                    format!("{p}/clip"),
                    format!("clip index {} out of range (0..{n})", c.clip),
                ));
            }
            if last.is_some_and(|l| c.clip <= l) {
                issues.push(ValidationIssue::new(
                    format!("{p}/clip"),
                    "chapters must be in increasing clip order, one per clip",
                ));
            }
            last = Some(c.clip);
            let len = c.text.trim().chars().count();
            if len == 0 || len > 40 {
                issues.push(ValidationIssue::new(
                    format!("{p}/text"),
                    "text must be 1-40 characters",
                ));
            }
        }
        for (i, c) in self.inserts.iter().enumerate() {
            let p = format!("/inserts/{i}");
            if c.clip >= n {
                issues.push(ValidationIssue::new(
                    format!("{p}/clip"),
                    format!("clip index {} out of range (0..{n})", c.clip),
                ));
            }
            if !(400..=15000).contains(&c.duration_ms) {
                issues.push(ValidationIssue::new(
                    format!("{p}/duration_ms"),
                    "duration_ms must be within 400..=15000",
                ));
            }
            let ext = std::path::Path::new(&c.path)
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase);
            if !ext.is_some_and(|e| {
                INSERT_EXTENSIONS.contains(&e.as_str())
                    || INSERT_VIDEO_EXTENSIONS.contains(&e.as_str())
            }) {
                issues.push(ValidationIssue::new(
                    format!("{p}/path"),
                    format!(
                        "inserts must be images ({}) or videos ({})",
                        INSERT_EXTENSIONS.join(", "),
                        INSERT_VIDEO_EXTENSIONS.join(", ")
                    ),
                ));
            }
        }
        if let Some(s) = &self.sound {
            if !(-30.0..=0.0).contains(&s.sfx_db) {
                issues.push(ValidationIssue::new(
                    "/sound/sfx_db",
                    "sfx_db must be within -30..=0",
                ));
            }
            if let Some(m) = &s.music {
                if !(-40.0..=-6.0).contains(&m.volume_db) {
                    issues.push(ValidationIssue::new(
                        "/sound/music/volume_db",
                        "volume_db must be within -40..=-6",
                    ));
                }
                if m.path.trim().is_empty() {
                    issues.push(ValidationIssue::new(
                        "/sound/music/path",
                        "music path is empty",
                    ));
                }
            }
        }
        if let Some(c) = &self.captions {
            let g = c.grouping;
            if g.min_words == 0 || g.max_words == 0 || g.min_words > g.max_words || g.max_words > 8
            {
                issues.push(ValidationIssue::new(
                    "/captions/grouping",
                    "require 1 <= min_words <= max_words <= 8",
                ));
            }
            if g.max_chars < 8 {
                issues.push(ValidationIssue::new(
                    "/captions/grouping/max_chars",
                    "max_chars must be >= 8",
                ));
            }
            if !used.is_empty() && used.iter().all(|ix| ix.transcript.is_empty()) {
                issues.push(ValidationIssue::new(
                    "/captions",
                    "captions requested but no source has a transcript; for on-screen text without speech use `texts`",
                ));
            }
        }
        if !(144..=4320).contains(&self.output.height) {
            issues.push(ValidationIssue::new(
                "/output/height",
                "height must be within 144..=4320",
            ));
        }
        for (i, t) in self.texts.iter().enumerate() {
            let p = format!("/texts/{i}");
            let len = t.text.trim().chars().count();
            if len == 0 || len > 120 {
                issues.push(ValidationIssue::new(
                    format!("{p}/text"),
                    "text must be 1-120 characters",
                ));
            }
            if t.clip >= n {
                issues.push(ValidationIssue::new(
                    format!("{p}/clip"),
                    format!("clip index {} out of range (0..{n})", t.clip),
                ));
            }
            if let Some(d) = t.duration_ms
                && !(300..=60000).contains(&d)
            {
                issues.push(ValidationIssue::new(
                    format!("{p}/duration_ms"),
                    "duration_ms must be within 300..=60000",
                ));
            }
            if let Some(e) = t.end_clip {
                if e >= n || e < t.clip {
                    issues.push(ValidationIssue::new(
                        format!("{p}/end_clip"),
                        format!("end_clip must be within {}..{n}", t.clip),
                    ));
                }
                if t.duration_ms.is_some() {
                    issues.push(ValidationIssue::new(
                        format!("{p}/end_clip"),
                        "give either duration_ms or end_clip, not both",
                    ));
                }
            }
        }
        if let Some(l) = &self.logo {
            let ext = std::path::Path::new(&l.path)
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase);
            if !ext.is_some_and(|e| INSERT_EXTENSIONS.contains(&e.as_str())) {
                issues.push(ValidationIssue::new(
                    "/logo/path",
                    format!("logo must be an image ({})", INSERT_EXTENSIONS.join(", ")),
                ));
            }
            if let Some(w) = l.width_pct
                && !(10..=60).contains(&w)
            {
                issues.push(ValidationIssue::new(
                    "/logo/width_pct",
                    "width_pct must be within 10..=60",
                ));
            }
        }
        if issues.is_empty() {
            Ok(())
        } else {
            Err(CoreError::Validation(issues))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_fields_and_raw_ffmpeg() {
        let bad = r#"{"clips":[{"start_ms":0,"end_ms":1000,"hook_score":0.5,"editorial_note":"x","zoompan":"z=1.5"}]}"#;
        let err = SemanticEdl::from_json(bad).unwrap_err().to_string();
        assert!(err.contains("unknown field"), "{err}");
    }

    #[test]
    fn rejects_pixel_coordinates_in_motion() {
        let bad = r#"{"clips":[{"start_ms":0,"end_ms":1000,"hook_score":0.5,"editorial_note":"x"}],
            "motion_cues":[{"clip":0,"preset":"SpringPunchIn","x":120}]}"#;
        assert!(SemanticEdl::from_json(bad).is_err());
    }

    #[test]
    fn defaults_are_applied() {
        let ok =
            r#"{"clips":[{"start_ms":0,"end_ms":1000,"hook_score":0.5,"editorial_note":"x"}]}"#;
        let edl = SemanticEdl::from_json(ok).unwrap();
        assert_eq!(edl.output.aspect, AspectPreset::Vertical9x16);
        assert_eq!(edl.output.height, 1920);
        assert_eq!(edl.motion_for(0).0, MotionPreset::None);
        assert_eq!(edl.transition_after(0).0, TransitionKind::Hard);
    }

    #[test]
    fn schema_exposes_presets_but_no_coordinates() {
        let schema = serde_json::to_string(&SemanticEdl::json_schema()).unwrap();
        for needle in [
            "SpringPunchIn",
            "AutoJCut",
            "Karaoke",
            "hook_score",
            "additionalProperties",
        ] {
            assert!(schema.contains(needle), "schema missing {needle}");
        }
        for forbidden in ["\"x\"", "zoompan", "filter"] {
            assert!(!schema.contains(forbidden), "schema leaks {forbidden}");
        }
    }
}
