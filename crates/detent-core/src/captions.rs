//! Syntactic caption chunking and Advanced SubStation Alpha (`.ass`) output.
//!
//! Words are first split into *phrases* at hard boundaries (long pauses,
//! sentence-final punctuation, clip boundaries). Each phrase is then broken
//! into 1..=`max_words` blocks by a Knuth–Plass-style dynamic program that
//! minimises a cost function: ending a block on a function word ("the",
//! "of", "and", ...) is heavily penalised, blocks shorter than `min_words`
//! are penalised, and ending on a comma is rewarded. The same routine picks
//! the line break inside a block when it is too long for one line.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::edl::{CaptionGrouping, CaptionPosition, CaptionStyle};

/// A word positioned on the output timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TimedWord {
    /// Surface text.
    pub text: String,
    /// Output start (µs).
    pub start_us: i64,
    /// Output end (µs).
    pub end_us: i64,
}

/// One on-screen caption.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CaptionBlock {
    /// Display start (µs, output timeline).
    pub start_us: i64,
    /// Display end (µs).
    pub end_us: i64,
    /// Words in the block.
    pub words: Vec<TimedWord>,
    /// Index of the first word of the second line, if the block wraps.
    pub line_break: Option<usize>,
}

/// Function words that must not end a line / block.
const WEAK_WORDS: &[&str] = &[
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "with", "from", "by", "into", "onto",
    "about", "as", "over", "under", "and", "but", "or", "nor", "so", "yet", "if", "that", "than",
    "my", "your", "his", "her", "its", "our", "their", "this", "these", "those", "is", "are",
    "was", "were", "be", "i", "we", "you", "it's", "i'm", "very", "not", "no",
];

fn normalized(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
        .to_lowercase()
}

/// True when `word` is a function word *without* trailing punctuation
/// (a comma or full stop after it means the clause ends there, which is fine).
pub fn is_weak(word: &str) -> bool {
    let w = word.trim();
    if w.ends_with([',', '.', '!', '?', ';', ':', '…']) {
        return false;
    }
    WEAK_WORDS.contains(&normalized(w).as_str())
}

fn ends_sentence(word: &str) -> bool {
    word.trim_end_matches(['"', '\'', ')', '”', '’'])
        .ends_with(['.', '!', '?', '…'])
}

fn ends_clause(word: &str) -> bool {
    word.trim_end().ends_with([',', ';', ':', '—', '-'])
}

const SHORT_COST: f64 = 4.0;
const NON_CLAUSE_END_COST: f64 = 1.0;

fn chars(words: &[TimedWord]) -> usize {
    words
        .iter()
        .map(|w| w.text.trim().chars().count())
        .sum::<usize>()
        + words.len().saturating_sub(1)
}

/// Lexicographic block cost: (overflowing blocks, weak endings, soft penalty).
/// Hard readability rules strictly dominate aesthetic preferences, so the DP
/// never trades a dangling "the" for slightly better-balanced blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cost(u32, u32, f64);

impl Cost {
    const ZERO: Self = Self(0, 0, 0.0);
    const INF: Self = Self(u32::MAX, u32::MAX, f64::INFINITY);

    fn add(self, o: Self) -> Self {
        Self(
            self.0.saturating_add(o.0),
            self.1.saturating_add(o.1),
            self.2 + o.2,
        )
    }

    fn lt(self, o: Self) -> bool {
        (self.0, self.1)
            .cmp(&(o.0, o.1))
            .then(self.2.total_cmp(&o.2))
            .is_lt()
    }
}

/// Optimal segmentation of `phrase` into blocks (returns block end indices).
fn segment(phrase: &[TimedWord], g: &CaptionGrouping) -> Vec<usize> {
    let n = phrase.len();
    let max_w = usize::from(g.max_words.max(1));
    let min_w = usize::from(g.min_words.max(1)).min(max_w);
    let ideal = (min_w + max_w) as f64 / 2.0;
    let mut best = vec![Cost::INF; n + 1];
    let mut back = vec![0_usize; n + 1];
    best[0] = Cost::ZERO;
    for j in 1..=n {
        for len in 1..=max_w.min(j) {
            let i = j - len;
            if best[i] == Cost::INF {
                continue;
            }
            let block = &phrase[i..j];
            let mut cost = Cost(0, 0, (len as f64 - ideal).powi(2) * 0.25);
            if chars(block) > usize::from(g.max_chars) && len > 1 {
                cost.0 += 1;
            }
            if len < min_w && !(i == 0 && j == n) {
                cost.2 += SHORT_COST * (min_w - len) as f64;
            }
            if j < n {
                let last = &block[len - 1].text;
                if is_weak(last) {
                    cost.1 += 1;
                } else if !ends_clause(last) {
                    cost.2 += NON_CLAUSE_END_COST;
                }
            }
            let total = best[i].add(cost);
            if total.lt(best[j]) {
                best[j] = total;
                back[j] = i;
            }
        }
    }
    let mut ends = Vec::new();
    let mut j = n;
    while j > 0 {
        ends.push(j);
        j = back[j];
    }
    ends.reverse();
    ends
}

/// Picks the line break for a block whose text exceeds `line_chars`.
/// Ranked lexicographically: no dangling function word, then no overflowing
/// line, then the most balanced split.
fn line_break(words: &[TimedWord], line_chars: usize) -> Option<usize> {
    if words.len() < 2 || chars(words) <= line_chars {
        return None;
    }
    (1..words.len()).min_by_key(|&k| {
        let (a, b) = (chars(&words[..k]), chars(&words[k..]));
        (
            is_weak(&words[k - 1].text),
            a.max(b).saturating_sub(line_chars),
            a.abs_diff(b),
        )
    })
}

/// Groups words into caption blocks.
///
/// `hard_breaks` lists word indices that must start a new block (e.g. the
/// first word of each clip). Display times never overlap: a block lingers up
/// to `linger_us` after its last word but always ends before the next starts.
pub fn chunk(
    words: &[TimedWord],
    hard_breaks: &[usize],
    g: &CaptionGrouping,
    linger_us: i64,
) -> Vec<CaptionBlock> {
    let pause_us = i64::from(g.pause_break_ms) * 1_000;
    let line_chars = (usize::from(g.max_chars) * 3 / 5).max(8);
    let mut blocks: Vec<CaptionBlock> = Vec::new();
    let mut start = 0;
    for i in 0..words.len() {
        let next_is_break = i + 1 == words.len()
            || hard_breaks.contains(&(i + 1))
            || words[i + 1].start_us - words[i].end_us >= pause_us
            || ends_sentence(&words[i].text);
        if !next_is_break {
            continue;
        }
        let phrase = &words[start..=i];
        let mut s = 0;
        for e in segment(phrase, g) {
            let ws = phrase[s..e].to_vec();
            let line_break = line_break(&ws, line_chars);
            let (Some(first), Some(last)) = (ws.first(), ws.last()) else {
                continue;
            };
            blocks.push(CaptionBlock {
                start_us: first.start_us,
                end_us: last.end_us,
                line_break,
                words: ws,
            });
            s = e;
        }
        start = i + 1;
    }
    // Linger, but never overlap the next block; enforce monotonic display.
    for k in 0..blocks.len() {
        let next_start = blocks.get(k + 1).map_or(i64::MAX, |b| b.start_us);
        let b = &mut blocks[k];
        b.end_us = (b.end_us + linger_us).min(next_start).max(b.start_us + 1);
    }
    blocks
}

/// Formats microseconds as an ASS timestamp `H:MM:SS.cc`.
pub fn ass_time(us: i64) -> String {
    let cs = (us.max(0) + 5_000) / 10_000;
    format!(
        "{}:{:02}:{:02}.{:02}",
        cs / 360_000,
        (cs / 6_000) % 60,
        (cs / 100) % 60,
        cs % 100
    )
}

/// Removes characters that would be parsed as ASS override syntax.
fn sanitize(text: &str) -> String {
    text.trim()
        .chars()
        .map(|c| match c {
            '{' => '(',
            '}' => ')',
            '\\' => '/',
            '\n' | '\r' => ' ',
            c => c,
        })
        .collect()
}

/// Renders blocks to a complete `.ass` document for a `width x height` canvas.
pub fn to_ass(
    blocks: &[CaptionBlock],
    style: CaptionStyle,
    position: CaptionPosition,
    width: u32,
    height: u32,
) -> String {
    let h = f64::from(height);
    let (align, margin_v) = match position {
        CaptionPosition::Lower => (2, (h * 0.22).round()),
        CaptionPosition::Center => (5, 0.0),
        CaptionPosition::Upper => (8, (h * 0.12).round()),
    };
    // Colours are &HAABBGGRR.
    let (font, size, bold, primary, secondary, outline, shadow, outline_w, shadow_d) = match style {
        CaptionStyle::BoldPop => (
            "Arial Black",
            h * 0.045,
            -1,
            "&H00FFFFFF",
            "&H00FFFFFF",
            "&H00000000",
            "&H80000000",
            h * 0.0055,
            h * 0.002,
        ),
        CaptionStyle::Minimal => (
            "Arial",
            h * 0.036,
            0,
            "&H00FFFFFF",
            "&H00FFFFFF",
            "&H00202020",
            "&H96000000",
            h * 0.0015,
            h * 0.0025,
        ),
        CaptionStyle::Karaoke => (
            "Arial Black",
            h * 0.045,
            -1,
            "&H0000E5FF",
            "&H00FFFFFF",
            "&H00000000",
            "&H80000000",
            h * 0.005,
            h * 0.002,
        ),
    };
    let mut out = String::with_capacity(512 + blocks.len() * 96);
    out.push_str("[Script Info]\n; Generated by detent\nScriptType: v4.00+\n");
    out.push_str(&format!("PlayResX: {width}\nPlayResY: {height}\n"));
    out.push_str("WrapStyle: 2\nScaledBorderAndShadow: yes\nYCbCr Matrix: TV.709\n\n");
    out.push_str("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n");
    out.push_str(&format!(
        "Style: Default,{font},{:.0},{primary},{secondary},{outline},{shadow},{bold},0,0,0,100,100,0,0,1,{:.1},{:.1},{align},{m},{m},{margin_v:.0},1\n\n",
        size,
        outline_w,
        shadow_d,
        m = (f64::from(width) * 0.06).round(),
    ));
    out.push_str("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n");
    for b in blocks {
        let mut text = String::new();
        if style == CaptionStyle::BoldPop {
            text.push_str(r"{\fscx85\fscy85\t(0,90,\fscx100\fscy100)}");
        }
        let mut cursor = b.start_us;
        for (i, w) in b.words.iter().enumerate() {
            if i > 0 {
                text.push_str(if b.line_break == Some(i) { r"\N" } else { " " });
            }
            if style == CaptionStyle::Karaoke {
                // \k durations in centiseconds, including any gap before the word.
                let end = if i + 1 == b.words.len() {
                    b.end_us
                } else {
                    w.end_us
                };
                let cs = ((end - cursor).max(0) + 5_000) / 10_000;
                text.push_str(&format!(r"{{\k{cs}}}"));
                cursor = end;
            }
            let word = sanitize(&w.text);
            if style == CaptionStyle::BoldPop {
                text.push_str(&word.to_uppercase());
            } else {
                text.push_str(&word);
            }
        }
        out.push_str(&format!(
            "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
            ass_time(b.start_us),
            ass_time(b.end_us),
            text
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn words(text: &str) -> Vec<TimedWord> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, t)| TimedWord {
                text: t.to_owned(),
                start_us: i as i64 * 300_000,
                end_us: i as i64 * 300_000 + 250_000,
            })
            .collect()
    }

    fn texts(blocks: &[CaptionBlock]) -> Vec<String> {
        blocks
            .iter()
            .map(|b| {
                b.words
                    .iter()
                    .map(|w| w.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    #[test]
    fn never_orphans_function_words() {
        let ws = words("we shipped the fastest version of the indexer to all of our users today");
        let blocks = chunk(&ws, &[], &CaptionGrouping::default(), 200_000);
        let t = texts(&blocks);
        for b in &t[..t.len() - 1] {
            let last = b.split(' ').next_back().unwrap();
            assert!(!is_weak(last), "block ends with weak word: {t:?}");
        }
        assert_eq!(
            t.join(" "),
            ws.iter()
                .map(|w| w.text.clone())
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    #[test]
    fn breaks_on_sentence_end_and_pause() {
        let mut ws = words("this is great. now watch this part closely");
        // Insert a long pause before "closely".
        let last = ws.len() - 1;
        ws[last].start_us += 2_000_000;
        ws[last].end_us += 2_000_000;
        let blocks = chunk(&ws, &[], &CaptionGrouping::default(), 0);
        let t = texts(&blocks);
        assert_eq!(t[0], "this is great.");
        assert_eq!(t.last().unwrap(), "closely");
    }

    #[test]
    fn respects_hard_breaks() {
        let ws = words("one two three four five six");
        let blocks = chunk(&ws, &[2], &CaptionGrouping::default(), 0);
        assert_eq!(texts(&blocks)[0], "one two");
    }

    #[test]
    fn line_break_avoids_weak_word() {
        let ws = words("unbelievably fast indexing of enormous files");
        let k = line_break(&ws[..5], 16).unwrap();
        assert!(!is_weak(&ws[k - 1].text));
    }

    #[test]
    fn ass_timestamps() {
        assert_eq!(ass_time(0), "0:00:00.00");
        assert_eq!(ass_time(61_234_567), "0:01:01.23");
        assert_eq!(ass_time(3_600_000_000), "1:00:00.00");
    }

    #[test]
    fn ass_document_structure_and_escaping() {
        let mut ws = words("hello {world} again");
        ws[1].text = r"{\b1}world".into();
        let blocks = chunk(&ws, &[], &CaptionGrouping::default(), 100_000);
        let doc = to_ass(
            &blocks,
            CaptionStyle::Karaoke,
            CaptionPosition::Lower,
            1080,
            1920,
        );
        assert!(doc.contains("PlayResX: 1080\nPlayResY: 1920"));
        assert!(doc.contains("[Events]"));
        assert!(doc.contains(r"{\k"));
        assert!(
            !doc.contains(r"{\b1}"),
            "user text must not inject overrides"
        );
        let dialogue: Vec<_> = doc.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert_eq!(dialogue.len(), blocks.len());
        let bold = to_ass(
            &blocks,
            CaptionStyle::BoldPop,
            CaptionPosition::Center,
            1080,
            1920,
        );
        assert!(bold.contains("HELLO"));
        assert!(bold.contains(",5,"), "centre alignment");
    }

    fn arb_word() -> impl Strategy<Value = String> {
        prop_oneof![
            3 => prop::sample::select(vec!["rust", "fast", "video", "editor", "cuts", "breath",
                "signal", "frames", "spring", "camera", "really", "works"]).prop_map(str::to_owned),
            2 => prop::sample::select(WEAK_WORDS.to_vec()).prop_map(str::to_owned),
            1 => prop::sample::select(vec!["done.", "wait,", "yes!", "why?"]).prop_map(str::to_owned),
        ]
    }

    proptest! {
        #[test]
        fn chunking_invariants(
            ws in prop::collection::vec((arb_word(), 50i64..400_000, 50_000i64..400_000), 1..80),
            min_w in 1u8..5, extra in 0u8..4,
        ) {
            let g = CaptionGrouping { min_words: min_w, max_words: min_w + extra, max_chars: 64, pause_break_ms: 300 };
            let mut t = 0;
            let words: Vec<TimedWord> = ws.iter().map(|(text, gap, dur)| {
                let start = t + gap;
                t = start + dur;
                TimedWord { text: text.clone(), start_us: start, end_us: start + dur }
            }).collect();
            let blocks = chunk(&words, &[], &g, 250_000);

            // 1. Every word exactly once, in order.
            let flat: Vec<&TimedWord> = blocks.iter().flat_map(|b| &b.words).collect();
            prop_assert_eq!(flat.len(), words.len());
            for (a, b) in flat.iter().zip(&words) { prop_assert_eq!(*a, b); }

            // 2. Size bound and monotonic, non-overlapping display.
            for b in &blocks {
                prop_assert!(!b.words.is_empty() && b.words.len() <= usize::from(g.max_words));
                prop_assert!(b.end_us > b.start_us);
            }
            for w in blocks.windows(2) {
                prop_assert!(w[0].end_us <= w[1].start_us);
            }

            // 3. If every window of max_words words in a phrase contains a
            //    content word, no non-final block of that phrase ends weak.
            let max_w = usize::from(g.max_words);
            let feasible = |phrase: &[TimedWord]| phrase.windows(max_w.min(phrase.len()))
                .all(|win| win.iter().any(|w| !is_weak(&w.text)));
            let mut i = 0;
            while i < blocks.len() {
                // Collect the blocks of one phrase (phrase ends at a hard boundary).
                let mut j = i;
                loop {
                    let last = blocks[j].words.last().unwrap();
                    let next = blocks.get(j + 1).map(|b| &b.words[0]);
                    let hard = match next {
                        None => true,
                        Some(n) => n.start_us - last.end_us >= 300_000 || ends_sentence(&last.text),
                    };
                    if hard { break; }
                    j += 1;
                }
                let phrase: Vec<TimedWord> = blocks[i..=j].iter().flat_map(|b| b.words.clone()).collect();
                if feasible(&phrase) {
                    for b in &blocks[i..j] {
                        prop_assert!(!is_weak(&b.words.last().unwrap().text),
                            "weak ending in {:?}", texts(&blocks[i..=j]));
                    }
                }
                i = j + 1;
            }
        }
    }
}
