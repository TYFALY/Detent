//! Acoustic re-timing of ASR word timestamps.
//!
//! whisper.cpp's per-word timestamps routinely stretch a word across the
//! silence that follows it (errors of 0.3-1 s are common), which would make
//! every pause look like speech to the solver. The acoustic pause detector,
//! by contrast, knows *where* the silences are but not *which* words they
//! separate. This module combines the two:
//!
//! 1. **Pause ↔ boundary matching.** Each long pause is matched to at most one
//!    word boundary (the gap between word `b-1` and word `b`, including the
//!    edges before the first and after the last word) by a monotonic dynamic
//!    program: cost = |boundary time − pause centre|, minus a bonus when word
//!    `b-1` ends with punctuation, or a fixed penalty to leave the pause
//!    unmatched (a long stop inside a word, music, a cough).
//! 2. **Island re-timing.** Matched pauses split the audio into speech
//!    islands. The whisper times of the words in each island are mapped
//!    affinely onto the island's acoustic extent, so word order and relative
//!    durations are kept while words no longer overlap matched pauses.

use crate::index::{SilenceInterval, WordToken};

/// Tunables for [`retime_words`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignConfig {
    /// Pauses shorter than this are ignored (ms).
    pub min_pause_ms: u64,
    /// Only boundaries within this distance of a pause are considered (ms).
    pub search_ms: u64,
    /// Cost reduction for a boundary after `. , ? ! ; :` (ms-equivalent).
    pub punctuation_bonus_ms: f64,
    /// Cost of leaving a pause unmatched (ms-equivalent).
    pub skip_penalty_ms: f64,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self {
            min_pause_ms: 200,
            search_ms: 2_500,
            punctuation_bonus_ms: 450.0,
            skip_penalty_ms: 900.0,
        }
    }
}

/// Matches pauses to boundaries. Returns `(pause_index, boundary)` pairs with
/// strictly increasing boundaries; boundary `b` lies before word `b`.
pub fn match_pauses(
    words: &[WordToken],
    pauses: &[&SilenceInterval],
    cfg: &AlignConfig,
) -> Vec<(usize, usize)> {
    let n = words.len();
    if n == 0 || pauses.is_empty() {
        return Vec::new();
    }
    let boundary_time = |b: usize| -> f64 {
        if b == 0 {
            words[0].start_ms as f64
        } else if b == n {
            words[n - 1].end_ms as f64
        } else {
            (words[b - 1].end_ms as f64 + words[b].start_ms as f64) / 2.0
        }
    };
    let punct = |b: usize| -> bool {
        b == 0
            || b == n
            || words[b - 1]
                .word
                .trim_end_matches(['"', '\'', ')', '”', '’'])
                .ends_with(['.', ',', '?', '!', ';', ':', '…'])
    };
    // Candidate boundaries per pause (sorted ascending).
    let cands: Vec<Vec<(usize, f64)>> = pauses
        .iter()
        .map(|p| {
            let centre = (p.start_ms + p.end_ms) as f64 / 2.0;
            let lo = centre - cfg.search_ms as f64;
            let hi = centre + cfg.search_ms as f64;
            let first = (0..=n).find(|&b| boundary_time(b) >= lo).unwrap_or(n + 1);
            (first..=n)
                .take_while(|&b| boundary_time(b) <= hi)
                .map(|b| {
                    let mut c = (boundary_time(b) - centre).abs();
                    if punct(b) {
                        c -= cfg.punctuation_bonus_ms;
                    }
                    (b, c)
                })
                .collect()
        })
        .collect();

    // DP state per pause: best total cost ending with this pause matched to
    // candidate k, or (index `len`) unmatched. `last` = highest boundary used.
    #[derive(Clone, Copy)]
    struct Cell {
        cost: f64,
        last: Option<usize>,
        prev: Option<usize>,
    }
    let mut table: Vec<Vec<Cell>> = Vec::with_capacity(pauses.len());
    for (i, cs) in cands.iter().enumerate() {
        let prev_row = if i == 0 { None } else { table.get(i - 1) };
        let start = Cell {
            cost: 0.0,
            last: None,
            prev: None,
        };
        let options: Vec<(usize, Cell)> = match prev_row {
            None => vec![(usize::MAX, start)],
            Some(row) => row.iter().copied().enumerate().collect(),
        };
        let mut row = Vec::with_capacity(cs.len() + 1);
        for &(b, c) in cs {
            let best = options
                .iter()
                .filter(|(_, cell)| cell.last.is_none_or(|l| l < b))
                .min_by(|x, y| x.1.cost.total_cmp(&y.1.cost));
            row.push(match best {
                Some(&(k, cell)) => Cell {
                    cost: cell.cost + c,
                    last: Some(b),
                    prev: (k != usize::MAX).then_some(k),
                },
                None => Cell {
                    cost: f64::INFINITY,
                    last: Some(b),
                    prev: None,
                },
            });
        }
        // Unmatched: carry the best previous state forward.
        let (k, cell) = options
            .iter()
            .copied()
            .min_by(|x, y| x.1.cost.total_cmp(&y.1.cost))
            .unwrap_or((usize::MAX, start));
        row.push(Cell {
            cost: cell.cost + cfg.skip_penalty_ms,
            last: cell.last,
            prev: (k != usize::MAX).then_some(k),
        });
        table.push(row);
    }
    // Backtrack.
    let mut out = Vec::new();
    let Some(last_row) = table.last() else {
        return out;
    };
    let mut k = last_row
        .iter()
        .enumerate()
        .min_by(|x, y| x.1.cost.total_cmp(&y.1.cost))
        .map(|(k, _)| k);
    for i in (0..table.len()).rev() {
        let Some(kk) = k else { break };
        let cell = table[i][kk];
        if kk < cands[i].len() {
            out.push((i, cands[i][kk].0));
        }
        k = cell.prev;
    }
    out.reverse();
    out
}

/// Re-times `words` so they do not overlap matched acoustic pauses.
pub fn retime_words(
    words: &[WordToken],
    pauses: &[SilenceInterval],
    duration_ms: u64,
    cfg: &AlignConfig,
) -> Vec<WordToken> {
    let long: Vec<&SilenceInterval> = pauses
        .iter()
        .filter(|p| p.end_ms.saturating_sub(p.start_ms) >= cfg.min_pause_ms)
        .collect();
    let matches = match_pauses(words, &long, cfg);
    if matches.is_empty() {
        return words.to_vec();
    }
    let n = words.len();
    // Island k spans words [w_lo, w_hi) and acoustic time [a_lo, a_hi].
    let mut cuts: Vec<(usize, u64, u64)> = matches
        .iter()
        .map(|&(p, b)| (b, long[p].start_ms, long[p].end_ms))
        .collect();
    cuts.sort_by_key(|c| c.0);
    let mut out = words.to_vec();
    let mut w_lo = 0;
    let mut a_lo: Option<u64> = None;
    for k in 0..=cuts.len() {
        let (w_hi, a_hi, next_lo) = match cuts.get(k) {
            Some(&(b, ps, pe)) => (b, Some(ps), Some(pe)),
            None => (n, None, None),
        };
        if w_hi > w_lo {
            let ws = &words[w_lo..w_hi];
            let w0 = ws[0].start_ms as f64;
            let w1 = ws[ws.len() - 1].end_ms as f64;
            let t0 = a_lo.map_or(w0.min(a_hi.map_or(w0, |h| h as f64)), |a| a as f64);
            let t1 = a_hi.map_or(w1.max(t0).min(duration_ms as f64), |a| a as f64);
            let (t0, t1) = (t0.min(t1), t1.max(t0));
            let scale = if w1 > w0 { (t1 - t0) / (w1 - w0) } else { 0.0 };
            let map = |t: u64| (t0 + (t as f64 - w0) * scale).round().clamp(t0, t1) as u64;
            for (o, w) in out[w_lo..w_hi].iter_mut().zip(ws) {
                o.start_ms = map(w.start_ms);
                o.end_ms = map(w.end_ms).max(o.start_ms);
            }
        }
        w_lo = w_hi;
        a_lo = next_lo;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::PauseKind;

    fn w(word: &str, s: u64, e: u64) -> WordToken {
        WordToken {
            word: word.into(),
            start_ms: s,
            end_ms: e,
            confidence: 0.9,
        }
    }

    fn pause(s: u64, e: u64) -> SilenceInterval {
        SilenceInterval {
            start_ms: s,
            end_ms: e,
            min_dbfs: -60.0,
            valley_ms: (s + e) / 2,
            kind: PauseKind::Silence,
        }
    }

    #[test]
    fn pulls_words_out_of_silences() {
        // Truth: "hello world." 0-1000, pause 1000-1800, "next one." 1800-2800.
        // Whisper smeared "world." across the pause and started late.
        let words = vec![
            w("hello", 0, 450),
            w("world.", 450, 1_700),
            w("next", 1_700, 2_300),
            w("one.", 2_300, 2_800),
        ];
        let out = retime_words(
            &words,
            &[pause(1_000, 1_800)],
            3_000,
            &AlignConfig::default(),
        );
        assert_eq!(out[1].end_ms, 1_000, "{out:?}");
        assert_eq!(out[2].start_ms, 1_800, "{out:?}");
        for x in &out {
            assert!(
                x.end_ms <= 1_000 || x.start_ms >= 1_800,
                "{x:?} overlaps the pause"
            );
        }
        // Order and monotonicity preserved.
        for p in out.windows(2) {
            assert!(p[0].start_ms <= p[1].start_ms && p[0].end_ms <= p[1].start_ms + 1);
        }
    }

    #[test]
    fn prefers_punctuation_and_stays_monotonic() {
        let words = vec![
            w("a", 0, 300),
            w("b.", 300, 900),
            w("c", 900, 1_200),
            w("d", 1_200, 1_500),
            w("e.", 1_500, 2_600),
            w("f", 2_600, 3_000),
        ];
        // Pause centres sit nearer the c|d boundary, but punctuation wins.
        let ps = [pause(950, 1_250), pause(2_200, 2_500)];
        let refs: Vec<&SilenceInterval> = ps.iter().collect();
        let m = match_pauses(&words, &refs, &AlignConfig::default());
        assert_eq!(m, vec![(0, 2), (1, 5)]);
    }

    #[test]
    fn leaves_far_pauses_unmatched() {
        let words = vec![w("only", 0, 400), w("words.", 400, 900)];
        let ps = [pause(20_000, 21_000)];
        let refs: Vec<&SilenceInterval> = ps.iter().collect();
        assert!(match_pauses(&words, &refs, &AlignConfig::default()).is_empty());
        let out = retime_words(&words, &ps, 30_000, &AlignConfig::default());
        assert_eq!(out, words);
    }

    #[test]
    fn leading_and_trailing_silence() {
        let words = vec![w("hi.", 0, 900), w("bye.", 900, 1_900)];
        let out = retime_words(
            &words,
            &[pause(0, 400), pause(1_500, 2_000)],
            2_000,
            &AlignConfig::default(),
        );
        assert!(out[0].start_ms >= 400, "{out:?}");
        assert!(out[1].end_ms <= 1_500, "{out:?}");
    }
}
