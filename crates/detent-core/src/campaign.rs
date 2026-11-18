//! Campaign rule checks: catch a rejection before posting.
//!
//! Paid clipping campaigns publish rules ("caption must be one of these
//! lines", "no music", "official footage only", "9:16 under 90 s"). A
//! [`CampaignRules`] file encodes them once; [`check`] tests a solved
//! timeline, the rendered file's facts and the post caption against them and
//! returns every rule with pass / fail / skipped, plus the reminders that
//! only a human can do on the platform (a paid-partnership toggle, the
//! submission window).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::timeline::ResolvedTimeline;

/// One campaign's rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct CampaignRules {
    /// Campaign name (shown in the report).
    pub name: String,
    /// Post caption rules.
    #[serde(default)]
    pub caption: CaptionRules,
    /// Rules on the edit and the rendered file.
    #[serde(default)]
    pub video: VideoRules,
    /// Words that may appear neither in the caption nor on screen (headline,
    /// texts, chapters, burned-in captions). Case-insensitive substrings.
    #[serde(default)]
    pub forbid_words: Vec<String>,
    /// Things only a person can check or do (paid-partnership label, submit
    /// within 30 minutes of posting, keep the post up 30 days...).
    #[serde(default)]
    pub reminders: Vec<String>,
}

/// Caption rules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct CaptionRules {
    /// The caption must be exactly one of these lines (whitespace runs are
    /// collapsed before comparing; everything else is word for word).
    #[serde(default)]
    pub exact_one_of: Vec<String>,
    /// Phrases the caption must contain (case-insensitive).
    #[serde(default)]
    pub must_include: Vec<String>,
    /// When set, every `#hashtag` must be in this list (case-insensitive).
    #[serde(default)]
    pub allowed_hashtags: Option<Vec<String>>,
    /// When set, every `@mention` must be in this list (case-insensitive).
    #[serde(default)]
    pub allowed_mentions: Option<Vec<String>>,
    /// Most hashtags allowed.
    #[serde(default)]
    pub max_hashtags: Option<u32>,
    /// No links in the caption.
    #[serde(default)]
    pub no_links: bool,
    /// Longest caption allowed (characters).
    #[serde(default)]
    pub max_chars: Option<u32>,
}

/// Rules on the edit and the rendered file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct VideoRules {
    /// Shortest allowed video (s).
    #[serde(default)]
    pub min_seconds: Option<f64>,
    /// Longest allowed video (s).
    #[serde(default)]
    pub max_seconds: Option<f64>,
    /// Must be 9:16 vertical.
    #[serde(default)]
    pub vertical: bool,
    /// Largest file allowed (MB, 1 MB = 1,000,000 bytes).
    #[serde(default)]
    pub max_size_mb: Option<f64>,
    /// No added music bed (original audio only).
    #[serde(default)]
    pub no_music: bool,
    /// No added sound effects (whooshes, pops).
    #[serde(default)]
    pub no_sound_effects: bool,
    /// Every source clip and insert must come from this folder (the
    /// campaign's official footage).
    #[serde(default)]
    pub footage_dir: Option<String>,
    /// Some on-screen text (headline or a text overlay) must be visible by
    /// this many seconds into the video (a hook in the first 2 s).
    #[serde(default)]
    pub hook_text_within_s: Option<f64>,
}

/// Facts about the rendered file (from a probe).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VideoFacts {
    /// File path.
    pub path: String,
    /// Size in bytes.
    pub bytes: u64,
    /// Duration (ms).
    pub duration_ms: u64,
    /// Picture width.
    pub width: u32,
    /// Picture height.
    pub height: u32,
}

/// What is being checked. Missing parts skip the rules that need them.
#[derive(Debug, Clone, Copy, Default)]
pub struct CheckInput<'a> {
    /// The solved edit.
    pub timeline: Option<&'a ResolvedTimeline>,
    /// The rendered file.
    pub video: Option<&'a VideoFacts>,
    /// The caption that will be posted.
    pub caption: Option<&'a str>,
}

/// Outcome of one rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// The rule holds.
    Pass,
    /// The rule is broken: fix before posting.
    Fail,
    /// Not checked (the needed input was not given).
    Skipped,
}

/// One checked rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RuleCheck {
    /// Stable rule id, e.g. `caption.exact_one_of`.
    pub rule: String,
    /// Outcome.
    pub status: Status,
    /// What was found.
    pub detail: String,
}

/// The full report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CampaignReport {
    /// Campaign name.
    pub campaign: String,
    /// True when no rule failed.
    pub ok: bool,
    /// Every rule that applies.
    pub checks: Vec<RuleCheck>,
    /// Manual steps to do on the platform.
    pub reminders: Vec<String>,
}

impl CampaignRules {
    /// Parses a rules file. Unknown fields are a hard error.
    pub fn from_json(s: &str) -> crate::Result<Self> {
        serde_json::from_str(s).map_err(|source| crate::CoreError::Json {
            what: "CampaignRules",
            source,
        })
    }
}

fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Tags starting with `sigil` (`#ad`, `@Ring`), without trailing dots;
/// `#a#b` is two tags.
fn tagged(caption: &str, sigil: char) -> Vec<String> {
    let chars: Vec<char> = caption.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != sigil {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < chars.len() && (chars[j].is_alphanumeric() || matches!(chars[j], '_' | '.')) {
            j += 1;
        }
        let tag: String = chars[i..j].iter().collect();
        let tag = tag.trim_end_matches('.');
        if tag.chars().count() > 1 {
            out.push(tag.to_owned());
        }
        i = j.max(i + 1);
    }
    out
}

/// Normalises a path for prefix comparison.
fn norm_path(p: &str) -> String {
    let s = p.replace('\\', "/");
    let s = s.strip_prefix("//?/").unwrap_or(&s).to_owned();
    if cfg!(windows) { s.to_lowercase() } else { s }
}

fn inside(path: &str, dir: &str) -> bool {
    let (p, d) = (norm_path(path), norm_path(dir));
    let d = d.trim_end_matches('/');
    p.starts_with(&format!("{d}/"))
}

/// Everything burned into the picture as text.
fn on_screen_text(t: &ResolvedTimeline) -> Vec<String> {
    let mut v: Vec<String> = t.texts.iter().map(|x| x.text.clone()).collect();
    if let Some(c) = &t.captions {
        v.extend(c.headline.clone());
        v.extend(c.chapters.iter().map(|c| c.text.clone()));
        v.extend(c.blocks.iter().map(|b| {
            b.words
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        }));
    }
    v
}

/// Checks `input` against `rules`.
pub fn check(rules: &CampaignRules, input: &CheckInput<'_>) -> CampaignReport {
    let mut checks = Vec::new();
    let mut push = |rule: &str, status: Status, detail: String| {
        checks.push(RuleCheck {
            rule: rule.to_owned(),
            status,
            detail,
        });
    };
    let pass_fail = |ok: bool| if ok { Status::Pass } else { Status::Fail };
    let skip = "no caption given (pass --caption)".to_owned();
    let c = &rules.caption;

    // Caption.
    if !c.exact_one_of.is_empty() {
        match input.caption {
            Some(cap) => {
                let got = collapse(cap);
                let ok = c.exact_one_of.iter().any(|l| collapse(l) == got);
                push(
                    "caption.exact_one_of",
                    pass_fail(ok),
                    if ok {
                        "caption is an approved line".into()
                    } else {
                        format!(
                            "caption {got:?} is not one of the {} approved lines, word for word: {}",
                            c.exact_one_of.len(),
                            c.exact_one_of
                                .iter()
                                .map(|l| format!("{l:?}"))
                                .collect::<Vec<_>>()
                                .join(" | ")
                        )
                    },
                );
            }
            None => push("caption.exact_one_of", Status::Skipped, skip.clone()),
        }
    }
    for phrase in &c.must_include {
        let rule = "caption.must_include";
        match input.caption {
            Some(cap) => {
                let ok = cap.to_lowercase().contains(&phrase.to_lowercase());
                push(
                    rule,
                    pass_fail(ok),
                    if ok {
                        format!("contains {phrase:?}")
                    } else {
                        format!("missing {phrase:?}")
                    },
                );
            }
            None => push(rule, Status::Skipped, skip.clone()),
        }
    }
    for (rule, sigil, allowed) in [
        ("caption.allowed_hashtags", '#', &c.allowed_hashtags),
        ("caption.allowed_mentions", '@', &c.allowed_mentions),
    ] {
        let Some(allowed) = allowed else { continue };
        match input.caption {
            Some(cap) => {
                let extra: Vec<String> = tagged(cap, sigil)
                    .into_iter()
                    .filter(|t| !allowed.iter().any(|a| a.eq_ignore_ascii_case(t)))
                    .collect();
                push(
                    rule,
                    pass_fail(extra.is_empty()),
                    if extra.is_empty() {
                        "no tags outside the allowed list".into()
                    } else {
                        format!(
                            "not allowed: {} (allowed: {})",
                            extra.join(" "),
                            allowed.join(" ")
                        )
                    },
                );
            }
            None => push(rule, Status::Skipped, skip.clone()),
        }
    }
    if let Some(max) = c.max_hashtags {
        match input.caption {
            Some(cap) => {
                let n = tagged(cap, '#').len();
                push(
                    "caption.max_hashtags",
                    pass_fail(n <= max as usize),
                    format!("{n} hashtag(s), at most {max}"),
                );
            }
            None => push("caption.max_hashtags", Status::Skipped, skip.clone()),
        }
    }
    if c.no_links {
        match input.caption {
            Some(cap) => {
                let l = cap.to_lowercase();
                let link = [
                    "http://",
                    "https://",
                    "www.",
                    ".com/",
                    "bit.ly",
                    "linktr.ee",
                ]
                .iter()
                .find(|k| l.contains(*k));
                push(
                    "caption.no_links",
                    pass_fail(link.is_none()),
                    link.map_or("no links".into(), |k| format!("contains a link ({k})")),
                );
            }
            None => push("caption.no_links", Status::Skipped, skip.clone()),
        }
    }
    if let Some(max) = c.max_chars {
        match input.caption {
            Some(cap) => {
                let n = cap.chars().count();
                push(
                    "caption.max_chars",
                    pass_fail(n <= max as usize),
                    format!("{n} characters, at most {max}"),
                );
            }
            None => push("caption.max_chars", Status::Skipped, skip.clone()),
        }
    }

    // Forbidden words in the caption and on screen.
    if !rules.forbid_words.is_empty() {
        let mut found = Vec::new();
        let mut looked = Vec::new();
        if let Some(cap) = input.caption {
            looked.push("caption");
            let l = cap.to_lowercase();
            found.extend(
                rules
                    .forbid_words
                    .iter()
                    .filter(|w| l.contains(&w.to_lowercase()))
                    .map(|w| format!("{w:?} in the caption")),
            );
        }
        if let Some(t) = input.timeline {
            looked.push("on-screen text");
            for text in on_screen_text(t) {
                let l = text.to_lowercase();
                found.extend(
                    rules
                        .forbid_words
                        .iter()
                        .filter(|w| l.contains(&w.to_lowercase()))
                        .map(|w| format!("{w:?} on screen ({text:?})")),
                );
            }
        }
        if looked.is_empty() {
            push(
                "forbid_words",
                Status::Skipped,
                "no caption or timeline given".into(),
            );
        } else {
            push(
                "forbid_words",
                pass_fail(found.is_empty()),
                if found.is_empty() {
                    format!(
                        "none of {} found in the {}",
                        rules.forbid_words.join(", "),
                        looked.join(" or ")
                    )
                } else {
                    found.join("; ")
                },
            );
        }
    }

    // The edit and the file.
    let v = &rules.video;
    let duration_s = input
        .video
        .map(|f| f.duration_ms as f64 / 1e3)
        .or_else(|| input.timeline.map(|t| t.duration_us as f64 / 1e6));
    for (rule, limit, longer) in [
        ("video.min_seconds", v.min_seconds, false),
        ("video.max_seconds", v.max_seconds, true),
    ] {
        let Some(limit) = limit else { continue };
        match duration_s {
            Some(d) => push(
                rule,
                pass_fail(if longer { d <= limit } else { d >= limit }),
                format!(
                    "{d:.1} s, {} {limit} s",
                    if longer { "at most" } else { "at least" }
                ),
            ),
            None => push(rule, Status::Skipped, "no timeline or video given".into()),
        }
    }
    if v.vertical {
        let dims = input
            .video
            .map(|f| (f.width, f.height))
            .or_else(|| input.timeline.map(|t| (t.output.width, t.output.height)));
        match dims {
            Some((w, h)) => {
                let r = f64::from(w) / f64::from(h.max(1));
                push(
                    "video.vertical",
                    pass_fail((r - 9.0 / 16.0).abs() < 0.01),
                    format!("{w}x{h}"),
                );
            }
            None => push(
                "video.vertical",
                Status::Skipped,
                "no timeline or video given".into(),
            ),
        }
    }
    if let Some(mb) = v.max_size_mb {
        match input.video {
            Some(f) => {
                let max = (mb * 1e6) as u64;
                push(
                    "video.max_size_mb",
                    pass_fail(f.bytes <= max),
                    format!(
                        "{:.2} MB, at most {mb} MB{}",
                        f.bytes as f64 / 1e6,
                        if f.bytes > max {
                            format!(" (re-render with --max-size-mb {mb})")
                        } else {
                            String::new()
                        }
                    ),
                );
            }
            None => push(
                "video.max_size_mb",
                Status::Skipped,
                "no rendered video given (pass --video)".into(),
            ),
        }
    }
    let need_t = "no timeline given (pass --timeline)".to_owned();
    if v.no_music {
        match input.timeline {
            Some(t) => {
                let m = t.sound.as_ref().and_then(|s| s.music.as_ref());
                push(
                    "video.no_music",
                    pass_fail(m.is_none()),
                    m.map_or("no music bed".into(), |m| format!("music bed {:?}", m.path)),
                );
            }
            None => push("video.no_music", Status::Skipped, need_t.clone()),
        }
    }
    if v.no_sound_effects {
        match input.timeline {
            Some(t) => {
                let n = t.sound.as_ref().map_or(0, |s| s.sfx.len());
                push(
                    "video.no_sound_effects",
                    pass_fail(n == 0),
                    format!("{n} added sound effect(s)"),
                );
            }
            None => push("video.no_sound_effects", Status::Skipped, need_t.clone()),
        }
    }
    if let Some(dir) = &v.footage_dir {
        match input.timeline {
            Some(t) => {
                let mut used: Vec<&str> = t
                    .segments
                    .iter()
                    .map(|s| t.source_of(s).path.as_str())
                    .collect();
                used.extend(t.inserts.iter().map(|i| i.path.as_str()));
                used.sort_unstable();
                used.dedup();
                let outside: Vec<&str> = used.into_iter().filter(|p| !inside(p, dir)).collect();
                push(
                    "video.footage_dir",
                    pass_fail(outside.is_empty()),
                    if outside.is_empty() {
                        format!("all footage is from {dir}")
                    } else {
                        format!("not from {dir}: {}", outside.join(", "))
                    },
                );
            }
            None => push("video.footage_dir", Status::Skipped, need_t.clone()),
        }
    }
    if let Some(s) = v.hook_text_within_s {
        match input.timeline {
            Some(t) => {
                let limit = (s * 1e6) as i64;
                let mut first: Option<i64> = t.texts.iter().map(|x| x.start_us).min();
                if t.captions.as_ref().is_some_and(|c| c.headline.is_some()) {
                    first = Some(0);
                }
                push(
                    "video.hook_text_within_s",
                    pass_fail(first.is_some_and(|f| f <= limit)),
                    first.map_or("no headline or on-screen text".into(), |f| {
                        format!("first text at {:.1} s, needed by {s} s", f as f64 / 1e6)
                    }),
                );
            }
            None => push("video.hook_text_within_s", Status::Skipped, need_t.clone()),
        }
    }
    let ok = checks.iter().all(|c| c.status != Status::Fail);
    CampaignReport {
        campaign: rules.name.clone(),
        ok,
        checks,
        reminders: rules.reminders.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> CampaignRules {
        CampaignRules::from_json(
            r##"{
            "name": "Demo",
            "caption": {
                "exact_one_of": ["first look at the Widget @Acme #ad", "Acme just showed off the Widget @Acme #ad"],
                "allowed_hashtags": ["#ad"],
                "allowed_mentions": ["@Acme"],
                "no_links": true
            },
            "video": {"max_seconds": 60, "vertical": true, "max_size_mb": 10, "no_music": true},
            "forbid_words": ["price"],
            "reminders": ["Turn on the paid partnership label"]
        }"##,
        )
        .unwrap()
    }

    #[test]
    fn approved_caption_passes_word_for_word() {
        let r = check(
            &rules(),
            &CheckInput {
                caption: Some("Acme just showed off   the Widget @Acme #ad"),
                ..CheckInput::default()
            },
        );
        assert!(r.ok, "{r:#?}");
        assert_eq!(r.reminders.len(), 1);
        let s = |id: &str| r.checks.iter().find(|c| c.rule == id).unwrap().status;
        assert_eq!(s("caption.exact_one_of"), Status::Pass);
        assert_eq!(s("video.no_music"), Status::Skipped);
    }

    #[test]
    fn extra_hashtags_mentions_links_and_words_fail() {
        let r = check(
            &rules(),
            &CheckInput {
                caption: Some(
                    "acme just showed off the Widget @Acme @Other #ad #fyp price www.x.io",
                ),
                ..CheckInput::default()
            },
        );
        assert!(!r.ok);
        let failed: Vec<&str> = r
            .checks
            .iter()
            .filter(|c| c.status == Status::Fail)
            .map(|c| c.rule.as_str())
            .collect();
        for id in [
            "caption.exact_one_of",
            "caption.allowed_hashtags",
            "caption.allowed_mentions",
            "caption.no_links",
            "forbid_words",
        ] {
            assert!(failed.contains(&id), "{id} should fail: {r:#?}");
        }
    }

    #[test]
    fn file_facts_are_checked() {
        let f = VideoFacts {
            path: "a.mp4".into(),
            bytes: 12_500_000,
            duration_ms: 75_000,
            width: 1080,
            height: 1350,
        };
        let r = check(
            &rules(),
            &CheckInput {
                video: Some(&f),
                ..CheckInput::default()
            },
        );
        let s = |id: &str| r.checks.iter().find(|c| c.rule == id).unwrap().status;
        assert_eq!(s("video.max_size_mb"), Status::Fail);
        assert_eq!(s("video.max_seconds"), Status::Fail);
        assert_eq!(s("video.vertical"), Status::Fail);
    }

    #[test]
    fn tags_and_paths() {
        assert_eq!(tagged("hi #ad,#Fyp @Ring. ok", '#'), ["#ad", "#Fyp"]);
        assert_eq!(tagged("hi #ad,#Fyp @Ring. ok", '@'), ["@Ring"]);
        assert!(inside("C:\\clips\\src\\a.mp4", "C:/clips/src/"));
        assert!(!inside("/clips/srcx/a.mp4", "/clips/src"));
    }
}
