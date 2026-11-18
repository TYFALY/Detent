//! Keeps the documented example EDL in sync with the schema.

use detent_core::SemanticEdl;
use detent_core::testkit::synthetic_index;

#[test]
fn example_edl_parses_and_solves() {
    let edl = SemanticEdl::from_json(include_str!("../../../examples/edl.json")).unwrap();
    assert_eq!(edl.clips.len(), 3);
    let idx = synthetic_index(600_000, 1);
    let res = detent_core::EdlSolver::default().solve(&edl, &idx).unwrap();
    assert!(res.timeline.verify().is_empty());
}

#[test]
fn shortform_example_parses_and_solves() {
    let edl = SemanticEdl::from_json(include_str!("../../../examples/shortform.json")).unwrap();
    assert!(edl.tighten.is_some() && edl.output.opening_pop);
    let idx = synthetic_index(600_000, 1);
    let res = detent_core::EdlSolver::default().solve(&edl, &idx).unwrap();
    let t = &res.timeline;
    assert!(t.verify().is_empty());
    assert_eq!(
        t.captions.as_ref().unwrap().headline_end_us,
        Some(3_000_000)
    );
    assert_eq!(t.splices.iter().filter(|s| s.flash).count(), 1);
}

#[test]
fn multisource_example_parses_and_solves() {
    let edl = SemanticEdl::from_json(include_str!("../../../examples/multisource.json")).unwrap();
    let idx: Vec<_> = (1..=3).map(|seed| synthetic_index(30_000, seed)).collect();
    let res = detent_core::EdlSolver::default()
        .solve_sources(&edl, &idx)
        .unwrap();
    let t = &res.timeline;
    assert!(t.verify().is_empty());
    assert_eq!(t.extra_sources.len(), 2);
    assert_eq!(t.texts.len(), 3);
    assert_eq!(t.segments.last().unwrap().speed, 0.5);
}

#[test]
fn campaign_example_parses_and_checks() {
    use detent_core::campaign::{CampaignRules, CheckInput, Status, check};
    let rules = CampaignRules::from_json(include_str!("../../../examples/campaign.json")).unwrap();
    let r = check(
        &rules,
        &CheckInput {
            caption: Some("Acme just showed off the Widget @acme #ad"),
            ..CheckInput::default()
        },
    );
    assert!(r.ok);
    assert!(r.checks.iter().any(|c| c.status == Status::Skipped));
    assert_eq!(r.reminders.len(), 3);
}

#[test]
fn demo_example_parses_and_solves() {
    let edl = SemanticEdl::from_json(include_str!("../../../examples/demo.json")).unwrap();
    let idx = synthetic_index(30_000, 7);
    let res = detent_core::EdlSolver::default().solve(&edl, &idx).unwrap();
    assert!(res.timeline.verify().is_empty());
    assert_eq!(res.timeline.texts.len(), 3);
}
