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
