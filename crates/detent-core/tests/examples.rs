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
