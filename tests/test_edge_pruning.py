"""
Offline tests for edge_pruning.py (identify_suspect_edges/prune_edges)
and edge_pruning_eval.py's A/B comparison. Fully synthetic, no
LLM/network needed.
"""
from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine
from grounded_reasoning.reasoning.edge_pruning import identify_suspect_edges, prune_edges
from grounded_reasoning.experiments.edge_pruning_eval import (
    run,
    run_mitigation_comparison,
    wilson_upper_bound,
)


def test_wilson_upper_bound_basic_properties():
    # zero observed successes still leaves a small but nonzero upper bound --
    # absence of evidence in a finite sample isn't proof of a zero true rate
    assert 0.0 < wilson_upper_bound(0, 100) < 0.05
    # a 95% upper bound must sit at or above the point estimate
    assert wilson_upper_bound(10, 100) > 10 / 100
    # more trials at the same observed rate -> tighter (smaller) upper bound
    assert wilson_upper_bound(100, 1000) < wilson_upper_bound(10, 100)
    assert wilson_upper_bound(0, 0) == 0.0


def test_blocks_a_spurious_edge_with_zero_true_support():
    edges = [("a", "b"), ("b", "c"), ("x", "y")]  # x->y is spurious, unrelated to a/b/c
    labeled = [("a", "c", True), ("x", "y", False)]
    blocked = identify_suspect_edges(edges, labeled)
    assert blocked == {("x", "y")}


def test_does_not_block_an_edge_with_any_true_support():
    # (a, b) appears on a TRUE claim's path AND happens to also be part of a
    # separately-labeled FALSE claim's path -- it must NOT be blocked, since
    # it has at least one true-claim vote (avoiding collateral damage to a
    # real edge)
    edges = [("a", "b"), ("b", "c"), ("b", "d")]
    labeled = [("a", "c", True), ("a", "d", False)]
    blocked = identify_suspect_edges(edges, labeled)
    assert ("a", "b") not in blocked
    assert ("b", "d") in blocked  # the actually-false edge, no true votes


def test_prune_edges_removes_exactly_the_blocked_set():
    edges = [("a", "b"), ("b", "c"), ("x", "y")]
    cleaned = prune_edges(edges, {("x", "y")})
    assert cleaned == [("a", "b"), ("b", "c")]


def test_no_false_paths_in_labeled_data_blocks_nothing():
    edges = [("a", "b"), ("b", "c")]
    labeled = [("a", "c", True)]
    assert identify_suspect_edges(edges, labeled) == set()


def test_pruned_graph_never_creates_a_fabricated_path():
    # sanity check: pruning can only REMOVE reachability, never invent it
    edges = [("a", "b"), ("b", "c"), ("x", "y")]
    labeled = [("x", "y", False)]
    blocked = identify_suspect_edges(edges, labeled)
    cleaned = prune_edges(edges, blocked)
    eng = FuzzyInferenceEngine(walk_len=8, alpha=0.6)
    for s, o in cleaned:
        eng.add_relation(s, o)
    assert eng.explain("a", "c") == ["a", "b", "c"]
    assert eng.explain("x", "y") is None


def test_cleaning_reduces_fpr_across_noise_regimes():
    # the aggregate empirical claim (not a per-trial guarantee): pruning
    # substantially reduces false-positive rate in every regime tested,
    # while coverage on the remaining graph stays close to target
    res = run(n_seeds=20)
    for label, m in res.items():
        assert m["cleaned_fpr"] < m["raw_fpr"] - 0.05, f"{label}: {m}"
        assert m["cleaned_coverage"] >= 0.9 - 0.05, f"{label}: {m}"


def test_larger_identify_split_and_min_evidence_cut_wrongly_blocked_rate():
    # the measured mitigation, checked in EVERY noise regime (not just one):
    # using a larger share of held-out data to identify suspect edges, plus
    # requiring a second corroborating false-claim encounter, substantially
    # reduces the rate at which a genuinely correct edge is wrongly removed
    # -- a real cost of the decision rule, not just a theoretical
    # possibility, and not eliminated by this mitigation, only reduced. The
    # upper-confidence-bound check is the precise, worst-case claim: even
    # accounting for sampling noise, the mitigated rate stays bounded.
    res = run_mitigation_comparison(n_seeds=40)
    for regime, configs in res.items():
        default = configs["default (identify_frac=0.5, min_evidence=1)"]
        safer = configs["safer (identify_frac=0.85, min_evidence=2)"]
        assert safer["pooled_wrongly_blocked_rate"] < default["pooled_wrongly_blocked_rate"], f"{regime}: {configs}"
        # worst measured regime (light spurious, fewest edges blocked) still
        # stays under a 95%-upper-confidence-bound on the wrongly-blocked rate
        assert safer["wrongly_blocked_upper_bound"] < 0.15, f"{regime}: {safer}"
        # the mitigation still cleans meaningfully, just less aggressively
        assert safer["cleaned_fpr"] < safer["raw_fpr"], f"{regime}: {safer}"
