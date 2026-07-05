"""
Offline lock for redundancy_conformal_eval.py: Mondrian (group-conditional)
conformal calibration by path-redundancy vs. a single global threshold.
Fully synthetic, no LLM/network needed.
"""
from grounded_reasoning.experiments.redundancy_conformal_eval import measure, run


def test_coverage_holds_in_both_scenarios():
    # the Mondrian guarantee (classical, not the empirical finding here): coverage
    # >= 1-alpha regardless of whether grouping happens to help efficiency
    res = run(alpha=0.1, n_seeds=20)
    for label, m in res.items():
        assert m["global_coverage"] >= 0.9 - 0.05, f"{label}: {m}"
        assert m["grouped_coverage"] >= 0.9 - 0.05, f"{label}: {m}"


def test_redundancy_grouping_improves_efficiency_under_dropout_dominant_noise():
    # the actual (falsifiable) empirical claim: grouping by path-redundancy
    # measurably lowers FPR when dropped edges are the dominant noise source
    res = run(alpha=0.1, n_seeds=20)
    m = res["dropout-dominant (p_drop=0.2, p_add=0.3)"]
    assert m["fpr_delta"] > 0.05, f"expected a clear FPR improvement, got {m}"


def test_redundancy_grouping_does_not_badly_break_efficiency_under_spurious_dominant_noise():
    # the honest limitation: no benefit (small cost tolerated) when the noise
    # doesn't interact with path-redundancy the way dropout does
    res = run(alpha=0.1, n_seeds=20)
    m = res["spurious-dominant (p_drop=0.0, p_add=0.3)"]
    assert m["fpr_delta"] > -0.1, f"grouping regressed efficiency too much here: {m}"


def test_single_trial_is_reproducible():
    r1 = measure(seed=0, alpha=0.1, p_drop=0.2, p_add=0.3)
    r2 = measure(seed=0, alpha=0.1, p_drop=0.2, p_add=0.3)
    assert r1 == r2
