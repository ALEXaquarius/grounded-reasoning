"""
Offline lock for the transitivity_calibration_eval experiment (Theorem M's
A/B demo): fully synthetic, no LLM/network needed.
"""
from grounded_reasoning.experiments.transitivity_calibration_eval import run


def test_calibrated_bound_holds_on_held_out_test_set():
    res = run(q=0.85, alpha=0.1, n_chains=200, seed=0)
    assert res["B_calibrated"]["held_on_held_out_test_set"]
    assert 0.0 < res["B_calibrated"]["precision_lower_bound"] < res["true_q"]


def test_bound_holds_at_the_stated_confidence_rate_not_every_single_time():
    # alpha=0.1 means the bound is EXPECTED to fail on ~10% of independent
    # draws by design (that's what "90% confidence" means) -- asserting every
    # single trial holds would be testing the wrong thing. Assert the
    # aggregate failure rate across many independent trials is consistent
    # with alpha instead, same style as theorem_conformal_reasoning's own check.
    trials = [
        run(q=q, alpha=0.1, n_chains=150, seed=seed)["B_calibrated"]["held_on_held_out_test_set"]
        for seed in range(40)
        for q in (0.6, 0.85, 0.99)
    ]
    coverage = sum(trials) / len(trials)
    assert coverage >= 0.9 - 0.05, f"coverage={coverage:.3f}, expected >= ~0.90"


def test_binary_alternative_has_no_risk_estimate_the_calibrated_one_does():
    res = run(q=0.85, seed=0)
    assert res["A_declared_silently_trusts_all"]["risk_reported"] is None
    assert res["A_undeclared_rejects_all"]["claims_kept"] == 0
    assert res["B_calibrated"]["precision_lower_bound"] > 0.0
