"""
Offline lock for the normalization_calibration_eval experiment (Theorem N's
A/B/C demo): fully synthetic, no LLM/network needed.
"""
from grounded_reasoning.experiments.normalization_calibration_eval import run


def test_no_normalization_never_produces_a_false_positive():
    # the Theorem G guarantee is untouched when normalize= isn't used at all
    for seed in range(10):
        assert run(seed=seed)["A_no_normalization"]["false_positives"] == 0


def test_calibrated_bound_holds_at_the_stated_confidence_rate():
    trials = [run(seed=seed)["C_fuzzy_calibrated"]["bound_held"] for seed in range(40)]
    coverage = sum(trials) / len(trials)
    assert coverage >= 0.9 - 0.07, f"coverage={coverage:.3f}, expected >= ~0.90"


def test_blind_fuzzy_trust_has_no_risk_estimate_the_calibrated_one_does():
    res = run(seed=0)
    assert res["B_fuzzy_trusted_blindly"]["risk_reported"] is None
    assert res["B_fuzzy_trusted_blindly"]["false_positives"] > 0  # the risk is real
    assert res["C_fuzzy_calibrated"]["merge_precision_lower_bound"] > 0.0
