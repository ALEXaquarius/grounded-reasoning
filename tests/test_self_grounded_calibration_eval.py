"""
Offline lock for self_grounded_calibration_eval.py: calibrate_transitivity
applied directly to SGDC's own self-verified output (facts sourced from the
LLM's own atomic assertions, no external KB) -- fully synthetic, no
LLM/network needed.
"""
from grounded_reasoning.experiments.self_grounded_calibration_eval import run


def test_bound_holds_on_default_scenario():
    res = run()
    b = res["B_sgdc_calibrated"]
    assert b["bound_le_true_precision"]
    assert b["bound_held_on_held_out_test_set"]
    assert 0.0 < b["precision_lower_bound"] < 1.0


def test_atomic_soundness_assumption_is_silently_wrong_when_violated():
    # "trusted blindly" assumes precision=1.0 (Theorem I's precondition); with
    # some wrong atomic facts, ACTUAL precision must be strictly below that,
    # and no risk number is reported for the gap -- exactly the blind spot
    # calibration exists to close.
    res = run(seed=1, p_wrong_atomic=0.15)
    a = res["A_sgdc_trusted_blindly"]
    assert a["assumed_precision"] == 1.0
    assert a["actual_precision"] < 1.0
    assert a["risk_reported"] is None


def test_bound_holds_at_the_stated_confidence_rate_not_every_single_time():
    # alpha=0.1 means the bound is EXPECTED to fail on ~10% of independent
    # draws by design -- assert the AGGREGATE rate across many trials is
    # consistent with alpha, same style as the sibling calibration evals.
    trials = [
        run(seed=seed, p_wrong_atomic=p)["B_sgdc_calibrated"]["bound_le_true_precision"]
        for seed in range(40)
        for p in (0.05, 0.15, 0.3)
    ]
    coverage = sum(trials) / len(trials)
    assert coverage >= 0.9 - 0.05, f"coverage={coverage:.3f}, expected >= ~0.90"


def test_zero_wrong_atomic_facts_gives_perfect_precision():
    # sanity check on the world-builder: with p_wrong_atomic=0, Theorem I's
    # actual precondition holds exactly, so SGDC's real precision must be 1.0
    res = run(seed=0, p_wrong_atomic=0.0)
    assert res["n_wrong_atomic_facts"] == 0
    assert res["A_sgdc_trusted_blindly"]["actual_precision"] == 1.0
