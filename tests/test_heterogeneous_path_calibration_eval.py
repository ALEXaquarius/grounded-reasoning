"""
Offline lock for heterogeneous_path_calibration_eval.py: fully synthetic,
no LLM/network needed.
"""
from grounded_reasoning.experiments.heterogeneous_path_calibration_eval import run


def test_verify_path_always_finds_every_constructed_chain():
    # Theorem G's guarantee is exact, not statistical -- this must hold every time
    for seed in range(15):
        assert run(seed=seed)["path_verification_matches_construction"]


def test_calibrated_bound_holds_at_the_stated_confidence_rate():
    # Compared against the TRUE q (known here -- synthetic ground truth), which
    # is what Clopper-Pearson actually guarantees. A noisier, more realistic
    # comparison against a finite held-out test SAMPLE (as a real deployment
    # would have to do, without knowing the true q) is illustrated in main(),
    # not asserted here, since it compounds two sources of sampling error and
    # isn't what the theorem's coverage statement is actually about.
    trials = [run(seed=seed)["bound_le_true_q"] for seed in range(80)]
    coverage = sum(trials) / len(trials)
    assert coverage >= 0.9 - 0.08, f"coverage={coverage:.3f}, expected >= ~0.90"
