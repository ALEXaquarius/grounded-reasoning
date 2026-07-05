"""
Offline lock for drift_conformal_eval.py and AdaptiveConformalReasoner:
Adaptive Conformal Inference (ACI) vs. a frozen split-conformal threshold
under a noise level that shifts partway through a stream. Fully synthetic,
no LLM/network needed.
"""
from grounded_reasoning import AdaptiveConformalReasoner, ConformalReasoner
from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine
from grounded_reasoning.experiments.drift_conformal_eval import run


def _averaged(n_trials=10):
    keys = ["static_pre_shift", "static_post_shift", "adaptive_pre_shift", "adaptive_post_shift"]
    agg = {k: [] for k in keys}
    for seed in range(n_trials):
        res = run(run_seed=seed)
        for k in keys:
            agg[k].append(res[k])
    return {k: sum(v) / len(v) for k, v in agg.items()}


def test_static_threshold_collapses_after_the_shift():
    # averaged over several seeds -- a single trial's per-batch (n=30) coverage
    # has real sampling noise, same reasoning as the aggregate-rate checks
    # used throughout the sibling calibration-eval tests
    means = _averaged()
    assert means["static_pre_shift"] >= 0.85
    assert means["static_post_shift"] < 0.7, "expected the frozen threshold to badly lose coverage"


def test_adaptive_threshold_recovers_after_the_shift():
    means = _averaged()
    assert means["adaptive_pre_shift"] >= 0.85
    assert means["adaptive_post_shift"] >= 0.85, "ACI should track the target coverage after the shift too"


def test_adaptive_beats_static_post_shift_across_many_seeds():
    n_trials = 15
    wins = 0
    for seed in range(n_trials):
        res = run(run_seed=seed)
        if res["adaptive_post_shift"] > res["static_post_shift"]:
            wins += 1
    assert wins == n_trials, f"adaptive should beat static post-shift in every trial, won {wins}/{n_trials}"


def test_adaptive_conformal_reasoner_basic_api():
    eng = FuzzyInferenceEngine(walk_len=6, alpha=0.7)
    eng.add_relation("a", "b")
    eng.add_relation("b", "c")
    acr = AdaptiveConformalReasoner(eng, alpha=0.1, init_scores=[0.1, 0.2, 0.3])
    assert acr.empirical_coverage is None  # no updates yet
    covered = acr.update("a", "c")
    assert covered in (True, False)
    assert acr.n_updates == 1
    assert acr.empirical_coverage == (1.0 if covered else 0.0)


def test_backward_compatible_conformal_reasoner_unaffected():
    # ConformalReasoner (the existing, non-adaptive class) must be completely
    # unaffected by adding AdaptiveConformalReasoner alongside it
    eng = FuzzyInferenceEngine(walk_len=6, alpha=0.7)
    eng.add_relation("a", "b")
    cr = ConformalReasoner(eng, alpha=0.1)
    tau = cr.calibrate([("a", "b")])
    assert tau == cr.tau
    assert cr.accept("a", "b")
