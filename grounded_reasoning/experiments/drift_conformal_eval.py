"""
Adaptive Conformal Inference (ACI, Gibbs & Candès 2021) vs. a frozen
split-conformal threshold, under a DRIFTING noise level — a genuinely
different problem than the Mondrian/redundancy grouping in
redundancy_conformal_eval.py, which only addresses HETEROGENEITY across a
static population, not DRIFT over time.

Both `ConformalReasoner` (Theorem K) and its Mondrian extension assume the
calibration and test data are exchangeable (drawn from the same
distribution). That is a real assumption a deployed system can silently
violate: an LLM-extraction pipeline processing many document batches over
time will not have IDENTICAL extraction quality in every batch -- later
documents may be cleaner, noisier, in a different domain, extracted by a
different model version, etc. A threshold calibrated once on an early batch
and then frozen has NO guarantee once the noise level shifts.

`AdaptiveConformalReasoner` instead updates its threshold from a STREAM of
confirmed-true examples using the classical ACI rule, with no stationarity
assumption needed for its OWN guarantee (the long-run average miscoverage
rate tracks alpha for any score sequence, including adversarial drift).

Scenario: a stream of query batches over a random relation graph, extracted
cleanly (p_drop=0.05) for the first half of the stream, then noisily
(p_drop=0.45) for the second half (simulating a shift in document/extraction
quality partway through). Measures per-batch coverage for both a STATIC
threshold (calibrated once on the very first batch, then frozen) and the
ADAPTIVE threshold (updated after every batch).

Run: python -m grounded_reasoning.experiments.drift_conformal_eval
(fully offline -- synthetic ground truth, no LLM call.)
"""
from __future__ import annotations

import random

from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine
from grounded_reasoning.reasoning.conformal_reasoning import AdaptiveConformalReasoner, conformal_threshold


def build(seed: int, p_drop: float, n: int = 45):
    """Same random-DAG construction as theorem_conformal_reasoning (integer
    node labels -- hash-seed-independent)."""
    rng = random.Random(seed)
    te = set()
    for j in range(1, n):
        for _ in range(rng.randint(1, 2)):
            te.add((rng.randint(0, j - 1), j))
    adj: dict[int, set[int]] = {}
    for a, b in te:
        adj.setdefault(a, set()).add(b)

    def closure(x: int) -> set[int]:
        seen: set[int] = set()
        frontier = list(adj.get(x, ()))
        while frontier:
            nf = []
            for u in frontier:
                if u not in seen:
                    seen.add(u)
                    nf.extend(adj.get(u, ()))
            frontier = nf
        return seen

    truth = {x: closure(x) for x in range(n)}
    eng = FuzzyInferenceEngine(walk_len=12, alpha=0.7)
    for a, b in sorted(te):  # sorted: set iteration order is hash-seed-dependent
        if rng.random() > p_drop:
            eng.add_relation(a, b)
    return n, truth, eng


def true_pair_scores(seed: int, p_drop: float, jit_seed: int, n: int = 45) -> list[float]:
    """A fresh random world at the given noise level; scores for every truly-related pair."""
    n, truth, eng = build(seed, p_drop, n)
    infc = {x: eng.infer(x) for x in range(n)}
    rng = random.Random(jit_seed)
    scores = [infc[x].get(b, 0.0) + rng.uniform(0, 1e-9) for x in range(n) for b in truth[x]]
    rng.shuffle(scores)
    return scores


def run(
    run_seed: int = 0, alpha: float = 0.1, n_batches: int = 40, batch_size: int = 30,
    gamma: float = 0.05, shift_at: float = 0.5,
) -> dict:
    def p_drop_at(i: int) -> float:
        return 0.05 if i < n_batches * shift_at else 0.45

    base = run_seed * 100_000
    init_scores = true_pair_scores(seed=base, p_drop=p_drop_at(0), jit_seed=base + 1)
    tau_static = conformal_threshold(init_scores[:batch_size], alpha)

    adaptive = AdaptiveConformalReasoner(
        engine=None, alpha=alpha, gamma=gamma, init_scores=init_scores[:batch_size],
    )

    static_cov, adaptive_cov = [], []
    for bidx in range(n_batches):
        batch = true_pair_scores(seed=base + bidx + 2, p_drop=p_drop_at(bidx), jit_seed=base + bidx + 9000)
        batch = batch[:batch_size]
        static_cov.append(sum(1 for s in batch if s >= tau_static) / len(batch))

        flags = [adaptive.update_score(s) for s in batch]
        adaptive_cov.append(sum(flags) / len(flags))

    h = int(n_batches * shift_at)
    return {
        "static_pre_shift": sum(static_cov[:h]) / h,
        "static_post_shift": sum(static_cov[h:]) / (n_batches - h),
        "adaptive_pre_shift": sum(adaptive_cov[:h]) / h,
        "adaptive_post_shift": sum(adaptive_cov[h:]) / (n_batches - h),
    }


def main() -> None:
    alpha = 0.1
    n_trials = 15
    keys = ["static_pre_shift", "static_post_shift", "adaptive_pre_shift", "adaptive_post_shift"]
    agg = {k: [] for k in keys}
    for t in range(n_trials):
        res = run(run_seed=t, alpha=alpha)
        for k in keys:
            agg[k].append(res[k])
    means = {k: sum(v) / len(v) for k, v in agg.items()}

    print("=" * 78)
    print("Frozen split-conformal threshold vs. Adaptive Conformal Inference (ACI),")
    print("noise shifting from p_drop=0.05 to p_drop=0.45 partway through a stream")
    print(f"({n_trials} independent trials)")
    print("=" * 78)
    print(f"\nSTATIC:   pre-shift coverage={means['static_pre_shift']:.1%}   "
          f"post-shift coverage={means['static_post_shift']:.1%}  (target >= {1-alpha:.0%})")
    print(f"ADAPTIVE: pre-shift coverage={means['adaptive_pre_shift']:.1%}   "
          f"post-shift coverage={means['adaptive_post_shift']:.1%}  (target >= {1-alpha:.0%})")
    print(
        f"\n=> The frozen threshold's coverage collapses after the shift "
        f"({means['static_post_shift']:.0%}, well below target) because split-conformal "
        f"assumes calibration and test data share a distribution -- violated here. "
        f"ACI recovers to ~target coverage after the shift by continuously adapting, "
        f"with no stationarity assumption needed for its own guarantee."
    )


if __name__ == "__main__":
    main()
