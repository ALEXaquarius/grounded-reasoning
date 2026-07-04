"""
OFFLINE test for the Conformal-on-LLM demo: simulates EXTRACTION NOISE (dropped
edges) in place of the LLM, locking down the invariant "coverage >= 1-alpha holds
EVEN WHEN the graph is noisy". No network calls.
"""
import random

from grounded_reasoning.experiments.conformal_llm_eval import build_ontology
from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine
from grounded_reasoning.reasoning.conformal_reasoning import conformal_threshold


def _coverage_under_drop(p_drop: float, alpha: float, seeds=range(12)) -> float:
    pos = []
    for k in seeds:
        words, gold, reach = build_ontology(seed=1000 + k)
        rng = random.Random(500 + k)
        eng = FuzzyInferenceEngine(walk_len=12, alpha=0.7)
        for a, b in gold:
            if rng.random() > p_drop:      # simulates the LLM dropping an edge (extraction noise)
                eng.add_relation(a, b)
        infc = {x: eng.infer(x) for x in words}
        for x in words:
            for y in words:
                if (x, y) in reach:
                    pos.append(infc[x].get(y, 0.0) + rng.uniform(0, 1e-9))
    rng = random.Random(0)
    rng.shuffle(pos)
    h = len(pos) // 2
    tau = conformal_threshold(pos[:h], alpha)
    return sum(1 for s in pos[h:] if s >= tau) / len(pos[h:])


def test_conformal_coverage_holds_under_extraction_noise():
    alpha = 0.1
    for p_drop in (0.0, 0.2, 0.4):
        cov = _coverage_under_drop(p_drop, alpha)
        # coverage >= 1-alpha (with a small tolerance for finite-sample discreteness) — HOLDS even with a noisy graph
        assert cov >= (1 - alpha) - 0.05, f"p_drop={p_drop} cov={cov}"
