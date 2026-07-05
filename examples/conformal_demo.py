"""
Demo: a distribution-free coverage guarantee that survives a noisy graph
(Theorem K).

The hard guard (GroundedReasoner.verify) needs a CLEAN graph -- every claim it
accepts is really true, but only because it's silent on anything the graph is
missing. If the graph itself is noisy (e.g. extracted by an LLM from raw
text, with edges dropped), the hard guard can't give any guarantee about
claims it silently can't see.

ConformalReasoner trades that hard precision for something that survives
noise: calibrate a threshold on operator confidence such that coverage
(the fraction of TRULY-related pairs whose score clears the threshold)
stays >= 1-alpha ON AVERAGE, REGARDLESS of how noisy the graph is -- a
distribution-free guarantee (Vovk et al.), at the cost of also letting
through some false positives (efficiency degrades with noise; validity
never does).

This expands on the quickstart notebook's one-cell version by also measuring
the false-positive rate on unrelated pairs -- the "price paid" half of the
tradeoff that a coverage number alone doesn't show -- and by comparing a
clean graph against a noisy one side by side.

Fully offline, synthetic ground truth and synthetic noise (no LLM call) --
see conformal_llm_eval.py for the same idea end-to-end on a REAL LLM's
extraction, and theory/theorems.py::theorem_conformal_reasoning for the
formal numerical verification this demo mirrors.

Run: python examples/conformal_demo.py
"""
from __future__ import annotations

import random

from grounded_reasoning import ConformalReasoner
from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine


def build_world(seed: int, p_drop: float, n: int = 45):
    """A random DAG (each node's parent picked from earlier nodes -- a small
    forest of shallow trees, not one long chain, so most true pairs stay
    within the engine's scoreable range), then a NOISY copy with p_drop
    fraction of edges missing (simulating imperfect extraction)."""
    rng = random.Random(seed)
    true_edges = set()
    for j in range(1, n):
        for _ in range(rng.randint(1, 2)):
            true_edges.add((rng.randint(0, j - 1), j))

    adj: dict[int, set[int]] = {}
    for a, b in true_edges:
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
    eng = FuzzyInferenceEngine(walk_len=10, alpha=0.7)
    n_dropped = 0
    for a, b in sorted(true_edges):
        if rng.random() > p_drop:
            eng.add_relation(a, b)
        else:
            n_dropped += 1
    return eng, truth, len(true_edges), n_dropped, rng


def measure(seed: int, alpha: float, p_drop: float, n: int = 45):
    eng, truth, n_edges, n_dropped, rng = build_world(seed, p_drop, n)
    true_pairs = [(x, b) for x in range(n) for b in truth[x]]
    false_pairs = [(x, y) for x in range(n) for y in range(n) if x != y and y not in truth[x]]

    rng.shuffle(true_pairs)  # calibration/test must be exchangeable -- always shuffle before splitting
    half = len(true_pairs) // 2
    cal_pairs, test_pairs = true_pairs[:half], true_pairs[half:]

    reasoner = ConformalReasoner(eng, alpha=alpha)
    tau = reasoner.calibrate(cal_pairs)
    coverage = sum(reasoner.accept(a, b) for a, b in test_pairs) / len(test_pairs)
    fpr = sum(reasoner.accept(a, b) for a, b in false_pairs) / len(false_pairs)
    return tau, coverage, fpr, n_edges, n_dropped


def main() -> None:
    alpha = 0.1
    n_seeds = 20  # averaged, not a single lucky/unlucky draw -- matches theorem_conformal_reasoning
    print("=" * 74)
    print(f"Random DAGs (kinship-like, 45 concepts, {n_seeds} seeds averaged), "
          "CLEAN vs. with 20% of edges randomly DROPPED (a noisy extraction).")
    print("=" * 74)

    for label, p_drop in [("clean (0% dropped)", 0.0), ("noisy (20% dropped)", 0.2)]:
        covs, fprs, n_edges_total, n_dropped_total = [], [], 0, 0
        for seed in range(n_seeds):
            tau, coverage, fpr, n_edges, n_dropped = measure(seed=seed, alpha=alpha, p_drop=p_drop)
            covs.append(coverage)
            fprs.append(fpr)
            n_edges_total += n_edges
            n_dropped_total += n_dropped
        avg_cov = sum(covs) / len(covs)
        avg_fpr = sum(fprs) / len(fprs)
        print(f"\n-- {label} --")
        print(f"   {n_dropped_total}/{n_edges_total} true edges missing across all seeds.")
        print(f"   Average coverage on held-out true pairs: {avg_cov:.1%} (target: >= {1 - alpha:.0%})")
        print(f"   Average false-positive rate on unrelated pairs: {avg_fpr:.1%}")
        ok = avg_cov >= (1 - alpha) - 0.03
        print(f"   {'PASS' if ok else 'FAIL'}: coverage guarantee holds")

    print("\n" + "=" * 74)
    print(
        "=> Coverage stays >= 1-alpha in BOTH cases (validity is distribution-free,\n"
        "   unaffected by how noisy the graph is). The false-positive rate is what\n"
        "   pays for the noise -- efficiency degrades, validity does not. The hard\n"
        "   guard alone could not have given ANY guarantee on the noisy graph: a\n"
        "   missing edge silently breaks a real path, with no way to know how much\n"
        "   was missed. See PAPER.md §7.1 for the proof."
    )


if __name__ == "__main__":
    main()
