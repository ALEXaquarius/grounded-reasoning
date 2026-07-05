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

Two further sections show the two ways to push efficiency further without
losing the coverage guarantee: `mondrian_demo` (a separate threshold per
path-redundancy group, best when the noise is dominated by DROPPED edges)
and `drift_demo` (a threshold that ADAPTS over time, needed when the noise
level itself changes between batches -- something no fixed calibration,
grouped or not, can handle).

Fully offline, synthetic ground truth and synthetic noise (no LLM call) --
see conformal_llm_eval.py for the same idea end-to-end on a REAL LLM's
extraction, theory/theorems.py::theorem_conformal_reasoning for the formal
numerical verification the main demo mirrors, and
redundancy_conformal_eval.py / drift_conformal_eval.py for the full
multi-seed studies behind the other two sections.

Run: python examples/conformal_demo.py
"""
from __future__ import annotations

import random

from grounded_reasoning import AdaptiveConformalReasoner, ConformalReasoner
from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine
from grounded_reasoning.reasoning.conformal_reasoning import conformal_threshold, redundancy_group


def build_world(seed: int, p_drop: float, n: int = 45, p_add: float = 0.0):
    """A random DAG (each node's parent picked from earlier nodes -- a small
    forest of shallow trees, not one long chain, so most true pairs stay
    within the engine's scoreable range), then a NOISY copy with p_drop
    fraction of edges missing (simulating imperfect extraction) and,
    optionally, some spurious edges added (p_add -- used by mondrian_demo,
    since pure edge-dropping alone can never fabricate a false positive: it
    only ever REMOVES edges from the true graph)."""
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
    for _ in range(int(p_add * len(true_edges))):
        a, b = rng.randint(0, n - 1), rng.randint(0, n - 1)
        if a != b:
            eng.add_relation(a, b)
    return eng, truth, len(true_edges), n_dropped, rng


def measure(seed: int, alpha: float, p_drop: float, n: int = 45, p_add: float = 0.0,
            use_redundancy_grouping: bool = False):
    eng, truth, n_edges, n_dropped, rng = build_world(seed, p_drop, n, p_add)
    infc = {x: eng.infer(x) for x in range(n)}
    true_pairs = [(x, b) for x in range(n) for b in truth[x]]
    # only pairs with SOME diffusion score are genuine false-positive candidates
    # (a pair scoring exactly 0 is never "at risk" of acceptance except in the
    # degenerate tau<=0 case, which would swamp the FPR metric with non-events)
    false_pairs = [
        (x, y) for x in range(n) for y in range(n)
        if x != y and y not in truth[x] and y in infc[x]
    ]

    rng.shuffle(true_pairs)  # calibration/test must be exchangeable -- always shuffle before splitting
    half = len(true_pairs) // 2
    cal_pairs, test_pairs = true_pairs[:half], true_pairs[half:]

    reasoner = ConformalReasoner(eng, alpha=alpha)
    group_fn = (lambda x, b: redundancy_group(eng, x, b)) if use_redundancy_grouping else None
    tau = reasoner.calibrate(cal_pairs, group_fn=group_fn)
    coverage = sum(reasoner.accept(a, b) for a, b in test_pairs) / len(test_pairs)
    # with pure edge-dropping (p_add=0) there may be NO false-positive candidates
    # at all (dropout can only remove true edges, never fabricate a false
    # connection) -- 0 candidates genuinely means 0% empirical FPR, not undefined
    fpr = sum(reasoner.accept(a, b) for a, b in false_pairs) / len(false_pairs) if false_pairs else 0.0
    return tau, coverage, fpr, n_edges, n_dropped


def main() -> None:
    alpha = 0.1
    n_seeds = 20  # averaged, not a single lucky/unlucky draw -- matches theorem_conformal_reasoning
    # a FIXED rate of spurious edges (p_add) throughout, matching
    # theorem_conformal_reasoning's own convention: pure edge-DROPPING alone
    # can never fabricate a false positive (it only ever removes true edges),
    # so a meaningful "efficiency pays for noise" story needs a real source of
    # false positives to degrade -- p_drop is what varies across the two rows.
    p_add = 0.3
    print("=" * 74)
    print(f"Random DAGs (kinship-like, 45 concepts, {n_seeds} seeds averaged), a FIXED rate of "
          f"spurious edges (p_add={p_add:.0%}) throughout, CLEAN vs. an ADDITIONAL 20% of true "
          f"edges randomly DROPPED on top.")
    print("=" * 74)

    for label, p_drop in [("clean (0% additionally dropped)", 0.0), ("noisy (20% additionally dropped)", 0.2)]:
        covs, fprs, n_edges_total, n_dropped_total = [], [], 0, 0
        for seed in range(n_seeds):
            tau, coverage, fpr, n_edges, n_dropped = measure(seed=seed, alpha=alpha, p_drop=p_drop, p_add=p_add)
            covs.append(coverage)
            fprs.append(fpr)
            n_edges_total += n_edges
            n_dropped_total += n_dropped
        avg_cov = sum(covs) / len(covs)
        avg_fpr = sum(fprs) / len(fprs)
        print(f"\n-- {label} --")
        print(f"   {n_dropped_total}/{n_edges_total} true edges missing across all seeds "
              f"(plus spurious edges at every noise level).")
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


def mondrian_demo() -> None:
    print("\n" + "=" * 74)
    print("Pushing efficiency further: a separate threshold per path-redundancy")
    print("group (Mondrian conformal), instead of one global threshold")
    print("=" * 74)

    # dropout-dominant noise (matching redundancy_conformal_eval.py's verified
    # scenario): some spurious edges too (p_add), so there are genuine false
    # positives to filter -- pure dropout alone can only REMOVE true edges,
    # never fabricate a false one, so there'd be nothing for grouping to improve
    alpha, n_seeds, p_drop, p_add = 0.1, 15, 0.2, 0.3
    global_fprs, grouped_fprs, global_covs, grouped_covs = [], [], [], []
    for seed in range(n_seeds):
        _, cov_g, fpr_g, *_ = measure(seed, alpha, p_drop, p_add=p_add, use_redundancy_grouping=False)
        _, cov_r, fpr_r, *_ = measure(seed, alpha, p_drop, p_add=p_add, use_redundancy_grouping=True)
        global_covs.append(cov_g)
        global_fprs.append(fpr_g)
        grouped_covs.append(cov_r)
        grouped_fprs.append(fpr_r)

    avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(f"\nGLOBAL threshold:  coverage={avg(global_covs):.1%}  fpr={avg(global_fprs):.1%}")
    print(f"GROUPED (redundancy_group): coverage={avg(grouped_covs):.1%}  fpr={avg(grouped_fprs):.1%}")
    print(
        "\nHow: ConformalReasoner.calibrate(cal_pairs, group_fn=lambda x, b: "
        "redundancy_group(engine, x, b)) -- a pair with more than one walk in the\n"
        "extracted graph gets its OWN threshold, since it survives a dropped edge\n"
        "more often than a singly-connected pair does. Coverage still holds either\n"
        "way; only efficiency changes. (This specific grouping helps when DROPPED\n"
        "edges dominate the noise -- see redundancy_conformal_eval.py for the honest\n"
        "case where it does NOT help, and PAPER.md §7.1 for a falsified alternative\n"
        "grouping that was tried and discarded first.)"
    )


def drift_demo() -> None:
    print("\n" + "=" * 74)
    print("A DIFFERENT problem: the noise level changing over time, not just")
    print("being heterogeneous (AdaptiveConformalReasoner / ACI)")
    print("=" * 74)

    alpha = 0.1
    n_batches, batch_size = 20, 25

    def batch_scores(seed: int, p_drop: float):
        eng, truth, *_ = build_world(seed, p_drop)
        rng = random.Random(1000 + seed)
        scores = [eng.infer(x).get(b, 0.0) + rng.uniform(0, 1e-9) for x in truth for b in truth[x]]
        rng.shuffle(scores)
        return scores[:batch_size]

    # clean for the first half of the stream, noisy for the second half
    init = batch_scores(seed=0, p_drop=0.05)
    tau_static = conformal_threshold(init, alpha)
    adaptive = AdaptiveConformalReasoner(engine=None, alpha=alpha, init_scores=init)

    static_pre, static_post, adaptive_pre, adaptive_post = [], [], [], []
    for i in range(n_batches):
        p_drop = 0.05 if i < n_batches // 2 else 0.45
        batch = batch_scores(seed=i + 1, p_drop=p_drop)
        static_cov = sum(1 for s in batch if s >= tau_static) / len(batch)
        adaptive_cov = sum(adaptive.update_score(s) for s in batch) / len(batch)
        (static_pre if i < n_batches // 2 else static_post).append(static_cov)
        (adaptive_pre if i < n_batches // 2 else adaptive_post).append(adaptive_cov)

    avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(f"\nNoise shifts from p_drop=0.05 to p_drop=0.45 halfway through {n_batches} batches.")
    print(f"FROZEN threshold:  pre-shift={avg(static_pre):.1%}  post-shift={avg(static_post):.1%}"
          f"  (target >= {1 - alpha:.0%})")
    print(f"ADAPTIVE (ACI):    pre-shift={avg(adaptive_pre):.1%}  post-shift={avg(adaptive_post):.1%}"
          f"  (target >= {1 - alpha:.0%})")
    print(
        "\nHow: acr = AdaptiveConformalReasoner(engine, alpha=0.1); "
        "acr.update(x, b) for each\nnewly-confirmed-true pair as they arrive over time, "
        "acr.accept(x, b) to check a claim\nagainst the CURRENT (adapting) threshold. "
        "No stationarity assumption needed for\nACI's own guarantee -- see "
        "drift_conformal_eval.py for the full 15-trial study."
    )


if __name__ == "__main__":
    main()
    mondrian_demo()
    drift_demo()
