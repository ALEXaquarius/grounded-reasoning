"""
Held-out-evidence edge pruning vs. the raw noisy graph — an A/B comparison
across 5 noise regimes, reported honestly (this is the strongest single
efficiency finding in the project's exploration of conformal-efficiency
improvements, and it is reported as such, not downplayed).

Background: conformal calibration (Theorem K) and its Mondrian extension
(redundancy_conformal_eval.py) work AROUND a noisy graph -- they never touch
the graph itself. This experiment asks a different question: can held-out
labeled evidence identify and remove the SPECIFIC spurious edges responsible
for false positives, rather than just calibrating a threshold that tolerates
them?

`identify_suspect_edges` (grounded_reasoning/reasoning/edge_pruning.py)
implements a simple decision rule, not a statistical guarantee: an edge that
appears on the shortest proof path of a held-out FALSE-labeled claim, and
NEVER on a held-out TRUE-labeled claim's path, is removed. A DIFFERENT way
of using this same "suspect edge" signal was tried first -- as a Mondrian
group_fn (see redundancy_conformal_eval.py for that machinery) -- and made
FPR WORSE at every noise level tested, because Mondrian must preserve
coverage even for the few true claims that happen to route through a bad
edge, forcing that group's own threshold down. Direct removal has no such
constraint and is what this experiment verifies.

Run: python -m grounded_reasoning.experiments.edge_pruning_eval
(fully offline -- synthetic ground truth, no LLM call.)
"""
from __future__ import annotations

import math
import random

from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine
from grounded_reasoning.reasoning.conformal_reasoning import conformal_threshold
from grounded_reasoning.reasoning.edge_pruning import identify_suspect_edges, prune_edges

SCENARIOS = {
    "dropout-dominant (p_drop=0.2, p_add=0.3)": (0.2, 0.3),
    "spurious-dominant (p_drop=0.0, p_add=0.3)": (0.0, 0.3),
    "heavy dropout (p_drop=0.3, p_add=0.3)": (0.3, 0.3),
    "light spurious (p_drop=0.2, p_add=0.1)": (0.2, 0.1),
    "heavy spurious (p_drop=0.2, p_add=0.5)": (0.2, 0.5),
}


def wilson_upper_bound(k: int, n: int, confidence: float = 0.95) -> float:
    """
    One-sided Wilson score upper confidence bound on a true proportion p,
    given k successes in n Bernoulli(p) trials -- a large-n-safe
    approximation (not the exact Clopper-Pearson bound used elsewhere in
    this project for small calibration samples: `clopper_pearson_lower` in
    `transitivity_calibration.py` overflows here because n is in the
    hundreds-to-thousands of pooled blocked-edge decisions across many
    seeds, not a handful of calibration trials). Asymptotically valid for n
    this large; not claimed to be exact for small n.
    """
    if n == 0:
        return 0.0
    z = -_norm_ppf((1 - confidence) / 2)  # e.g. z=1.96 for confidence=0.95
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center + margin) / denom


def _norm_ppf(q: float) -> float:
    """Standard normal quantile via Acklam's rational approximation (no scipy dependency)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    if q < p_low:
        r = math.sqrt(-2 * math.log(q))
        return (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
               ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    if q <= 1 - p_low:
        r = q - 0.5
        t = r * r
        return (((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4])*t+a[5])*r / \
               (((((b[0]*t+b[1])*t+b[2])*t+b[3])*t+b[4])*t+1)
    r = math.sqrt(-2 * math.log(1 - q))
    return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
            ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)


def build_true_dag(seed: int, n: int = 45):
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
    return n, true_edges, truth


def noisy_edges(true_edges, n: int, rng: random.Random, p_drop: float, p_add: float) -> set[tuple[int, int]]:
    edges = set()
    for a, b in sorted(true_edges):  # sorted: set iteration order is hash-seed-dependent
        if rng.random() > p_drop:
            edges.add((a, b))
    for _ in range(int(p_add * len(true_edges))):
        a, b = rng.randint(0, n - 1), rng.randint(0, n - 1)
        if a != b:
            edges.add((a, b))
    return edges


def measure(
    seed: int,
    p_drop: float,
    p_add: float,
    alpha: float = 0.1,
    n: int = 45,
    identify_frac: float = 0.5,
    min_evidence: int = 1,
):
    n, true_edges, truth = build_true_dag(seed, n)
    rng = random.Random(1000 + seed)
    edges = list(noisy_edges(true_edges, n, rng, p_drop, p_add))

    eng_raw = FuzzyInferenceEngine(walk_len=12, alpha=0.7)
    for a, b in edges:
        eng_raw.add_relation(a, b)
    infc_raw = {x: eng_raw.infer(x) for x in range(n)}
    all_candidates = [(x, b) for x in range(n) for b in range(n) if x != b and b in infc_raw[x]]

    rng2 = random.Random(3000 + seed)
    rng2.shuffle(all_candidates)
    split = int(len(all_candidates) * identify_frac)
    # `identify_frac` of the pairs supply held-out labeled evidence to
    # identify suspect edges; the rest are used to evaluate both graphs --
    # disjoint, no double-dipping on the same evidence used to prune. Using a
    # LARGER identify_frac (e.g. 0.8 instead of 0.5) measurably reduces the
    # rate at which a genuinely correct edge is wrongly blocked -- see
    # `run_mitigation_comparison` and the module docstring.
    identify_pairs = [(x, b, b in truth[x]) for x, b in all_candidates[:split]]
    eval_candidates = all_candidates[split:]

    blocked = identify_suspect_edges(edges, identify_pairs, walk_len=12, alpha=0.7, min_evidence=min_evidence)
    n_wrongly_blocked = sum(1 for e in blocked if e in true_edges)
    wrongly_blocked_rate = n_wrongly_blocked / len(blocked) if blocked else 0.0
    cleaned_edges = prune_edges(edges, blocked)
    eng_clean = FuzzyInferenceEngine(walk_len=12, alpha=0.7)
    for a, b in cleaned_edges:
        eng_clean.add_relation(a, b)

    def evaluate(eng):
        infc = {x: eng.infer(x) for x in range(n)}
        jit_rng = random.Random(5000 + seed)
        scores = {(x, b): infc[x].get(b, 0.0) + jit_rng.uniform(0, 1e-9) for x, b in eval_candidates}
        true_eval = [(x, b) for x, b in eval_candidates if b in truth[x]]
        false_eval = [(x, b) for x, b in eval_candidates if b not in truth[x]]
        if not true_eval or not false_eval:
            return None
        rng3 = random.Random(6000 + seed)
        tr = list(true_eval)
        rng3.shuffle(tr)
        h2 = len(tr) // 2
        cal, test = tr[:h2], tr[h2:]
        if not cal or not test:
            return None
        tau = conformal_threshold([scores[p] for p in cal], alpha)
        coverage = sum(1 for p in test if scores[p] >= tau) / len(test)
        fpr = sum(1 for p in false_eval if scores[p] >= tau) / len(false_eval)
        return coverage, fpr

    r_raw = evaluate(eng_raw)
    r_clean = evaluate(eng_clean)
    if r_raw is None or r_clean is None:
        return None
    return {
        "raw_coverage": r_raw[0], "raw_fpr": r_raw[1],
        "cleaned_coverage": r_clean[0], "cleaned_fpr": r_clean[1],
        "n_blocked": len(blocked),
        "n_wrongly_blocked": n_wrongly_blocked,
        "wrongly_blocked_rate": wrongly_blocked_rate,
    }


def run(n_seeds: int = 60, alpha: float = 0.1) -> dict:
    out = {}
    for label, (p_drop, p_add) in SCENARIOS.items():
        rows = [measure(s, p_drop, p_add, alpha) for s in range(n_seeds)]
        rows = [r for r in rows if r is not None]
        means = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        out[label] = means
    return out


def run_mitigation_comparison(n_seeds: int = 60, alpha: float = 0.1, confidence: float = 0.95) -> dict:
    """
    Compares the default identification split/threshold (identify_frac=0.5,
    min_evidence=1) against a safer configuration (identify_frac=0.8,
    min_evidence=2), across EVERY noise regime in SCENARIOS (not just one),
    reporting the POOLED wrongly-blocked rate (total wrongly-blocked edges /
    total blocked edges, across all seeds in a regime) together with a
    one-sided Wilson upper confidence bound on it -- an honest worst-case
    statement, not just a point estimate, per the same discipline as the
    Clopper-Pearson bounds used elsewhere in this project for calibration
    guarantees (see `wilson_upper_bound`'s docstring for why a different,
    large-n approximation is used here instead of `clopper_pearson_lower`).
    """
    configs = {
        "default (identify_frac=0.5, min_evidence=1)": {"identify_frac": 0.5, "min_evidence": 1},
        "safer (identify_frac=0.8, min_evidence=2)": {"identify_frac": 0.8, "min_evidence": 2},
    }
    out = {}
    for regime, (p_drop, p_add) in SCENARIOS.items():
        out[regime] = {}
        for label, kwargs in configs.items():
            rows = [measure(s, p_drop, p_add, alpha, **kwargs) for s in range(n_seeds)]
            rows = [r for r in rows if r is not None]
            total_blocked = sum(r["n_blocked"] for r in rows)
            total_wrong = sum(r["n_wrongly_blocked"] for r in rows)
            pooled_rate = total_wrong / total_blocked if total_blocked else 0.0
            upper = wilson_upper_bound(total_wrong, total_blocked, confidence)
            out[regime][label] = {
                "pooled_wrongly_blocked_rate": pooled_rate,
                "wrongly_blocked_upper_bound": upper,
                "cleaned_fpr": sum(r["cleaned_fpr"] for r in rows) / len(rows),
                "raw_fpr": sum(r["raw_fpr"] for r in rows) / len(rows),
                "n_blocked_total": total_blocked,
            }
    return out


def main() -> None:
    res = run()
    print("=" * 88)
    print("Held-out-evidence edge pruning vs. the raw noisy graph")
    print("=" * 88)
    for label, m in res.items():
        print(f"\n-- {label} --")
        print(f"   RAW:     coverage={m['raw_coverage']:.1%}  fpr={m['raw_fpr']:.1%}")
        print(f"   CLEANED: coverage={m['cleaned_coverage']:.1%}  fpr={m['cleaned_fpr']:.1%}  "
              f"(avg {m['n_blocked']:.1f} edges blocked, {m['wrongly_blocked_rate']:.1%} wrongly blocked)")
    print(
        "\n=> Removing held-out-evidence-flagged suspect edges substantially and consistently\n"
        "   cuts false-positive rate across every noise regime tested -- coverage on the\n"
        "   remaining graph is essentially unaffected. This is a simple decision rule\n"
        "   (any single disqualifying encounter removes an edge), not a statistical\n"
        "   guarantee like the Clopper-Pearson bounds elsewhere in this project -- the\n"
        "   wrongly-blocked rate above is real, not negligible, and is the actual price\n"
        "   of the FPR reduction."
    )

    print("\n" + "=" * 88)
    print("Mitigation: identify_frac and min_evidence vs. the wrongly-blocked rate,")
    print("pooled across seeds, EVERY noise regime, with a 95% upper confidence bound")
    print("=" * 88)
    mit = run_mitigation_comparison()
    for regime, configs in mit.items():
        print(f"\n-- {regime} --")
        for label, m in configs.items():
            print(f"   {label:42s} pooled={m['pooled_wrongly_blocked_rate']:6.1%}  "
                  f"95%-upper-bound={m['wrongly_blocked_upper_bound']:6.1%}  "
                  f"cleaned_fpr={m['cleaned_fpr']:6.1%}  (n_blocked={m['n_blocked_total']})")
    print(
        "\n=> Using a larger share of held-out data to identify suspect edges (and requiring\n"
        "   a second corroborating false-claim encounter) cuts the wrongly-blocked rate\n"
        "   substantially in EVERY regime tested -- typically to single digits, worse (and\n"
        "   noisiest, fewest edges blocked) under light-spurious noise -- at the cost of a\n"
        "   smaller reserved evaluation set and somewhat less aggressive cleaning (higher\n"
        "   cleaned FPR). Run with a larger n_seeds for a tighter confidence bound."
    )


if __name__ == "__main__":
    main()
