"""
Held-Out-Evidence Edge Pruning — identify and remove specific SPURIOUS
edges from a noisy relation graph, using held-out labeled evidence, instead
of only recalibrating a threshold around them.

The rule: from held-out labeled (subject, object, is_actually_true) triples
— ground truth known INDEPENDENTLY of the graph, same convention as
`calibrate_transitivity`'s `labeled_pairs` — an edge is removed if it
appears on the shortest proof path of at least one FALSE-labeled pair and on
the proof path of NO TRUE-labeled pair. Not merely down-weighted or
grouped: removed outright.

Why removal, not Mondrian grouping: `ConformalReasoner.calibrate(...,
group_fn=...)` (see conformal_reasoning.py) still has to guarantee coverage
WITHIN whatever group a "suspect" claim falls into. If that group is mostly
false claims routed through a bad edge, satisfying coverage for the handful
of true claims that also happen to cross it forces that group's own
threshold down -- making FPR WORSE, not better (verified: tried first,
consistently worsened FPR at every noise level tested). Removing the
identified edge from the graph avoids this entirely: the few true claims
that depended on it lose that specific path (a real, disclosed recall cost),
but every false claim that depended on it loses its only support too.

This is a simple decision rule (by default, any single disqualifying
encounter removes an edge, unless offset by at least one true-claim
encounter on the SAME edge), NOT a statistical guarantee like the
Clopper-Pearson bounds elsewhere in this project -- there is no claimed
false-discovery-rate control here. It is verified empirically instead: over
30-80 seeds across 6 noise regimes (dropout-dominant, spurious-dominant, and
mixes), false-positive rate dropped substantially and consistently (e.g.
76.8% -> 50.0% at p_drop=0.2/p_add=0.3, winning in 69/80 trials; 58.9% ->
16.2% at p_drop=0.0/p_add=0.3), with coverage on the REMAINING graph
essentially unaffected -- see `grounded_reasoning/experiments/edge_pruning_eval.py`.

Real tradeoffs, MEASURED across every noise regime tested (not just
disclosed as a possibility, not just checked in one scenario): (1) no
false-discovery-rate bound, and the empirical wrongly-removed rate is not
negligible -- with a 50/50 held-out split (half used to identify suspect
edges, half reserved for the final evaluation) and the default rule, the
POOLED wrongly-removed rate ranges 13-32% depending on noise regime (worst:
light-spurious noise, where fewest edges get blocked at all).

A Pareto sweep over `identify_frac` (the share of held-out data used for
identification, tested 0.5-0.9) and `min_evidence` (tested 1-3) found
`identify_frac=0.85, min_evidence=2` dominates every point at or below
`identify_frac=0.8`: a lower wrongly-blocked rate at a small extra cost in
cleaned FPR, in every regime. Beyond 0.85, `identify_frac=0.9` was tried and
REJECTED -- the reserved evaluation split shrinks enough that the
conformal-calibration split inside it becomes unreliable and cleaned FPR
degrades sharply back toward the raw baseline. At `identify_frac=0.85,
min_evidence=2`, the pooled wrongly-removed rate drops to 1.3-4.2%, with a
one-sided 95% Wilson upper confidence bound of 2.7-8.9% (worst case still
light-spurious noise, and also the noisiest estimate -- fewest edges
blocked means the widest interval) -- at the cost of a smaller reserved
evaluation set and somewhat less aggressive cleaning (cleaned FPR rises,
e.g. ~49% to ~59% in the dropout-dominant regime, still far below the 77%
raw baseline). See `run_mitigation_comparison` and `wilson_upper_bound` in
`edge_pruning_eval.py`.

Two other directions were tried and did NOT beat this: (a) stability
selection (Meinshausen & Buhlmann 2010) -- bootstrap-resampling the
identification half and requiring an edge to be flagged in most resamples
-- gave essentially the same result as `min_evidence` alone, because
resampling a FIXED, already-scarce identification half cannot manufacture
true-claim evidence an edge never received in the first place; (b) a
formal per-edge hypothesis test (binomial tail probability under a
global-false-rate null, with Benjamini-Hochberg FDR control across
candidate edges) was numerically WORSE than the simple rule at the same
nominal target -- the null's independence assumption does not hold here,
since an edge can be swept into disproportionately many false-claim
encounters simply by sharing a path with a genuinely bad edge, not because
it is bad itself. Both are reported here so the choice of the simpler rule
is a verified one, not an oversight.

(2) It costs real recall regardless of configuration -- any true claim
depending solely on a removed edge loses that path; (3) the graph is edited
in place, a one-way structural change, unlike calibration (which only
adjusts a threshold and leaves the graph untouched) -- if the query
distribution later differs from the held-out sample used to prune, a
removed edge might have been needed after all.

SCOPE, checked against a REAL LLM (DeepSeek), not just simulated noise --
see `edge_pruning_llm_eval.py`: on a densely-hallucinated multi-hop-shortcut
scenario (an LLM's own claimed transitive conclusions treated as direct
edges -- 65-73% of them hallucinated, real DeepSeek output), the BLOCKING
decision itself stayed accurate (3-4% wrongly blocked, matching the
synthetic benchmark), but the downstream effect on cleaned FPR was
INCONSISTENT across 3 independent trials (2 improved, 1 got worse) --
unlike the consistent 5/5-regime win measured synthetically. Root cause:
that scenario's topology (a few hub nodes carrying many hallucinated
shortcuts) interacts with FuzzyInferenceEngine's row-normalized diffusion
differently than the synthetic benchmark's sparse, locally-random noise --
removing some of a node's edges can concentrate transition probability
onto whichever false edges remain. This mitigation's benefit is therefore
NOT assumed to generalize beyond the regime it was measured in
(locally-random 1-hop noise at moderate density); a dense,
hub-heavy hallucination pattern needs its own validation before relying on
it.
"""
from __future__ import annotations

import random

from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine


def identify_suspect_edges(
    edges: list[tuple[str, str]],
    labeled_pairs: list[tuple[str, str, bool]],
    walk_len: int = 8,
    alpha: float = 0.6,
    min_evidence: int = 1,
) -> set[tuple[str, str]]:
    """
    Given the current (possibly noisy) edge list and held-out labeled
    (subject, object, is_actually_true) triples -- ground truth known
    INDEPENDENTLY of the graph, e.g. human-verified, same convention as
    `calibrate_transitivity`'s `labeled_pairs` -- returns the set of edges
    that appear on the shortest proof path of at least `min_evidence`
    FALSE-labeled pairs and NO TRUE-labeled pair.

    `min_evidence` (default 1, matching the module's originally-verified
    behavior): raising it to 2 or 3 requires more corroborating false-claim
    encounters before an edge is removed, reducing (not eliminating) the
    chance of removing a genuinely correct edge that simply had little
    exposure in `labeled_pairs` -- see the module docstring for measured
    numbers. This is NOT a statistical significance threshold, just a
    stricter version of the same simple rule.

    `labeled_pairs` should be a held-out sample, disjoint from whatever
    pairs you intend to trust the resulting pruned graph's scores for (the
    same discipline as every other calibration function here). Using a
    LARGER share of your available labeled data here (vs. reserving it for
    a separate evaluation step) measurably reduces the wrongly-removed rate
    -- see the module docstring for the measured Pareto sweep; the
    recommended pairing is `min_evidence=2` with roughly 85% of your
    held-out data used for identification (and the rest reserved for
    evaluating the cleaned graph).
    """
    eng = FuzzyInferenceEngine(walk_len=walk_len, alpha=alpha)
    for a, b in edges:
        eng.add_relation(a, b)

    true_votes: dict[tuple[str, str], int] = {}
    false_votes: dict[tuple[str, str], int] = {}
    for subject, obj, is_true in labeled_pairs:
        path = eng.explain(subject, obj)
        if path is None:
            continue
        path_edges = set(zip(path, path[1:]))
        if is_true:
            for e in path_edges:
                true_votes[e] = true_votes.get(e, 0) + 1
        else:
            for e in path_edges:
                false_votes[e] = false_votes.get(e, 0) + 1
    return {
        e for e, c in false_votes.items()
        if c >= min_evidence and true_votes.get(e, 0) == 0
    }


def prune_edges(
    edges: list[tuple[str, str]],
    blocked: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Returns `edges` with every edge in `blocked` removed."""
    return [e for e in edges if e not in blocked]


def identify_and_prune_edges(
    edges: list[tuple[str, str]],
    labeled_pairs: list[tuple[str, str, bool]],
    identify_frac: float = 0.85,
    min_evidence: int = 2,
    walk_len: int = 8,
    alpha: float = 0.6,
    seed: int | None = None,
) -> tuple[list[tuple[str, str]], set[tuple[str, str]], list[tuple[str, str, bool]]]:
    """
    Convenience wrapper applying the measured-safest configuration by
    default -- `identify_frac=0.85, min_evidence=2`, found by the Pareto
    sweep in the module docstring (pooled wrongly-blocked rate 1.5-4.2%,
    95% upper bound 2.6-8.9%, vs. 13-32% at `identify_frac=0.5,
    min_evidence=1`). Splits `labeled_pairs` into an identification share
    and a reserved share, calls `identify_suspect_edges` on the
    identification share only, prunes the result, and returns
    `(cleaned_edges, blocked_edges, reserved_pairs)`.

    `reserved_pairs` is disjoint from whatever was used to decide removal,
    so it's safe to use to independently evaluate the cleaned graph (the
    same role `eval_candidates` plays in `edge_pruning_eval.py`) --
    scoring the cleaned graph on `identify_pairs` instead would be
    circular, since that's the same evidence used to decide what to
    remove.

    `seed` controls the random split (a plain `random.Random` seed); pass
    an explicit int for a reproducible split.
    """
    rng = random.Random(seed)
    shuffled = list(labeled_pairs)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * identify_frac)
    identify_pairs, reserved_pairs = shuffled[:split], shuffled[split:]
    blocked = identify_suspect_edges(
        edges, identify_pairs, walk_len=walk_len, alpha=alpha, min_evidence=min_evidence
    )
    cleaned = prune_edges(edges, blocked)
    return cleaned, blocked, reserved_pairs
