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

Real tradeoffs, MEASURED, not just disclosed as a possibility: (1) no
false-discovery-rate bound, and the empirical wrongly-removed rate is not
negligible -- with a 50/50 held-out split (half used to identify suspect
edges, half reserved for the final evaluation), ~17-19% of removed edges
were genuinely correct edges that simply drew zero true-claim traffic in the
identification half by chance. Two measured mitigations, NOT a formal fix:
using a LARGER share of the held-out data for identification (e.g. 80/20
instead of 50/50) drops this to ~5.5%, and additionally requiring
`min_evidence=2` (below) drops it further to ~4.5% -- at the cost of a
smaller reserved evaluation set and slightly less aggressive cleaning
(cleaned FPR ~58% instead of ~49% in the same scenario). (2) a real recall
cost -- any true claim depending solely on a removed edge loses that path;
(3) the graph is edited in place, a one-way structural change, unlike
calibration (which only adjusts a threshold and leaves the graph untouched)
-- if the query distribution later differs from the held-out sample used to
prune, a removed edge might have been needed after all.
"""
from __future__ import annotations

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
    -- see the module docstring.
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
