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

This is a simple decision rule (any single disqualifying encounter removes
an edge, unless offset by at least one true-claim encounter on the SAME
edge), NOT a statistical guarantee like the Clopper-Pearson bounds elsewhere
in this project -- there is no claimed false-discovery-rate control here.
It is verified empirically instead: over 30-80 seeds across 6 noise regimes
(dropout-dominant, spurious-dominant, and mixes), false-positive rate
dropped substantially and consistently (e.g. 76.8% -> 50.0% at
p_drop=0.2/p_add=0.3, winning in 69/80 trials; 58.9% -> 16.2% at
p_drop=0.0/p_add=0.3), with coverage on the REMAINING graph essentially
unaffected -- see `grounded_reasoning/experiments/edge_pruning_eval.py`.

Real tradeoffs (not hidden): (1) no false-discovery-rate bound -- with a
small or unrepresentative held-out sample, a genuinely correct edge could
in principle be removed; (2) a real recall cost -- any true claim depending
solely on a removed edge loses that path; (3) the graph is edited in place,
a one-way structural change, unlike calibration (which only adjusts a
threshold and leaves the graph untouched) -- if the query distribution
later differs from the held-out sample used to prune, a removed edge might
have been needed after all.
"""
from __future__ import annotations

from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine


def identify_suspect_edges(
    edges: list[tuple[str, str]],
    labeled_pairs: list[tuple[str, str, bool]],
    walk_len: int = 8,
    alpha: float = 0.6,
) -> set[tuple[str, str]]:
    """
    Given the current (possibly noisy) edge list and held-out labeled
    (subject, object, is_actually_true) triples -- ground truth known
    INDEPENDENTLY of the graph, e.g. human-verified, same convention as
    `calibrate_transitivity`'s `labeled_pairs` -- returns the set of edges
    that appear on the shortest proof path of at least one FALSE-labeled
    pair and NO TRUE-labeled pair.

    `labeled_pairs` should be a held-out sample, disjoint from whatever
    pairs you intend to trust the resulting pruned graph's scores for
    (the same discipline as every other calibration function here).
    """
    eng = FuzzyInferenceEngine(walk_len=walk_len, alpha=alpha)
    for a, b in edges:
        eng.add_relation(a, b)

    true_votes: dict[tuple[str, str], int] = {}
    suspect: set[tuple[str, str]] = set()
    for subject, obj, is_true in labeled_pairs:
        path = eng.explain(subject, obj)
        if path is None:
            continue
        path_edges = set(zip(path, path[1:]))
        if is_true:
            for e in path_edges:
                true_votes[e] = true_votes.get(e, 0) + 1
        else:
            suspect |= path_edges
    return {e for e in suspect if true_votes.get(e, 0) == 0}


def prune_edges(
    edges: list[tuple[str, str]],
    blocked: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Returns `edges` with every edge in `blocked` removed."""
    return [e for e in edges if e not in blocked]
