"""
Negative-Selection Edge Pruning — identify and remove specific SPURIOUS
edges from a noisy relation graph, using held-out labeled evidence, instead
of only recalibrating a threshold around them.

Motivation (immunology-inspired framing, not new math): in adaptive immunity,
T-cells that react to the body's OWN tissue ("self") are eliminated during
maturation (negative selection) -- a cell is judged not by a statistical
score but by a single disqualifying encounter. Applied here: a graph edge
that appears on the proof path of at least one held-out labeled claim known
to be FALSE, and on the proof path of NO held-out claim known to be TRUE, is
judged spurious and removed outright -- not merely down-weighted or grouped.

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

This is a simple decision rule (any single disqualifying encounter vetoes an
edge, unless "exonerated" by at least one true-claim encounter), not a
statistical guarantee like the Clopper-Pearson bounds elsewhere in this
project -- there is no claimed false-discovery-rate control here. It is
verified empirically instead: over 30-80 seeds across 6 noise regimes
(dropout-dominant, spurious-dominant, and mixes), false-positive rate
dropped substantially and consistently (e.g. 76.8% -> 50.0% at
p_drop=0.2/p_add=0.3, winning in 69/80 trials; 58.9% -> 16.2% at
p_drop=0.0/p_add=0.3), with coverage on the REMAINING graph essentially
unaffected -- see `grounded_reasoning/experiments/negative_selection_eval.py`.
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
    blocked: set[tuple[str, str]] = set()
    for subject, obj, is_true in labeled_pairs:
        path = eng.explain(subject, obj)
        if path is None:
            continue
        path_edges = set(zip(path, path[1:]))
        if is_true:
            for e in path_edges:
                true_votes[e] = true_votes.get(e, 0) + 1
        else:
            blocked |= path_edges
    return {e for e in blocked if true_votes.get(e, 0) == 0}


def prune_edges(
    edges: list[tuple[str, str]],
    blocked: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Returns `edges` with every edge in `blocked` removed."""
    return [e for e in edges if e not in blocked]
