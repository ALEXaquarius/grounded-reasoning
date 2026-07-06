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

This is a simple decision rule, NOT a statistical guarantee like the
Clopper-Pearson bounds elsewhere in this project: there is no
false-discovery-rate control. This project's Clopper-Pearson guarantees
apply to calibrating a single THRESHOLD (`calibrate_transitivity`,
`ConformalReasoner`) -- a well-behaved, low-dimensional object under
exchangeability. Edge removal is a per-edge SELECTION problem instead: a
held-out claim's true/false label is evidence about its whole proof path,
not any one edge on it, so when several edges share a path with a
genuinely bad one, no resampling or hypothesis-testing scheme tried gives
a valid bound on the wrongly-removed rate (an attribution/identifiability
obstacle, not a tunable parameter -- see CHANGELOG/git history for the
specific constructions ruled out). This rule is verified empirically
instead.

Measured properties (`edge_pruning_eval.py`, 5 noise regimes, 60+ seeds
each): false-positive rate drops substantially and consistently (e.g.
77% -> 49% under dropout-dominant noise, 59% -> 16% under
spurious-dominant noise), with coverage on the remaining graph essentially
unaffected.

Real, MEASURED tradeoffs: (1) the wrongly-removed rate is not negligible.
At the default configuration (`identify_frac=0.5, min_evidence=1`), it
pools to 13-32% across regimes. At the recommended configuration
(`identify_frac=0.85, min_evidence=2` -- found by a Pareto sweep, and the
default of `identify_and_prune_edges` below), it drops to 1.5-4.2%, with a
95% Wilson upper confidence bound of 2.6-8.9% -- at the cost of a smaller
reserved evaluation set and a somewhat higher cleaned FPR (e.g. ~49% to
~59% in the dropout-dominant regime, still far below the 77% raw
baseline). See `run_mitigation_comparison` in `edge_pruning_eval.py`.
(2) It costs real recall regardless of configuration -- any true claim
depending solely on a removed edge loses that path. (3) The graph is
edited in place, a one-way structural change, unlike calibration (which
only adjusts a threshold and leaves the graph untouched) -- if the query
distribution later differs from the held-out sample used to prune, a
removed edge might have been needed after all.

SCOPE, checked against a REAL LLM (DeepSeek, not just simulated noise) --
see `edge_pruning_llm_eval.py`: on a densely-hallucinated multi-hop-shortcut
scenario, the blocking decision stayed accurate on real hallucinations,
matching the synthetic benchmark's precision, but the improvement in
downstream FPR held in only 73% of random identify/eval splits of that
data (mean 69.6% -> 63.5%), not the near-universal win measured on the
synthetic benchmark. This mitigation's benefit should therefore NOT be
assumed to generalize beyond the regime it was measured in
(locally-random 1-hop noise at moderate density) -- a dense, hub-heavy
hallucination pattern still helps more often than not, but needs its own
validation before relying on it.

A topology-aware variant (requiring less evidence to remove an edge
pointing at a high-centrality "hub" node, more for a peripheral one, since
a bad hub-attachment does outsized damage) was tried against both
benchmarks: on the synthetic one it was statistically indistinguishable
from just raising `min_evidence` uniformly (no real structural gain); on
the real hallucination data it touched ~5x fewer edges for comparable or
slightly better mean FPR, but did not improve the underlying 73% split
reliability. Not adopted -- a different tradeoff point (much less graph
editing), not a resolved improvement.

A supervised classifier (logistic regression over vote counts) was also
tried, fitted on synthetic data and cross-validated for training-run
stability -- but it decides on an ABSOLUTE, fitted feature scale, and
failed outright on the real data, whose evidence counts run ~100x larger
than what it was fitted on (see `edge_pruning_llm_eval.py`'s docstring).
A quantile threshold -- blocking a candidate once its false-vote count
reaches a given fraction of the false-vote-count distribution FOR THAT
GRAPH, rather than a fixed integer -- was tried as a closed-form,
scale-invariant replacement requiring no fitting, on the same principle
that makes this project's conformal calibration distribution-free (a rank
within a sample, not an absolute cutoff). It transfers across that same
~100x scale gap between synthetic and real data with no retuning at all
(unlike the classifier), landing at the same 11/15 split-reliability as
the count-based rule above. But on the synthetic benchmark it is
uniformly more conservative than `min_evidence=2` (fewer edges blocked,
worse FPR in every regime tested) rather than a strict improvement, so it
is not adopted here -- the scale-invariance property is real and
answers what a fitted classifier was actually buying (before it broke
under distribution shift), but this project only ships a decision rule
once it beats what is already shipped, not merely matches it under a
different parameterization.
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


def identify_suspect_edges_propagated(
    edges: list[tuple[str, str]],
    labeled_pairs: list[tuple[str, str, bool]],
    walk_len: int = 8,
    alpha: float = 0.6,
    min_evidence: int = 2,
    propagated_min_evidence: int = 1,
) -> set[tuple[str, str]]:
    """
    A deterministic two-pass refinement of `identify_suspect_edges` -- no
    fitted parameters, no learned weights. Once ANY edge into a target
    node is confirmed suspect (>=min_evidence false votes, zero true
    votes), that target is treated as a "hallucination magnet": its OTHER
    candidate incoming edges (also zero true votes) only need
    `propagated_min_evidence` (lower) false-claim corroboration to be
    removed too, instead of independently clearing `min_evidence` alone.

    Why: `identify_suspect_edges` decides each edge independently. When a
    hub node has SEVERAL bad incoming edges but only some individually
    clear `min_evidence`, the ones left behind benefit from
    FuzzyInferenceEngine's row-normalized diffusion (P = D^-1 W):
    removing some of a node's incoming edges elsewhere in the graph does
    not directly change this, but removing some of a *source* node's
    OTHER outgoing edges concentrates that source's transition
    probability onto whichever of its edges remain -- including any
    still-bad one landing on this same magnet target that wasn't
    individually blocked. This is the mechanism diagnosed behind the
    real-LLM inconsistency in the module docstring. Once one bad edge
    into a target is confirmed, that itself is corroborating evidence
    that other suspicious-looking edges into the SAME target are not
    just identification noise.

    Measured (`edge_pruning_eval.py` / `edge_pruning_llm_eval.py`): on the
    synthetic benchmark (identify_frac=0.85, 200 seeds/regime),
    essentially identical to `identify_suspect_edges(min_evidence=2)` --
    sparse, locally-random noise rarely puts multiple suspect edges on
    the same target, so the propagation step rarely fires (no
    regression). On the real DeepSeek hallucination data (hub-node-heavy
    by construction), it blocks ~2.6x more edges than the plain rule
    while keeping the pooled wrongly-blocked rate low (~3.4%) and the
    same 11/15 (73%) split-reliability against the raw graph, with a
    slightly better mean FPR (62.0% vs 63.5%) -- a real, deterministic
    improvement on the specific regime the plain rule was weakest on, not
    a full fix (the underlying 73% reliability is unchanged, not solved).
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

    candidates = {e for e in false_votes if true_votes.get(e, 0) == 0}
    blocked = {e for e in candidates if false_votes[e] >= min_evidence}
    magnet_targets = {b for _, b in blocked}
    for e in candidates - blocked:
        _, b = e
        if b in magnet_targets and false_votes[e] >= propagated_min_evidence:
            blocked.add(e)
    return blocked


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
    use_propagation: bool = False,
    propagated_min_evidence: int = 1,
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

    `use_propagation=True` swaps in `identify_suspect_edges_propagated`
    instead of the plain rule -- opt-in, not the default, since its
    measured benefit is specific to hub-heavy graphs (see that function's
    docstring): essentially no effect on locally-random noise, but blocks
    substantially more real hallucinated edges on dense, hub-node-heavy
    data (e.g. an LLM's own multi-hop shortcuts converging on popular
    concepts) at the same reliability and a slightly better mean FPR.
    Recommended when your deployment's graph has that shape; the plain
    default is the safer choice when it doesn't or you're unsure.

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
    if use_propagation:
        blocked = identify_suspect_edges_propagated(
            edges, identify_pairs, walk_len=walk_len, alpha=alpha,
            min_evidence=min_evidence, propagated_min_evidence=propagated_min_evidence,
        )
    else:
        blocked = identify_suspect_edges(
            edges, identify_pairs, walk_len=walk_len, alpha=alpha, min_evidence=min_evidence
        )
    cleaned = prune_edges(edges, blocked)
    return cleaned, blocked, reserved_pairs
