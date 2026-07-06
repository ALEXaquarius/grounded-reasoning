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
see `edge_pruning_llm_eval.py`: an EARLIER pass of this validation had a
node-namespace bug that silently merged distinct per-query DAGs together,
inflating per-edge evidence counts with spurious cross-query overlap
(fixed -- see CHANGELOG/git history). Re-run on correctly-namespaced real
data (each query's DAG genuinely disjoint, matching how a real deployment
issues independent queries), the picture differs materially from what
synthetic noise predicts: with each candidate edge backed by EXACTLY one
labeled encounter (no query repeated), `min_evidence` thresholds of 2+ --
including `identify_suspect_edges_propagated` below, whose magnet
mechanism needs some edge to first clear that bar -- never fire at all (0
edges blocked, a pure no-op on this data). Lowering to `min_evidence=1`
does block real hallucinated edges, but on THIS regime -- many distinct
source nodes, each with several direct 1-hop shortcut claims, mostly
false -- makes downstream FPR WORSE than doing nothing (beat raw in only
4/15 splits, mean FPR 63.0% -> 70.7%), not better.

Traced to `FuzzyInferenceEngine`'s row-normalized diffusion (P = D^-1 W):
a reserved (never-blocked) edge's score is proportional to
1/out-degree(source); removing that SAME source's OTHER (blocked) edges
mechanically concentrates transition mass onto whatever the source has
left, inflating the score of any still-present false edge -- worse the
MORE aggressively a source's edges get pruned, which is exactly what
happens on this hub/multi-shortcut-per-source topology. This is a
structural cost of REMOVING edges outright on this graph shape, not a
defect in which specific edges get chosen. `masked_infer` below sidesteps
it by normalizing each source's transition probabilities by its ORIGINAL
out-degree (from the full, unpruned edge list) -- removing a blocked edge
then only ever REMOVES its confidence mass, never redistributes it onto
surviving edges. Paired with the plain `min_evidence=1` rule on this same
corrected data, it recovers a genuine improvement: beats raw in 12/15
splits, mean FPR 63.0% -> 54.0% (vs. `min_evidence=1` alone making FPR
*worse*). On the synthetic benchmark (5 regimes, 60 seeds each) it is
statistically indistinguishable from ordinary pruned-graph scoring -- no
regression -- since that benchmark's noise is sparse per-node, giving the
degree-concentration effect little room to act either way. Use
`masked_infer` for scoring whenever pruning is applied to a graph built
from many direct, per-source shortcut claims (e.g. an LLM's own
multi-hop over-claims); plain post-prune scoring remains fine for
generic, locally-random noise.

A topology-aware variant (requiring less evidence to remove an edge
pointing at a high-centrality "hub" node, more for a peripheral one) and
`identify_suspect_edges_propagated` below were both measured as real
improvements over the plain rule in an earlier pass over this real data --
but that pass used the since-corrected, namespace-buggy dataset, and
neither improvement reproduced on the fix (both depend on some edge
individually clearing `min_evidence>=2`, which never happens here -- see
above). They are kept for graphs where an edge genuinely does get
independently corroborated 2+ times (still help there, and do not
regress the synthetic benchmark), but neither is a fix for the
single-encounter regime `masked_infer` targets.
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

    Measured (`edge_pruning_eval.py`): on the synthetic benchmark
    (identify_frac=0.85, 200 seeds/regime), essentially identical to
    `identify_suspect_edges(min_evidence=2)` -- sparse, locally-random
    noise rarely puts multiple suspect edges on the same target, so the
    propagation step rarely fires (no regression).

    On real DeepSeek hallucination data where each candidate edge is
    backed by only ONE labeled encounter (the realistic case when queries
    aren't repeated -- see `edge_pruning_llm_eval.py`), this function's
    magnet mechanism never fires at all: it requires some edge to first
    independently clear `min_evidence` (default 2), which never happens
    when no edge ever gets a second vote, so it degenerates to a pure
    no-op there. An EARLIER pass measured a real improvement on this same
    real data, but that pass used a dataset with a node-namespace bug
    that spuriously inflated per-edge evidence counts above 1 (fixed --
    see CHANGELOG/git history); the improvement did not reproduce once
    fixed. Kept for graphs where an edge genuinely IS independently
    corroborated 2+ times (still helps there, and does not regress the
    synthetic benchmark) -- for the single-encounter, hub-heavy real
    regime, see `masked_infer` below instead (module docstring above).
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


def masked_infer(
    edges: list[tuple[str, str]],
    blocked: set[tuple[str, str]],
    source: str,
    walk_len: int = 8,
    alpha: float = 0.6,
) -> dict[str, float]:
    """
    An alternative to running `FuzzyInferenceEngine.infer` on
    `prune_edges(edges, blocked)`: mirrors that same diffusion exactly,
    except each node's transition probabilities are normalized by its
    ORIGINAL out-degree (computed from the full `edges`, including
    blocked ones) instead of the degree of the pruned graph -- so
    removing a blocked edge only ever REMOVES its confidence mass, never
    redistributes it onto the source's surviving edges.

    Why this matters (see the module docstring's real-data discussion):
    plain post-prune scoring's transition probabilities are `P = D^-1 W`
    on the PRUNED graph, so a source's out-degree shrinks by however many
    of its edges got blocked -- concentrating its remaining transition
    mass onto whatever edges are left, including any still-present false
    edge that individual voting didn't happen to block. This is a real
    cost, not a corner case, on a graph built from many direct,
    per-source shortcut claims (e.g. an LLM's own multi-hop over-claims):
    measured on real DeepSeek hallucination data where each candidate
    edge is backed by only one labeled encounter, plain `min_evidence=1`
    pruning made downstream FPR WORSE than no pruning at all (63.0% ->
    70.7%, beating raw in only 4/15 splits), while `masked_infer` on the
    SAME blocked set recovered a genuine improvement (63.0% -> 54.0%,
    12/15). On the synthetic benchmark (5 regimes, 60 seeds each) it is
    statistically indistinguishable from plain post-prune scoring -- no
    regression -- since that benchmark's noise is sparse per-node, giving
    the concentration effect little room to act either way.

    `blocked` need not come from any specific function above -- any
    edge subset works, since this only changes how transition
    probabilities are normalized, not which edges are considered removed.
    """
    raw: dict[str, dict[str, float]] = {}
    for a, b in edges:
        raw.setdefault(a, {})
        raw[a][b] = raw[a].get(b, 0.0) + 1.0
    adj: dict[str, list[tuple[str, float]]] = {}
    for u, nbrs in raw.items():
        deg = sum(nbrs.values()) or 1.0  # ORIGINAL degree -- includes blocked edges
        adj[u] = [(v, w / deg) for v, w in nbrs.items() if (u, v) not in blocked]

    x = {source: 1.0}
    out: dict[str, float] = {}
    coef = 1.0
    for _ in range(walk_len):
        nx: dict[str, float] = {}
        for u, xu in x.items():
            for v, p in adj.get(u, ()):
                nx[v] = nx.get(v, 0.0) + xu * p
        x = nx
        coef *= alpha
        for v, xv in x.items():
            if xv > 0:
                out[v] = out.get(v, 0.0) + coef * xv
    return out


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
    instead of the plain rule -- opt-in, not the default. Its magnet
    mechanism only ever fires for a target that already has an edge
    independently corroborated `min_evidence` (default 2) times; if your
    `labeled_pairs` never repeats the same query (each candidate edge
    backed by exactly one encounter -- the common case for an LLM's
    per-query shortcut claims), this is a no-op, same as plain
    `min_evidence=2` alone (see the module docstring's real-data
    discussion). Where it DOES have repeated corroboration to work with,
    it does not regress the synthetic benchmark.

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
