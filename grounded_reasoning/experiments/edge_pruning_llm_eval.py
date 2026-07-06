"""
Held-out-evidence edge pruning on REAL LLM hallucinations (DeepSeek), not
the synthetic dropout/spurious-edge simulation used in edge_pruning_eval.py.

Reuses nl_ontology_eval.py's dense-abstract-DAG scenario, which already
reliably collapses LLM precision on multi-hop composition (SGDC's premise:
LLMs are accurate on atomic facts, unreliable on composing them). Each of
the LLM's claimed transitive conclusions for a query "x relates to Z?" is
treated as a direct SHORTCUT EDGE x->Z added to the graph -- exactly the
kind of self-asserted multi-hop "fact" a self-grounded pipeline might trust
-- so the false ones are genuine, LLM-sourced spurious edges, not simulated
ones.

Run: DEEPSEEK_API_KEY=... python -m grounded_reasoning.experiments.edge_pruning_llm_eval

RESULT (n_seeds=4, top_k=10, 3 DeepSeek-call batches pooled into one
dataset of 1960 labeled pairs / 1242 edges, 69% hallucinated;
`collect_shortcut_claims` namespaces each seed's nodes to keep pooled
batches from colliding, since `build_dense_dag` draws from a fixed word
list regardless of seed): `identify_and_prune_edges`'s blocking decision
stays accurate on real hallucinations, matching the synthetic benchmark's
precision. But across 15 independent identify/eval random splits of that
same real data (splitting is free -- no extra API calls), cleaned FPR beat
raw FPR in only **11/15 (73%)** of splits (mean raw FPR 69.6% -> mean
cleaned FPR 63.5%) -- a real average improvement, but with genuine
per-split variance, unlike the near-universal win measured in every
synthetic regime in edge_pruning_eval.py.

Traced partly to `FuzzyInferenceEngine`'s row-normalized diffusion
(`P = D^-1 W`): a handful of hub nodes here carry many hallucinated
shortcut edges, and removing some of a node's edges can concentrate
transition probability onto whichever false edges remain -- unlike the
synthetic benchmark's sparse, locally-random noise, where this effect has
little room to act. `masked_infer` below is a mass-conserving alternative
that was tried against this same effect and did not resolve the
inconsistency (same 11/15 hit rate, on a mostly different subset of
splits) -- kept here as a tested, not-adopted alternative.

CONCLUSION: this mitigation's benefit is specific to the topology it was
measured on (locally-random 1-hop noise at moderate density). It still
helps more often than not on this densely-hallucinated, hub-heavy real
scenario, but not as reliably as on the synthetic benchmark, and should
not be assumed to generalize without separate validation on data
resembling the deployment's actual topology.

A deterministic (no ML, no fitted parameters) refinement,
`identify_suspect_edges_propagated` in `edge_pruning.py`, targets this
hub-heavy case directly: once one incoming edge to a target is confirmed
bad, treat that target as a "hallucination magnet" and lower the evidence
bar for its OTHER candidate incoming edges too. `run_propagation_comparison`
below is the analysis behind its docstring's claim: on this same real
data, it blocks ~2.6x more edges than plain pruning (9847 vs 3788 across
15 splits) while keeping the pooled wrongly-blocked rate low (3.4% vs
1.0%) and the SAME 11/15 split-reliability against raw, with a slightly
better mean FPR (62.0% vs 63.5%). A supervised (logistic regression)
classifier over these same features was ALSO tried, cross-validated for
stability across 5 independent training runs (consistent results), but
failed to generalize from synthetic training data to this real data at
all -- it blocked 0 edges outright until its per-feature normalization was
manually adapted to the real data's scale, and even then performed WORSE
than both raw and plain pruning (mean FPR 78.2%). Not adopted, and not
reflected in edge_pruning.py, since a fitted model that fails this badly
under distribution shift is a real generalization risk, not a validated
alternative -- kept here only as a cautionary result.
"""
from __future__ import annotations

import random

from grounded_reasoning.experiments.nl_ontology_eval import build_dense_dag, parse
from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine
from grounded_reasoning.reasoning.conformal_reasoning import conformal_threshold
from grounded_reasoning.reasoning.edge_pruning import identify_and_prune_edges


def collect_shortcut_claims(n_seeds: int, top_k: int, model: str, alpha: float = 0.1, seed_offset: int = 100):
    from grounded_reasoning.reasoning.llm_client import DeepSeekClient

    client = DeepSeekClient(model=model)
    base_edges: list[tuple[str, str]] = []
    shortcut_edges: list[tuple[str, str]] = []
    labeled_pairs: list[tuple[str, str, bool]] = []
    gold_edges: set[tuple[str, str]] = set()

    for seed in range(n_seeds):
        global_seed = seed_offset + seed
        alg, words, edges = build_dense_dag(seed=global_seed, n=22)
        # build_dense_dag always draws from the SAME fixed word list
        # (ABSTRACT_WORDS[:n]) regardless of seed -- only the edge structure
        # varies. Pooling multiple seeds' edges/labels into one graph WITHOUT
        # namespacing would let the same node name mean two different,
        # possibly contradictory things across seeds (e.g. "vora relates to
        # sigma" true in one seed's DAG, false in another's), corrupting the
        # pooled ground truth. Tag every node with its GLOBAL seed
        # (seed_offset + seed), not the local loop index -- calling this
        # function more than once with different seed_offsets (as done to
        # build a larger pooled dataset) would otherwise reuse the same
        # "s0_", "s1_", ... prefixes across calls, silently colliding two
        # DIFFERENT DAGs under one node identity (verified: build_dense_dag
        # produces genuinely different edge sets at different seeds despite
        # sharing the same fixed vocabulary).
        tag = lambda w: f"s{global_seed}_{w}"  # noqa: E731
        edges = [(tag(a), tag(b)) for a, b in edges]
        universe = {tag(w) for w in words}
        factstr = "\n".join(f"- {a} relates to {b}." for a, b in sorted(edges))
        base_edges.extend(edges)
        gold_edges |= set(edges)

        srcs = sorted({a for a, _ in edges}, key=lambda x: -len(alg.closure(x[len(f"s{global_seed}_"):], "relates to")))[:top_k]
        for x_tagged in srcs:
            x = x_tagged[len(f"s{global_seed}_"):]
            truth = {tag(t) for t in alg.closure(x, "relates to")}
            prompt = (
                f"Facts (use ONLY these):\n{factstr}\n\n"
                f"Rule: if A relates to B and B relates to C then A relates to C (transitive).\n"
                f'List EVERY Z such that "{x_tagged} relates to Z" is deducible (all levels). '
                f"JSON array only."
            )
            claimed = parse(client.ask(prompt, temperature=0.0), universe)
            for z in claimed:
                if z == x_tagged:
                    continue
                shortcut_edges.append((x_tagged, z))
                labeled_pairs.append((x_tagged, z, z in truth))

    return base_edges, shortcut_edges, labeled_pairs, gold_edges, client


def masked_infer(edges, blocked, source, walk_len: int = 8, alpha: float = 0.6):
    """
    A candidate fix for the row-normalization side effect described in the
    module docstring, TRIED and found NOT to resolve the inconsistency
    (same ~73% hit rate as topological pruning, on a mostly-different set
    of splits) -- kept here, not in edge_pruning.py, because it is not
    adopted: shipping it would suggest it's a validated improvement.

    Mirrors FuzzyInferenceEngine's diffusion exactly, except transition
    probabilities are normalized by each node's ORIGINAL out-degree
    (computed from the FULL `edges`, including blocked ones) rather than
    the degree of the pruned graph -- so removing a blocked edge only ever
    REMOVES its confidence mass, never redistributes it onto surviving
    edges from the same node.
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


def score_graph(edges, eval_pairs, walk_len: int, alpha_diffusion: float, alpha_conformal: float, seed: int = 0):
    eng = FuzzyInferenceEngine(walk_len=walk_len, alpha=alpha_diffusion)
    for a, b in edges:
        eng.add_relation(a, b)
    xs = {x for x, _, _ in eval_pairs}
    infc = {x: eng.infer(x) for x in xs}
    jit_rng = random.Random(seed + 777)
    scores = {(x, z): infc[x].get(z, 0.0) + jit_rng.uniform(0, 1e-9) for x, z, _ in eval_pairs}
    true_eval = [(x, z) for x, z, t in eval_pairs if t]
    false_eval = [(x, z) for x, z, t in eval_pairs if not t]
    if not true_eval or not false_eval:
        return None
    r = random.Random(seed + 555)
    tr = list(true_eval)
    r.shuffle(tr)
    h = len(tr) // 2
    cal, test = tr[:h], tr[h:]
    if not cal or not test:
        return None
    tau = conformal_threshold([scores[p] for p in cal], alpha_conformal)
    coverage = sum(1 for p in test if scores[p] >= tau) / len(test)
    fpr = sum(1 for p in false_eval if scores[p] >= tau) / len(false_eval)
    return coverage, fpr


def run(n_seeds: int = 4, top_k: int = 10, model: str = "deepseek-chat",
        alpha: float = 0.1, walk_len: int = 8, diffusion_alpha: float = 0.6,
        seed_offset: int = 100, verbose: bool = True) -> dict:
    base_edges, shortcut_edges, labeled_pairs, gold_edges, client = collect_shortcut_claims(
        n_seeds, top_k, model, alpha, seed_offset
    )
    n_true = sum(1 for _, _, t in labeled_pairs if t)
    n_false = len(labeled_pairs) - n_true

    full_edges = list(set(base_edges) | set(shortcut_edges))
    cleaned_edges, blocked, reserved_pairs = identify_and_prune_edges(
        full_edges, labeled_pairs, walk_len=walk_len, alpha=diffusion_alpha, seed=1
    )
    n_wrong = sum(1 for e in blocked if e in gold_edges)

    r_raw = score_graph(full_edges, reserved_pairs, walk_len, diffusion_alpha, alpha)
    r_clean = score_graph(cleaned_edges, reserved_pairs, walk_len, diffusion_alpha, alpha)

    res = {
        "n_shortcut_claims": len(shortcut_edges), "n_true": n_true, "n_false_hallucinated": n_false,
        "n_blocked": len(blocked), "n_wrongly_blocked": n_wrong,
        "llm_tokens": client.total_tokens,
    }
    if r_raw and r_clean:
        res.update({
            "raw_coverage": round(r_raw[0], 3), "raw_fpr": round(r_raw[1], 3),
            "cleaned_coverage": round(r_clean[0], 3), "cleaned_fpr": round(r_clean[1], 3),
        })
    if verbose:
        import json
        print(json.dumps(res, indent=2))
        if r_raw and r_clean:
            verdict = "IMPROVED" if r_clean[1] < r_raw[1] else "DID NOT IMPROVE (see module docstring)"
            print(
                f"\n{n_false}/{len(shortcut_edges)} of the LLM's shortcut claims were hallucinated. "
                f"Blocking decision: {res['n_blocked']} flagged, {n_wrong} wrongly ({n_wrong/max(res['n_blocked'],1):.1%}).\n"
                f"Downstream FPR: raw={r_raw[1]:.1%} -> cleaned={r_clean[1]:.1%} -- {verdict}."
            )
    return res


def run_multi_split(n_seeds: int = 4, top_k: int = 10, model: str = "deepseek-chat",
                     alpha: float = 0.1, walk_len: int = 8, diffusion_alpha: float = 0.6,
                     seed_offset: int = 100, n_splits: int = 15, verbose: bool = True) -> dict:
    """
    The analysis behind this module's documented RESULT: fetches real
    LLM-hallucinated shortcut claims ONCE, then evaluates `n_splits`
    different random identify/eval splits of that SAME data (free -- no
    extra API calls) for both topological pruning (identify_and_prune_edges
    + a fresh engine on the cleaned edge list) and `masked_infer`, comparing
    each against the raw (unpruned) graph on every split.
    """
    base_edges, shortcut_edges, labeled_pairs, gold_edges, client = collect_shortcut_claims(
        n_seeds, top_k, model, alpha, seed_offset
    )
    full_edges = list(set(base_edges) | set(shortcut_edges))

    topo_beats_raw = masked_beats_raw = n_valid = 0
    raw_fprs, topo_fprs, masked_fprs = [], [], []
    for split_seed in range(n_splits):
        cleaned_edges, blocked, reserved_pairs = identify_and_prune_edges(
            full_edges, labeled_pairs, walk_len=walk_len, alpha=diffusion_alpha, seed=split_seed
        )
        r_raw = score_graph(full_edges, reserved_pairs, walk_len, diffusion_alpha, alpha, split_seed)
        r_topo = score_graph(cleaned_edges, reserved_pairs, walk_len, diffusion_alpha, alpha, split_seed)
        xs = {x for x, _, _ in reserved_pairs}
        infc_masked = {x: masked_infer(full_edges, blocked, x, walk_len, diffusion_alpha) for x in xs}
        r_masked = _score_from_infc(infc_masked, reserved_pairs, alpha, split_seed)
        if r_raw is None or r_topo is None or r_masked is None:
            continue
        n_valid += 1
        raw_fprs.append(r_raw[1])
        topo_fprs.append(r_topo[1])
        masked_fprs.append(r_masked[1])
        topo_beats_raw += r_topo[1] < r_raw[1]
        masked_beats_raw += r_masked[1] < r_raw[1]

    res = {
        "n_valid_splits": n_valid,
        "topo_beats_raw": topo_beats_raw, "masked_beats_raw": masked_beats_raw,
        "mean_raw_fpr": round(sum(raw_fprs) / n_valid, 3),
        "mean_topo_fpr": round(sum(topo_fprs) / n_valid, 3),
        "mean_masked_fpr": round(sum(masked_fprs) / n_valid, 3),
        "llm_tokens": client.total_tokens,
    }
    if verbose:
        import json
        print(json.dumps(res, indent=2))
        print(
            f"\nAcross {n_valid} random identify/eval splits of the same real LLM-hallucinated "
            f"data:\n topological pruning beat raw in {topo_beats_raw}/{n_valid}; "
            f"masked_infer beat raw in {masked_beats_raw}/{n_valid}."
        )
    return res


def run_propagation_comparison(n_seeds: int = 4, top_k: int = 10, model: str = "deepseek-chat",
                                alpha: float = 0.1, walk_len: int = 8, diffusion_alpha: float = 0.6,
                                seed_offset: int = 100, n_splits: int = 15, verbose: bool = True) -> dict:
    """
    The analysis behind `identify_suspect_edges_propagated`'s docstring
    claim: fetches real LLM-hallucinated shortcut claims ONCE, then
    compares plain topological pruning (`use_propagation=False`) against
    the propagated variant (`use_propagation=True`) across `n_splits`
    random identify/eval splits of that SAME data (free -- no extra API
    calls), on both downstream FPR and the wrongly-blocked rate (pooled
    across splits, since this data's hub-heavy topology is exactly where
    propagation is expected to matter).
    """
    base_edges, shortcut_edges, labeled_pairs, gold_edges, client = collect_shortcut_claims(
        n_seeds, top_k, model, alpha, seed_offset
    )
    full_edges = list(set(base_edges) | set(shortcut_edges))

    topo_beats_raw = prop_beats_raw = n_valid = 0
    raw_fprs, topo_fprs, prop_fprs = [], [], []
    topo_blocked_total = topo_wrong_total = prop_blocked_total = prop_wrong_total = 0
    for split_seed in range(n_splits):
        cleaned_topo, blocked_topo, reserved_pairs = identify_and_prune_edges(
            full_edges, labeled_pairs, walk_len=walk_len, alpha=diffusion_alpha, seed=split_seed
        )
        cleaned_prop, blocked_prop, _ = identify_and_prune_edges(
            full_edges, labeled_pairs, walk_len=walk_len, alpha=diffusion_alpha, seed=split_seed,
            use_propagation=True,
        )
        topo_blocked_total += len(blocked_topo)
        topo_wrong_total += sum(1 for e in blocked_topo if e in gold_edges)
        prop_blocked_total += len(blocked_prop)
        prop_wrong_total += sum(1 for e in blocked_prop if e in gold_edges)

        r_raw = score_graph(full_edges, reserved_pairs, walk_len, diffusion_alpha, alpha, split_seed)
        r_topo = score_graph(cleaned_topo, reserved_pairs, walk_len, diffusion_alpha, alpha, split_seed)
        r_prop = score_graph(cleaned_prop, reserved_pairs, walk_len, diffusion_alpha, alpha, split_seed)
        if r_raw is None or r_topo is None or r_prop is None:
            continue
        n_valid += 1
        raw_fprs.append(r_raw[1])
        topo_fprs.append(r_topo[1])
        prop_fprs.append(r_prop[1])
        topo_beats_raw += r_topo[1] < r_raw[1]
        prop_beats_raw += r_prop[1] < r_raw[1]

    res = {
        "n_valid_splits": n_valid,
        "topo_beats_raw": topo_beats_raw, "prop_beats_raw": prop_beats_raw,
        "mean_raw_fpr": round(sum(raw_fprs) / n_valid, 3),
        "mean_topo_fpr": round(sum(topo_fprs) / n_valid, 3),
        "mean_prop_fpr": round(sum(prop_fprs) / n_valid, 3),
        "topo_n_blocked": topo_blocked_total, "topo_wrongly_blocked_rate": round(topo_wrong_total / max(topo_blocked_total, 1), 3),
        "prop_n_blocked": prop_blocked_total, "prop_wrongly_blocked_rate": round(prop_wrong_total / max(prop_blocked_total, 1), 3),
        "llm_tokens": client.total_tokens,
    }
    if verbose:
        import json
        print(json.dumps(res, indent=2))
        print(
            f"\nAcross {n_valid} random identify/eval splits of the same real LLM-hallucinated data:\n"
            f" plain pruning: beat raw {topo_beats_raw}/{n_valid}, blocked {topo_blocked_total} edges "
            f"({res['topo_wrongly_blocked_rate']:.1%} wrongly).\n"
            f" propagated:    beat raw {prop_beats_raw}/{n_valid}, blocked {prop_blocked_total} edges "
            f"({res['prop_wrongly_blocked_rate']:.1%} wrongly)."
        )
    return res


def _score_from_infc(infc, eval_pairs, alpha_conformal, seed):
    jit_rng = random.Random(seed + 777)
    scores = {(x, z): infc[x].get(z, 0.0) + jit_rng.uniform(0, 1e-9) for x, z, _ in eval_pairs}
    true_eval = [(x, z) for x, z, t in eval_pairs if t]
    false_eval = [(x, z) for x, z, t in eval_pairs if not t]
    if not true_eval or not false_eval:
        return None
    r = random.Random(seed + 555)
    tr = list(true_eval)
    r.shuffle(tr)
    h = len(tr) // 2
    cal, test = tr[:h], tr[h:]
    if not cal or not test:
        return None
    tau = conformal_threshold([scores[p] for p in cal], alpha_conformal)
    coverage = sum(1 for p in test if scores[p] >= tau) / len(test)
    fpr = sum(1 for p in false_eval if scores[p] >= tau) / len(false_eval)
    return coverage, fpr


if __name__ == "__main__":
    run()
