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

CORRECTNESS NOTE on an earlier version of this result: `build_dense_dag`
draws node names from the SAME fixed word list regardless of seed -- only
edge structure varies by seed. An earlier version of `collect_shortcut_claims`
pooled multiple seeds' edges/labels into one graph WITHOUT namespacing nodes
per seed, so the same node name could carry contradictory ground truth
across seeds (e.g. "vora relates to sigma" true in one seed's DAG, false in
another's) once merged -- corrupting the pooled dataset. Fixed by tagging
every node with its seed. The result below is from the CORRECTED version;
it changed the specific numbers but not the qualitative conclusion.

RESULT (verified on corrected data, n_seeds=4, top_k=10, 3 independent
DeepSeek-call batches pooled into one dataset of 1960 labeled pairs / 1242
edges, 1348/1960 = 69% hallucinated): identify_and_prune_edges's BLOCKING
DECISION stays accurate on real hallucinations. But across 15 independent
identify/eval RANDOM SPLITS of that same real data (no extra API calls --
splitting is free), cleaned FPR beat raw FPR in only **11/15 (73%)** of
splits (mean raw FPR 69.6% -> mean cleaned FPR 63.5%) -- a real average
improvement, but with genuine per-split variance, unlike the near-universal
win measured in every synthetic regime in edge_pruning_eval.py. The 4
splits where cleaning did NOT help were exactly the splits where the raw
graph's FPR on that particular reserved-eval slice happened to already be
low (30-45%, vs. a typical 60-90%) -- less headroom, more downside risk
from the pruning decision's own noise.

Root cause (confirmed by inspecting score distributions on a smaller
repro): this scenario's topology is qualitatively different from the
synthetic benchmark's sparse, locally-random noise -- a handful of hub
nodes here carry many hallucinated shortcut edges. FuzzyInferenceEngine's
diffusion is row-normalized (P = D^-1 W); removing some of a node's
outgoing edges concentrates transition probability onto whichever edges
REMAIN, including any still-false ones that weren't blocked (blocking is
strict: skipped if it ever saw a true-claim vote or too few false-claim
votes). That redistribution can raise remaining false scores enough to
outweigh the direct benefit of removing the edges that WERE blocked. The
same row-normalization side effect was independently observed earlier in
this project's exploration of reinforcement-style edge weighting.

A candidate fix was tried and did NOT resolve it: `masked_infer` (mass-
conserving inference -- normalize by each node's ORIGINAL out-degree,
including blocked edges, so pruning only ever REMOVES confidence mass
instead of ever redistributing it onto surviving edges) was tested
against the same 15 splits. On the synthetic benchmark it reproduces
topological pruning's numbers almost exactly (as expected -- low-out-degree
nodes there give the redistribution effect little room to act). On this
real, hub-heavy hallucination data it ALSO beat raw in 11/15 splits --
identical hit rate to plain topological pruning, on a mostly-different set
of splits. So the row-normalization mechanism is A real contributor, but
not THE whole explanation -- something about this topology's variance
across splits is more fundamental than that one side effect, and remains
unresolved. Not shipped; recorded as a tried-and-inconclusive direction.

CONCLUSION: identify_and_prune_edges's benefit, as measured across every
regime in edge_pruning_eval.py, is specific to that regime's topology
(locally-random 1-hop noise at moderate density). It still helps MORE
OFTEN THAN NOT on this densely-hallucinated, hub-heavy real scenario
(11/15 splits, mean FPR improves), but not with the same reliability as the
synthetic benchmark, and should not be assumed to generalize without
separate validation on data resembling the deployment's actual topology.
This is recorded as a real, measured limitation, not swept under a "still
helps on average" claim.
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
        alg, words, edges = build_dense_dag(seed=seed_offset + seed, n=22)
        # build_dense_dag always draws from the SAME fixed word list
        # (ABSTRACT_WORDS[:n]) regardless of seed -- only the edge structure
        # varies. Pooling multiple seeds' edges/labels into one graph WITHOUT
        # namespacing would let the same node name mean two different,
        # possibly contradictory things across seeds (e.g. "vora relates to
        # sigma" true in one seed's DAG, false in another's), corrupting the
        # pooled ground truth. Tag every node with its seed to keep each
        # seed's DAG a fully disjoint subgraph.
        tag = lambda w: f"s{seed}_{w}"  # noqa: E731
        edges = [(tag(a), tag(b)) for a, b in edges]
        universe = {tag(w) for w in words}
        factstr = "\n".join(f"- {a} relates to {b}." for a, b in sorted(edges))
        base_edges.extend(edges)
        gold_edges |= set(edges)

        srcs = sorted({a for a, _ in edges}, key=lambda x: -len(alg.closure(x[len(f"s{seed}_"):], "relates to")))[:top_k]
        for x_tagged in srcs:
            x = x_tagged[len(f"s{seed}_"):]
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
