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

RESULT (verified, n_seeds=4, top_k=10, 3 independent trials with disjoint
ontology seeds -- real DeepSeek calls, ~40 per trial): 65-73% of the LLM's
shortcut claims were hallucinated in every trial (488/744, 586/805,
553/761). identify_and_prune_edges's BLOCKING DECISION was consistently
accurate -- 3.1%, 3.9%, 3.7% wrongly blocked -- matching the synthetic
benchmark's precision. But the DOWNSTREAM effect on cleaned FPR was
INCONSISTENT, unlike every synthetic regime in edge_pruning_eval.py (which
improved in 5/5 regimes, every pooled seed):

| Trial | Raw FPR | Cleaned FPR | Result |
|---|---:|---:|---|
| 0 | 62.9% | 84.3% | WORSE |
| 1 | 89.4% | 70.2% | better |
| 2 | 95.0% | 75.0% | better |

2 of 3 improved, 1 regressed -- not the clean, consistent win seen
synthetically. Reported as-is, not averaged into a single reassuring
number.

Root cause (confirmed by inspecting score distributions on a smaller
repro): this scenario's topology is qualitatively different from the
synthetic benchmark's sparse, locally-random noise -- a handful of hub
nodes here carry many hallucinated shortcut edges. FuzzyInferenceEngine's
diffusion is row-normalized (P = D^-1 W); removing some of a node's
outgoing edges concentrates transition probability onto whichever edges
REMAIN, including any still-false ones that weren't blocked (blocking is
strict: skipped if it ever saw a true-claim vote or too few false-claim
votes). That redistribution can raise remaining false scores enough to
outweigh the direct benefit of removing the edges that WERE blocked, and
whether it does depends on which specific edges survive per trial -- hence
the inconsistency. The same row-normalization side effect was
independently observed earlier in this project's exploration of
reinforcement-style edge weighting.

CONCLUSION: identify_and_prune_edges's benefit, as measured across every
regime in edge_pruning_eval.py, is specific to that regime's topology
(locally-random 1-hop noise at moderate density). It should NOT be assumed
to generalize to a densely-hallucinated multi-hop-shortcut regime like this
one without separate validation -- here the effect on downstream FPR is
inconsistent (worse in 1 of 3 real trials) despite the blocking decision
itself being accurate. This is recorded as a real, measured limitation, not
swept under a "still helps on average" claim.
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
        universe = set(words)
        factstr = "\n".join(f"- {a} relates to {b}." for a, b in sorted(edges))
        base_edges.extend(edges)
        gold_edges |= set(edges)

        srcs = sorted({a for a, _ in edges}, key=lambda x: -len(alg.closure(x, "relates to")))[:top_k]
        for x in srcs:
            truth = alg.closure(x, "relates to")
            prompt = (
                f"Facts (use ONLY these):\n{factstr}\n\n"
                f"Rule: if A relates to B and B relates to C then A relates to C (transitive).\n"
                f'List EVERY Z such that "{x} relates to Z" is deducible (all levels). '
                f"JSON array only."
            )
            claimed = parse(client.ask(prompt, temperature=0.0), universe)
            for z in claimed:
                if z == x:
                    continue
                shortcut_edges.append((x, z))
                labeled_pairs.append((x, z, z in truth))

    return base_edges, shortcut_edges, labeled_pairs, gold_edges, client


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


if __name__ == "__main__":
    run()
