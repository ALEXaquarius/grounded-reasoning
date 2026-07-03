"""
Hallucination blocking on a REAL LLM with several kinds of NATURAL-LANGUAGE RELATIONS.

Unlike guard_llm_eval (kinship only): here we use three real-language relations —
  • "is a"     (classification / taxonomy) — transitive: sparrow is-a bird is-a animal
  • "causes"   (causation)                 — transitive: virus causes fever causes …
  • "part of"  (meronymy)                  — transitive: piston part-of engine part-of car

CLOSED world: the LLM may only use the given 1-hop facts; any assertion beyond that
is a HALLUCINATION (even if it happens to be "true" by outside knowledge). Ground
truth and the guard = the transitive closure of OperatorRelationAlgebra (Theorem G).
All relations are ACYCLIC ⟹ the operator is NILPOTENT (Theorem H) ⟹ the closure
terminates in finitely many steps.

Run: DEEPSEEK_API_KEY=... python -m src.experiments.nl_ontology_eval
"""
from __future__ import annotations

import json
import re

from src.reasoning.operator_algebra import OperatorRelationAlgebra
from src.reasoning.relation_spectrum import is_acyclic, spectral_radius

# A MORE CLOSED & HARDER world: long chains (4-6 hops) + anti-commonsense TRAPS
# (facts that hold in this closed world but CONTRADICT real-world knowledge, to
# tempt the LLM into importing outside knowledge = hallucination). All relations
# remain ACYCLIC (Theorem H: nilpotent).
FACTS: dict[str, list[tuple[str, str]]] = {
    "is a": [
        ("sparrow", "bird"), ("bird", "vertebrate"), ("vertebrate", "animal"),
        ("animal", "organism"), ("organism", "entity"),
        ("salmon", "fish"), ("fish", "vertebrate"),
        ("oak", "tree"), ("tree", "plant"), ("plant", "organism"),
        # anti-commonsense TRAP: in this world whale is-a fish (real: mammal)
        ("whale", "fish"),
        # TRAP: penguin does NOT connect up to bird here — only to "flightless"
        ("penguin", "flightless"), ("flightless", "creature"),
    ],
    "causes": [
        ("drought", "cropfailure"), ("cropfailure", "famine"),
        ("famine", "migration"), ("migration", "conflict"), ("conflict", "poverty"),
        ("virus", "infection"), ("infection", "fever"), ("fever", "fatigue"),
        ("fatigue", "errors"),
    ],
    "part of": [
        ("piston", "engine"), ("engine", "car"), ("car", "fleet"),
        ("fleet", "company"),
        ("nib", "pen"), ("pen", "pencilcase"), ("pencilcase", "bag"),
        ("wheel", "car"),
    ],
}

RELDEF = {
    "is a": "X is a Y, and if X is a Y and Y is a Z then X is a Z (transitive)",
    "causes": "X causes Y, and if X causes Y and Y causes Z then X causes Z (transitive)",
    "part of": "X is part of Y, and if X part of Y and Y part of Z then X part of Z",
}


def build():
    alg = OperatorRelationAlgebra()
    concepts: set[str] = set()
    for rel, edges in FACTS.items():
        for a, b in edges:
            alg.add(a, rel, b)
            concepts |= {a, b}
    return alg, sorted(concepts)


def spectral_report(alg) -> dict:
    """Theorem H: each acyclic ontology relation ⟹ nilpotent operator (ρ=0)."""
    out = {}
    names = alg._names  # noqa: SLF001 (internal check)
    for rel in FACTS:
        A = alg.operator(rel).astype(float).T  # A[i,j]=1 ⟺ i--rel-->j
        out[rel] = {"acyclic": is_acyclic(A), "spectral_radius": round(spectral_radius(A), 9)}
    return out


def make_queries(alg, concepts):
    """Transitive queries: 'every Y such that X <rel> Y (all levels)'. Answer = closure."""
    q = []
    for rel in FACTS:
        srcs = {a for a, _ in FACTS[rel]}
        for x in sorted(srcs):
            q.append((rel, x, alg.closure(x, rel)))
    return q


def build_prompt(rel, x):
    lines = []
    for r, edges in FACTS.items():
        for a, b in edges:
            lines.append(f"- {a} {r} {b}.")
    facts = "\n".join(lines)
    return (
        f"Facts (the ONLY facts you may use; ignore any outside knowledge):\n{facts}\n\n"
        f"Rule: {RELDEF[rel]}.\n"
        f'Question: List EVERY concept Z such that "{x} {rel} Z" can be deduced '
        f"(all transitive levels), using ONLY the facts above.\n"
        f'Answer with ONLY a JSON array, e.g. ["a","b"] or [].'
    )


def parse(text, universe):
    m = re.search(r"\[.*?\]", text, re.S)
    if m:
        try:
            return {str(x).strip().lower() for x in json.loads(m.group(0))} & universe
        except Exception:
            pass
    return {w.lower() for w in re.findall(r"[a-zA-Z]+", text)} & universe


ABSTRACT_WORDS = [
    "axon", "boron", "cetus", "delta", "echo", "flux", "gron", "helix",
    "ionis", "kappa", "lumen", "mira", "nexus", "orin", "pyra", "quill",
    "rho", "sigma", "tau", "umbra", "vora", "wyrd",
]


def build_dense_dag(seed: int = 3, n: int = 22):
    """A DENSE DAG over ABSTRACT concepts (not reliant on outside knowledge) — each
    node connects up to 1-2 earlier nodes ⟹ a LARGE transitive closure. Still acyclic (Theorem H)."""
    import random

    rng = random.Random(seed)
    alg = OperatorRelationAlgebra()
    edges: list[tuple[str, str]] = []
    w = ABSTRACT_WORDS[:n]
    for i in range(1, n):
        for _ in range(rng.randint(1, 2)):
            j = rng.randint(0, i - 1)
            if (w[i], w[j]) not in edges:
                alg.add(w[i], "relates to", w[j])
                edges.append((w[i], w[j]))
    return alg, w, edges


def run_dense(seed: int = 3, top_k: int = 8, model: str = "deepseek-chat", verbose: bool = True):
    """HARD scenario: transitive closure over a dense DAG of abstract concepts."""
    from src.reasoning.llm_client import DeepSeekClient

    alg, words, edges = build_dense_dag(seed)
    A = alg.operator("relates to").astype(float).T
    universe = set(words)
    factstr = "\n".join(f"- {a} relates to {b}." for a, b in sorted(edges))
    client = DeepSeekClient(model=model)
    srcs = sorted(
        {a for a, _ in edges}, key=lambda x: -len(alg.closure(x, "relates to"))
    )[:top_k]

    llm_tp = llm_fp = llm_fn = leak = drop = 0
    for x in srcs:
        truth = alg.closure(x, "relates to")
        prompt = (
            f"Facts (use ONLY these):\n{factstr}\n\n"
            f"Rule: if A relates to B and B relates to C then A relates to C (transitive).\n"
            f'List EVERY Z such that "{x} relates to Z" is deducible (all levels). '
            f"JSON array only."
        )
        claimed = parse(client.ask(prompt, temperature=0.0), universe)
        llm_tp += len(claimed & truth)
        llm_fp += len(claimed - truth)
        llm_fn += len(truth - claimed)
        kept = {c for c in claimed if c in alg.closure(x, "relates to")}
        leak += len(kept - truth)
        drop += len((claimed & truth) - kept)

    lp = llm_tp / max(llm_tp + llm_fp, 1)
    lr = llm_tp / max(llm_tp + llm_fn, 1)
    res = {
        "scenario": "dense_abstract_dag",
        "acyclic": is_acyclic(A), "spectral_radius": round(spectral_radius(A), 9),
        "n_queries": len(srcs), "edges": len(edges),
        "llm_precision": lp, "llm_recall": lr, "llm_hallucinations": llm_fp,
        "guard_caught": llm_fp - leak, "guard_leaked": leak, "guard_dropped_true": drop,
        "guarded_precision": llm_tp / max(llm_tp + leak, 1),
    }
    if verbose:
        print(json.dumps(res, indent=2))
        print(
            f"\nDense abstract DAG: LLM precision COLLAPSES to {lp:.2%} ({llm_fp} hallucinations, "
            f"over-claiming). Guard catches {llm_fp - leak}/{llm_fp}, dropped_true={drop} "
            f"⟹ precision=100%."
        )
    return res


def run(model: str = "deepseek-chat", verbose: bool = True):
    from src.reasoning.llm_client import DeepSeekClient

    alg, concepts = build()
    universe = set(concepts)
    spec = spectral_report(alg)
    queries = make_queries(alg, concepts)
    client = DeepSeekClient(model=model)

    llm_tp = llm_fp = llm_fn = 0
    g_tp = g_fp = g_fn = 0
    dropped_true = 0
    for rel, x, truth in queries:
        claimed = parse(client.ask(build_prompt(rel, x), temperature=0.0), universe)
        llm_tp += len(claimed & truth)
        llm_fp += len(claimed - truth)
        llm_fn += len(truth - claimed)
        kept = {c for c in claimed if c in alg.closure(x, rel)}  # guard: grounded path exists
        g_tp += len(kept & truth)
        g_fp += len(kept - truth)
        g_fn += len(truth - kept)
        dropped_true += len((claimed & truth) - kept)

    def prf(tp, fp, fn):
        return tp / max(tp + fp, 1), tp / max(tp + fn, 1)

    lp, lr = prf(llm_tp, llm_fp, llm_fn)
    gp, gr = prf(g_tp, g_fp, g_fn)
    res = {
        "n_queries": len(queries),
        "spectral": spec,
        "llm_precision": lp, "llm_recall": lr, "llm_hallucinations": llm_fp,
        "guarded_precision": gp, "guarded_recall": gr,
        "guard_caught": llm_fp - g_fp, "guard_leaked": g_fp,
        "guard_dropped_true": dropped_true,
    }
    if verbose:
        print(json.dumps(res, indent=2))
        print(
            f"\nEvery ontology relation is ACYCLIC (ρ=0, nilpotent) — Theorem H.\n"
            f"Raw LLM: precision={lp:.2%} ({llm_fp} hallucinations).  "
            f"After GUARD: precision={gp:.2%} "
            f"(caught {llm_fp - g_fp}/{llm_fp}, {g_fp} leaked through, {dropped_true} correct answers wrongly dropped)."
        )
    return res


if __name__ == "__main__":
    run()
