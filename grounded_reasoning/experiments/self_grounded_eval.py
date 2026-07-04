"""
Self-Grounded Deductive Consistency (SGDC) — hallucination verification WITHOUT
external knowledge, exploiting the LLM's confidence asymmetry (atomic facts are
reliable, compositions hallucinate).

Protocol (2 LLM calls, ground truth is NEVER used to filter):
  1. Ask the LLM for the ATOMIC (1-hop) FACTS of a world — the step the LLM is RELIABLE at.
  2. Ask the LLM for MULTI-HOP CONCLUSIONS (transitive closure) — the step the LLM
     tends to HALLUCINATE at.
  3. Build an operator from the LLM's OWN atomic facts; compute the certified closure.
  4. SGDC: reject any multi-hop conclusion that falls OUTSIDE the LLM's own closure.

Evaluation (ground truth is used ONLY for scoring, NEVER for filtering): does SGDC
recover the same precision as using an external graph? If so ⟹ no external graph
is needed (removes a core limitation).

Survival condition (falsifiable): atomic-fact precision must exceed multi-hop precision.
If atomic facts also hallucinate heavily ⟹ SGDC is useless (recorded honestly).

Run: DEEPSEEK_API_KEY=... python -m grounded_reasoning.experiments.self_grounded_eval
"""
from __future__ import annotations

import json
import re

from grounded_reasoning.reasoning.operator_algebra import OperatorRelationAlgebra
from grounded_reasoning.reasoning.relation_spectrum import is_acyclic, spectral_radius

# The REAL closed world (ground truth for SCORING; never used to filter prompts).
GROUND = [
    ("sparrow", "bird"), ("bird", "vertebrate"), ("vertebrate", "animal"),
    ("animal", "organism"), ("organism", "entity"),
    ("salmon", "fish"), ("fish", "vertebrate"),
    ("oak", "tree"), ("tree", "plant"), ("plant", "organism"),
    ("whale", "fish"),  # anti-commonsense trap
]


def _truth_alg():
    alg = OperatorRelationAlgebra()
    for a, b in GROUND:
        alg.add(a, "is a", b)
    return alg


def _universe():
    u = set()
    for a, b in GROUND:
        u |= {a, b}
    return u


def _parse_pairs(text, universe):
    """Extract [x,y] pairs meaning 'x is a y' from JSON."""
    m = re.search(r"\[.*\]", text, re.S)
    pairs = []
    if m:
        try:
            for it in json.loads(m.group(0)):
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    a, b = str(it[0]).strip().lower(), str(it[1]).strip().lower()
                    if a in universe and b in universe and a != b:
                        pairs.append((a, b))
        except Exception:
            pass
    return pairs


def run(model: str = "deepseek-chat", verbose: bool = True):
    from grounded_reasoning.reasoning.llm_client import DeepSeekClient

    truth = _truth_alg()
    universe = _universe()
    concepts = sorted(universe)
    client = DeepSeekClient(model=model)

    # ---- 1. LLM provides ATOMIC FACTS (1-hop) ----
    atom_prompt = (
        "Consider these concepts: " + ", ".join(concepts) + ".\n"
        "State ONLY the DIRECT, one-step 'is a' facts among them that you are confident "
        "about (immediate category, no transitive steps). "
        'Answer as a JSON array of [x, y] pairs meaning "x is a y". e.g. [["sparrow","bird"]].'
    )
    atom_pairs = _parse_pairs(client.ask(atom_prompt, temperature=0.0), universe)

    # SELF-GROUNDED graph from the LLM's own facts
    llm_alg = OperatorRelationAlgebra()
    for a, b in atom_pairs:
        llm_alg.add(a, "is a", b)

    # atomic-fact reliability (scored against the 1-hop ground truth)
    gt_atoms = set(GROUND)
    atom_tp = len(set(atom_pairs) & gt_atoms)
    atom_fp = len(set(atom_pairs) - gt_atoms)
    atom_prec = atom_tp / max(atom_tp + atom_fp, 1)

    # ---- 2. LLM provides MULTI-HOP CONCLUSIONS (transitive closure) ----
    srcs = sorted({a for a, _ in GROUND})
    multi_claims: dict[str, set[str]] = {}
    for x in srcs:
        p = (
            "Using ONLY real-world common sense about these concepts "
            + ", ".join(concepts) + ",\n"
            f'list EVERY category Z such that "{x} is a Z" holds transitively '
            f"(all levels up). JSON array of names only."
        )
        m = re.search(r"\[.*?\]", client.ask(p, temperature=0.0), re.S)
        s = set()
        if m:
            try:
                s = {str(z).strip().lower() for z in json.loads(m.group(0))} & universe
            except Exception:
                pass
        multi_claims[x] = s

    # ---- 3+4. score: raw vs SGDC (filtered by the LLM's OWN closure) vs external ----
    def score(filter_alg):
        tp = fp = fn = 0
        for x in srcs:
            tr = truth.closure(x, "is a")
            claimed = multi_claims[x]
            if filter_alg is None:
                kept = claimed
            else:
                kept = {c for c in claimed if c in filter_alg.closure(x, "is a")}
            tp += len(kept & tr)
            fp += len(kept - tr)
            fn += len(tr - kept)
        return tp / max(tp + fp, 1), tp / max(tp + fn, 1)

    raw_p, raw_r = score(None)
    sgdc_p, sgdc_r = score(llm_alg)      # filtered by the LLM's OWN graph (no external knowledge)
    ext_p, ext_r = score(truth)          # filtered by the external graph (upper bound)

    # ---- spectral: did the LLM assert a contradictory cycle on its own? ----
    A = llm_alg.operator("is a").astype(float).T if atom_pairs else None
    contradiction_cycle = (not is_acyclic(A)) if A is not None and A.size else False
    rho = round(spectral_radius(A), 6) if A is not None and A.size else 0.0

    res = {
        "atomic_precision": round(atom_prec, 4),
        "atomic_facts_llm": len(atom_pairs),
        "raw_multi_precision": round(raw_p, 4), "raw_multi_recall": round(raw_r, 4),
        "sgdc_precision": round(sgdc_p, 4), "sgdc_recall": round(sgdc_r, 4),
        "external_precision": round(ext_p, 4), "external_recall": round(ext_r, 4),
        "llm_asserted_contradiction_cycle": contradiction_cycle,
        "spectral_radius_llm_graph": rho,
        "survival_condition": atom_prec > raw_p,  # SGDC's survival condition
    }
    if verbose:
        print(json.dumps(res, indent=2))
        print(
            f"\nAsymmetry: atomic-fact precision={atom_prec:.0%} vs raw multi-hop="
            f"{raw_p:.0%}.  SGDC (self-grounded, 0 external knowledge): precision "
            f"{raw_p:.0%}→{sgdc_p:.0%} (external upper bound={ext_p:.0%}).  "
            f"{'✔ SGDC effective' if res['survival_condition'] and sgdc_p >= raw_p else '✘ failure recorded'}"
        )
    return res


if __name__ == "__main__":
    run()
