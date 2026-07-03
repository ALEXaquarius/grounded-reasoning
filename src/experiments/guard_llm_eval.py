"""
HALLUCINATION-BLOCKING experiment on a REAL LLM (DeepSeek) using the grounded inference engine.

Protocol:
  1. Generate a relation tree (kinship) and give the LLM only the 1-HOP FACTS (parent).
  2. Ask the LLM MULTI-HOP inference questions (grandparent, great-grandparent,
     ancestor, sibling-by-analogy) — including TRAP questions whose correct answer is EMPTY.
  3. The LLM answers (JSON). We have GROUND TRUTH from OperatorRelationAlgebra (Theorem G).
  4. HallucinationGuard accepts a name proposed by the LLM iff a grounded path exists.

Measures: (i) how much the LLM hallucinates on its own; (ii) how many hallucinations
the guard catches while NOT dropping any correct answer (precision must = 1.0 by Theorem G).

Run: DEEPSEEK_API_KEY=... python -m src.experiments.guard_llm_eval
(reads the key from .env/environment variable; never hardcode it).
"""
from __future__ import annotations

import json
import random
import re

from src.reasoning.operator_algebra import OperatorRelationAlgebra


def build_family(seed: int = 0):
    """Multi-generation kinship tree. Returns (facts[(child,parent)], alg, names)."""
    rng = random.Random(seed)
    names = [
        "Al", "Bo", "Cy", "Di", "Ed", "Fi", "Gu", "Ha",
        "Io", "Ju", "Ka", "Lu", "Mo", "Ni", "Op", "Pa",
    ]
    rng.shuffle(names)
    alg = OperatorRelationAlgebra()
    facts: list[tuple[str, str]] = []
    # assign each person (except the 4 roots) a parent from the "older" group → a tiered tree
    for i, child in enumerate(names):
        if i >= 4:
            parent = names[rng.randint(0, i - 1)]
            alg.add(child, "parent", parent)
            facts.append((child, parent))
    return facts, alg, names


def make_queries(alg: OperatorRelationAlgebra, names: list[str]):
    """Multi-hop inference questions + correct (grounded) answers."""
    q = []
    for person in names:
        q.append(("grandparent", person, alg.follow(person, ["parent", "parent"])))
        q.append((
            "great-grandparent",
            person,
            alg.follow(person, ["parent", "parent", "parent"]),
        ))
        q.append(("ancestor", person, alg.closure(person, "parent")))
    return q


def build_prompt(facts, kind, person):
    lines = "\n".join(f"- {c}'s parent is {p}." for c, p in facts)
    defs = {
        "grandparent": "a grandparent is a parent of a parent",
        "great-grandparent": "a great-grandparent is a parent of a grandparent",
        "ancestor": "an ancestor is a parent, or a parent of an ancestor (all levels up)",
    }
    return (
        f"Facts (the ONLY facts you may use):\n{lines}\n\n"
        f"Definition: {defs[kind]}.\n"
        f"Question: List every {kind} of {person}, deduced ONLY from the facts above.\n"
        f"If there are none, answer with an empty list.\n"
        f'Answer with ONLY a JSON array of names, e.g. ["Al","Bo"] or [].'
    )


def parse_names(text: str, universe: set[str]) -> set[str]:
    m = re.search(r"\[.*?\]", text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            return {str(x).strip() for x in arr} & universe
        except Exception:
            pass
    return {w for w in re.findall(r"[A-Z][a-z]+", text)} & universe


def run(seed: int = 0, model: str = "deepseek-chat", verbose: bool = True):
    from src.reasoning.llm_client import DeepSeekClient

    facts, alg, names = build_family(seed)
    universe = set(names)
    queries = make_queries(alg, names)
    client = DeepSeekClient(model=model)

    # LLM self-assessment (no guard)
    llm_tp = llm_fp = llm_fn = 0
    # after guard filtering
    g_tp = g_fp = g_fn = 0
    guard_dropped_true = 0

    for kind, person, truth in queries:
        prompt = build_prompt(facts, kind, person)
        raw = client.ask(prompt, temperature=0.0)
        claimed = parse_names(raw, universe)

        # raw LLM
        llm_tp += len(claimed & truth)
        llm_fp += len(claimed - truth)          # HALLUCINATION: proposed an incorrect name
        llm_fn += len(truth - claimed)

        # guard: keep a name proposed by the LLM ONLY IF a grounded path matching kind exists
        kept = {c for c in claimed if _grounded(alg, kind, person, c)}
        g_tp += len(kept & truth)
        g_fp += len(kept - truth)
        g_fn += len(truth - kept)
        guard_dropped_true += len((claimed & truth) - kept)

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return p, r

    lp, lr = prf(llm_tp, llm_fp, llm_fn)
    gp, gr = prf(g_tp, g_fp, g_fn)
    res = {
        "n_queries": len(queries),
        "llm_precision": lp, "llm_recall": lr, "llm_hallucinations": llm_fp,
        "guarded_precision": gp, "guarded_recall": gr, "guard_caught": llm_fp - g_fp,
        "guard_leaked": g_fp, "guard_dropped_true": guard_dropped_true,
    }
    if verbose:
        print(json.dumps(res, indent=2))
        print(
            f"\nRaw LLM: precision={lp:.2%}  ({llm_fp} hallucinated names)\n"
            f"After GUARD: precision={gp:.2%}  (caught {llm_fp - g_fp}/{llm_fp} hallucinations, "
            f"{g_fp} leaked through, {guard_dropped_true} correct answers wrongly dropped)"
        )
    return res


def _grounded(alg, kind, person, cand) -> bool:
    if kind == "grandparent":
        return cand in alg.follow(person, ["parent", "parent"])
    if kind == "great-grandparent":
        return cand in alg.follow(person, ["parent", "parent", "parent"])
    if kind == "ancestor":
        return cand in alg.closure(person, "parent")
    return False


if __name__ == "__main__":
    run()
