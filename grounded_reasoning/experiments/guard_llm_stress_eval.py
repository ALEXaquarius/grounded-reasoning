"""
HARDER hallucination-blocking stress test on a REAL LLM (DeepSeek).

`guard_llm_eval.py` already shows the guard catches hallucinations on a small,
clean 16-person family tree. This stress test tries to make the LLM hallucinate
MORE, and checks the guard still catches ALL of it:

  1. A LARGER, deeper family tree (48 people, up to 7 generations) — more
     names to confuse, longer chains to track.
  2. DISTRACTOR facts injected alongside the real "parent" facts: sibling and
     spouse relations, which are NOT ancestry but sound related — a classic
     LLM confusion trap (e.g. inferring "sibling's parent" as an extra
     grandparent, or treating a spouse as a blood ancestor).
  3. Facts given as a shuffled NATURAL-LANGUAGE PARAGRAPH (not a clean bullet
     list) — harder to track than `guard_llm_eval.py`'s bullet-point facts.
  4. GUARANTEED-EMPTY trap questions: ask for ancestors of root-generation
     people (by construction, they have none) — any non-empty answer is a
     pure hallucination with a KNOWN-in-advance right answer.
  5. HIGHER temperature (default 0.7, vs. 0.0 elsewhere) to intentionally
     increase the LLM's hallucination rate, and MULTIPLE seeds/trials
     aggregated together (not just one run) for statistical weight.

Prediction (Theorem G/H, not statistical): however much the LLM hallucinates
under these harder conditions, the guard's precision must stay 1.0 — grounded
verification is exact local graph membership, not a noise-tolerant estimate.
If guard precision ever drops below 1.0 here, that is a real bug, not noise.

Run: DEEPSEEK_API_KEY=... python -m grounded_reasoning.experiments.guard_llm_stress_eval
"""
from __future__ import annotations

import json
import random
import re

from grounded_reasoning.reasoning.operator_algebra import OperatorRelationAlgebra

NAMES = [
    "Al", "Bo", "Cy", "Di", "Ed", "Fi", "Gu", "Ha", "Io", "Ju", "Ka", "Lu",
    "Mo", "Ni", "Op", "Pa", "Qi", "Ry", "Su", "Ty", "Uv", "Vy", "Wu", "Xi",
    "Yo", "Zu", "Ab", "Bc", "Cd", "De", "Ef", "Fg", "Gh", "Hi", "Ij", "Jk",
    "Kl", "Lm", "Mn", "No", "Op2", "Pq", "Qr", "Rs", "St", "Tu", "Uv2", "Vw",
]


def build_family(seed: int, n: int = 48):
    """Deep multi-generation kinship tree + sibling/spouse DISTRACTOR facts
    (real facts, but NOT ancestry — a trap for the LLM to mistake for ancestry)."""
    rng = random.Random(seed)
    names = NAMES[:n]
    rng.shuffle(names)
    alg = OperatorRelationAlgebra()
    facts: list[tuple[str, str]] = []
    roots = names[:6]                      # generation 0: guaranteed NO ancestors
    parent_of: dict[str, str] = {}
    for i, child in enumerate(names):
        if i >= 6:
            parent = names[rng.randint(0, i - 1)]
            alg.add(child, "parent", parent)
            facts.append((child, parent))
            parent_of[child] = parent

    # sibling facts: pairs sharing a parent (TRUE fact, NOT ancestry)
    by_parent: dict[str, list[str]] = {}
    for c, p in facts:
        by_parent.setdefault(p, []).append(c)
    siblings = [
        (a, b) for kids in by_parent.values() for i, a in enumerate(kids)
        for b in kids[i + 1:]
    ]
    # spouse facts: random unrelated pairing among non-root people (TRUE fact, NOT ancestry)
    pool = [x for x in names if x not in roots]
    rng.shuffle(pool)
    spouses = [(pool[i], pool[i + 1]) for i in range(0, len(pool) - 1, 2)]

    return facts, siblings, spouses, alg, names, roots


def make_queries(alg: OperatorRelationAlgebra, names: list[str], roots: list[str]):
    """Multi-hop questions, INCLUDING guaranteed-empty traps for root-generation people."""
    q = []
    for person in names:
        q.append(("grandparent", person, alg.follow(person, ["parent", "parent"])))
        q.append((
            "great-grandparent",
            person,
            alg.follow(person, ["parent", "parent", "parent"]),
        ))
        q.append(("ancestor", person, alg.closure(person, "parent")))
    # explicit traps: roots have NO ancestors of any kind by construction
    for person in roots:
        q.append(("ancestor", person, set()))
        q.append(("grandparent", person, set()))
    return q


def render_facts(facts, siblings, spouses, rng) -> str:
    """Render ALL facts (parent + sibling + spouse distractors) as a shuffled
    natural-language paragraph, harder to track than a clean bullet list."""
    lines = [f"{c} is a child of {p}." for c, p in facts]
    lines += [f"{a} and {b} are siblings." for a, b in siblings]
    lines += [f"{a} is married to {b}." for a, b in spouses]
    rng.shuffle(lines)
    return " ".join(lines)


def build_prompt(text, kind, person):
    defs = {
        "grandparent": "a grandparent is a parent of a parent",
        "great-grandparent": "a great-grandparent is a parent of a grandparent",
        "ancestor": "an ancestor is a parent, or a parent of an ancestor (all levels up)",
    }
    return (
        f"Facts (the ONLY facts you may use; note some describe siblings or marriages, "
        f"NOT parent/child):\n{text}\n\n"
        f"Definition: {defs[kind]}. Siblings and spouses are NOT ancestors by themselves.\n"
        f"Question: List every {kind} of {person}, deduced ONLY from parent/child facts above.\n"
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
    return {w for w in re.findall(r"[A-Za-z][a-z0-9]*", text)} & universe


def _grounded(alg, kind, person, cand) -> bool:
    if kind == "grandparent":
        return cand in alg.follow(person, ["parent", "parent"])
    if kind == "great-grandparent":
        return cand in alg.follow(person, ["parent", "parent", "parent"])
    if kind == "ancestor":
        return cand in alg.closure(person, "parent")
    return False


def run(seeds=range(5), n: int = 48, temperature: float = 0.7,
        model: str = "deepseek-chat", timeout: float = 180.0,
        max_queries_per_trial: int | None = None, verbose: bool = True):
    """
    timeout: reasoning ("thinking") models can take much longer per call than
    a plain chat model — bump this if the model in use has extended thinking
    enabled, or calls may time out mid-run.
    max_queries_per_trial: if set, subsamples each trial's query list down to
    this many questions (ALWAYS keeping every guaranteed-empty trap query,
    since those are the cheapest, highest-signal hallucination check) — use
    this to keep a live run's wall-clock time bounded when each LLM call is
    slow, at the cost of a smaller sample.
    """
    from grounded_reasoning.reasoning.llm_client import DeepSeekClient

    client = DeepSeekClient(model=model, timeout=timeout)
    llm_tp = llm_fp = llm_fn = 0
    g_tp = g_fp = g_fn = 0
    guard_dropped_true = 0
    n_queries = 0
    n_trap_queries = n_trap_hallucinated = 0

    for seed in seeds:
        facts, siblings, spouses, alg, names, roots = build_family(seed, n)
        universe = set(names)
        queries = make_queries(alg, names, roots)
        rng = random.Random(seed)
        text = render_facts(facts, siblings, spouses, rng)

        if max_queries_per_trial is not None and len(queries) > max_queries_per_trial:
            traps = [q for q in queries if q[1] in roots and not q[2]]
            rest = [q for q in queries if not (q[1] in roots and not q[2])]
            budget = max(0, max_queries_per_trial - len(traps))
            queries = traps + rng.sample(rest, k=min(budget, len(rest)))

        for kind, person, truth in queries:
            n_queries += 1
            prompt = build_prompt(text, kind, person)
            raw = client.ask(prompt, temperature=temperature)
            claimed = parse_names(raw, universe)

            llm_tp += len(claimed & truth)
            llm_fp += len(claimed - truth)
            llm_fn += len(truth - claimed)

            is_trap = person in roots and not truth
            if is_trap:
                n_trap_queries += 1
                if claimed:
                    n_trap_hallucinated += 1

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
        "n_trials": len(list(seeds)), "n_queries": n_queries, "temperature": temperature,
        "n_trap_queries": n_trap_queries, "n_trap_hallucinated": n_trap_hallucinated,
        "llm_precision": round(lp, 3), "llm_recall": round(lr, 3),
        "llm_hallucinations": llm_fp,
        "guarded_precision": round(gp, 3), "guarded_recall": round(gr, 3),
        "guard_caught": llm_fp - g_fp, "guard_leaked": g_fp,
        "guard_dropped_true": guard_dropped_true,
        "llm_tokens": client.total_tokens,
    }
    if verbose:
        print(json.dumps(res, indent=2))
        print(
            f"\nHARDER stress test ({res['n_trials']} trials x {n} people, T={temperature}, "
            f"distractor sibling/spouse facts, shuffled prose):\n"
            f"Raw LLM: precision={lp:.1%} recall={lr:.1%} "
            f"({llm_fp} hallucinated names total, {n_trap_hallucinated}/{n_trap_queries} "
            f"guaranteed-empty TRAP questions hallucinated).\n"
            f"After GUARD: precision={gp:.1%} "
            f"(caught {llm_fp - g_fp}/{max(llm_fp, 1)} hallucinations, {g_fp} leaked, "
            f"{guard_dropped_true} correct answers wrongly dropped).\n"
            + ("✔ guard precision = 100% even under stress\n" if gp >= 0.999 - 1e-9
               else "✘ GUARD LEAK — this is a real bug, not expected noise\n")
        )
    return res


if __name__ == "__main__":
    run()
