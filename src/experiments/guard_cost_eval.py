"""
TOKEN COST of using the guard — measured, not guessed.

Two ways to fix LLM hallucinations, compared on the SAME inference output:

  A. GUARD (ours): local verification via operator algebra (Theorem G+H). LLM
     token cost = 0 (just an O(n²·K) matrix multiplication on CPU). Precision → 100% (guaranteed).
  B. LLM SELF-VERIFY: call the LLM a SECOND time to have it re-filter its own claims.
     Costs EXTRA tokens, and comes with NO guarantee (it can still hallucinate).

Question: does using the guard INCREASE token cost? Measure total_tokens for both.

Run: DEEPSEEK_API_KEY=... python -m src.experiments.guard_cost_eval
"""
from __future__ import annotations

import json
import time

from src.experiments.nl_ontology_eval import build_dense_dag, parse


def run(seed: int = 3, top_k: int = 6, model: str = "deepseek-chat", verbose: bool = True):
    from src.reasoning.llm_client import DeepSeekClient

    alg, words, edges = build_dense_dag(seed)
    universe = set(words)
    factstr = "\n".join(f"- {a} relates to {b}." for a, b in sorted(edges))
    srcs = sorted(
        {a for a, _ in edges}, key=lambda x: -len(alg.closure(x, "relates to"))
    )[:top_k]

    reason_client = DeepSeekClient(model=model)   # inference step (shared by A and B)
    verify_client = DeepSeekClient(model=model)   # dedicated to self-verify (measures B's tokens)

    guard_tp = guard_fp = 0
    verify_tp = verify_fp = 0
    guard_seconds = 0.0

    for x in srcs:
        truth = alg.closure(x, "relates to")
        prompt = (
            f"Facts (use ONLY these):\n{factstr}\n\n"
            f"Rule: if A relates to B and B relates to C then A relates to C (transitive).\n"
            f'List EVERY Z such that "{x} relates to Z" is deducible (all levels). '
            f"JSON array only."
        )
        claimed = parse(reason_client.ask(prompt, temperature=0.0), universe)

        # --- A. Local GUARD: 0 tokens, algebra only ---
        t0 = time.perf_counter()
        kept = {c for c in claimed if c in alg.closure(x, "relates to")}
        guard_seconds += time.perf_counter() - t0
        guard_tp += len(kept & truth)
        guard_fp += len(kept - truth)

        # --- B. LLM SELF-VERIFY: call the LLM a second time to self-filter (costs extra tokens) ---
        vprompt = (
            f"Facts (the ONLY truth):\n{factstr}\n\n"
            f'Someone claims these are all Z with "{x} relates to Z": '
            f"{sorted(claimed)}.\n"
            f"Keep ONLY the ones actually deducible by transitivity from the facts. "
            f"JSON array only."
        )
        vkept = parse(verify_client.ask(vprompt, temperature=0.0), universe)
        verify_tp += len(vkept & truth)
        verify_fp += len(vkept - truth)

    def prec(tp, fp):
        return tp / max(tp + fp, 1)

    res = {
        "n_queries": len(srcs),
        "reasoning_tokens": reason_client.total_tokens,   # shared by A and B
        "guard_extra_tokens": 0,                          # the guard does NOT call the LLM
        "guard_cpu_ms": round(1000 * guard_seconds, 3),
        "guard_precision": prec(guard_tp, guard_fp),
        "self_verify_extra_tokens": verify_client.total_tokens,  # extra cost incurred by B
        "self_verify_precision": prec(verify_tp, verify_fp),
    }
    if verbose:
        print(json.dumps(res, indent=2))
        base = res["reasoning_tokens"]
        sv = res["self_verify_extra_tokens"]
        print(
            f"\nInference (shared): {base} tokens.\n"
            f"A. GUARD: +0 LLM tokens, +{res['guard_cpu_ms']} ms CPU, "
            f"precision={res['guard_precision']:.0%} (guaranteed).\n"
            f"B. LLM self-verify: +{sv} tokens "
            f"(+{100 * sv / max(base, 1):.0f}% over inference), "
            f"precision={res['self_verify_precision']:.0%} (not guaranteed).\n"
            f"⟹ Using the guard does NOT increase token cost; self-verification via LLM does."
        )
    return res


if __name__ == "__main__":
    run()
