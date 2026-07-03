"""
CHI PHÍ TOKEN của việc dùng guard — đo thật, không suy đoán.

Hai cách sửa ảo giác của LLM, so trên CÙNG một đầu ra suy diễn:

  A. GUARD (của ta): kiểm chứng cục bộ bằng đại số toán tử (Định lý G+H). Chi phí
     token LLM = 0 (chỉ nhân ma trận O(n²·K) trên CPU). Precision → 100% (đảm bảo).
  B. LLM SELF-VERIFY: gọi LLM LẦN 2 để nó tự lọc lại claim. Tốn THÊM token, và
     KHÔNG có bảo đảm (vẫn có thể ảo giác tiếp).

Câu hỏi: dùng guard có làm TĂNG chi phí token không? Đo total_tokens ở cả hai.

Chạy: DEEPSEEK_API_KEY=... python -m src.experiments.guard_cost_eval
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

    reason_client = DeepSeekClient(model=model)   # bước suy diễn (chung cả A và B)
    verify_client = DeepSeekClient(model=model)   # riêng cho self-verify (đo token B)

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

        # --- A. GUARD cục bộ: 0 token, chỉ đại số ---
        t0 = time.perf_counter()
        kept = {c for c in claimed if c in alg.closure(x, "relates to")}
        guard_seconds += time.perf_counter() - t0
        guard_tp += len(kept & truth)
        guard_fp += len(kept - truth)

        # --- B. LLM SELF-VERIFY: gọi LLM lần 2 để tự lọc (tốn thêm token) ---
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
        "reasoning_tokens": reason_client.total_tokens,   # chung cho cả A và B
        "guard_extra_tokens": 0,                          # guard KHÔNG gọi LLM
        "guard_cpu_ms": round(1000 * guard_seconds, 3),
        "guard_precision": prec(guard_tp, guard_fp),
        "self_verify_extra_tokens": verify_client.total_tokens,  # phần TĂNG thêm của B
        "self_verify_precision": prec(verify_tp, verify_fp),
    }
    if verbose:
        print(json.dumps(res, indent=2))
        base = res["reasoning_tokens"]
        sv = res["self_verify_extra_tokens"]
        print(
            f"\nSuy diễn (chung): {base} token.\n"
            f"A. GUARD: +0 token LLM, +{res['guard_cpu_ms']} ms CPU, "
            f"precision={res['guard_precision']:.0%} (đảm bảo).\n"
            f"B. LLM self-verify: +{sv} token "
            f"(+{100 * sv / max(base, 1):.0f}% so với suy diễn), "
            f"precision={res['self_verify_precision']:.0%} (không đảm bảo).\n"
            f"⟹ Dùng guard KHÔNG tăng chi phí token; tự-kiểm-bằng-LLM thì có."
        )
    return res


if __name__ == "__main__":
    run()
