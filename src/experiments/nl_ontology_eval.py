"""
Chặn ảo giác trên LLM THẬT với QUAN HỆ NGÔN NGỮ TỰ NHIÊN nhiều loại.

Khác guard_llm_eval (chỉ kinship): ở đây dùng ba quan hệ ngôn ngữ thật —
  • "is a"     (phân loại / taxonomy)   — bắc cầu: sparrow is-a bird is-a animal
  • "causes"   (nhân quả)               — bắc cầu: virus causes fever causes …
  • "part of"  (bộ phận / meronymy)     — bắc cầu: piston part-of engine part-of car

Thế giới ĐÓNG (closed-world): LLM chỉ được dùng các fact 1-bước đưa vào; mọi khẳng
định ngoài đó là ẢO GIÁC (kể cả khi "đúng" theo tri thức ngoài). Ground truth và
guard = đóng kín bắc cầu của OperatorRelationAlgebra (Định lý G). Các quan hệ đều
ACYCLIC ⟹ toán tử NILPOTENT (Định lý H) ⟹ đóng kín dừng hữu hạn.

Chạy: DEEPSEEK_API_KEY=... python -m src.experiments.nl_ontology_eval
"""
from __future__ import annotations

import json
import re

from src.reasoning.operator_algebra import OperatorRelationAlgebra
from src.reasoning.relation_spectrum import is_acyclic, spectral_radius

# Thế giới đóng HƠN & KHÓ HƠN: chuỗi dài (4-6 bước) + BẪY phản-tri-thức (fact đúng
# trong thế giới đóng nhưng NGƯỢC tri thức thật, để dụ LLM nhập kiến thức ngoài =
# ảo giác). Mọi quan hệ vẫn ACYCLIC (Định lý H: nilpotent).
FACTS: dict[str, list[tuple[str, str]]] = {
    "is a": [
        ("sparrow", "bird"), ("bird", "vertebrate"), ("vertebrate", "animal"),
        ("animal", "organism"), ("organism", "entity"),
        ("salmon", "fish"), ("fish", "vertebrate"),
        ("oak", "tree"), ("tree", "plant"), ("plant", "organism"),
        # BẪY phản-tri-thức: trong thế giới này whale is-a fish (thật: mammal)
        ("whale", "fish"),
        # BẪY: penguin KHÔNG nối lên bird ở đây — chỉ tới "flightless"
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
    """Định lý H: mỗi quan hệ ontology acyclic ⟹ toán tử nilpotent (ρ=0)."""
    out = {}
    names = alg._names  # noqa: SLF001 (kiểm chứng nội bộ)
    for rel in FACTS:
        A = alg.operator(rel).astype(float).T  # A[i,j]=1 ⟺ i--rel-->j
        out[rel] = {"acyclic": is_acyclic(A), "spectral_radius": round(spectral_radius(A), 9)}
    return out


def make_queries(alg, concepts):
    """Câu bắc cầu: 'mọi Y mà X <rel> (mọi cấp)'. Đáp án = đóng kín."""
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
    """DAG DÀY từ khái niệm TRỪU TƯỢNG (không dựa tri thức ngoài) — mỗi đỉnh nối
    lên 1-2 đỉnh trước ⟹ đóng kín bắc cầu LỚN. Vẫn acyclic (Định lý H)."""
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
    """Kịch bản KHÓ: đóng kín bắc cầu trên DAG dày, khái niệm trừu tượng."""
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
            f"\nDAG dày trừu tượng: LLM SỤP về precision={lp:.2%} (ảo giác {llm_fp}, "
            f"over-claim). Guard bắt {llm_fp - leak}/{llm_fp}, dropped_true={drop} "
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
        kept = {c for c in claimed if c in alg.closure(x, rel)}  # guard: đường đi grounded
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
            f"\nMỗi quan hệ ontology ACYCLIC (ρ=0, nilpotent) — Định lý H.\n"
            f"LLM thô: precision={lp:.2%} (ảo giác {llm_fp}).  "
            f"Sau GUARD: precision={gp:.2%} "
            f"(bắt {llm_fp - g_fp}/{llm_fp}, lọt {g_fp}, loại nhầm đúng {dropped_true})."
        )
    return res


if __name__ == "__main__":
    run()
