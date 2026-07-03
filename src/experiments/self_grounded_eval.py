"""
Self-Grounded Deductive Consistency (SGDC) — kiểm chứng ảo giác KHÔNG cần tri thức
ngoài, khai thác bất đối xứng độ tin cậy của LLM (fact nguyên tử vững, hợp thành ảo).

Giao thức (2 lời gọi LLM, KHÔNG dùng ground truth ngoài để lọc):
  1. Hỏi LLM các FACT NGUYÊN TỬ (1-bước) của một thế giới — bước LLM ĐÁNG TIN.
  2. Hỏi LLM các KẾT LUẬN NHIỀU BƯỚC (đóng kín bắc cầu) — bước LLM HAY ẢO GIÁC.
  3. Dựng toán tử từ CHÍNH fact nguyên tử của LLM; tính đóng kín được chứng nhận.
  4. SGDC: bác bỏ mọi kết luận multi-hop nằm NGOÀI đóng kín của chính LLM.

Đánh giá (ground truth CHỈ để chấm điểm, KHÔNG để lọc): SGDC có khôi phục precision
như dùng đồ thị ngoài không? Nếu có ⟹ không cần đồ thị ngoài (xóa giới hạn cốt lõi).

Điều kiện sống-còn (falsifiable): precision fact nguyên tử phải > precision multi-hop.
Nếu fact nguyên tử cũng ảo giác nặng ⟹ SGDC vô dụng (ghi lại trung thực).

Chạy: DEEPSEEK_API_KEY=... python -m src.experiments.self_grounded_eval
"""
from __future__ import annotations

import json
import re

from src.reasoning.operator_algebra import OperatorRelationAlgebra
from src.reasoning.relation_spectrum import is_acyclic, spectral_radius

# Thế giới đóng THẬT (ground truth để CHẤM, không đưa vào prompt lọc).
GROUND = [
    ("sparrow", "bird"), ("bird", "vertebrate"), ("vertebrate", "animal"),
    ("animal", "organism"), ("organism", "entity"),
    ("salmon", "fish"), ("fish", "vertebrate"),
    ("oak", "tree"), ("tree", "plant"), ("plant", "organism"),
    ("whale", "fish"),  # bẫy phản-tri-thức
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
    """Trích các cặp [x,y] nghĩa 'x is a y' từ JSON."""
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
    from src.reasoning.llm_client import DeepSeekClient

    truth = _truth_alg()
    universe = _universe()
    concepts = sorted(universe)
    client = DeepSeekClient(model=model)

    # ---- 1. LLM đưa FACT NGUYÊN TỬ (1-bước) ----
    atom_prompt = (
        "Consider these concepts: " + ", ".join(concepts) + ".\n"
        "State ONLY the DIRECT, one-step 'is a' facts among them that you are confident "
        "about (immediate category, no transitive steps). "
        'Answer as a JSON array of [x, y] pairs meaning "x is a y". e.g. [["sparrow","bird"]].'
    )
    atom_pairs = _parse_pairs(client.ask(atom_prompt, temperature=0.0), universe)

    # đồ thị TỰ-GROUNDED từ chính LLM
    llm_alg = OperatorRelationAlgebra()
    for a, b in atom_pairs:
        llm_alg.add(a, "is a", b)

    # độ tin cậy fact nguyên tử (chấm bằng ground truth 1-bước)
    gt_atoms = set(GROUND)
    atom_tp = len(set(atom_pairs) & gt_atoms)
    atom_fp = len(set(atom_pairs) - gt_atoms)
    atom_prec = atom_tp / max(atom_tp + atom_fp, 1)

    # ---- 2. LLM đưa KẾT LUẬN NHIỀU BƯỚC (đóng kín) ----
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

    # ---- 3+4. chấm: raw vs SGDC (lọc bằng đóng kín của CHÍNH LLM) vs external ----
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
    sgdc_p, sgdc_r = score(llm_alg)      # lọc bằng đồ thị TỰ LLM (không tri thức ngoài)
    ext_p, ext_r = score(truth)          # lọc bằng đồ thị ngoài (trần trên)

    # ---- spectral: LLM có tự khẳng định chu trình mâu thuẫn không? ----
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
        "survival_condition": atom_prec > raw_p,  # điều kiện sống-còn của SGDC
    }
    if verbose:
        print(json.dumps(res, indent=2))
        print(
            f"\nBất đối xứng: precision fact nguyên tử={atom_prec:.0%} vs multi-hop thô="
            f"{raw_p:.0%}.  SGDC (tự-grounded, 0 tri thức ngoài): precision "
            f"{raw_p:.0%}→{sgdc_p:.0%} (trần ngoài={ext_p:.0%}).  "
            f"{'✔ SGDC hiệu lực' if res['survival_condition'] and sgdc_p >= raw_p else '✘ ghi nhận thất bại'}"
        )
    return res


if __name__ == "__main__":
    run()
