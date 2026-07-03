"""
Test OFFLINE cho demo Conformal-trên-LLM: mô phỏng NHIỄU TRÍCH (bỏ cạnh) thay LLM,
khóa bất biến "phủ ≥ 1−α giữ NGAY CẢ khi đồ thị bẩn". Không gọi mạng.
"""
import random

from src.experiments.conformal_llm_eval import build_ontology
from src.reasoning.abstract_inference import FuzzyInferenceEngine
from src.reasoning.conformal_reasoning import conformal_threshold


def _coverage_under_drop(p_drop: float, alpha: float, seeds=range(12)) -> float:
    pos = []
    for k in seeds:
        words, gold, reach = build_ontology(seed=1000 + k)
        rng = random.Random(500 + k)
        eng = FuzzyInferenceEngine(walk_len=12, alpha=0.7)
        for a, b in gold:
            if rng.random() > p_drop:      # mô phỏng LLM bỏ sót cạnh (nhiễu trích)
                eng.add_relation(a, b)
        infc = {x: eng.infer(x) for x in words}
        for x in words:
            for y in words:
                if (x, y) in reach:
                    pos.append(infc[x].get(y, 0.0) + rng.uniform(0, 1e-9))
    rng = random.Random(0)
    rng.shuffle(pos)
    h = len(pos) // 2
    tau = conformal_threshold(pos[:h], alpha)
    return sum(1 for s in pos[h:] if s >= tau) / len(pos[h:])


def test_conformal_coverage_holds_under_extraction_noise():
    alpha = 0.1
    for p_drop in (0.0, 0.2, 0.4):
        cov = _coverage_under_drop(p_drop, alpha)
        # phủ ≥ 1−α (biên, dung sai nhỏ do rời rạc/hữu hạn) — GIỮ dù đồ thị bẩn
        assert cov >= (1 - alpha) - 0.05, f"p_drop={p_drop} cov={cov}"
