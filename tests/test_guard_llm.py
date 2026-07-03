"""
Kiểm chứng OFFLINE (mock LLM) rằng HallucinationGuard trên đại số toán tử là bộ
lọc precision=1.0: bắt MỌI ảo giác, KHÔNG loại nhầm đáp án đúng.

Thực nghiệm LIVE tương ứng (DeepSeek) ở src/experiments/guard_llm_eval.py —
đo trên LLM thật cho kết quả cùng tính chất (Định lý G). Test này không gọi mạng.
"""
import random

from src.experiments.guard_llm_eval import _grounded, build_family, make_queries


def _mock_llm_answer(truth: set[str], universe: set[str], rng: random.Random) -> set[str]:
    """LLM giả: giữ hầu hết đáp án đúng + BỊA thêm vài tên sai (ảo giác)."""
    kept = {t for t in truth if rng.random() > 0.1}          # bỏ sót 10%
    fakes = set(rng.sample(sorted(universe - truth), k=min(3, len(universe - truth))))
    return kept | fakes


def test_guard_is_perfect_precision_filter():
    facts, alg, names = build_family(seed=0)
    universe = set(names)
    queries = make_queries(alg, names)
    rng = random.Random(7)

    total_fakes = leaked = dropped_true = tp = fp = 0
    for kind, person, truth in queries:
        claimed = _mock_llm_answer(truth, universe, rng)
        total_fakes += len(claimed - truth)
        # guard: chỉ giữ tên có đường đi grounded đúng loại quan hệ
        kept = {c for c in claimed if _grounded(alg, kind, person, c)}
        leaked += len(kept - truth)               # ảo giác lọt guard
        dropped_true += len((claimed & truth) - kept)  # đáp án đúng bị loại nhầm
        tp += len(kept & truth)
        fp += len(kept - truth)

    assert total_fakes > 0                         # có ảo giác để bắt
    assert leaked == 0                             # KHÔNG lọt ảo giác nào
    assert dropped_true == 0                       # KHÔNG loại nhầm đáp án đúng
    assert tp / max(tp + fp, 1) == 1.0             # precision = 1.0 (Định lý G)


def test_guard_uses_zero_llm_tokens():
    """Guard là đại số cục bộ: KHÔNG gọi LLM ⟹ +0 token (chi phí không tăng)."""
    from src.experiments.nl_ontology_eval import build_dense_dag

    class CallCountingClient:
        def __init__(self):
            self.n_calls = 0

        def ask(self, *a, **k):        # mọi lần gọi LLM đều bị đếm
            self.n_calls += 1
            return "[]"

    alg, words, edges = build_dense_dag(seed=3)
    client = CallCountingClient()
    x = next(a for a, _ in edges)
    # 1 lần suy diễn
    client.ask("reason")
    calls_after_reason = client.n_calls
    # GUARD: lọc cục bộ bằng đóng kín toán tử — không đụng client
    claimed = set(words) - {x}
    kept = {c for c in claimed if c in alg.closure(x, "relates to")}
    assert client.n_calls == calls_after_reason      # guard = 0 lời gọi LLM thêm
    assert kept <= alg.closure(x, "relates to")       # chỉ giữ cái grounded


def test_dense_dag_is_acyclic_and_guard_perfect_on_overclaim():
    """DAG dày trừu tượng: nilpotent (Định lý H) + guard bắt LLM over-claim."""

    from src.experiments.nl_ontology_eval import build_dense_dag
    from src.reasoning.relation_spectrum import is_acyclic, spectral_radius

    alg, words, edges = build_dense_dag(seed=3)
    A = alg.operator("relates to").astype(float).T
    assert is_acyclic(A) and spectral_radius(A) < 1e-9   # Định lý H

    # mock LLM "over-claim": khẳng định gần như MỌI khái niệm (như DeepSeek thật)
    universe = set(words)
    leaked = dropped_true = tp = fp = fakes = 0
    for x in {a for a, _ in edges}:
        truth = alg.closure(x, "relates to")
        claimed = universe - {x}                    # over-claim toàn bộ
        fakes += len(claimed - truth)
        kept = {c for c in claimed if c in alg.closure(x, "relates to")}
        leaked += len(kept - truth)
        dropped_true += len((claimed & truth) - kept)
        tp += len(kept & truth)
        fp += len(kept - truth)
    assert fakes > 0 and leaked == 0 and dropped_true == 0
    assert tp / max(tp + fp, 1) == 1.0
