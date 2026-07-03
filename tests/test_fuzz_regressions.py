"""
Test hồi quy từ FUZZ NGẪU NHIÊN (đối chiếu GroundedReasoner với BFS/DFS độc lập trên
~9500 đồ thị ngẫu nhiên: dày/thưa, có/không chu trình, self-loop, multi-edge, tên
Unicode đa ngữ). Tìm ra 2 bug thật, đã sửa; khóa lại để không tái phát.

Bug 1 — `GroundedReasoner._path_via`: BFS seed `prev={subject: None}` trước khi
chạy khiến điều kiện `v not in prev` luôn False khi `v==subject`, nên KHÔNG BAO GIỜ
phát hiện được đường quay lại subject (self-loop/chu trình khi obj==subject) — dù đồ
thị nhỏ hay verify(x,x,via=None) qua engine khác vẫn đúng. Ảnh hưởng: sau khi tối ưu
hiệu năng (BFS O(V+E) thay ma trận), verify(x,x,via=rel) LUÔN trả False dù có chu
trình thật.

Bug 2 — `FuzzyInferenceEngine.explain`: case đặc biệt `if a==b: return [a]` coi
self-identity là "grounded" qua đường-0-bước, TRÁI với chính Định lý Guard Soundness
đã tuyên bố (explain(a,b)≠None ⟺ b reachable từ a). Hệ quả: verify(x,x,via=None) và
HallucinationGuard.verify(x,x) chấp nhận claim "x liên hệ x" dù KHÔNG có fact nào hỗ
trợ (confidence=0.0 nhưng grounded=True) — vi phạm đúng lời hứa cốt lõi của guard.

Cả hai đã sửa bằng cùng một mẫu: kiểm tra đích TRƯỚC khi gate theo tập visited, và
tái tạo đường đi bằng cách đi ngược từ đỉnh liền trước (u) thay vì từ đích (v) — vì
khi obj==subject, v trùng subject ngay từ đầu nên không dùng làm điều kiện dừng được.
"""
import random

from grounded_reasoning import GroundedReasoner
from src.reasoning.abstract_inference import FuzzyInferenceEngine, HallucinationGuard


class TestSelfCycleRegression:
    """Bug 1: verify(via=rel) phải phát hiện chu trình quay lại subject."""

    def test_exact_fuzz_repro_multiedge_cycle(self):
        # đồ thị tìm được từ fuzz (seed=578895315): wa_2 -> hl_0 -> wa_2 là chu trình
        # thật, nhưng lib từng trả False.
        edges = [
            ("hl_0", "wa_2"), ("hl_0", "io_1"), ("io_1", "hl_0"),
            ("io_1", "hl_0"), ("wa_2", "hl_0"), ("hl_0", "hl_0"),
        ]
        gr = GroundedReasoner()
        gr.add_facts([(s, "r", o) for s, o in edges])
        v = gr.verify("wa_2", "wa_2", via="r")
        assert v.grounded
        assert v.proof[0] == "wa_2" and v.proof[-1] == "wa_2"
        adjset = set(edges)
        for a, b in zip(v.proof, v.proof[1:]):
            assert (a, b) in adjset, f"cạnh {a}->{b} không có thật trong đồ thị"

    def test_direct_self_loop_grounded(self):
        gr = GroundedReasoner()
        gr.add_facts([("x", "r", "x")])
        v = gr.verify("x", "x", via="r")
        assert v.grounded and v.proof == ["x", "x"]

    def test_no_cycle_self_query_not_grounded(self):
        # p có cạnh ra nhưng KHÔNG quay lại p ⟹ verify(p,p) phải False
        gr = GroundedReasoner()
        gr.add_facts([("p", "r", "q")])
        v = gr.verify("p", "p", via="r")
        assert not v.grounded and v.proof is None

    def test_longer_cycle_via_rel(self):
        gr = GroundedReasoner()
        gr.add_facts([("a", "r", "b"), ("b", "r", "c"), ("c", "r", "a")])
        v = gr.verify("a", "a", via="r")
        assert v.grounded and v.proof[0] == "a" and v.proof[-1] == "a"
        # verify các node khác trên chu trình vẫn đúng (không hồi quy)
        assert gr.verify("b", "c", via="r").grounded
        assert gr.verify("c", "b", via="r").grounded  # đi vòng qua a


class TestSelfIdentitySoundnessRegression:
    """Bug 2: explain/verify KHÔNG được coi self-identity là grounded nếu vô căn cứ."""

    def test_engine_explain_no_cycle_returns_none(self):
        e = FuzzyInferenceEngine()
        e.add_relation("x", "y")   # x có cạnh ra, không cycle
        assert e.explain("x", "x") is None
        assert e.confidence("x", "x") == 0.0   # nhất quán với explain

    def test_engine_explain_real_cycle_returns_path(self):
        e = FuzzyInferenceEngine()
        for a, b in [("a", "b"), ("b", "c"), ("c", "a")]:
            e.add_relation(a, b)
        path = e.explain("a", "a")
        assert path is not None and path[0] == "a" and path[-1] == "a"
        adjset = {("a", "b"), ("b", "c"), ("c", "a")}
        for x, y in zip(path, path[1:]):
            assert (x, y) in adjset

    def test_guard_rejects_ungrounded_self_claim(self):
        e = FuzzyInferenceEngine()
        e.add_relation("a", "b")
        g = HallucinationGuard(e)
        ok, path = g.verify("a", "a")
        assert not ok and path is None    # KHÔNG chấp nhận claim vô căn cứ

    def test_guard_accepts_real_self_cycle(self):
        e = FuzzyInferenceEngine()
        for a, b in [("a", "b"), ("b", "a")]:
            e.add_relation(a, b)
        ok, path = HallucinationGuard(e).verify("a", "a")
        assert ok and path == ["a", "b", "a"]

    def test_groundedreasoner_verify_via_none_consistent_with_via_rel(self):
        # via=None (FuzzyInferenceEngine) và via=rel (OperatorRelationAlgebra) phải
        # NHẤT QUÁN về việc self-identity có grounded hay không.
        gr = GroundedReasoner()
        gr.add_facts([("p", "r", "q")])   # không cycle
        assert gr.verify("p", "p").grounded == gr.verify("p", "p", via="r").grounded
        assert not gr.verify("p", "p").grounded

        gr2 = GroundedReasoner()
        gr2.add_facts([("a", "r", "b"), ("b", "r", "a")])   # cycle thật
        assert gr2.verify("a", "a").grounded and gr2.verify("a", "a", via="r").grounded


class TestBoundedFuzzProperty:
    """
    Property test fuzz THU GỌN (seed cố định, nhanh cho CI): đối chiếu verify/
    contradictions với BFS/DFS độc lập trên nhiều đồ thị ngẫu nhiên nhỏ.
    """

    @staticmethod
    def _baseline_reachable(edges, src, dst):
        adj: dict[str, list[str]] = {}
        for s, o in edges:
            adj.setdefault(s, []).append(o)
        seen: set[str] = set()
        frontier = list(adj.get(src, ()))
        while frontier:
            nf = []
            for v in frontier:
                if v == dst:
                    return True
                if v not in seen:
                    seen.add(v)
                    nf.extend(adj.get(v, ()))
            frontier = nf
        return False

    def test_random_graphs_match_baseline_reachability(self):
        rng = random.Random(20260702)  # seed cố định ⟹ tái lập được, không flaky
        for _ in range(200):
            n = rng.randint(1, 12)
            names = [f"n{i}" for i in range(n)]
            edges = []
            for _ in range(rng.randint(1, n * 3)):
                s, o = rng.choice(names), rng.choice(names)
                edges.append((s, o))
            gr = GroundedReasoner()
            gr.add_facts([(s, "r", o) for s, o in edges])
            for _ in range(5):
                s, o = rng.choice(names), rng.choice(names)
                v = gr.verify(s, o, via="r")
                expected = self._baseline_reachable(edges, s, o)
                assert v.grounded == expected, (s, o, edges)
                if v.grounded:
                    adjset = set(edges)
                    assert v.proof[0] == s and v.proof[-1] == o
                    for a, b in zip(v.proof, v.proof[1:]):
                        assert (a, b) in adjset
