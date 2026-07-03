"""
Robustness / edge-case / hang-safety cho lớp agent (GroundedReasoner + tool).
Bao các lỗi từng gây crash hoặc chậm: quan hệ/khái niệm lạ, đồ thị rỗng, chu trình,
self-loop, đầu vào tool sai dạng, và đồ thị LỚN (không treo).
"""
import time

import pytest

from grounded_reasoning import GroundedReasoner, verify_relation


class TestNoCrash:
    def test_unknown_relation_returns_false(self):
        g = GroundedReasoner()
        g.add_facts([("a", "r", "b")])
        v = g.verify("a", "b", via="nope")
        assert not v.grounded and v.proof is None and v.confidence == 0.0

    def test_empty_graph(self):
        g = GroundedReasoner()
        assert not g.verify("a", "b").grounded
        assert not g.verify("a", "b", via="r").grounded
        assert g.contradictions("r") == []

    def test_unknown_entities(self):
        g = GroundedReasoner()
        g.add_facts([("a", "r", "b")])
        assert not g.verify("zzz", "b", via="r").grounded
        assert not g.verify("a", "zzz", via="r").grounded

    def test_add_facts_malformed_raises_clean(self):
        with pytest.raises(ValueError):
            GroundedReasoner().add_facts([("a", "b")])            # 2-tuple


class TestCyclesNoHang:
    def test_cycle_reachability(self):
        g = GroundedReasoner()
        g.add_facts([("a", "r", "b"), ("b", "r", "c"), ("c", "r", "a")])
        assert g.verify("a", "c", via="r").grounded
        assert g.verify("c", "b", via="r").grounded              # vòng ⟹ tới được

    def test_self_loop(self):
        g = GroundedReasoner()
        g.add_facts([("x", "r", "x")])
        assert g.contradictions("r") == [["x"]]

    def test_contradiction_detection(self):
        g = GroundedReasoner()
        g.add_facts([("cat", "isa", "mammal"), ("mammal", "isa", "animal"),
                     ("animal", "isa", "cat"), ("dog", "isa", "mammal")])
        cyc = g.contradictions("isa")
        assert cyc and set(cyc[0]) == {"cat", "mammal", "animal"}
        g2 = GroundedReasoner()
        g2.add_facts([("cat", "isa", "mammal"), ("mammal", "isa", "animal")])
        assert g2.contradictions("isa") == []


class TestTool:
    def test_tool_skips_malformed_facts(self):
        r = verify_relation([["a", "b"], ["a", "r", "b"]], "a", "r", "b")
        assert r["grounded"] and r.get("skipped_facts") == 1

    def test_tool_empty_and_none(self):
        assert verify_relation([], "a", "r", "b")["grounded"] is False
        assert verify_relation(None, "a", "r", "b")["grounded"] is False


class TestScale:
    def test_deep_chain_is_fast(self):
        g = GroundedReasoner()
        n = 20000
        g.add_facts([(f"n{i}", "r", f"n{i+1}") for i in range(n)])
        t = time.time()
        v = g.verify("n0", f"n{n}", via="r")
        assert v.grounded and len(v.proof) == n + 1
        assert time.time() - t < 5.0                              # không treo
        assert not g.verify("n0", "absent", via="r").grounded

    def test_large_cycle_contradiction_is_fast(self):
        g = GroundedReasoner()
        n = 2000
        g.add_facts([(f"c{i}", "r", f"c{(i+1) % n}") for i in range(n)])
        t = time.time()
        assert g.contradictions("r")                              # phát hiện chu trình
        assert time.time() - t < 2.0
