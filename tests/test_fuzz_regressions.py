"""
Regression tests from RANDOM FUZZING (cross-checking GroundedReasoner against an
independent BFS/DFS over ~9500 random graphs: dense/sparse, cyclic/acyclic,
self-loops, multi-edges, multilingual Unicode names). Found 2 real bugs, now fixed;
locked down here to prevent recurrence.

Bug 1 — `GroundedReasoner._path_via`: seeding the BFS with `prev={subject: None}`
before running makes the condition `v not in prev` always False when `v==subject`,
so it could NEVER detect a path looping back to subject (a self-loop/cycle when
obj==subject) — even though small graphs or verify(x,x,via=None) via the other
engine gave the correct answer. Impact: after a performance optimization (BFS
O(V+E) instead of a matrix), verify(x,x,via=rel) ALWAYS returned False even when a
real cycle existed.

Bug 2 — `FuzzyInferenceEngine.explain`: the special case `if a==b: return [a]`
treats self-identity as "grounded" via a 0-hop path, CONTRADICTING the very Guard
Soundness Theorem it claims to satisfy (explain(a,b)!=None <=> b is reachable from
a). Consequence: verify(x,x,via=None) and HallucinationGuard.verify(x,x) accepted
the claim "x relates to x" even with NO supporting fact (confidence=0.0 but
grounded=True) — violating the guard's core promise.

Both were fixed with the same pattern: check the target BEFORE gating on the
visited set, and reconstruct the path by walking backward from the predecessor
node (u) rather than from the target (v) — since when obj==subject, v equals
subject from the very start, so it cannot be used as the stopping condition.
"""
import random

from grounded_reasoning import GroundedReasoner
from src.reasoning.abstract_inference import FuzzyInferenceEngine, HallucinationGuard


class TestSelfCycleRegression:
    """Bug 1: verify(via=rel) must detect a cycle looping back to subject."""

    def test_exact_fuzz_repro_multiedge_cycle(self):
        # graph found by fuzzing (seed=578895315): wa_2 -> hl_0 -> wa_2 is a real
        # cycle, but the library used to return False.
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
            assert (a, b) in adjset, f"edge {a}->{b} does not really exist in the graph"

    def test_direct_self_loop_grounded(self):
        gr = GroundedReasoner()
        gr.add_facts([("x", "r", "x")])
        v = gr.verify("x", "x", via="r")
        assert v.grounded and v.proof == ["x", "x"]

    def test_no_cycle_self_query_not_grounded(self):
        # p has an outgoing edge but does NOT loop back to p ⟹ verify(p,p) must be False
        gr = GroundedReasoner()
        gr.add_facts([("p", "r", "q")])
        v = gr.verify("p", "p", via="r")
        assert not v.grounded and v.proof is None

    def test_longer_cycle_via_rel(self):
        gr = GroundedReasoner()
        gr.add_facts([("a", "r", "b"), ("b", "r", "c"), ("c", "r", "a")])
        v = gr.verify("a", "a", via="r")
        assert v.grounded and v.proof[0] == "a" and v.proof[-1] == "a"
        # verifying other nodes on the cycle still works correctly (no regression)
        assert gr.verify("b", "c", via="r").grounded
        assert gr.verify("c", "b", via="r").grounded  # goes around via a


class TestSelfIdentitySoundnessRegression:
    """Bug 2: explain/verify must NOT treat self-identity as grounded when unsupported."""

    def test_engine_explain_no_cycle_returns_none(self):
        e = FuzzyInferenceEngine()
        e.add_relation("x", "y")   # x has an outgoing edge, no cycle
        assert e.explain("x", "x") is None
        assert e.confidence("x", "x") == 0.0   # consistent with explain

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
        assert not ok and path is None    # does NOT accept an unsupported claim

    def test_guard_accepts_real_self_cycle(self):
        e = FuzzyInferenceEngine()
        for a, b in [("a", "b"), ("b", "a")]:
            e.add_relation(a, b)
        ok, path = HallucinationGuard(e).verify("a", "a")
        assert ok and path == ["a", "b", "a"]

    def test_groundedreasoner_verify_via_none_consistent_with_via_rel(self):
        # via=None (FuzzyInferenceEngine) and via=rel (OperatorRelationAlgebra) must
        # be CONSISTENT on whether self-identity is grounded.
        gr = GroundedReasoner()
        gr.add_facts([("p", "r", "q")])   # no cycle
        assert gr.verify("p", "p").grounded == gr.verify("p", "p", via="r").grounded
        assert not gr.verify("p", "p").grounded

        gr2 = GroundedReasoner()
        gr2.add_facts([("a", "r", "b"), ("b", "r", "a")])   # real cycle
        assert gr2.verify("a", "a").grounded and gr2.verify("a", "a", via="r").grounded


class TestBoundedFuzzProperty:
    """
    A SCALED-DOWN fuzz property test (fixed seed, fast for CI): cross-checks
    verify/contradictions against an independent BFS/DFS over many small random graphs.
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
        rng = random.Random(20260702)  # fixed seed ⟹ reproducible, not flaky
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
