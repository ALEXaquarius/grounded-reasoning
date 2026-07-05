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
from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine, HallucinationGuard
from grounded_reasoning.reasoning.transitivity_calibration import calibrate_transitivity


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


class TestWiderFuzzProperty:
    """
    A deeper fuzz pass than TestBoundedFuzzProperty: cross-checks code paths that
    class left uncovered (via=None, verify_path, contradictions, normalize=,
    filter_claims dispatch) against independent baselines, over ~1k+ random
    graphs each (self-loops, multi-edges, mixed relation types). All fixed-seed
    for CI reproducibility. 0 failures found when this was run at much larger
    scale (5k-10k trials/suite) during development -- these are scaled-down
    reproductions of that pass, not new discoveries on their own.
    """

    @staticmethod
    def _random_graph(rng, n_max=10, rel_pool=("r1", "r2", "r3")):
        n = rng.randint(1, n_max)
        names = [f"n{i}" for i in range(n)]
        edges = [
            (rng.choice(names), rng.choice(rel_pool), rng.choice(names))
            for _ in range(rng.randint(0, n_max * 3))
        ]
        return names, edges

    @staticmethod
    def _reach(adj, src, dst):
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

    def test_any_relation_reachability_matches_baseline(self):
        # verify(via=None) must match unbounded any-relation BFS reachability
        rng = random.Random(1)
        for _ in range(300):
            names, edges = self._random_graph(rng)
            gr = GroundedReasoner()
            gr.add_facts(edges)
            adj: dict[str, list[str]] = {}
            for s, _, o in edges:
                adj.setdefault(s, []).append(o)
            for _ in range(5):
                s, o = rng.choice(names), rng.choice(names)
                v = gr.verify(s, o, via=None)
                assert v.grounded == self._reach(adj, s, o), (s, o, edges)

    def test_verify_path_matches_exact_hop_sequence_baseline(self):
        rng = random.Random(2)
        for _ in range(300):
            names, edges = self._random_graph(rng, rel_pool=("r1", "r2"))
            gr = GroundedReasoner()
            gr.add_facts(edges)
            by_rel: dict[str, dict[str, list[str]]] = {}
            for s, r, o in edges:
                by_rel.setdefault(r, {}).setdefault(s, []).append(o)
            for _ in range(5):
                s, o = rng.choice(names), rng.choice(names)
                via = [rng.choice(("r1", "r2")) for _ in range(rng.randint(1, 3))]
                v = gr.verify_path(s, o, via)

                frontier = {s}
                for rel in via:
                    frontier = {w for u in frontier for w in by_rel.get(rel, {}).get(u, ())}
                expected = o in frontier
                assert v.grounded == expected, (s, o, via, edges)
                if v.grounded:
                    assert len(v.proof) == len(via) + 1

    def test_contradictions_matches_dfs_cycle_baseline(self):
        rng = random.Random(3)
        for _ in range(300):
            names, edges = self._random_graph(rng, rel_pool=("r1", "r2"))
            gr = GroundedReasoner()
            gr.add_facts(edges)
            for rel in ("r1", "r2"):
                adj: dict[str, list[str]] = {}
                for s, r, o in edges:
                    if r == rel:
                        adj.setdefault(s, []).append(o)
                color: dict[str, int] = {}

                def dfs(u):
                    color[u] = 1
                    for v in adj.get(u, ()):
                        c = color.get(v, 0)
                        if c == 1 or (c == 0 and dfs(v)):
                            return True
                    color[u] = 2
                    return False

                has_cycle = any(dfs(u) for u in list(adj) if color.get(u, 0) == 0)
                cycles = gr.contradictions(rel)
                assert bool(cycles) == has_cycle, (rel, edges)
                if cycles:
                    cyc = cycles[0]
                    adjset = {(a, b) for a, r, b in edges if r == rel}
                    for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                        assert (a, b) in adjset

    def test_normalize_never_loses_reachability_vs_exact_match(self):
        # folding case/whitespace aliases of the SAME entity must never make a
        # previously-reachable pair (under exact string matching) unreachable
        aliases_of = {
            "bob": ["bob", "Bob", " bob ", "BOB"],
            "carol": ["carol", "Carol", " carol "],
            "dave": ["dave", "DAVE"],
        }
        rng = random.Random(4)
        for _ in range(150):
            canon = list(aliases_of)
            edges = []
            for _ in range(rng.randint(2, 12)):
                a, b = rng.choice(canon), rng.choice(canon)
                edges.append((rng.choice(aliases_of[a]), "r", rng.choice(aliases_of[b])))
            gr_norm = GroundedReasoner(normalize=lambda s: s.strip().casefold())
            gr_norm.add_facts(edges)
            gr_raw = GroundedReasoner()
            gr_raw.add_facts(edges)
            for a in canon:
                for b in canon:
                    if a == b:
                        continue
                    if gr_raw.verify(a, b, via="r").grounded:
                        assert gr_norm.verify(a, b, via="r").grounded, (a, b, edges)

    def test_filter_claims_agrees_with_direct_verify_calls(self):
        rng = random.Random(5)
        for _ in range(300):
            names, edges = self._random_graph(rng, rel_pool=("r1", "r2"))
            gr = GroundedReasoner()
            gr.add_facts(edges)
            claims = []
            for _ in range(8):
                s, o = rng.choice(names), rng.choice(names)
                mode = rng.choice(("none", "str", "list"))
                if mode == "none":
                    claims.append((s, o))
                elif mode == "str":
                    claims.append((s, o, rng.choice(("r1", "r2"))))
                else:
                    claims.append((s, o, [rng.choice(("r1", "r2")) for _ in range(rng.randint(1, 2))]))
            for c, v in gr.filter_claims(claims):
                subj, obj = c[0], c[1]
                via = c[2] if len(c) > 2 else None
                direct = gr.verify_path(subj, obj, via) if isinstance(via, list) else gr.verify(subj, obj, via=via)
                assert (v.grounded, v.proof, v.confidence) == (direct.grounded, direct.proof, direct.confidence)

    def test_transitive_relations_guard_matches_allowlist(self):
        rng = random.Random(6)
        pool = ["r1", "r2", "r3", "r4"]
        for _ in range(150):
            allowed = set(rng.sample(pool, k=rng.randint(1, 3)))
            gr = GroundedReasoner(transitive_relations=allowed)
            gr.add_facts([("a", rng.choice(pool), "b")])
            for rel in pool:
                if rel in allowed:
                    gr.verify("a", "b", via=rel)  # must NOT raise
                else:
                    try:
                        gr.verify("a", "b", via=rel)
                        assert False, f"expected ValueError for undeclared via={rel!r}"
                    except ValueError:
                        pass
            # verify_path must NOT consult the allowlist at all (documented behavior)
            gr.verify_path("a", "b", [rng.choice(pool)])

    def test_calibrate_transitivity_bound_never_exceeds_point_estimate(self):
        # a lower confidence bound can never legitimately exceed the raw k/n estimate
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randint(1, 200)
            k = rng.randint(0, n)
            alpha = rng.choice((0.05, 0.1, 0.2))
            grounded_pairs = [(f"a{i}", f"b{i}") for i in range(n)]
            ground_truth = {(f"a{i}", f"b{i}"): (i < k) for i in range(n)}
            res = calibrate_transitivity(grounded_pairs, ground_truth, alpha=alpha)
            bound = res["precision_lower_bound"]
            assert 0.0 <= bound <= 1.0
            assert bound <= (k / n if n else 0.0) + 1e-9
            assert res["n_grounded"] == n and res["n_confirmed"] == k

    def test_confidence_is_not_bounded_to_01_under_a_strong_self_loop(self):
        # documents a discovered, non-obvious property (not a bug): confidence
        # is Sum_{k=1}^{K} alpha^k*(P^k)[a,b], a sum of probabilities across
        # hop-counts, not itself a probability -- a pure self-loop pushes it
        # above 1.0. grounded=True is still correct; only the numeric magnitude
        # is unbounded. See Verdict.confidence's docstring.
        gr = GroundedReasoner(walk_len=8, alpha=0.6)
        gr.add_facts([("x", "r", "x")])
        v = gr.verify("x", "x", via=None)
        assert v.grounded
        expected = sum(0.6**k for k in range(1, 9))
        assert v.confidence == expected
        assert v.confidence > 1.0
