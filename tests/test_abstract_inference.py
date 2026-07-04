"""
Tests for the abstract inference algorithm (FuzzyInferenceEngine) + Theorem F.
"""
from grounded_reasoning.reasoning.abstract_inference import (
    FuzzyInferenceEngine, TypedInferenceEngine, HallucinationGuard,
)
from grounded_reasoning.experiments.inference_eval import evaluate
from grounded_reasoning.theory.theorems import theorem_fuzzy_inference


def _chain_engine():
    e = FuzzyInferenceEngine(walk_len=8, alpha=0.6)
    for a, b in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]:
        e.add_relation(a, b)
    return e


class TestFuzzyInference:
    def test_deep_chaining(self):
        e = _chain_engine()
        # infers a→e (4 hops) even though there is no direct edge
        assert e.confidence("a", "e") > 0

    def test_confidence_decays_with_depth(self):
        e = _chain_engine()
        c = {b: e.confidence("a", b) for b in ["b", "c", "d", "e"]}
        assert c["b"] > c["c"] > c["d"] > c["e"] > 0  # farther = less certain

    def test_grounded_no_hallucination(self):
        e = _chain_engine()
        e.add_relation("x", "y")  # disconnected component
        assert e.confidence("a", "y") == 0.0   # no path ⟹ no inferred relation
        assert e.confidence("y", "a") == 0.0

    def test_explain_returns_path(self):
        e = _chain_engine()
        assert e.explain("a", "e") == ["a", "b", "c", "d", "e"]
        assert e.explain("a", "z") is None      # never fabricates a path

    def test_directional(self):
        e = _chain_engine()
        assert e.confidence("a", "e") > 0
        assert e.confidence("e", "a") == 0.0    # directed relation


class TestTypedInference:
    def _kin(self):
        e = TypedInferenceEngine()
        for a, b in [("dave", "carol"), ("carol", "al"), ("al", "x1"), ("eve", "dave")]:
            e.add(a, "parent", b)
        return e

    def test_compositional_grandparent(self):
        e = self._kin()
        assert e.follow("dave", ["parent", "parent"]) == {"al"}          # parent∘parent
        assert e.follow("eve", ["parent", "parent", "parent"]) == {"al"}  # 3-way composition

    def test_analogy(self):
        e = self._kin()
        # dave:carol :: eve:? → dave (same parent relation)
        assert e.analogy("dave", "carol", "eve") == {"dave"}


class TestHallucinationGuard:
    def test_guard_perfect_on_grounded_facts(self):
        e = FuzzyInferenceEngine()
        for a, b in [("a", "b"), ("b", "c"), ("c", "d")]:
            e.add_relation(a, b)
        g = HallucinationGuard(e)
        ok, path = g.verify("a", "d")       # real (transitive)
        assert ok and path == ["a", "b", "c", "d"]
        ok2, path2 = g.verify("a", "z")     # fabricated
        assert not ok2 and path2 is None
        ok3, _ = g.verify("d", "a")         # wrong direction (hallucination)
        assert not ok3


class TestInferenceAB:
    def test_fuzzy_pareto_dominates(self):
        r = evaluate(seed=0)
        # Fuzzy: BOTH 100% deep recall AND zero hallucinations — a Pareto point no other strategy reaches
        assert r["B_fuzzy"]["deep_recall"] == 1.0
        assert r["B_fuzzy"]["halluc_rate"] == 0.0
        # one-hop does not hallucinate but cannot infer deep chains
        assert r["A_one_hop"]["deep_recall"] == 0.0
        assert r["A_one_hop"]["halluc_rate"] == 0.0
        # guesser has high coverage but hallucinates heavily
        assert r["C_guesser"]["halluc_rate"] > 0.5


class TestTheoremF:
    def test_fuzzy_inference_confirmed(self):
        r = theorem_fuzzy_inference()
        assert "CONFIRMED" in r["conclusion"], r["conclusion"]
        assert r["calibrated"] and r["false_inferences"] == 0
        assert r["deep_chain_recall"] == 1.0
