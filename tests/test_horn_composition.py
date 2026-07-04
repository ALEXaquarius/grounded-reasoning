"""
Tests for Horn inference (forward-chaining) + the generic composition algebra.
"""
from grounded_reasoning.reasoning.composition_algebra import fold, learn_composition
from grounded_reasoning.reasoning.horn import entails, explain, forward_chain


class TestHorn:
    def test_forward_chain_least_model(self):
        facts = {"a"}
        rules = [(frozenset({"a"}), "b"), (frozenset({"a", "b"}), "c"),
                 (frozenset({"d"}), "e")]  # d is absent → e is not derived
        M = forward_chain(facts, rules)
        assert M == {"a", "b", "c"}

    def test_entails_and_explain_grounded(self):
        facts = {"a"}
        rules = [(frozenset({"a"}), "b"), (frozenset({"b"}), "c")]
        assert entails(facts, rules, "c")
        assert not entails(facts, rules, "z")
        proof = explain(facts, rules, "c")
        assert proof == [(frozenset({"a"}), "b"), (frozenset({"b"}), "c")]
        assert explain(facts, rules, "z") is None      # never fabricates a proof

    def test_transitive_closure_is_one_rule_horn(self):
        nodes = ["x", "y", "z"]
        facts = {("e", "x", "y"), ("e", "y", "z")}
        rules = [(frozenset({("e", a, b), ("e", b, c)}), ("e", a, c))
                 for a in nodes for b in nodes for c in nodes]
        M = forward_chain(facts, rules)
        assert ("e", "x", "z") in M                     # derived transitively


class TestCompositionAlgebra:
    def test_learn_and_fold_associative(self):
        # additive monoid Z_4: comp(a,b)=(a+b)%4
        def g(seq):
            acc = seq[0]
            for x in seq[1:]:
                acc = (acc + x) % 4
            return acc
        chains = []
        for a in range(4):
            for b in range(4):
                chains.append(((a, b), g((a, b))))
        table, conf, _ = learn_composition(chains)
        assert conf == 0
        assert fold((1, 2, 3, 2), table) == g((1, 2, 3, 2))   # 8%4=0

    def test_conflict_detects_non_associative(self):
        # conflicting data for the same key ⟹ conflict > 0
        chains = [((0, 0), "x"), ((0, 0), "y")]
        _, conf, _ = learn_composition(chains)
        assert conf > 0
