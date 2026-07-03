"""
OFFLINE tests for Self-Grounded Deductive Consistency (SGDC) + the spectral
contradiction certificate. No network calls — locks down the algebraic LOGIC; the
LLM-side measurement lives in the experiment.
"""

from src.reasoning.operator_algebra import OperatorRelationAlgebra
from src.reasoning.relation_spectrum import cycle_members, is_acyclic, spectral_radius


def _alg(pairs):
    a = OperatorRelationAlgebra()
    for x, y in pairs:
        a.add(x, "is a", y)
    return a


class TestSGDCLogic:
    def test_self_closure_filters_unentailed_claims(self):
        # atomic facts (correct) → certified closure
        atoms = [("sparrow", "bird"), ("bird", "animal"), ("animal", "organism")]
        alg = _alg(atoms)
        # multi-hop LLM claim: mixes CORRECT (entailed) with HALLUCINATED (not entailed)
        claimed = {"animal", "organism", "plant"}   # plant cannot be derived
        kept = {c for c in claimed if c in alg.closure("sparrow", "is a")}
        assert kept == {"animal", "organism"}        # correctly drops the hallucinated 'plant'
        assert "plant" not in kept

    def test_sgdc_precision_one_when_atoms_correct(self):
        # if the atomic facts are CORRECT, every claim surviving the filter is grounded ⟹ precision=1
        atoms = [("a", "b"), ("b", "c"), ("c", "d")]
        alg = _alg(atoms)
        truth = alg.closure("a", "is a")             # {b,c,d}
        claimed = {"b", "c", "d", "x", "y"}          # x,y are fabricated
        kept = {c for c in claimed if c in alg.closure("a", "is a")}
        fp = len(kept - truth)
        assert fp == 0                               # 0 false positives


class TestSpectralContradiction:
    def test_cycle_in_asserted_is_a_is_certified(self):
        # LLM self-asserts a contradictory is-a cycle: cat→mammal→animal→cat
        alg = _alg([("cat", "mammal"), ("mammal", "animal"), ("animal", "cat"),
                    ("dog", "mammal")])
        A = alg.operator("is a").astype(float).T
        assert not is_acyclic(A)                      # contradiction detected
        assert spectral_radius(A) >= 1.0 - 1e-9       # spectral certificate rho>=1
        members = {alg._names[i] for i in cycle_members(A)}  # idx → name
        assert {"cat", "mammal", "animal"} <= members  # cycle localized

    def test_consistent_hierarchy_has_zero_radius(self):
        alg = _alg([("cat", "mammal"), ("mammal", "animal"), ("animal", "organism")])
        A = alg.operator("is a").astype(float).T
        assert is_acyclic(A) and spectral_radius(A) < 1e-9
        assert cycle_members(A) == set()
