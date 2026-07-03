"""
Tests OFFLINE cho Self-Grounded Deductive Consistency (SGDC) + chứng chỉ mâu thuẫn
phổ. Không gọi mạng — khóa phần LOGIC (đại số), phần LLM đo ở experiment.
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
        # fact nguyên tử (đúng) → đóng kín chứng nhận
        atoms = [("sparrow", "bird"), ("bird", "animal"), ("animal", "organism")]
        alg = _alg(atoms)
        # LLM claim multi-hop: có cái ĐÚNG (entailed) + cái ẢO GIÁC (không entailed)
        claimed = {"animal", "organism", "plant"}   # plant không suy ra được
        kept = {c for c in claimed if c in alg.closure("sparrow", "is a")}
        assert kept == {"animal", "organism"}        # loại đúng 'plant' ảo giác
        assert "plant" not in kept

    def test_sgdc_precision_one_when_atoms_correct(self):
        # nếu fact nguyên tử ĐÚNG, mọi claim còn lại sau lọc đều grounded ⟹ precision=1
        atoms = [("a", "b"), ("b", "c"), ("c", "d")]
        alg = _alg(atoms)
        truth = alg.closure("a", "is a")             # {b,c,d}
        claimed = {"b", "c", "d", "x", "y"}          # x,y bịa
        kept = {c for c in claimed if c in alg.closure("a", "is a")}
        fp = len(kept - truth)
        assert fp == 0                               # 0 dương-tính-giả


class TestSpectralContradiction:
    def test_cycle_in_asserted_is_a_is_certified(self):
        # LLM tự khẳng định chu trình is-a mâu thuẫn: cat→mammal→animal→cat
        alg = _alg([("cat", "mammal"), ("mammal", "animal"), ("animal", "cat"),
                    ("dog", "mammal")])
        A = alg.operator("is a").astype(float).T
        assert not is_acyclic(A)                      # phát hiện mâu thuẫn
        assert spectral_radius(A) >= 1.0 - 1e-9       # chứng chỉ phổ ρ≥1
        members = {alg._names[i] for i in cycle_members(A)}  # idx → tên
        assert {"cat", "mammal", "animal"} <= members  # định vị chu trình

    def test_consistent_hierarchy_has_zero_radius(self):
        alg = _alg([("cat", "mammal"), ("mammal", "animal"), ("animal", "organism")])
        A = alg.operator("is a").astype(float).T
        assert is_acyclic(A) and spectral_radius(A) < 1e-9
        assert cycle_members(A) == set()
