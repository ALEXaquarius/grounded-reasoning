"""
Tests cho đại số TOÁN TỬ của quan hệ (OperatorRelationAlgebra).
"""
from src.reasoning.operator_algebra import OperatorRelationAlgebra


def _kin():
    a = OperatorRelationAlgebra()
    for x, y in [("dave", "carol"), ("carol", "al"), ("al", "x1"), ("eve", "dave")]:
        a.add(x, "parent", y)
    return a


class TestOperatorAlgebra:
    def test_composition_is_matrix_product(self):
        a = _kin()
        # grandparent = parent∘parent
        assert a.follow("dave", ["parent", "parent"]) == {"al"}
        assert a.follow("eve", ["parent", "parent", "parent"]) == {"al"}

    def test_closure_is_ancestor(self):
        a = _kin()
        # đóng kín bắc cầu: mọi tổ tiên của eve
        assert a.closure("eve", "parent") == {"dave", "carol", "al", "x1"}

    def test_inverse_follow_is_transpose(self):
        a = _kin()
        # ai có parent = carol? → dave
        assert a.inverse_follow("carol", "parent") == {"dave"}
        assert a.inverse_follow("al", "parent") == {"carol"}

    def test_analogy_applies_inferred_relation(self):
        a = _kin()
        # dave:carol :: eve:? — suy 'parent', áp lên eve → dave
        assert a.analogy("dave", "carol", "eve") == {"dave"}

    def test_no_path_returns_empty(self):
        a = _kin()
        assert a.follow("x1", ["parent"]) == set()      # x1 không có parent
        assert a.follow("dave", ["owns"]) == set()       # quan hệ không tồn tại
