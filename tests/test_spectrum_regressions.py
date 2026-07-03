"""
Test hồi quy từ fuzz `relation_spectrum.py` (~3000 lượt, ma trận ngẫu nhiên đối
chiếu DFS/BFS thuần) — tìm ra 2 bug thật, đã sửa; khóa lại để không tái phát.

Bug 1 — `is_nilpotent`/`cycle_members` dùng TRỰC TIẾP ma trận trọng số để kiểm tra
nilpotency (Aⁿ). Với chu trình THẬT có trọng số cạnh <1 (vd 0.5) và đủ dài, các
phần tử Aⁿ suy giảm theo hàm mũ và tụt dưới `tol=1e-9` THUẦN VÌ SỐ HỌC — không phải
vì đồ thị acyclic. Kết quả: `is_acyclic(A)=True` dù `spectral_radius(A)=0.5>0` —
mâu thuẫn ngay với chính docstring "ACYCLIC ⟺ ρ(A)=0". Sửa: nhị phân hóa ma trận
trước khi kiểm tra nilpotency (acyclicity là thuộc tính TÔ-PÔ, không phụ thuộc
trọng số cạnh).

Bug 2 — `katz_resolvent` không kiểm tra điều kiện hội tụ α·ρ(A)<1 của chuỗi
Neumann. Khi α·ρ(A)≥1 (miền phân kỳ), `np.linalg.inv` vẫn chạy được (ma trận
không suy biến CHÍNH XÁC) và trả về một ma trận "trông hợp lý" nhưng KHÔNG bằng
Σα^k A^k nữa (chuỗi thật sự phân kỳ) — sai âm thầm, không cảnh báo. Sửa: raise
ValueError rõ ràng khi α·ρ(A)≥1.
"""
import numpy as np
import pytest

from src.reasoning.relation_spectrum import (
    cycle_members,
    is_acyclic,
    katz_resolvent,
    spectral_radius,
)


class TestWeightedCycleRegression:
    """Bug 1: acyclicity phải là thuộc tính CẤU TRÚC, không phụ thuộc trọng số."""

    def test_exact_fuzz_repro_weighted_long_cycle(self):
        # đồ thị fuzz tìm được (seed=414909873): chu trình trọng số 0.5 dài, sau
        # matrix-power sẽ suy giảm dưới tol nếu không nhị phân hóa.
        import random
        rng = random.Random(414909873)
        n = 30
        A = np.zeros((n, n))
        for _ in range(int(n * 0.2)):
            i, j = rng.randrange(n), rng.randrange(n)
            A[i, j] = rng.choice([1.0, 2.0, 0.5])
        assert not is_acyclic(A)          # phải phát hiện ĐÚNG là có chu trình
        assert len(cycle_members(A)) > 0
        assert spectral_radius(A) > 1e-6  # nhất quán với is_acyclic

    def test_small_weight_cycle_still_detected(self):
        # chu trình 2 node, trọng số RẤT nhỏ nhưng KHÔNG bằng 0 — vẫn phải acyclic=False
        A = np.array([[0.0, 0.01], [0.01, 0.0]])
        assert not is_acyclic(A)
        assert cycle_members(A) == {0, 1}

    def test_weighted_dag_still_acyclic(self):
        # DAG thật (không chu trình) dù có trọng số — phải vẫn acyclic=True
        A = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
        assert is_acyclic(A)
        assert cycle_members(A) == set()
        assert spectral_radius(A) < 1e-9

    def test_is_acyclic_consistent_with_spectral_radius(self):
        # bất biến chung: is_acyclic ⟺ spectral_radius≈0 (Định lý H), phải luôn đúng
        cases = [
            np.array([[0.0, 3.0], [0.0, 0.0]]),          # DAG trọng số
            np.array([[0.0, 0.3], [0.3, 0.0]]),           # chu trình trọng số nhỏ
            np.zeros((4, 4)),                              # rỗng
        ]
        for A in cases:
            acyc = is_acyclic(A)
            rho = spectral_radius(A)
            assert acyc == (rho < 1e-6), f"A={A} acyc={acyc} rho={rho}"


class TestKatzDivergenceRegression:
    """Bug 2: katz_resolvent phải raise rõ khi α·ρ(A)≥1, không âm thầm sai."""

    def test_divergent_regime_raises(self):
        B = np.array([[0.0, 2.0], [2.0, 0.0]])  # rho=2
        with pytest.raises(ValueError, match="diverges"):
            katz_resolvent(B, 0.6)               # alpha*rho=1.2 ≥ 1

    def test_boundary_alpha_rho_equals_one_raises(self):
        B = np.array([[0.0, 1.0], [1.0, 0.0]])  # rho=1
        with pytest.raises(ValueError):
            katz_resolvent(B, 1.0)               # alpha*rho=1.0 (biên, KHÔNG hội tụ)

    def test_convergent_regime_still_works(self):
        B = np.array([[0.0, 2.0], [2.0, 0.0]])  # rho=2
        kz = katz_resolvent(B, 0.4)              # alpha*rho=0.8 < 1
        assert kz.shape == (2, 2)
        assert np.isfinite(kz).all()

    def test_acyclic_zero_alpha_rho_never_raises(self):
        A = np.array([[0.0, 1.0], [0.0, 0.0]])  # DAG, rho=0
        kz = katz_resolvent(A, 0.99)             # alpha*0=0 < 1 luôn OK dù alpha lớn
        assert np.isfinite(kz).all()
