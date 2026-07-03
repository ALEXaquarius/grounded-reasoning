"""
Regression tests from fuzzing `relation_spectrum.py` (~3000 runs, random matrices
cross-checked against plain DFS/BFS) — found 2 real bugs, now fixed; locked down
here to prevent recurrence.

Bug 1 — `is_nilpotent`/`cycle_members` used the weighted matrix DIRECTLY to test
nilpotency (A^n). For a REAL cycle with edge weight <1 (e.g. 0.5) and enough
length, the entries of A^n decay exponentially and fall below `tol=1e-9` PURELY
FOR NUMERICAL REASONS — not because the graph is acyclic. Result:
`is_acyclic(A)=True` even though `spectral_radius(A)=0.5>0` — directly
contradicting the docstring's own claim "ACYCLIC <=> rho(A)=0". Fix: binarize the
matrix before testing nilpotency (acyclicity is a TOPOLOGICAL property, independent
of edge weights).

Bug 2 — `katz_resolvent` did not check the Neumann series convergence condition
alpha*rho(A)<1. When alpha*rho(A)>=1 (the divergent regime), `np.linalg.inv` still
runs successfully (the matrix isn't EXACTLY singular) and returns a
"plausible-looking" matrix that no longer equals Sum(alpha^k A^k) (the series
truly diverges) — a silent wrong answer with no warning. Fix: raise a clear
ValueError when alpha*rho(A)>=1.
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
    """Bug 1: acyclicity must be a STRUCTURAL property, independent of edge weights."""

    def test_exact_fuzz_repro_weighted_long_cycle(self):
        # graph found by fuzzing (seed=414909873): a long cycle with weight 0.5,
        # which decays below tol after matrix-power unless binarized first.
        import random
        rng = random.Random(414909873)
        n = 30
        A = np.zeros((n, n))
        for _ in range(int(n * 0.2)):
            i, j = rng.randrange(n), rng.randrange(n)
            A[i, j] = rng.choice([1.0, 2.0, 0.5])
        assert not is_acyclic(A)          # must correctly detect a cycle
        assert len(cycle_members(A)) > 0
        assert spectral_radius(A) > 1e-6  # consistent with is_acyclic

    def test_small_weight_cycle_still_detected(self):
        # 2-node cycle with a VERY small but nonzero weight — must still be acyclic=False
        A = np.array([[0.0, 0.01], [0.01, 0.0]])
        assert not is_acyclic(A)
        assert cycle_members(A) == {0, 1}

    def test_weighted_dag_still_acyclic(self):
        # a real DAG (no cycle) despite having weights — must still be acyclic=True
        A = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
        assert is_acyclic(A)
        assert cycle_members(A) == set()
        assert spectral_radius(A) < 1e-9

    def test_is_acyclic_consistent_with_spectral_radius(self):
        # general invariant: is_acyclic <=> spectral_radius~=0 (Theorem H), must always hold
        cases = [
            np.array([[0.0, 3.0], [0.0, 0.0]]),          # weighted DAG
            np.array([[0.0, 0.3], [0.3, 0.0]]),           # small-weight cycle
            np.zeros((4, 4)),                              # empty
        ]
        for A in cases:
            acyc = is_acyclic(A)
            rho = spectral_radius(A)
            assert acyc == (rho < 1e-6), f"A={A} acyc={acyc} rho={rho}"


class TestKatzDivergenceRegression:
    """Bug 2: katz_resolvent must raise clearly when alpha*rho(A)>=1, not fail silently."""

    def test_divergent_regime_raises(self):
        B = np.array([[0.0, 2.0], [2.0, 0.0]])  # rho=2
        with pytest.raises(ValueError, match="diverges"):
            katz_resolvent(B, 0.6)               # alpha*rho=1.2 ≥ 1

    def test_boundary_alpha_rho_equals_one_raises(self):
        B = np.array([[0.0, 1.0], [1.0, 0.0]])  # rho=1
        with pytest.raises(ValueError):
            katz_resolvent(B, 1.0)               # alpha*rho=1.0 (boundary, does NOT converge)

    def test_convergent_regime_still_works(self):
        B = np.array([[0.0, 2.0], [2.0, 0.0]])  # rho=2
        kz = katz_resolvent(B, 0.4)              # alpha*rho=0.8 < 1
        assert kz.shape == (2, 2)
        assert np.isfinite(kz).all()

    def test_acyclic_zero_alpha_rho_never_raises(self):
        A = np.array([[0.0, 1.0], [0.0, 0.0]])  # DAG, rho=0
        kz = katz_resolvent(A, 0.99)             # alpha*0=0 < 1 always OK regardless of alpha
        assert np.isfinite(kz).all()
