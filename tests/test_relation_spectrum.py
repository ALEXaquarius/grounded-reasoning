"""
Tests for the relation operator SPECTRUM (relation_spectrum) — Theorem H.
"""
import numpy as np

from src.reasoning.relation_spectrum import (
    cycle_members,
    diffusion_sum,
    is_acyclic,
    katz_resolvent,
    row_stochastic,
    spectral_radius,
)


def _dag(n=6):
    A = np.zeros((n, n))
    for i in range(1, n):
        A[i, i - 1] = 1.0  # chain i -> i-1 (DAG)
    return A


class TestRelationSpectrum:
    def test_acyclic_is_nilpotent_rho_zero(self):
        A = _dag()
        assert is_acyclic(A)
        assert spectral_radius(A) < 1e-9

    def test_cycle_raises_spectral_radius(self):
        A = _dag()
        A[0, A.shape[0] - 1] = 1.0  # closes the loop → a cycle
        assert not is_acyclic(A)
        assert spectral_radius(A) >= 1.0 - 1e-9

    def test_cycle_members_detected(self):
        n = 5
        A = np.zeros((n, n))
        A[0, 1] = A[1, 2] = A[2, 0] = 1.0     # cycle {0,1,2}
        A[3, 4] = 1.0                          # disconnected acyclic branch
        assert cycle_members(A) == {0, 1, 2}

    def test_katz_equals_finite_diffusion_on_dag(self):
        A = _dag()
        err = np.abs(diffusion_sum(A, 0.5, A.shape[0]) - katz_resolvent(A, 0.5)).max()
        assert err < 1e-9

    def test_row_stochastic_rows_sum_to_one_or_zero(self):
        A = _dag()
        P = row_stochastic(A)
        sums = P.sum(axis=1)
        assert np.all((np.abs(sums - 1.0) < 1e-9) | (sums == 0.0))
