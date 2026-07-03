"""
The SPECTRUM of the relation operator — inference structure via eigenvalues.

Connects the operator algebra (Theorem G) to spectral theory. For a relation with
adjacency matrix A (A[i,j] = 1 iff i --r--> j), the spectrum of A reveals the
inference structure:

  - ACYCLIC <=> NILPOTENT <=> spectral radius rho(A)=0.
      A genuine hierarchy (parent, part-of) has a nilpotent A: A^n=0, so
      transitive closure HALTS after <= n steps (no infinite loop).
  - CYCLES: concept i lies on a cycle iff (Sum_{k=1}^{n} A^k)[i,i] > 0
      iff a nonzero eigenvalue exists. These are EQUIVALENCE CLASSES (mutual
      reachability).
  - RESOLVENT = DIFFUSION: with P = D^-1 A (row-stochastic), diffused belief
      Sum_{k>=1} alpha^k P^k = (I-alpha*P)^-1 - I (the Neumann series / Katz index),
      converging iff alpha*rho(P) < 1. Hence FuzzyInferenceEngine IS exactly the
      truncated resolvent — fuzzy inference is spectral analysis of the relation
      operator.

This is abstract inference viewed through a SPECTRAL LENS: acyclicity, cycles, and
transitive belief are all spectral invariants of the same operator.
"""
from __future__ import annotations

import numpy as np


def spectral_radius(A: np.ndarray) -> float:
    """rho(A) = max |eigenvalue|."""
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def is_nilpotent(A: np.ndarray, tol: float = 1e-9) -> bool:
    """
    A is nilpotent (a STRUCTURAL property of the relation graph, independent of
    edge weights) iff A^n = 0 once BINARIZED (presence/absence of an edge only)
    iff the graph is ACYCLIC.

    We BINARIZE before checking: using the weighted matrix directly, a REAL cycle
    with edge weights <1 would make entries of A^n decay exponentially and drop
    below `tol` purely from arithmetic (not because the graph is acyclic),
    producing a false negative. Acyclicity is a TOPOLOGICAL property of the graph,
    not a property of the weights.
    """
    n = A.shape[0]
    if n == 0:
        return True
    B = (A != 0).astype(float)
    return float(np.abs(np.linalg.matrix_power(B, n)).max()) <= tol


def is_acyclic(A: np.ndarray, tol: float = 1e-9) -> bool:
    """The relation has no cycle (a DAG) iff the operator is nilpotent."""
    return is_nilpotent(A, tol)


def cycle_members(A: np.ndarray, tol: float = 1e-9) -> set[int]:
    """
    Indices of concepts lying on AT LEAST one cycle (self-reachable in >=1 step).
    BINARIZED before accumulating (same reasoning as `is_nilpotent`): this is a
    STRUCTURAL property independent of weights, avoiding a false negative when
    edge weights <1 make (A^k)[i,i] decay below tol despite a real cycle.
    """
    n = A.shape[0]
    B = (A != 0).astype(float)
    acc = np.zeros((n, n))
    P = np.eye(n)
    for _ in range(n):
        P = P @ B
        acc += P
    return {i for i in range(n) if acc[i, i] > tol}


def katz_resolvent(A: np.ndarray, alpha: float) -> np.ndarray:
    """
    Sum_{k>=1} alpha^k A^k = (I-alpha*A)^-1 - I (the Katz index). Converges iff
    alpha*rho(A) < 1. For nilpotent (acyclic) A, the sum is FINITE and the
    identity is exact.

    Raises
    ------
    ValueError if alpha*rho(A) >= 1: the Neumann series DIVERGES; the matrix
    inverse may still be computable (not exactly singular) but it NO LONGER equals
    Sum alpha^k A^k — better to fail loudly than to silently return a
    "plausible-looking" but meaningless matrix.
    """
    n = A.shape[0]
    if n > 0:
        rho = spectral_radius(A)
        if alpha * rho >= 1.0:
            raise ValueError(
                f"katz_resolvent diverges: alpha*rho(A) = {alpha}*{rho} = {alpha * rho:.6g} >= 1 "
                "(the Neumann series needs alpha*rho(A) < 1 to converge)"
            )
    eye = np.eye(n)
    return np.linalg.inv(eye - alpha * A) - eye


def row_stochastic(A: np.ndarray) -> np.ndarray:
    """P = D^-1 A: row-normalized (the diffusion transition matrix)."""
    deg = A.sum(axis=1)
    P = np.zeros_like(A, dtype=float)
    nz = deg > 0
    P[nz] = A[nz] / deg[nz, None]
    return P


def diffusion_sum(A: np.ndarray, alpha: float, K: int) -> np.ndarray:
    """Sum_{k=1}^{K} alpha^k A^k — the truncated diffusion sum (matches FuzzyInferenceEngine)."""
    n = A.shape[0]
    S = np.zeros((n, n))
    P = np.eye(n)
    for k in range(1, K + 1):
        P = P @ A
        S += (alpha ** k) * P
    return S
