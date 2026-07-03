"""
The OPERATOR algebra for relations — the linear-algebra foundation of
compositional inference & analogy.

Connects set-based inference (TypedInferenceEngine) to the project's operator
theory ("relational inference is a linear operator on a concept space"). Each
relation r becomes a boolean OPERATOR R_r on R^n (concepts = basis vectors e_i):

    R_r[i, j] = 1  iff  (concept_j) --r--> (concept_i)

Inference then becomes exact LINEAR ALGEBRA:

  - COMPOSITION  r o s  ->  operator product  R_r . R_s   (Theorem G: matches follow exactly)
  - INVERSE  r^-1  ->  transpose     R_r^T        (backward inference)
  - TRANSITIVE CLOSURE  r*  ->  Sum_k R_r^k        (any number of steps)
  - ANALOGY  A:B::C:?  ->  apply the operator inferred from (A->B) to C

This is ABSTRACT inference with an algebraic foundation: NEW relations
(grandparent, ancestor) are generated from BASE relations purely by operations on
operators — no learning, no embeddings.
"""
from __future__ import annotations

import numpy as np


class OperatorRelationAlgebra:
    """A typed relation graph, representing each relation as a boolean operator."""

    def __init__(self) -> None:
        self._idx: dict[str, int] = {}
        self._names: list[str] = []
        # relation -> list of (a, b) edges, pending (lazy) matrix construction
        self._rel_edges: dict[str, list[tuple[str, str]]] = {}
        self._ops: dict[str, np.ndarray] | None = None

    def _id(self, c: str) -> int:
        if c not in self._idx:
            self._idx[c] = len(self._names)
            self._names.append(c)
            self._ops = None  # dimension changed -> rebuild
        return self._idx[c]

    def add(self, a: str, rel: str, b: str) -> None:
        self._id(a)
        self._id(b)
        self._rel_edges.setdefault(rel, []).append((a, b))
        self._ops = None

    # -- building the operators ---------------------------------------------
    def _build(self) -> dict[str, np.ndarray]:
        if self._ops is not None:
            return self._ops
        n = len(self._names)
        ops: dict[str, np.ndarray] = {}
        for rel, edges in self._rel_edges.items():
            M = np.zeros((n, n), dtype=bool)
            for a, b in edges:
                M[self._idx[b], self._idx[a]] = True  # column a -> row b
            ops[rel] = M
        self._ops = ops
        return ops

    def operator(self, rel: str) -> np.ndarray:
        """The boolean operator R_rel (n x n)."""
        return self._build()[rel]

    def _onehot(self, c: str) -> np.ndarray:
        v = np.zeros(len(self._names), dtype=bool)
        if c in self._idx:  # a concept never seen before -> the zero vector
            v[self._idx[c]] = True
        return v

    def _support(self, v: np.ndarray) -> set[str]:
        return {self._names[i] for i in np.nonzero(v)[0]}

    @staticmethod
    def _apply(M: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Boolean matrix-vector product: (M v)_i = OR_j M_ij AND v_j."""
        return (M & v[None, :]).any(axis=1)

    # -- inference ------------------------------------------------------------
    def follow(self, src: str, rels: list[str]) -> set[str]:
        """Composition via the OPERATOR PRODUCT: apply R_{r1}, R_{r2}, ... to e_src."""
        ops = self._build()
        v = self._onehot(src)
        for r in rels:
            if r not in ops:
                return set()
            v = self._apply(ops[r], v)
        return self._support(v)

    def inverse_follow(self, src: str, rel: str) -> set[str]:
        """Backward inference via the transpose: {a : a --rel--> src} = support(R_rel^T e_src)."""
        ops = self._build()
        if rel not in ops:
            return set()
        return self._support(self._apply(ops[rel].T, self._onehot(src)))

    def closure(self, src: str, rel: str, max_steps: int | None = None) -> set[str]:
        """TRANSITIVE closure r*: every concept reachable via >=1 rel-step."""
        ops = self._build()
        if rel not in ops or src not in self._idx:  # unknown relation/concept -> empty
            return set()
        M = ops[rel]
        n = len(self._names)
        max_steps = n if max_steps is None else max_steps
        v = self._apply(M, self._onehot(src))
        acc = v.copy()
        for _ in range(max_steps - 1):
            v = self._apply(M, v)
            new = acc | v
            if np.array_equal(new, acc):
                break
            acc = new
        return self._support(acc)

    def relations_between(self, a: str, b: str) -> list[str]:
        ops = self._build()
        ia, ib = self._idx.get(a), self._idx.get(b)
        if ia is None or ib is None:
            return []
        return [r for r, M in ops.items() if M[ib, ia]]

    def analogy(self, a: str, b: str, c: str) -> set[str]:
        """A:B::C:? — apply every relation operator connecting (a→b) to c."""
        ops = self._build()
        out: set[str] = set()
        for r in self.relations_between(a, b):
            out |= self._support(self._apply(ops[r], self._onehot(c)))
        return out
