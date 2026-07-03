"""
Đại số TOÁN TỬ cho quan hệ — nền tảng tuyến tính của suy diễn hợp thành & loại suy.

Nối suy diễn tập-hợp (TypedInferenceEngine) với lý thuyết toán tử của dự án
("Retrieval là toán tử tuyến tính trên không gian khái niệm"). Mỗi quan hệ r trở
thành một TOÁN TỬ boolean R_r trên ℝ^n (khái niệm = vector cơ sở e_i):

    R_r[i, j] = 1  ⟺  (concept_j) --r--> (concept_i)

Khi đó suy diễn trở thành ĐẠI SỐ TUYẾN TÍNH chính xác:

  • HỢP THÀNH  r∘s  →  tích toán tử  R_r · R_s   (định lý G: đúng khớp với follow)
  • NGHỊCH ĐẢO r⁻¹  →  chuyển vị     R_rᵀ        (suy diễn ngược)
  • BẮC CẦU (đóng kín) r*  →  Σ_k R_r^k        (mọi số bước)
  • LOẠI SUY A:B::C:?  →  áp toán tử suy từ (A→B) lên C

Đây là suy diễn TRỪU TƯỢNG có nền tảng đại số: quan hệ MỚI (grandparent, ancestor)
sinh ra từ quan hệ GỐC bằng phép toán trên toán tử — không học, không embedding.
"""
from __future__ import annotations

import numpy as np


class OperatorRelationAlgebra:
    """Đồ thị quan hệ có kiểu, biểu diễn mỗi quan hệ bằng một toán tử boolean."""

    def __init__(self) -> None:
        self._idx: dict[str, int] = {}
        self._names: list[str] = []
        # quan hệ -> danh sách cạnh (a, b) chờ dựng ma trận (lazy)
        self._rel_edges: dict[str, list[tuple[str, str]]] = {}
        self._ops: dict[str, np.ndarray] | None = None

    def _id(self, c: str) -> int:
        if c not in self._idx:
            self._idx[c] = len(self._names)
            self._names.append(c)
            self._ops = None  # kích thước đổi → dựng lại
        return self._idx[c]

    def add(self, a: str, rel: str, b: str) -> None:
        self._id(a)
        self._id(b)
        self._rel_edges.setdefault(rel, []).append((a, b))
        self._ops = None

    # -- dựng toán tử ------------------------------------------------------
    def _build(self) -> dict[str, np.ndarray]:
        if self._ops is not None:
            return self._ops
        n = len(self._names)
        ops: dict[str, np.ndarray] = {}
        for rel, edges in self._rel_edges.items():
            M = np.zeros((n, n), dtype=bool)
            for a, b in edges:
                M[self._idx[b], self._idx[a]] = True  # cột a → hàng b
            ops[rel] = M
        self._ops = ops
        return ops

    def operator(self, rel: str) -> np.ndarray:
        """Toán tử boolean R_rel (n×n)."""
        return self._build()[rel]

    def _onehot(self, c: str) -> np.ndarray:
        v = np.zeros(len(self._names), dtype=bool)
        if c in self._idx:  # khái niệm chưa từng xuất hiện → vector không
            v[self._idx[c]] = True
        return v

    def _support(self, v: np.ndarray) -> set[str]:
        return {self._names[i] for i in np.nonzero(v)[0]}

    @staticmethod
    def _apply(M: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Nhân boolean ma trận-vector: (M v)_i = OR_j M_ij ∧ v_j."""
        return (M & v[None, :]).any(axis=1)

    # -- suy diễn ----------------------------------------------------------
    def follow(self, src: str, rels: list[str]) -> set[str]:
        """Hợp thành bằng TÍCH TOÁN TỬ: áp R_{r1}, R_{r2}, … lên e_src."""
        ops = self._build()
        v = self._onehot(src)
        for r in rels:
            if r not in ops:
                return set()
            v = self._apply(ops[r], v)
        return self._support(v)

    def inverse_follow(self, src: str, rel: str) -> set[str]:
        """Suy diễn NGƯỢC qua chuyển vị: {a : a --rel--> src} = support(R_relᵀ e_src)."""
        ops = self._build()
        if rel not in ops:
            return set()
        return self._support(self._apply(ops[rel].T, self._onehot(src)))

    def closure(self, src: str, rel: str, max_steps: int | None = None) -> set[str]:
        """ĐÓNG KÍN bắc cầu r* : mọi khái niệm tới được qua ≥1 bước quan hệ rel."""
        ops = self._build()
        if rel not in ops or src not in self._idx:  # quan hệ/khái niệm chưa biết → rỗng
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
        """A:B::C:? — áp mọi toán tử quan hệ nối (a→b) lên c."""
        ops = self._build()
        out: set[str] = set()
        for r in self.relations_between(a, b):
            out |= self._support(self._apply(ops[r], self._onehot(c)))
        return out
