"""
Abstract Fuzzy Inference — thuật toán SUY DIỄN TRỪU TƯỢNG trên đồ thị quan hệ.

Tái định hướng (từ retrieval → REASONING): ngôn ngữ/khái niệm là TRỪU TƯỢNG,
không cần khớp chính xác, nhưng phải SUY DIỄN được quan hệ gián tiếp. Đây là năng
lực mà:
  - Embedding KHÔNG có (chỉ đo tương tự MỘT bước, không chuỗi).
  - LLM có nhưng ẢO GIÁC (suy diễn ra quan hệ không tồn tại).

Máy suy diễn này (dựa khuếch tán phổ — Định lý Multi-Hop Bridging) có 3 tính chất
mà kiểm chứng số xác nhận (theorem_fuzzy_inference):

  1. FUZZY CALIBRATED: niềm tin giảm ĐƠN ĐIỆU theo độ sâu suy diễn (α^k) —
     suy xa hơn = kém chắc chắn hơn. Không cần "chính xác", mà cần ĐÚNG THỨ TỰ.
  2. DEEP CHAINING: suy ra quan hệ N-bước (heart→pulse→…→cardiac) mà khớp
     1-bước bó tay hoàn toàn.
  3. GROUNDED (không ảo giác): CHỈ suy ra khi tồn tại đường đi thật; hai thành
     phần rời → niềm tin = 0 (Định lý no-false-bridge). Suy diễn CÓ BẢO ĐẢM.

Niềm tin suy diễn: conf(a→b) = Σ_{k=1}^{K} α^k · (P^k)[a,b],  P = D⁻¹W
(tổng có trọng số xác suất đi từ a đến b qua ≤ K bước; chuỗi dài bị chiết khấu α).

Khác retrieval: đây KHÔNG nhắm điểm nDCG, mà nhắm NĂNG LỰC SUY DIỄN có giải thích
được (trả về đường đi) và không bịa — một trục khác hẳn, nơi toán học này mạnh.
"""

from __future__ import annotations


class FuzzyInferenceEngine:
    """
    Đồ thị quan hệ có hướng + suy diễn bắc cầu mờ qua khuếch tán.

    Parameters
    ----------
    walk_len : độ sâu suy diễn tối đa K.
    alpha    : chiết khấu mỗi bước ∈ (0,1) — chuỗi dài kém chắc chắn hơn.
    """

    def __init__(self, walk_len: int = 8, alpha: float = 0.6) -> None:
        self.walk_len = walk_len
        self.alpha = alpha
        self._edges: dict[str, dict[str, float]] = {}

    def add_relation(self, a: str, b: str, weight: float = 1.0) -> None:
        """Thêm quan hệ a → b (có hướng, trọng số)."""
        self._edges.setdefault(a, {})
        self._edges[a][b] = self._edges[a].get(b, 0.0) + weight

    def _transition(self) -> dict[str, list[tuple[str, float]]]:
        adj: dict[str, list[tuple[str, float]]] = {}
        for u, nbrs in self._edges.items():
            deg = sum(nbrs.values()) or 1.0
            adj[u] = [(v, w / deg) for v, w in nbrs.items()]
        return adj

    def infer(self, source: str) -> dict[str, float]:
        """
        Suy diễn từ `source`: trả về {concept: niềm tin} cho mọi khái niệm suy ra
        được (qua ≤ walk_len bước). Niềm tin giảm theo độ sâu.
        """
        adj = self._transition()
        x = {source: 1.0}
        out: dict[str, float] = {}
        coef = 1.0
        for _ in range(self.walk_len):
            nx: dict[str, float] = {}
            for u, xu in x.items():
                for v, p in adj.get(u, ()):
                    nx[v] = nx.get(v, 0.0) + xu * p
            x = nx
            coef *= self.alpha
            for v, xv in x.items():
                if xv > 0:
                    out[v] = out.get(v, 0.0) + coef * xv
        return out

    def confidence(self, a: str, b: str) -> float:
        """Niềm tin suy diễn a → b (0 nếu không có đường đi)."""
        return self.infer(a).get(b, 0.0)

    def explain(self, a: str, b: str) -> list[str] | None:
        """
        GIẢI THÍCH suy diễn: trả về đường đi ngắn nhất a → b (BFS) QUA ≥1 BƯỚC, hoặc
        None nếu không suy ra được. Đây là tính GROUNDED — suy diễn có bằng chứng,
        không bịa. KỂ CẢ khi a==b: chỉ grounded nếu có chu trình/self-loop THẬT quay
        lại a (không coi self-identity là hiển nhiên qua "đường rỗng" 0-bước) — khớp
        đúng Định lý Guard Soundness (explain(a,b)≠None ⟺ b reachable từ a).
        """
        prev: dict[str, str] = {}
        visited = {a}
        frontier = [a]
        while frontier:
            nf = []
            for u in frontier:
                for v in self._edges.get(u, {}):
                    if v == b:
                        chain = [u]
                        while chain[-1] in prev:
                            chain.append(prev[chain[-1]])
                        chain.reverse()      # a → … → u
                        chain.append(v)      # a → … → u → b
                        return chain
                    if v not in visited:
                        visited.add(v)
                        prev[v] = u
                        nf.append(v)
            frontier = nf
        return None


class TypedInferenceEngine:
    """
    Suy diễn HỢP THÀNH và LOẠI SUY trên đồ thị quan hệ CÓ KIỂU.

    - follow(src, [r1,r2,…]): đi theo CHUỖI quan hệ (compose) → grandparent =
      parent∘parent, great-grandparent = parent∘parent∘parent.
    - analogy(a, b, c): A:B::C:? — suy kiểu quan hệ từ (a,b) rồi áp lên c.

    Đây là suy diễn TRỪU TƯỢNG: derive quan hệ MỚI (grandparent) từ quan hệ GỐC
    (parent) bằng hợp thành — năng lực compositional mà tương tự vector không có.
    """

    def __init__(self) -> None:
        self._e: dict[tuple[str, str], set[str]] = {}  # (a, rel) -> {b,...}

    def add(self, a: str, rel: str, b: str) -> None:
        self._e.setdefault((a, rel), set()).add(b)

    def follow(self, src: str, rels: list[str]) -> set[str]:
        """Suy diễn hợp thành: đi theo chuỗi quan hệ."""
        cur = {src}
        for r in rels:
            nxt: set[str] = set()
            for x in cur:
                nxt |= self._e.get((x, r), set())
            cur = nxt
        return cur

    def relations_between(self, a: str, b: str) -> list[str]:
        return [r for (x, r), ys in self._e.items() if x == a and b in ys]

    def analogy(self, a: str, b: str, c: str) -> set[str]:
        """A:B::C:? — suy kiểu quan hệ (a→b) rồi áp lên c."""
        out: set[str] = set()
        for r in self.relations_between(a, b):
            out |= self._e.get((c, r), set())
        return out


class HallucinationGuard:
    """
    Bọc ngoài LLM để CHẶN ẢO GIÁC: LLM đề xuất quan hệ, máy suy diễn có bảo đảm
    kiểm chứng bằng ĐƯỜNG ĐI. Chấp nhận claim ⟺ tồn tại đường đi grounded.

    Định lý Guard Soundness: trên đồ thị, explain(a,b)≠None ⟺ b reachable từ a
    (BFS đúng). ⟹ Guard KHÔNG BAO GIỜ chấp nhận claim không có đường đi
    (precision=1.0 trên fact bắc cầu) — LLM không thể "lừa" bằng ảo giác.
    """

    def __init__(self, engine: FuzzyInferenceEngine) -> None:
        self.engine = engine

    def verify(self, a: str, b: str) -> tuple[bool, list[str] | None]:
        """Trả về (chấp nhận?, đường đi bằng chứng | None)."""
        path = self.engine.explain(a, b)
        return (path is not None, path)
