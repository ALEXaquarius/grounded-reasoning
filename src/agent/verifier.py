"""
GroundedReasoner — facade tích hợp cho AGENT/LLM.

Một mặt tiền gọn bọc lõi suy diễn (operator algebra + diffusion + guard) thành API
mà agent dùng ngay: nạp fact, KIỂM CHỨNG claim quan hệ nhiều bước TRƯỚC khi khẳng
định, trả về đường đi bằng chứng + độ tin cậy. Bắt ảo giác quan hệ với **0 token LLM**
và **precision đảm bảo** (Định lý G: chấp nhận ⟺ có đường đi grounded).

Ví dụ:
    gr = GroundedReasoner()
    for s, r, o in facts:            # fact 1-bước (agent cấp hoặc LLM tự trích)
        gr.add_fact(s, r, o)
    v = gr.verify("alice", "carol", via="parent")   # claim nhiều bước
    if not v.grounded:               # ảo giác → chặn
        ...
    print(v.proof)                   # ['alice','bob','carol'] — bằng chứng
"""
from __future__ import annotations

from dataclasses import dataclass

from src.reasoning.abstract_inference import FuzzyInferenceEngine
from src.reasoning.operator_algebra import OperatorRelationAlgebra


@dataclass
class Verdict:
    """Kết quả kiểm chứng một claim quan hệ."""

    grounded: bool
    proof: list[str] | None = None      # đường đi bằng chứng (None nếu không grounded)
    confidence: float = 0.0             # niềm tin khuếch tán (giảm theo độ sâu)
    relation: str | None = None

    def as_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "proof": self.proof,
            "confidence": round(self.confidence, 6),
            "relation": self.relation,
        }


class GroundedReasoner:
    """
    Đồ thị quan hệ có kiểu + các phép kiểm chứng grounded cho agent.

    Parameters
    ----------
    walk_len : độ sâu suy diễn tối đa cho điểm tin cậy.
    alpha    : chiết khấu mỗi bước ∈ (0,1).
    """

    def __init__(self, walk_len: int = 8, alpha: float = 0.6) -> None:
        self._alg = OperatorRelationAlgebra()
        self._eng = FuzzyInferenceEngine(walk_len=walk_len, alpha=alpha)
        self._typed: dict[str, dict[str, set[str]]] = {}  # rel -> a -> {b}
        self._relations: set[str] = set()

    # -- xây đồ thị --------------------------------------------------------
    def add_fact(self, subject: str, relation: str, obj: str) -> None:
        """Thêm một fact 1-bước (subject --relation--> obj)."""
        self._alg.add(subject, relation, obj)
        self._eng.add_relation(subject, obj)                 # đồ thị any-relation
        self._typed.setdefault(relation, {}).setdefault(subject, set()).add(obj)
        self._relations.add(relation)

    def add_facts(self, triples) -> None:
        for i, t in enumerate(triples):
            if len(t) != 3:
                raise ValueError(
                    f"fact #{i} phải là (subject, relation, object), nhận: {t!r}"
                )
            self.add_fact(*t)

    # -- kiểm chứng --------------------------------------------------------
    def verify(self, subject: str, obj: str, via: str | None = None) -> Verdict:
        """
        Kiểm chứng claim: subject có liên hệ (bắc cầu) tới obj không?

        via=None  → qua đường đi BẤT KỲ quan hệ (grounded path tồn tại?).
        via=rel   → qua đóng kín bắc cầu CỦA quan hệ rel (subject --rel*--> obj?).

        Chấp nhận ⟺ có đường đi thật ⟹ KHÔNG bao giờ chấp nhận ảo giác (Định lý G).
        """
        if via is None:
            path = self._eng.explain(subject, obj)
            conf = self._eng.confidence(subject, obj)
            return Verdict(path is not None, path, conf, None)
        # BFS trên đồ thị-theo-quan-hệ: O(V+E), tránh ma trận dày (scale tới đồ thị lớn).
        path = self._path_via(subject, obj, via)
        reachable = path is not None
        conf = self._eng.confidence(subject, obj) if reachable else 0.0
        return Verdict(reachable, path, conf, via)

    def filter_claims(self, claims) -> list[tuple[tuple, Verdict]]:
        """
        Lọc một LÔ claim (subject, obj[, via]) của LLM — giữ cái grounded, chặn ảo
        giác. Trả [(claim, Verdict)]. 0 token, precision đảm bảo.
        """
        out = []
        for c in claims:
            subj, obj = c[0], c[1]
            via = c[2] if len(c) > 2 else None
            out.append((c, self.verify(subj, obj, via=via)))
        return out

    # -- soundness / mâu thuẫn --------------------------------------------
    def contradictions(self, relation: str) -> list[list[str]]:
        """
        Phát hiện MÂU THUẪN: nếu `relation` đáng ra là thứ tự (acyclic) mà có chu
        trình ⟹ trả các khái niệm trên MỘT chu trình (0 token). Rỗng = nhất quán.

        Dùng DFS phát hiện back-edge — O(V+E), không dùng trị riêng (tránh O(n³)).
        (Tương đương phổ ρ>0 đã chứng minh ở Định lý H.)
        """
        adj = self._typed.get(relation, {})
        if not adj:
            return []
        color: dict[str, int] = {}   # 0=chưa thăm, 1=đang trên stack, 2=xong
        parent: dict[str, str] = {}
        for root in list(adj.keys()):
            if color.get(root, 0) != 0:
                continue
            stack = [(root, iter(adj.get(root, ())))]
            color[root] = 1
            while stack:
                node, it = stack[-1]
                for nxt in it:
                    c = color.get(nxt, 0)
                    if c == 0:
                        color[nxt] = 1
                        parent[nxt] = node
                        stack.append((nxt, iter(adj.get(nxt, ()))))
                        break
                    if c == 1:  # back-edge node→nxt ⟹ có chu trình
                        cyc = [node]
                        x = node
                        while x != nxt and x in parent:
                            x = parent[x]
                            cyc.append(x)
                        return [list(reversed(cyc))]
                else:
                    color[node] = 2
                    stack.pop()
        return []

    # -- nội bộ ------------------------------------------------------------
    def _path_via(self, subject: str, obj: str, rel: str) -> list[str] | None:
        """
        BFS ngắn nhất subject→obj qua ≥1 bước quan hệ rel. Kiểm tra `v==obj` TRƯỚC
        khi gate theo `visited`, để phát hiện được đường quay lại CHÍNH subject
        (self-loop hoặc chu trình khi obj==subject) — subject vẫn nằm trong
        `visited` (seed sẵn, KHÔNG có entry trong `prev`) để không mở rộng lại nó
        vô hạn khi obj≠subject.

        Tái tạo đường đi đi NGƯỢC từ `u` (đỉnh nguồn của cạnh vừa khớp) qua `prev`
        — không đi ngược từ `v` — vì khi obj==subject, `v` trùng subject ngay từ
        phần tử đầu nên không thể dùng làm điều kiện dừng vòng lặp.
        """
        adj = self._typed.get(rel, {})
        prev: dict[str, str] = {}
        visited = {subject}
        frontier = [subject]
        while frontier:
            nf = []
            for u in frontier:
                for v in adj.get(u, ()):
                    if v == obj:
                        chain = [u]
                        while chain[-1] in prev:
                            chain.append(prev[chain[-1]])
                        chain.reverse()      # subject → … → u
                        chain.append(v)      # subject → … → u → obj
                        return chain
                    if v not in visited:
                        visited.add(v)
                        prev[v] = u
                        nf.append(v)
            frontier = nf
        return None
