"""
Suy diễn Horn (forward-chaining) — TỔNG QUÁT HÓA guard bắc cầu sang logic đầy đủ.

Guard hợp thành quan hệ (Định lý G) = trường hợp riêng của Horn với MỘT luật
edge(a,b)∧edge(b,c)→edge(a,c). Forward-chaining tính MÔ HÌNH NHỎ NHẤT (least model)
của chương trình Horn: sound (mọi fact suy ra đều có chứng minh) + complete (mọi
fact suy được đều suy ra) + dừng. Đây là bộ kiểm chứng 0-token cho suy diễn LLM ở
dạng luật tổng quát (∧ trong thân) — mở đường tới ProofWriter/EntailmentBank.

Trung thực: đây là ngữ nghĩa Datalog cổ điển (không mới), giá trị = đóng gói thành
lớp kiểm chứng có bảo đảm cho đầu ra LLM, thống nhất với đại số toán tử.
"""
from __future__ import annotations

Rule = tuple[frozenset, object]  # (thân: tập literal, đầu: literal)


def forward_chain(facts: set, rules: list[Rule]) -> set:
    """Mô hình nhỏ nhất: đóng kín facts dưới rules tới bất động (O(#luật · vòng))."""
    derived = set(facts)
    changed = True
    while changed:
        changed = False
        for body, head in rules:
            if head not in derived and body <= derived:
                derived.add(head)
                changed = True
    return derived


def entails(facts: set, rules: list[Rule], goal) -> bool:
    """Goal có suy ra được không (goal ∈ least model)."""
    return goal in forward_chain(facts, rules)


def explain(facts: set, rules: list[Rule], goal) -> list[Rule] | None:
    """
    Chứng minh grounded cho goal: dãy luật kích hoạt dẫn tới goal (hoặc None).
    Đây là tính GROUNDED — guard chấp nhận claim ⟺ có chứng minh.
    """
    derived = set(facts)
    proof: list[Rule] = []
    if goal in derived:
        return proof
    changed = True
    while changed:
        changed = False
        for body, head in rules:
            if head not in derived and body <= derived:
                derived.add(head)
                proof.append((body, head))
                if head == goal:
                    return proof
                changed = True
    return None
