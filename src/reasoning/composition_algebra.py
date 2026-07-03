"""
Đại số HỢP THÀNH tổng quát — học bảng hợp thành của một monoid/nhóm từ ví dụ chuỗi,
bằng closure (fixpoint). Trừu tượng hóa `learn_table_closure` của CLUTRR (DRY).

Cho tập token R và phép hợp thành kết hợp comp: R×R → R. Quan sát = các chuỗi
(t₁,…,t_k) kèm nhãn GOLD = hợp thành toàn chuỗi. Học comp bằng lan truyền:
  • cặp trực tiếp (chuỗi độ dài 2) cho comp(t₁,t₂)=gold;
  • chuỗi dài rút gọn bằng luật đã biết tới còn 2 phần tử ⟹ suy luật còn thiếu.
Lặp tới bất động. Vì hợp thành KẾT HỢP, rút gọn ở bất kỳ vị trí đều hợp lệ.
"""
from __future__ import annotations


def fold(seq: tuple, table: dict) -> object | None:
    """
    Rút gọn chuỗi bằng bảng hợp thành (CYK bất kỳ vị trí). None nếu thiếu luật
    HOẶC chuỗi rỗng (không có phần tử identity ngầm định, không bịa kết quả).
    """
    if not seq:
        return None
    spans = list(seq)
    while len(spans) > 1:
        reduced = False
        for i in range(len(spans) - 1):
            key = (spans[i], spans[i + 1])
            if key in table:
                spans[i:i + 2] = [table[key]]
                reduced = True
                break
        if not reduced:
            return None
    return spans[0]


def learn_composition(chains: list[tuple[tuple, object]], max_iter: int = 1000):
    """
    Học bảng comp[(a,b)]→c từ (chuỗi, gold) bằng fixpoint.
    Trả về (table, conflicts, iters). conflicts>0 ⟹ dữ liệu KHÔNG kết hợp/nhất quán.
    """
    table: dict[tuple, object] = {}
    conflicts = 0
    for it in range(1, max_iter + 1):
        changed = False
        for seq, gold in chains:
            spans = list(seq)
            # rút gọn tối đa bằng luật hiện có
            i = 0
            while len(spans) > 2 and i < len(spans) - 1:
                key = (spans[i], spans[i + 1])
                if key in table:
                    spans[i:i + 2] = [table[key]]
                    i = 0
                else:
                    i += 1
            if len(spans) == 2:
                key = (spans[0], spans[1])
                if key not in table:
                    table[key] = gold
                    changed = True
                elif table[key] != gold:
                    conflicts += 1
        if not changed:
            return table, conflicts, it
    return table, conflicts, max_iter
