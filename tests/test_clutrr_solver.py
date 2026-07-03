"""
Test OFFLINE cho solver hợp thành CLUTRR (path_relations + solve) — không gọi mạng.
Dùng bảng hợp thành thủ công + row giả để khóa LOGIC fold (Định lý G trên dữ liệu).
"""
from src.experiments.clutrr_eval import (
    clean_chain,
    learn_table_closure,
    path_relations,
    solve,
)


def test_path_relations_shortest():
    row = {
        "story_edges": "[(0, 1), (1, 2), (2, 3), (3, 1)]",  # có cạnh lùi (3→1)
        "edge_types": "['son', 'daughter', 'mother', 'x']",
        "query_edge": "(0, 2)",
    }
    rels = path_relations(row)
    assert rels == [("son", 1), ("daughter", 2)]  # đường ngắn nhất 0→1→2


def test_path_relations_self_cycle_regression():
    """
    Fuzz (~2000 lượt, đối chiếu BFS độc lập) tìm ra: path_relations có cùng lớp bug
    self-cycle đã sửa ở GroundedReasoner._path_via/FuzzyInferenceEngine.explain — BFS
    seed prev={src:None} trước vòng lặp khiến KHÔNG BAO GIỜ phát hiện được đường quay
    lại chính src (query_edge=(x,x) qua chu trình). Hàm này không được `run()` dùng
    (clean_chain luôn cho src≠dst) nên không ảnh hưởng số liệu benchmark đã công bố,
    nhưng vẫn là bug thật trên hàm công khai — sửa & khóa lại.
    """
    row = {"story_edges": [(0, 1), (1, 0)], "edge_types": ["a", "b"], "query_edge": (0, 0)}
    assert path_relations(row) == [("a", 1), ("b", 0)]

    # tự-chu-trình dài hơn (3 node)
    row2 = {"story_edges": [(0, 1), (1, 2), (2, 0)], "edge_types": ["a", "b", "c"],
            "query_edge": (0, 0)}
    assert path_relations(row2) == [("a", 1), ("b", 2), ("c", 0)]

    # không có chu trình quay lại src -> vẫn phải None (không bịa)
    row3 = {"story_edges": [(0, 1), (1, 2)], "edge_types": ["a", "b"], "query_edge": (0, 0)}
    assert path_relations(row3) is None


def test_clean_chain_filters_noise():
    chain = {
        "story_edges": "[(0, 1), (1, 2)]", "query_edge": "(0, 2)",
        "edge_types": "['son', 'daughter']",
    }
    noisy = {
        "story_edges": "[(0, 1), (1, 2), (2, 0)]", "query_edge": "(0, 2)",
        "edge_types": "['son', 'daughter', 'x']",
    }
    assert clean_chain(chain) == [("son", 1), ("daughter", 2)]
    assert clean_chain(noisy) is None


def test_solve_folds_composition():
    # bảng: son∘daughter(→nữ) = granddaughter; granddaughter∘brother(→nam)=grandson
    table = {
        ("son", "daughter", "female"): "granddaughter",
        ("granddaughter", "brother", "male"): "grandson",
    }
    gorder = ["male", "male", "female", "male"]      # node 0..3 genders
    rels = [("son", 1), ("daughter", 2), ("brother", 3)]
    assert solve(rels, gorder, table) == "grandson"


def test_solve_returns_none_when_rule_missing():
    table = {("son", "daughter", "female"): "granddaughter"}
    gorder = ["male", "male", "female", "male"]
    rels = [("son", 1), ("daughter", 2), ("brother", 3)]  # thiếu luật bước 2
    assert solve(rels, gorder, table) is None


def test_closure_learns_missing_rule_from_longer_chain():
    # hop-2 dạy luật cơ sở; hop-3 (gold) suy ra luật hợp thành còn thiếu
    def row(story_edges, edge_types, genders, target, k):
        return {
            "story_edges": str(story_edges), "edge_types": str(edge_types),
            "query_edge": str((0, k)), "genders": genders, "target_text": target,
        }
    train = [
        # son∘daughter (→female) = granddaughter  (hop-2)
        row([(0, 1), (1, 2)], ["son", "daughter"], "A:male,B:male,C:female",
            "granddaughter", 2),
        # son-daughter-brother (→male) = grandson  (hop-3, gold cho luật mới)
        row([(0, 1), (1, 2), (2, 3)], ["son", "daughter", "brother"],
            "A:male,B:male,C:female,D:male", "grandson", 3),
    ]
    table = learn_table_closure(train)
    # luật cơ sở đã học
    assert table[("son", "daughter", "female")] == "granddaughter"
    # luật hợp thành (granddaughter∘brother→male) suy ra được từ chuỗi hop-3
    assert table[("granddaughter", "brother", "male")] == "grandson"
    # ⟹ giải được chuỗi hop-3 đầy đủ
    assert solve([("son", 1), ("daughter", 2), ("brother", 3)],
                 ["male", "male", "female", "male"], table) == "grandson"
