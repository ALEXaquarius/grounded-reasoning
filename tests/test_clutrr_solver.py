"""
OFFLINE test for the CLUTRR composition solver (path_relations + solve) — no network calls.
Uses a manual composition table + fake rows to lock down the fold LOGIC (Theorem G on data).
"""
from grounded_reasoning.experiments.clutrr_eval import (
    clean_chain,
    learn_table_closure,
    path_relations,
    solve,
)


def test_path_relations_shortest():
    row = {
        "story_edges": "[(0, 1), (1, 2), (2, 3), (3, 1)]",  # has a back edge (3→1)
        "edge_types": "['son', 'daughter', 'mother', 'x']",
        "query_edge": "(0, 2)",
    }
    rels = path_relations(row)
    assert rels == [("son", 1), ("daughter", 2)]  # shortest path 0→1→2


def test_path_relations_self_cycle_regression():
    """
    Fuzzing (~2000 runs, cross-checked against an independent BFS) found that
    path_relations has the same self-cycle bug class fixed in
    GroundedReasoner._path_via/FuzzyInferenceEngine.explain — seeding the BFS with
    prev={src:None} before the loop means it can NEVER detect a path looping back
    to src itself (query_edge=(x,x) via a cycle). This function isn't used by
    `run()` (clean_chain always ensures src≠dst), so it does not affect the
    published benchmark numbers, but it is still a real bug on a public function —
    fixed and locked down here.
    """
    row = {"story_edges": [(0, 1), (1, 0)], "edge_types": ["a", "b"], "query_edge": (0, 0)}
    assert path_relations(row) == [("a", 1), ("b", 0)]

    # longer self-cycle (3 nodes)
    row2 = {"story_edges": [(0, 1), (1, 2), (2, 0)], "edge_types": ["a", "b", "c"],
            "query_edge": (0, 0)}
    assert path_relations(row2) == [("a", 1), ("b", 2), ("c", 0)]

    # no cycle back to src -> must still be None (no fabrication)
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
    # table: son∘daughter(→female) = granddaughter; granddaughter∘brother(→male)=grandson
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
    rels = [("son", 1), ("daughter", 2), ("brother", 3)]  # missing rule for step 2
    assert solve(rels, gorder, table) is None


def test_closure_learns_missing_rule_from_longer_chain():
    # hop-2 teaches the base rule; hop-3 (gold) infers the missing composed rule
    def row(story_edges, edge_types, genders, target, k):
        return {
            "story_edges": str(story_edges), "edge_types": str(edge_types),
            "query_edge": str((0, k)), "genders": genders, "target_text": target,
        }
    train = [
        # son∘daughter (→female) = granddaughter  (hop-2)
        row([(0, 1), (1, 2)], ["son", "daughter"], "A:male,B:male,C:female",
            "granddaughter", 2),
        # son-daughter-brother (→male) = grandson  (hop-3, gold for the new rule)
        row([(0, 1), (1, 2), (2, 3)], ["son", "daughter", "brother"],
            "A:male,B:male,C:female,D:male", "grandson", 3),
    ]
    table = learn_table_closure(train)
    # base rule learned
    assert table[("son", "daughter", "female")] == "granddaughter"
    # composed rule (granddaughter∘brother→male) inferred from the hop-3 chain
    assert table[("granddaughter", "brother", "male")] == "grandson"
    # ⟹ the full hop-3 chain can now be solved
    assert solve([("son", 1), ("daughter", 2), ("brother", 3)],
                 ["male", "male", "female", "male"], table) == "grandson"
