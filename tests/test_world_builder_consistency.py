"""
Invariants for the build-world/ground-truth-closure functions in
src/experiments/*_eval.py — locked down from ~700 fuzzing runs (cross-checked
against an independent BFS over the edges they generate): 0 bugs found, but this
ground truth is USED AS THE CORRECT ANSWER in the LLM experiments reported in
PAPER.md, so it deserves permanent protection against silent regressions.
"""


def _baseline_reach(edges, nodes):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    reach = set()
    for src in nodes:
        seen = set()
        frontier = list(adj.get(src, ()))
        while frontier:
            nf = []
            for v in frontier:
                if v not in seen:
                    seen.add(v)
                    nf.extend(adj.get(v, ()))
            frontier = nf
        for v in seen:
            reach.add((src, v))
    return reach


class TestConformalOntologyGroundTruth:
    def test_reach_matches_independent_bfs(self):
        from src.experiments.conformal_llm_eval import build_ontology

        for seed in (0, 1, 5, 10):
            words, edges, reach = build_ontology(seed, n=12)
            assert reach == _baseline_reach(edges, words)


class TestGuardFamilyGroundTruth:
    def test_ancestor_query_matches_independent_bfs(self):
        from src.experiments.guard_llm_eval import build_family, make_queries

        for seed in (0, 1, 5):
            facts, alg, names = build_family(seed)
            expected = _baseline_reach(facts, names)
            for kind, person, ans in make_queries(alg, names):
                if kind == "ancestor":
                    exp = {b for (a, b) in expected if a == person}
                    assert set(ans) == exp, (seed, person)


class TestInferenceChainWorldGroundTruth:
    def test_truth_pairs_match_independent_bfs(self):
        from src.experiments.inference_eval import build_chain_world

        for seed in (0, 1, 2):
            idx, eng, edges, truth = build_chain_world(n_concepts=30, seed=seed)
            flat = [(u, v) for u, vs in edges.items() for v in vs]
            expected = _baseline_reach(flat, range(idx))
            assert set(truth.keys()) == expected


class TestDenseDagGroundTruth:
    def test_closure_matches_independent_bfs(self):
        from src.experiments.nl_ontology_eval import build_dense_dag

        for seed in (0, 1, 3):
            alg, w, edges = build_dense_dag(seed, n=10)
            expected = _baseline_reach(edges, w)
            for src in w:
                got = alg.closure(src, "relates to")
                exp = {b for (a, b) in expected if a == src}
                assert got == exp, (seed, src)
