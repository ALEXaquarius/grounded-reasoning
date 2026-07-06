"""
Offline tests for edge_pruning.py (identify_suspect_edges/prune_edges)
and edge_pruning_eval.py's A/B comparison. Fully synthetic, no
LLM/network needed.
"""
from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine
from grounded_reasoning.reasoning.edge_pruning import (
    identify_and_prune_edges,
    identify_suspect_edges,
    identify_suspect_edges_propagated,
    prune_edges,
)
from grounded_reasoning.experiments.edge_pruning_eval import (
    run,
    run_mitigation_comparison,
    wilson_upper_bound,
)


def test_wilson_upper_bound_basic_properties():
    # zero observed successes still leaves a small but nonzero upper bound --
    # absence of evidence in a finite sample isn't proof of a zero true rate
    assert 0.0 < wilson_upper_bound(0, 100) < 0.05
    # a 95% upper bound must sit at or above the point estimate
    assert wilson_upper_bound(10, 100) > 10 / 100
    # more trials at the same observed rate -> tighter (smaller) upper bound
    assert wilson_upper_bound(100, 1000) < wilson_upper_bound(10, 100)
    assert wilson_upper_bound(0, 0) == 0.0


def test_blocks_a_spurious_edge_with_zero_true_support():
    edges = [("a", "b"), ("b", "c"), ("x", "y")]  # x->y is spurious, unrelated to a/b/c
    labeled = [("a", "c", True), ("x", "y", False)]
    blocked = identify_suspect_edges(edges, labeled)
    assert blocked == {("x", "y")}


def test_does_not_block_an_edge_with_any_true_support():
    # (a, b) appears on a TRUE claim's path AND happens to also be part of a
    # separately-labeled FALSE claim's path -- it must NOT be blocked, since
    # it has at least one true-claim vote (avoiding collateral damage to a
    # real edge)
    edges = [("a", "b"), ("b", "c"), ("b", "d")]
    labeled = [("a", "c", True), ("a", "d", False)]
    blocked = identify_suspect_edges(edges, labeled)
    assert ("a", "b") not in blocked
    assert ("b", "d") in blocked  # the actually-false edge, no true votes


def test_prune_edges_removes_exactly_the_blocked_set():
    edges = [("a", "b"), ("b", "c"), ("x", "y")]
    cleaned = prune_edges(edges, {("x", "y")})
    assert cleaned == [("a", "b"), ("b", "c")]


def test_no_false_paths_in_labeled_data_blocks_nothing():
    edges = [("a", "b"), ("b", "c")]
    labeled = [("a", "c", True)]
    assert identify_suspect_edges(edges, labeled) == set()


def test_pruned_graph_never_creates_a_fabricated_path():
    # sanity check: pruning can only REMOVE reachability, never invent it
    edges = [("a", "b"), ("b", "c"), ("x", "y")]
    labeled = [("x", "y", False)]
    blocked = identify_suspect_edges(edges, labeled)
    cleaned = prune_edges(edges, blocked)
    eng = FuzzyInferenceEngine(walk_len=8, alpha=0.6)
    for s, o in cleaned:
        eng.add_relation(s, o)
    assert eng.explain("a", "c") == ["a", "b", "c"]
    assert eng.explain("x", "y") is None


def test_cleaning_reduces_fpr_across_noise_regimes():
    # the aggregate empirical claim (not a per-trial guarantee): pruning
    # substantially reduces false-positive rate in every regime tested,
    # while coverage on the remaining graph stays close to target
    res = run(n_seeds=20)
    for label, m in res.items():
        assert m["cleaned_fpr"] < m["raw_fpr"] - 0.05, f"{label}: {m}"
        assert m["cleaned_coverage"] >= 0.9 - 0.05, f"{label}: {m}"


def test_larger_identify_split_and_min_evidence_cut_wrongly_blocked_rate():
    # the measured mitigation, checked in EVERY noise regime (not just one):
    # using a larger share of held-out data to identify suspect edges, plus
    # requiring a second corroborating false-claim encounter, substantially
    # reduces the rate at which a genuinely correct edge is wrongly removed
    # -- a real cost of the decision rule, not just a theoretical
    # possibility, and not eliminated by this mitigation, only reduced. The
    # upper-confidence-bound check is the precise, worst-case claim: even
    # accounting for sampling noise, the mitigated rate stays bounded.
    res = run_mitigation_comparison(n_seeds=40)
    for regime, configs in res.items():
        default = configs["default (identify_frac=0.5, min_evidence=1)"]
        safer = configs["safer (identify_frac=0.85, min_evidence=2)"]
        assert safer["pooled_wrongly_blocked_rate"] < default["pooled_wrongly_blocked_rate"], f"{regime}: {configs}"
        # worst measured regime (light spurious, fewest edges blocked) still
        # stays under a 95%-upper-confidence-bound on the wrongly-blocked rate
        assert safer["wrongly_blocked_upper_bound"] < 0.15, f"{regime}: {safer}"
        # the mitigation still cleans meaningfully, just less aggressively
        assert safer["cleaned_fpr"] < safer["raw_fpr"], f"{regime}: {safer}"


def test_identify_and_prune_edges_splits_without_overlap_or_loss():
    edges = [("a", "b"), ("b", "c"), ("x", "y")]
    labeled = [("a", "c", True), ("x", "y", False), ("a", "b", True), ("b", "c", True)]
    cleaned, blocked, reserved = identify_and_prune_edges(edges, labeled, seed=0)
    # every input pair ends up in exactly the reserved set or was consumed
    # for identification -- none dropped, none duplicated
    assert len(reserved) == len(labeled) - int(len(labeled) * 0.85)
    assert set(reserved) <= set(labeled)
    assert cleaned == prune_edges(edges, blocked)


def test_identify_and_prune_edges_default_config_matches_documented_recommendation():
    # the defaults ARE the measured-safest configuration, not left for the
    # caller to remember and pass explicitly each time
    import inspect
    sig = inspect.signature(identify_and_prune_edges)
    assert sig.parameters["identify_frac"].default == 0.85
    assert sig.parameters["min_evidence"].default == 2


def test_identify_and_prune_edges_is_deterministic_given_a_seed():
    edges = [("a", "b"), ("b", "c"), ("x", "y"), ("y", "z")]
    labeled = [("a", "c", True), ("x", "z", False)] * 5
    r1 = identify_and_prune_edges(edges, labeled, seed=42)
    r2 = identify_and_prune_edges(edges, labeled, seed=42)
    assert r1 == r2


def test_propagation_blocks_a_second_edge_sharing_a_confirmed_bad_target():
    # p1->hub gets 2 false votes (clears min_evidence=2 under the plain
    # rule); p2->hub only gets 1 (would NOT clear it alone). Once hub is a
    # confirmed "magnet", p2->hub should be swept in too by propagation.
    # q1->other also gets only 1 false vote but "other" has no confirmed-bad
    # edge -- it must NOT be blocked, proving propagation is target-specific,
    # not a blanket lowering of the threshold everywhere.
    edges = [("p1", "hub"), ("p2", "hub"), ("q1", "other"), ("a", "b"), ("b", "c")]
    labeled = [
        ("p1", "hub", False), ("p1", "hub", False),
        ("p2", "hub", False),
        ("q1", "other", False),
        ("a", "c", True),
    ]
    plain = identify_suspect_edges(edges, labeled, min_evidence=2)
    assert plain == {("p1", "hub")}

    propagated = identify_suspect_edges_propagated(
        edges, labeled, min_evidence=2, propagated_min_evidence=1
    )
    assert propagated == {("p1", "hub"), ("p2", "hub")}
    assert ("q1", "other") not in propagated


def test_propagation_never_blocks_an_edge_with_true_support_even_via_a_magnet():
    # p1->hub confirmed bad; p2->hub has a TRUE vote too -- must survive
    # propagation exactly as it survives the plain rule (a true vote is
    # always decisive, regardless of the target's magnet status).
    edges = [("p1", "hub"), ("p2", "hub")]
    labeled = [
        ("p1", "hub", False), ("p1", "hub", False),
        ("p2", "hub", True), ("p2", "hub", False),
    ]
    propagated = identify_suspect_edges_propagated(edges, labeled, min_evidence=2, propagated_min_evidence=1)
    assert propagated == {("p1", "hub")}
    assert ("p2", "hub") not in propagated


def test_use_propagation_flag_is_opt_in_and_off_by_default():
    edges = [("p1", "hub"), ("p2", "hub"), ("a", "b"), ("b", "c")]
    labeled = [
        ("p1", "hub", False), ("p1", "hub", False),
        ("p2", "hub", False),
        ("a", "c", True),
    ]
    _, blocked_default, _ = identify_and_prune_edges(edges, labeled, identify_frac=1.0, min_evidence=2, seed=0)
    _, blocked_propagated, _ = identify_and_prune_edges(
        edges, labeled, identify_frac=1.0, min_evidence=2, seed=0, use_propagation=True
    )
    assert ("p2", "hub") not in blocked_default
    assert ("p2", "hub") in blocked_propagated


def test_propagation_does_not_regress_synthetic_benchmark_fpr():
    # the measured claim: propagation is statistically indistinguishable
    # from the plain rule on sparse, locally-random noise (it should not
    # make things WORSE there even though its benefit is elsewhere)
    from grounded_reasoning.experiments.edge_pruning_eval import SCENARIOS, build_true_dag, noisy_edges
    from grounded_reasoning.reasoning.conformal_reasoning import conformal_threshold
    import random

    def measure(seed, p_drop, p_add, use_propagation, n=45, identify_frac=0.85, alpha=0.1):
        n, true_edges, truth = build_true_dag(seed, n)
        rng = random.Random(1000 + seed)
        edges = list(noisy_edges(true_edges, n, rng, p_drop, p_add))
        eng_raw = FuzzyInferenceEngine(walk_len=12, alpha=0.7)
        for a, b in edges:
            eng_raw.add_relation(a, b)
        infc_raw = {x: eng_raw.infer(x) for x in range(n)}
        all_candidates = [(x, b) for x in range(n) for b in range(n) if x != b and b in infc_raw[x]]
        rng2 = random.Random(3000 + seed)
        rng2.shuffle(all_candidates)
        labeled_pairs = [(x, b, b in truth[x]) for x, b in all_candidates]
        cleaned, blocked, reserved = identify_and_prune_edges(
            edges, labeled_pairs, identify_frac=identify_frac, walk_len=12, alpha=0.7,
            seed=seed, use_propagation=use_propagation,
        )
        eng_clean = FuzzyInferenceEngine(walk_len=12, alpha=0.7)
        for a, b in cleaned:
            eng_clean.add_relation(a, b)
        infc = {x: eng_clean.infer(x) for x in range(n)}
        jit_rng = random.Random(5000 + seed)
        scores = {(x, b): infc[x].get(b, 0.0) + jit_rng.uniform(0, 1e-9) for x, b, _ in reserved}
        true_eval = [(x, b) for x, b, t in reserved if t]
        false_eval = [(x, b) for x, b, t in reserved if not t]
        if not true_eval or not false_eval:
            return None
        r = random.Random(6000 + seed)
        tr = list(true_eval)
        r.shuffle(tr)
        h = len(tr) // 2
        cal, test = tr[:h], tr[h:]
        if not cal or not test:
            return None
        tau = conformal_threshold([scores[p] for p in cal], alpha)
        fpr = sum(1 for p in false_eval if scores[p] >= tau) / len(false_eval)
        return fpr

    n_seeds = 30
    for label, (p_drop, p_add) in SCENARIOS.items():
        plain_fprs, prop_fprs = [], []
        for seed in range(n_seeds):
            fp = measure(seed, p_drop, p_add, use_propagation=False)
            fq = measure(seed, p_drop, p_add, use_propagation=True)
            if fp is not None:
                plain_fprs.append(fp)
            if fq is not None:
                prop_fprs.append(fq)
        mean_plain = sum(plain_fprs) / len(plain_fprs)
        mean_prop = sum(prop_fprs) / len(prop_fprs)
        # propagation must not be MEANINGFULLY worse on this regime
        assert mean_prop < mean_plain + 0.10, f"{label}: plain={mean_plain:.1%} prop={mean_prop:.1%}"
