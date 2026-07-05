"""
OFFLINE verification (mock LLM) for the HARDER stress scenario in
grounded_reasoning/experiments/guard_llm_stress_eval.py: bigger tree, sibling/
spouse distractors, and guaranteed-empty trap questions. Simulates a much
noisier LLM (drops more correct answers, fabricates more names, including on
trap questions) and locks that the guard's precision stays EXACTLY 1.0
regardless — this is Theorem G/H's exact local verification, not a
statistical estimate, so no amount of simulated noise should ever leak a
single hallucination through. No network calls.
"""
import random

from grounded_reasoning.experiments.guard_llm_stress_eval import (
    _grounded,
    build_family,
    make_queries,
)


def _mock_noisy_llm(truth: set[str], universe: set[str], rng: random.Random) -> set[str]:
    """A worse fake LLM than test_guard_llm.py's: drops 30% of correct answers and
    fabricates up to 6 wrong names (vs. 10%/3 in the baseline mock) — simulating the
    higher temperature and distractor confusion of the stress scenario."""
    kept = {t for t in truth if rng.random() > 0.3}
    pool = sorted(universe - truth)
    fakes = set(rng.sample(pool, k=min(6, len(pool)))) if pool else set()
    return kept | fakes


def test_guard_is_perfect_precision_filter_under_heavier_noise():
    facts, siblings, spouses, alg, names, roots = build_family(seed=0, n=48)
    universe = set(names)
    queries = make_queries(alg, names, roots)
    rng = random.Random(11)

    total_fakes = leaked = dropped_true = tp = fp = 0
    n_trap = n_trap_hallucinated_raw = n_trap_hallucinated_guarded = 0
    for kind, person, truth in queries:
        claimed = _mock_noisy_llm(truth, universe, rng)
        total_fakes += len(claimed - truth)

        is_trap = person in roots and not truth
        if is_trap:
            n_trap += 1
            if claimed:
                n_trap_hallucinated_raw += 1

        kept = {c for c in claimed if _grounded(alg, kind, person, c)}
        if is_trap and kept:
            n_trap_hallucinated_guarded += 1

        leaked += len(kept - truth)
        dropped_true += len((claimed & truth) - kept)
        tp += len(kept & truth)
        fp += len(kept - truth)

    assert total_fakes > 0                    # heavier noise really does fabricate names
    assert n_trap > 0 and n_trap_hallucinated_raw > 0  # the raw mock DOES hallucinate on traps
    assert n_trap_hallucinated_guarded == 0   # but the guard catches every single one
    assert leaked == 0                        # NO hallucination leaked through anywhere
    assert dropped_true == 0                  # NO correct answer was wrongly dropped
    assert tp / max(tp + fp, 1) == 1.0         # precision = 1.0 (Theorem G), even under stress


def test_trap_questions_have_empty_ground_truth_by_construction():
    """Sanity check on the test harness itself: root-generation people must have
    NO ancestors/grandparents, so any claimed name on those queries is pure hallucination
    with a known-in-advance right answer -- otherwise the trap doesn't test what it claims to."""
    facts, siblings, spouses, alg, names, roots = build_family(seed=0, n=48)
    queries = make_queries(alg, names, roots)
    trap_truths = [truth for kind, person, truth in queries if person in roots]
    assert trap_truths and all(t == set() for t in trap_truths)
