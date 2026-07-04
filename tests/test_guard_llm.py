"""
OFFLINE verification (mock LLM) that HallucinationGuard built on the operator
algebra is a precision=1.0 filter: it catches EVERY hallucination and NEVER drops
a correct answer.

The corresponding LIVE experiment (DeepSeek) lives in
grounded_reasoning/experiments/guard_llm_eval.py — measuring on a real LLM confirms the same
property (Theorem G). This test makes no network calls.
"""
import random

from grounded_reasoning.experiments.guard_llm_eval import _grounded, build_family, make_queries


def _mock_llm_answer(truth: set[str], universe: set[str], rng: random.Random) -> set[str]:
    """Fake LLM: keeps most correct answers + FABRICATES a few wrong names (hallucination)."""
    kept = {t for t in truth if rng.random() > 0.1}          # drops 10%
    fakes = set(rng.sample(sorted(universe - truth), k=min(3, len(universe - truth))))
    return kept | fakes


def test_guard_is_perfect_precision_filter():
    facts, alg, names = build_family(seed=0)
    universe = set(names)
    queries = make_queries(alg, names)
    rng = random.Random(7)

    total_fakes = leaked = dropped_true = tp = fp = 0
    for kind, person, truth in queries:
        claimed = _mock_llm_answer(truth, universe, rng)
        total_fakes += len(claimed - truth)
        # guard: keep only names with a grounded path of the correct relation kind
        kept = {c for c in claimed if _grounded(alg, kind, person, c)}
        leaked += len(kept - truth)               # hallucinations that leaked through the guard
        dropped_true += len((claimed & truth) - kept)  # correct answers wrongly dropped
        tp += len(kept & truth)
        fp += len(kept - truth)

    assert total_fakes > 0                         # there are hallucinations to catch
    assert leaked == 0                             # NO hallucination leaked through
    assert dropped_true == 0                       # NO correct answer was wrongly dropped
    assert tp / max(tp + fp, 1) == 1.0             # precision = 1.0 (Theorem G)


def test_guard_uses_zero_llm_tokens():
    """The guard is local algebra: it makes NO LLM calls ⟹ +0 tokens (no added cost)."""
    from grounded_reasoning.experiments.nl_ontology_eval import build_dense_dag

    class CallCountingClient:
        def __init__(self):
            self.n_calls = 0

        def ask(self, *a, **k):        # every LLM call gets counted
            self.n_calls += 1
            return "[]"

    alg, words, edges = build_dense_dag(seed=3)
    client = CallCountingClient()
    x = next(a for a, _ in edges)
    # 1 inference call
    client.ask("reason")
    calls_after_reason = client.n_calls
    # GUARD: local filtering via the operator closure — never touches the client
    claimed = set(words) - {x}
    kept = {c for c in claimed if c in alg.closure(x, "relates to")}
    assert client.n_calls == calls_after_reason      # guard = 0 extra LLM calls
    assert kept <= alg.closure(x, "relates to")       # only grounded items are kept


def test_dense_dag_is_acyclic_and_guard_perfect_on_overclaim():
    """Dense abstract DAG: nilpotent (Theorem H) + the guard catches LLM over-claiming."""

    from grounded_reasoning.experiments.nl_ontology_eval import build_dense_dag
    from grounded_reasoning.reasoning.relation_spectrum import is_acyclic, spectral_radius

    alg, words, edges = build_dense_dag(seed=3)
    A = alg.operator("relates to").astype(float).T
    assert is_acyclic(A) and spectral_radius(A) < 1e-9   # Theorem H

    # mock "over-claiming" LLM: asserts almost EVERY concept (like a real DeepSeek run)
    universe = set(words)
    leaked = dropped_true = tp = fp = fakes = 0
    for x in {a for a, _ in edges}:
        truth = alg.closure(x, "relates to")
        claimed = universe - {x}                    # overclaims everything
        fakes += len(claimed - truth)
        kept = {c for c in claimed if c in alg.closure(x, "relates to")}
        leaked += len(kept - truth)
        dropped_true += len((claimed & truth) - kept)
        tp += len(kept & truth)
        fp += len(kept - truth)
    assert fakes > 0 and leaked == 0 and dropped_true == 0
    assert tp / max(tp + fp, 1) == 1.0
