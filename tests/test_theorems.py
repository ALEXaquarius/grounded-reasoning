"""
Numerical verification of the grounded-reasoning theorems (F-L).
"""
import pytest

from grounded_reasoning.theory.theorems import (
    ALL_THEOREMS,
    theorem_closure_completeness,
    theorem_conformal_reasoning,
    theorem_fuzzy_inference,
    theorem_horn_least_model,
    theorem_operator_compositional_equivalence,
    theorem_relation_spectrum,
    theorem_sgdc_recall_bound,
    theorem_transitivity_calibration,
)


@pytest.fixture(scope="module")
def theorem_results():
    """
    Run every theorem's (seeded, Monte-Carlo) verification exactly once per test
    session and share the results — the individual tests below assert on this
    cache instead of each re-invoking the same expensive functions again.
    """
    return {fn: fn() for fn in ALL_THEOREMS}


def test_all_theorems_confirmed(theorem_results):
    """Every theorem must be CONFIRMED by its numerical check."""
    for fn, r in theorem_results.items():
        assert "CONFIRMED" in r["conclusion"], f"{r['theorem']}: {r['conclusion']}"


def test_fuzzy_inference_calibrated_and_grounded(theorem_results):
    r = theorem_results[theorem_fuzzy_inference]
    assert "CONFIRMED" in r["conclusion"]


def test_operator_equivalence_exact(theorem_results):
    r = theorem_results[theorem_operator_compositional_equivalence]
    assert r["composition_mismatch"] == 0
    assert r["closure_mismatch"] == 0
    assert r["inverse_mismatch"] == 0


def test_relation_spectrum_engine_equals_resolvent(theorem_results):
    r = theorem_results[theorem_relation_spectrum]
    assert r["acyclic_rho0_nilpotent"] and r["cyclic_rho_ge_1"]
    assert r["cycle_members_exact"]
    assert r["katz_eq_diffusion_err"] < 1e-9
    assert r["engine_eq_resolvent_err"] < 1e-6


def test_closure_completeness(theorem_results):
    r = theorem_results[theorem_closure_completeness]
    assert r["sound_zero_conflict"] and r["cov_equals_acc"]
    assert r["coverage_in_alphabet"] == 1.0
    assert r["rules_eq_prefix_times_atoms"]
    assert r["naive_generation_insufficient"]  # the naive hypothesis is refuted


def test_conformal_reasoning_valid_under_noise(theorem_results):
    r = theorem_results[theorem_conformal_reasoning]
    assert r["validity_holds_under_noise"]  # coverage >= 1-alpha even under noise
    assert r["efficiency_degrades_with_noise"]  # FPR grows with noise
    assert r["coverage_noisy"] >= r["target_coverage"] - 0.02


def test_sgdc_recall_bound_two_sided(theorem_results):
    r = theorem_results[theorem_sgdc_recall_bound]
    assert r["precision_always_one_when_sound"]  # (1) precision=1 when sound
    assert r["frechet_bound_holds"]  # (2) rho_S >= max(0, rho_M+c-1)
    assert r["worst_slack"] >= -1e-9  # bound holds (and is tight: ~0)
    assert r["frechet_active_trials"] > 50  # the bound is actually binding
    assert r["completeness_gives_equality"]  # (3) completeness => rho_S = rho_M


def test_horn_least_model(theorem_results):
    r = theorem_results[theorem_horn_least_model]
    assert "CONFIRMED" in r["conclusion"]


def test_transitivity_calibration_coverage(theorem_results):
    r = theorem_results[theorem_transitivity_calibration]
    assert r["empirical_coverage"] >= r["target_coverage"] - 0.02
    assert r["zero_evidence_gives_zero_bound"]  # k=0 -> bound=0.0, never overconfident
    assert r["full_evidence_bound_below_one"]   # k=n -> bound<1.0, never claims certainty
