"""
Direct tests for the Clopper-Pearson calibration module (Theorem M), separate
from the numerical Monte-Carlo verification in tests/test_theorems.py.
"""
import pytest

from grounded_reasoning.reasoning.transitivity_calibration import (
    calibrate_transitivity,
    clopper_pearson_lower,
)


class TestClopperPearsonLower:
    def test_zero_successes_gives_zero_bound(self):
        assert clopper_pearson_lower(0, 40, 0.1) == 0.0

    def test_all_successes_closed_form(self):
        # k=n: P(X>=n|p)=p^n=alpha => p=alpha^(1/n), exact closed form
        assert clopper_pearson_lower(10, 10, 0.1) == pytest.approx(0.1 ** (1 / 10), abs=1e-9)

    def test_matches_scipy_beta_quantile(self):
        # cross-checked once against scipy.stats.beta.ppf(alpha, k, n-k+1) to
        # machine precision during development (0.00e+00 diff at k=0,20,38,40,
        # n=40, alpha=0.1); pinned here as exact regression values so a future
        # change to the bisection can't silently drift from the closed form.
        assert clopper_pearson_lower(20, 40, 0.1) == pytest.approx(0.388297, abs=1e-5)
        assert clopper_pearson_lower(38, 40, 0.1) == pytest.approx(0.872372, abs=1e-5)

    def test_monotonic_in_k(self):
        bounds = [clopper_pearson_lower(k, 50, 0.1) for k in range(0, 51, 5)]
        assert all(bounds[i] < bounds[i + 1] for i in range(len(bounds) - 1))

    def test_tighter_alpha_gives_lower_bound(self):
        # smaller alpha (more confidence required) must give a MORE conservative
        # (lower) bound, never a higher one, for the same evidence
        loose = clopper_pearson_lower(30, 40, 0.2)
        tight = clopper_pearson_lower(30, 40, 0.01)
        assert tight < loose

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            clopper_pearson_lower(5, 0, 0.1)     # n<1
        with pytest.raises(ValueError):
            clopper_pearson_lower(-1, 10, 0.1)   # k<0
        with pytest.raises(ValueError):
            clopper_pearson_lower(11, 10, 0.1)   # k>n
        with pytest.raises(ValueError):
            clopper_pearson_lower(5, 10, 0.0)    # alpha out of (0,1)
        with pytest.raises(ValueError):
            clopper_pearson_lower(5, 10, 1.0)


class TestCalibrateTransitivity:
    def test_basic_tally_and_bound(self):
        pairs = [("a", "b"), ("c", "d"), ("e", "f")]
        truth = {("a", "b"): True, ("c", "d"): True, ("e", "f"): False}
        res = calibrate_transitivity(pairs, truth, alpha=0.1)
        assert res["n_grounded"] == 3
        assert res["n_confirmed"] == 2
        assert 0.0 < res["precision_lower_bound"] < (2 / 3)

    def test_missing_ground_truth_pairs_are_skipped_not_counted(self):
        pairs = [("a", "b"), ("c", "d")]
        truth = {("a", "b"): True}  # ("c","d") has no known ground truth
        res = calibrate_transitivity(pairs, truth, alpha=0.1)
        assert res["n_grounded"] == 1 and res["n_confirmed"] == 1

    def test_empty_input_gives_zero_bound_not_a_crash(self):
        res = calibrate_transitivity([], {}, alpha=0.1)
        assert res["n_grounded"] == 0 and res["precision_lower_bound"] == 0.0
