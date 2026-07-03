"""
Regression tests from fuzzing `composition_algebra.py` (800 runs, cross-checked
against real addition-mod-N) and `conformal_reasoning.py` (300 coverage configs +
500 ConformalReasoner runs + edge cases) — found 2 real crash bugs, now fixed;
locked down here to prevent recurrence.

Bug 1 — `composition_algebra.fold(())`: an EMPTY sequence leaves `spans=[]`, the
`while len(spans)>1` loop never runs, and `return spans[0]` then crashes with
IndexError on an empty list. Fix: return None for an empty sequence (consistent
with the "None when no reduction is possible" contract, rather than fabricating a
nonexistent identity element).

Bug 2 — `conformal_reasoning.conformal_threshold([], alpha>=1/(n+1))`: with 0
calibration points AND alpha large enough that `k=int(alpha*(n+1))>=1` (n=0, so
alpha>=1 suffices), the expression `s[min(k,n)-1]` = `s[min(k,0)-1]` = `s[-1]` on
an empty list ⟹ IndexError. The bug propagates to `ConformalReasoner.calibrate([])`
(a real code path when true_pairs is empty, e.g. an agent forgetting to supply
calibration examples). Fix: raise a clear ValueError — with 0 calibration points,
the coverage guarantee >=1-alpha CANNOT be established mathematically (the
exchangeability argument needs >=1 sample); better to error out than to silently
return a baseless threshold.
"""
import pytest

from src.reasoning.composition_algebra import fold, learn_composition
from src.reasoning.conformal_reasoning import ConformalReasoner, conformal_threshold


class TestCompositionEmptySequenceRegression:
    def test_fold_empty_sequence_returns_none(self):
        assert fold((), {}) is None
        assert fold((), {("a", "b"): "c"}) is None  # still None even with a nonempty table

    def test_fold_single_element_unaffected(self):
        assert fold((5,), {}) == 5

    def test_learn_composition_empty_chains(self):
        table, conflicts, iters = learn_composition([])
        assert table == {} and conflicts == 0


class TestConformalEmptyCalibrationRegression:
    def test_exact_fuzz_repro_empty_scores_alpha_one(self):
        with pytest.raises(ValueError):
            conformal_threshold([], 1.0)

    def test_empty_scores_various_alpha_raise(self):
        for alpha in (0.0, 0.5, 1.0, 1.5, -0.1):
            with pytest.raises(ValueError):
                conformal_threshold([], alpha)

    def test_nonempty_still_works_after_fix(self):
        # the old invariant (k<1 -> -inf) still holds for n>=1
        assert conformal_threshold([1.0, 2.0, 3.0], 0.0) == float("-inf")
        tau = conformal_threshold([1.0, 2.0, 3.0], 1.0)
        assert tau == 3.0   # alpha=1: only accept the single highest score

    def test_conformal_reasoner_calibrate_empty_raises(self):
        class _Engine:
            def infer(self, x):
                return {}

        cr = ConformalReasoner(_Engine(), alpha=1.0)
        with pytest.raises(ValueError):
            cr.calibrate([])   # empty true_pairs -> cannot calibrate

    def test_conformal_reasoner_calibrate_nonempty_still_works(self):
        class _Engine:
            def infer(self, x):
                return {"b": 0.8, "c": 0.2}

        cr = ConformalReasoner(_Engine(), alpha=0.1)
        cr.calibrate([("a", "b")])
        assert cr.accept("a", "b")
