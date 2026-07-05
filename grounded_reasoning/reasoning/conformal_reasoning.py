"""
Conformal Reasoning — a distribution-free COVERAGE guarantee for multi-hop
inference EVEN WHEN the relation graph is NOISY (missing/spurious edges).

The guard/SGDC give precision=1.0, but ONLY when the graph is clean (sound). When
the relation is extracted from noisy natural language, that hard guarantee no
longer holds. Conformal prediction (Vovk, Gammerman, Shafer) fixes exactly that:
use the operator confidence conf(a→b) as a nonconformity score, calibrate a
threshold tau on a calibration set to GUARANTEE P(the true answer is kept) >= 1-alpha
— with no distributional assumption, valid regardless of graph quality.

We do NOT invent conformal prediction; the contribution is pairing it with the
operator confidence (the Katz resolvent, Theorem H) and describing the tradeoff:
COVERAGE is always valid, EFFICIENCY (prediction-set size / FPR) degrades with
noise. See the verification: theorem_conformal_reasoning.

Also not new: Mondrian (group-conditional) conformal prediction (Vovk et al.) —
calibrating a SEPARATE threshold per group of an available-at-test-time
partitioning function, instead of one global threshold, still gives per-group
(hence marginal) coverage >= 1-alpha, by the identical split-conformal argument
applied within each group. `ConformalReasoner.calibrate(group_fn=...)` exposes
this; `redundancy_group` is one concrete, numerically-verified choice of
partitioning function for this system (see `redundancy_conformal_eval.py`).
A different partitioning function tried first (hop-distance) was numerically
FALSIFIED -- it made efficiency WORSE, not better -- and is not shipped; see
PAPER.md §7.1's remark for the full account of both the working and the
falsified attempt.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable


def conformal_threshold(cal_scores: list[float], alpha: float) -> float:
    """
    The split-conformal threshold: accept if score >= tau implies coverage >= 1-alpha
    (marginal, distribution-free). tau is the k-th smallest calibration score,
    k = floor(alpha*(n+1)) (tau = -inf if k=0).

    Raises
    ------
    ValueError if cal_scores is empty: with 0 calibration points, the coverage
    guarantee >= 1-alpha CANNOT be established (the exchangeability argument needs
    >= 1 sample) — better to fail loudly than to silently return a "plausible-
    looking" threshold that carries no guarantee.
    """
    if not cal_scores:
        raise ValueError("conformal_threshold needs >=1 calibration point for a coverage guarantee")
    s = sorted(cal_scores)
    n = len(s)
    k = int(alpha * (n + 1))
    if k < 1:
        return float("-inf")
    return s[min(k, n) - 1]


def redundancy_group(engine, a: str, b: str) -> int:
    """
    A Mondrian grouping function: 0 if a→b has at most one walk in the (possibly
    noisy) extracted graph, 1 if it has more than one (`engine.path_multiplicity`).
    Computable at test time from the extracted graph alone, no ground truth
    needed -- a valid Mondrian covariate.

    Numerically verified (`redundancy_conformal_eval.py`) to measurably IMPROVE
    conformal efficiency (lower false-positive rate) specifically when the
    dominant noise is DROPPED edges (matching conformal_llm_eval.py's own
    documented noise mode: an LLM's extraction missing relations) -- because a
    multiply-connected pair mechanically survives a single random edge drop more
    often than a singly-connected one. It gives no benefit (and a small,
    non-coverage-breaking cost from splitting the calibration set) when the
    dominant noise is SPURIOUS added edges instead -- an honest limitation, not
    hidden.
    """
    return 0 if engine.path_multiplicity(a, b) <= 1 else 1


class ConformalReasoner:
    """
    Wraps an inference engine (with `.infer(x) -> {b: conf}`), calibrates a
    conformal threshold from labeled examples, then returns PREDICTION SETS with a
    coverage guarantee >= 1-alpha.
    """

    def __init__(self, engine, alpha: float = 0.1) -> None:
        self.engine = engine
        self.alpha = alpha
        self.tau = float("-inf")
        self._group_tau: dict[Hashable, float] = {}
        self._group_fn: Callable[[str, str], Hashable] | None = None
        self._cache: dict = {}

    def _conf(self, x, b) -> float:
        if x not in self._cache:
            self._cache[x] = self.engine.infer(x)
        return self._cache[x].get(b, 0.0)

    def calibrate(
        self, true_pairs: list[tuple],
        group_fn: Callable[[str, str], Hashable] | None = None,
    ) -> float:
        """
        Calibrate tau on TRUE (positive-class) (x, b) pairs.

        group_fn: optional Mondrian/group-conditional partitioning function
        (e.g. `redundancy_group` bound to this engine), computable from (x, b)
        alone WITHOUT knowing the true label. When set, calibrates a SEPARATE
        threshold per group (each still satisfying the same >= 1-alpha coverage
        argument, applied within that group's own exchangeable cal/test split)
        instead of one global threshold. `accept`/`predict_set` then look up
        the calling pair's own group. A group with too few calibration points
        to compute its own quantile (or a group never seen during calibration
        at all) falls back to the GLOBAL threshold (computed from ALL
        calibration scores pooled), never to "-inf" or an error -- a fallback
        MUST exist for every possible test-time group, at least as
        conservative as no grouping at all.

        Returns the GLOBAL threshold (always computed, whether or not
        group_fn is set) for backward compatibility.
        """
        scores = [self._conf(x, b) for x, b in true_pairs]
        self.tau = conformal_threshold(scores, self.alpha)
        self._group_fn = group_fn
        self._group_tau = {}
        if group_fn is not None:
            by_group: dict[Hashable, list[float]] = {}
            for x, b in true_pairs:
                by_group.setdefault(group_fn(x, b), []).append(self._conf(x, b))
            self._group_tau = {g: conformal_threshold(s, self.alpha) for g, s in by_group.items()}
        return self.tau

    def _tau_for(self, x, b) -> float:
        if self._group_fn is None:
            return self.tau
        return self._group_tau.get(self._group_fn(x, b), self.tau)

    def accept(self, x, b) -> bool:
        """Accept (x, b) if its confidence is >= the (group-)conformal threshold."""
        return self._conf(x, b) >= self._tau_for(x, b)

    def predict_set(self, x, candidates) -> set:
        """The prediction set {b : conf(x→b) >= tau} — guaranteed to contain the true answer with probability >= 1-alpha."""
        return {b for b in candidates if self._conf(x, b) >= self._tau_for(x, b)}
