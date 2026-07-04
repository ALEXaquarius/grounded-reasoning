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
"""
from __future__ import annotations


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
        self._cache: dict = {}

    def _conf(self, x, b) -> float:
        if x not in self._cache:
            self._cache[x] = self.engine.infer(x)
        return self._cache[x].get(b, 0.0)

    def calibrate(self, true_pairs: list[tuple]) -> float:
        """Calibrate tau on TRUE (positive-class) (x, b) pairs."""
        scores = [self._conf(x, b) for x, b in true_pairs]
        self.tau = conformal_threshold(scores, self.alpha)
        return self.tau

    def accept(self, x, b) -> bool:
        """Accept (x, b) if its confidence is >= the conformal threshold."""
        return self._conf(x, b) >= self.tau

    def predict_set(self, x, candidates) -> set:
        """The prediction set {b : conf(x→b) >= tau} — guaranteed to contain the true answer with probability >= 1-alpha."""
        return {b for b in candidates if self._conf(x, b) >= self.tau}
