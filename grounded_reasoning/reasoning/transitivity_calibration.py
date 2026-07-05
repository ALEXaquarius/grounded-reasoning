"""
Empirical Transitivity Calibration — a measured, falsifiable alternative to
declaring a relation "transitive" as a blind, binary modeling assumption.

`GroundedReasoner(transitive_relations={...})` (see agent/verifier.py) closes
one real gap — Theorem G says nothing about whether `via` composes
transitively in reality — with a binary allowlist: declare it, or the guard
raises. That is sound but coarse: it cannot distinguish "genuinely
transitive, just undeclared" from "actually not transitive", and it gives no
sense of HOW confident the declaration should be.

This module answers a narrower, calibratable question instead: given a held-
out sample of (subject, object) pairs the GRAPH marks as grounded via `rel`
(i.e. a composed path exists), each paired with an INDEPENDENTLY KNOWN ground
truth label (not derived from the graph — e.g. human-verified, or checked
against a trusted oracle), what confidence can we place in "a graph-grounded
composed claim for `rel` is actually true"?

The Clopper-Pearson interval (Clopper & Pearson, 1934) gives an EXACT,
distribution-free one-sided lower confidence bound on that true rate from a
finite i.i.d./exchangeable sample — no distributional assumption on the
underlying process, in the same spirit as the conformal guarantee elsewhere
in this project (Theorem K), though the two are classically distinct tools
applied to different questions here. Not new math; the contribution is
pairing it with this system's specific gap.
"""
from __future__ import annotations

import math


def clopper_pearson_lower(k: int, n: int, alpha: float, iters: int = 100) -> float:
    """
    One-sided Clopper-Pearson lower confidence bound (confidence level 1-alpha)
    on the true success probability p, given k successes in n i.i.d./exchangeable
    Bernoulli(p) trials.

    Defined as the p solving P(Binomial(n, p) >= k) = alpha (the standard exact
    interval; equivalently the alpha-quantile of Beta(k, n-k+1)). Computed here
    by bisection on the binomial survival function so no scipy/special-function
    dependency is needed — `P(X >= k | n, p)` is monotonically increasing in p,
    so the root is unique and bisection converges to it.

    Raises
    ------
    ValueError if n < 1, k < 0, k > n, or alpha not in (0, 1): these inputs
    don't correspond to a meaningful calibration sample or confidence level.
    """
    if n < 1:
        raise ValueError(f"clopper_pearson_lower needs n>=1 calibration trials, got n={n}")
    if not (0 <= k <= n):
        raise ValueError(f"k must satisfy 0<=k<=n, got k={k}, n={n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if k == 0:
        return 0.0
    if k == n:
        return alpha ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        cdf_below_k = sum(math.comb(n, i) * mid**i * (1 - mid) ** (n - i) for i in range(k))
        surv = 1 - cdf_below_k  # P(X >= k | n, mid), increasing in mid
        if surv > alpha:
            hi = mid
        else:
            lo = mid
    return lo


def calibrate_transitivity(
    grounded_pairs: list[tuple[object, object]],
    ground_truth: dict[tuple[object, object], bool],
    alpha: float = 0.1,
) -> dict:
    """
    Calibrate confidence that a graph-grounded composed claim is actually true.

    Parameters
    ----------
    grounded_pairs : (subject, object) pairs the graph marks `grounded=True`
        for some relation `rel` (typically `GroundedReasoner.verify(..., via=rel)`
        filtered to `.grounded`). These must be held-out calibration pairs, not
        the pairs you intend to trust the bound for.
    ground_truth : independently-known truth for each pair in `grounded_pairs`
        (NOT derived from the graph itself — see module docstring). Pairs
        missing from this mapping are ignored (skipped, not counted).
    alpha : miscoverage rate; the returned bound holds with confidence 1-alpha.

    Returns
    -------
    dict with `n_grounded` (calibration pairs actually scored), `n_confirmed`
    (how many were true), `precision_lower_bound` (Clopper-Pearson, confidence
    1-alpha), and `alpha`.
    """
    scored = [ground_truth[p] for p in grounded_pairs if p in ground_truth]
    n = len(scored)
    k = sum(1 for v in scored if v)
    bound = clopper_pearson_lower(k, n, alpha) if n > 0 else 0.0
    return {
        "n_grounded": n,
        "n_confirmed": k,
        "precision_lower_bound": bound,
        "alpha": alpha,
    }
