"""
A/B comparison: the binary `transitive_relations=` allowlist (declare-or-reject)
versus the calibrated `calibrate_transitivity()` bound (Theorem M), on a
synthetic relation that is NOT perfectly transitive — the exact situation
neither the algebra nor a blind declaration can safely resolve on its own.

Setup: a "trusts"-like relation where a 2-hop composed claim (A trusts B,
B trusts C) is actually true in the world with probability q < 1 (ground
truth is simulated here so it's known for scoring — in a real deployment it
would come from a trusted oracle/human check on a held-out sample).

  A. BINARY (transitive_relations=...): either declare "trusts" transitive
     (every grounded claim is asserted with silent full confidence, including
     the (1-q) fraction that's actually wrong) or don't (every grounded claim
     is rejected outright, losing the q fraction that's actually right). No
     middle ground, no number attached to the risk either way.
  B. CALIBRATED (calibrate_transitivity): measures a lower confidence bound on
     the true composed-claim precision from held-out labeled pairs, and
     reports it honestly instead of guessing.

Run: python -m grounded_reasoning.experiments.transitivity_calibration_eval
(fully offline — this experiment doesn't call an LLM, it evaluates the
calibration MACHINERY itself against a known synthetic ground truth).
"""
from __future__ import annotations

import random

from grounded_reasoning.agent import GroundedReasoner


def build_world(seed: int, q: float, n_chains: int = 200):
    """n_chains disjoint (A,B,C) triples; each composed A-trusts-C claim is
    actually true with probability q (independent of the graph, which always
    has the A->B->C edges present -- the graph can't see q at all)."""
    rng = random.Random(seed)
    gr = GroundedReasoner()
    pairs, truth = [], {}
    for i in range(n_chains):
        a, b, c = f"A{i}", f"B{i}", f"C{i}"
        gr.add_facts([(a, "trusts", b), (b, "trusts", c)])
        pairs.append((a, c))
        truth[(a, c)] = rng.random() < q
    return gr, pairs, truth


def run(q: float = 0.85, alpha: float = 0.1, n_chains: int = 200, seed: int = 0) -> dict:
    gr, pairs, truth = build_world(seed, q, n_chains)
    rng = random.Random(1000 + seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    cal_pairs, test_pairs = shuffled[:half], shuffled[half:]

    # A. binary: "declared" trusts everything grounded (no calibration at all)
    declared_precision = sum(truth[p] for p in test_pairs) / len(test_pairs)
    # A. binary: "undeclared" rejects everything -> recovers 0 of the q fraction
    #    that was actually true (the cost of the safe-but-blind alternative)
    undeclared_recovered = 0

    # B. calibrated: bound from the calibration half, checked against the test half
    labeled_cal = [(a, c, truth[(a, c)]) for a, c in cal_pairs]
    cal = gr.calibrate_transitivity("trusts", labeled_cal, alpha=alpha)
    actual_test_precision = sum(truth[p] for p in test_pairs) / len(test_pairs)
    bound_holds = cal["precision_lower_bound"] <= actual_test_precision

    res = {
        "true_q": q,
        "n_calibration": cal["n_grounded"],
        "A_declared_silently_trusts_all": {
            "claims_kept": len(test_pairs),
            "actual_precision": round(declared_precision, 4),
            "risk_reported": None,  # the binary mechanism reports NO risk estimate
        },
        "A_undeclared_rejects_all": {
            "claims_kept": undeclared_recovered,
            "true_claims_lost": round(q * len(test_pairs)),
        },
        "B_calibrated": {
            "n_confirmed_in_calibration": cal["n_confirmed"],
            "precision_lower_bound": round(cal["precision_lower_bound"], 4),
            "held_on_held_out_test_set": bound_holds,
        },
    }
    return res


def main() -> None:
    res = run()
    print("=" * 74)
    print(f"Synthetic 'trusts' world: true composed-claim precision q={res['true_q']}")
    print("=" * 74)
    print(f"\nA. binary, DECLARED transitive: keeps all {res['A_declared_silently_trusts_all']['claims_kept']} "
          f"test claims, actual precision={res['A_declared_silently_trusts_all']['actual_precision']:.0%}, "
          f"but reports NO risk number -- silently wrong {1-res['A_declared_silently_trusts_all']['actual_precision']:.0%} of the time.")
    print(f"A. binary, UNDECLARED: keeps 0 claims, loses "
          f"{res['A_undeclared_rejects_all']['true_claims_lost']} claims that were actually TRUE.")
    print(f"\nB. calibrated: from {res['n_calibration']} held-out calibration pairs "
          f"({res['B_calibrated']['n_confirmed_in_calibration']} confirmed), "
          f"reports precision >= {res['B_calibrated']['precision_lower_bound']:.0%} "
          f"with 90% confidence -- and this bound held on a fresh held-out test set: "
          f"{res['B_calibrated']['held_on_held_out_test_set']}.")
    print("\n=> The calibrated bound gives an honest, checkable number in the exact "
          "situation where the binary mechanism can only guess blindly or reject everything.")


if __name__ == "__main__":
    main()
