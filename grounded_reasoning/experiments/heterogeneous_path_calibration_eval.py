"""
Demonstrates `verify_path`/`calibrate_path`: verifying and calibrating a claim
that chains through DIFFERENT relation types, not a single relation's closure.

Scenario: "financially_dependent_on" is a derived claim composed from
`parent` then `employer` ("X's parent works for Y" -> "X is financially
dependent on Y"). This is a real semantic hypothesis, not something the
algebra can confirm on its own (Theorem G only guarantees the PATH exists,
not that the composed claim means what we intend it to mean — same boundary
as Theorem M, now demonstrated on a heterogeneous chain instead of a
same-relation closure). The claim holds with probability q < 1 (e.g. the
parent may be self-employed, retired, or the connection may be too indirect
to count as "dependent").

Run: python -m grounded_reasoning.experiments.heterogeneous_path_calibration_eval
(fully offline -- synthetic ground truth, no LLM call.)
"""
from __future__ import annotations

import random

from grounded_reasoning.agent import GroundedReasoner


def build_world(seed: int, q: float, n_chains: int = 150):
    rng = random.Random(seed)
    gr = GroundedReasoner()
    pairs, truth = [], {}
    for i in range(n_chains):
        x, parent, employer = f"X{i}", f"P{i}", f"Y{i}"
        gr.add_facts([(x, "parent", parent), (parent, "employer", employer)])
        pairs.append((x, employer))
        truth[(x, employer)] = rng.random() < q
    return gr, pairs, truth


def run(q: float = 0.7, alpha: float = 0.1, n_chains: int = 150, seed: int = 0) -> dict:
    gr, pairs, truth = build_world(seed, q, n_chains)
    via = ["parent", "employer"]

    # sanity: verify_path finds every constructed chain (Theorem G's guarantee,
    # exposed for a heterogeneous path -- this part has NOTHING to do with q)
    all_found = all(gr.verify_path(x, y, via).grounded for x, y in pairs)

    rng = random.Random(1000 + seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    cal_pairs, test_pairs = shuffled[:half], shuffled[half:]

    labeled_cal = [(x, y, truth[(x, y)]) for x, y in cal_pairs]
    calibration = gr.calibrate_path(via, labeled_cal, alpha=alpha)
    actual_test_precision = sum(truth[p] for p in test_pairs) / len(test_pairs)
    # Two different checks, deliberately kept separate:
    #  - bound_le_true_q: what Clopper-Pearson actually guarantees (>=1-alpha of
    #    the time, against the TRUE parameter q -- known here because q is
    #    synthetic ground truth, not observable in a real deployment).
    #  - bound_held_on_held_out_test_set: what a real user would see (compared
    #    against a finite held-out SAMPLE, not the true q) -- realistic, but
    #    noisier than the theorem's own guarantee, since it compounds the
    #    calibration sample's randomness with the test sample's randomness.
    bound_le_true_q = calibration["precision_lower_bound"] <= q
    bound_holds = calibration["precision_lower_bound"] <= actual_test_precision

    return {
        "true_q": q,
        "path_verification_matches_construction": all_found,
        "n_calibration": calibration["n_grounded"],
        "n_confirmed": calibration["n_confirmed"],
        "precision_lower_bound": round(calibration["precision_lower_bound"], 4),
        "actual_test_precision": round(actual_test_precision, 4),
        "bound_le_true_q": bool(bound_le_true_q),
        "bound_held_on_held_out_test_set": bool(bound_holds),
    }


def main() -> None:
    res = run()
    print("=" * 74)
    print("Heterogeneous path: subject --parent--> X --employer--> object")
    print(f"('financially_dependent_on', true composed-claim precision q={res['true_q']})")
    print("=" * 74)
    print(f"\nverify_path finds every constructed chain (Theorem G, exposed for a "
          f"mixed-relation path): {res['path_verification_matches_construction']}")
    print(f"\ncalibrate_path from {res['n_calibration']} held-out pairs "
          f"({res['n_confirmed']} confirmed): reports precision >= "
          f"{res['precision_lower_bound']:.0%} with 90% confidence.")
    print(f"True q: {res['true_q']:.0%} -- bound <= true q: {res['bound_le_true_q']} "
          f"(this is what Clopper-Pearson actually guarantees, >=90% of the time).")
    print(f"Held-out test SAMPLE precision: {res['actual_test_precision']:.0%} -- "
          f"bound held there too: {res['bound_held_on_held_out_test_set']} "
          f"(a noisier, realistic proxy — real deployments don't know the true q).")
    print("\n=> Same calibration machinery as Theorem M, applied to a FIXED "
          "heterogeneous relation chain instead of a single relation's closure.")


if __name__ == "__main__":
    main()
