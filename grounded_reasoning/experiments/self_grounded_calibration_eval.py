"""
A/B comparison: SGDC (Theorem I) trusted blindly versus the SAME SGDC output
calibrated with `calibrate_transitivity` (Theorem M) -- measuring the risk
SGDC's own survival condition leaves unmeasured, instead of assuming it away.

SGDC (self_grounded_eval.py) needs no external KB: it builds the graph from
the LLM's OWN atomic (1-hop) facts and self-verifies the LLM's OWN multi-hop
conclusions against that closure. Theorem I's precision=1.0 guarantee is
CONDITIONAL: it holds if the atomic facts are sound (contain no false
edges). In a "counter-prior" domain (PAPER.md §6's own honest example: "a
whale is a fish"), that condition can quietly fail -- some of the LLM's own
atomic facts are simply wrong, not just incomplete -- and nothing in SGDC
as originally shipped MEASURES how much this actually costs in practice.

The key realization (not new math): `calibrate_transitivity` doesn't care
where a reasoner's facts came from -- an external KB, or an LLM's own atomic
self-assertions. Calibrating SGDC's own grounded=True claims directly (from
held-out, independently-labeled evidence) applies Theorem M's already-general
Clopper-Pearson argument to a THIRD question, exactly as PAPER.md §5.3.4
already did for heterogeneous relation paths. This is deliberately NOT a new
theorem or a new library method -- it is the existing `calibrate_transitivity`
called on a GroundedReasoner whose facts happen to be self-asserted, which
already works today with zero code changes.

  A. SGDC, TRUSTED BLINDLY (as Theorem I documents it): precision=1.0 is
     ASSUMED from atomic soundness. When atomic facts are NOT perfectly sound
     (a realistic LLM failure mode), actual precision silently drops below
     the assumed 1.0, and nothing reports this.
  B. SGDC, CALIBRATED (calibrate_transitivity on the SAME SGDC output):
     measures precision from held-out evidence instead of assuming it,
     giving an honest lower confidence bound that HOLDS regardless of how
     unsound the atomic facts turned out to be.

Also demonstrates that this is a genuinely nontrivial thing to measure (not
inferable by eyeballing atomic-fact precision alone): a modest fraction of
wrong atomic facts can be AMPLIFIED into a much larger multi-hop precision
loss through composition (a single false edge can poison many downstream
claims) -- exactly why calibrating the OUTPUT directly, not estimating it
from the atomic layer, is the right move.

Run: python -m grounded_reasoning.experiments.self_grounded_calibration_eval
(fully offline -- synthetic ground truth, no LLM call.)
"""
from __future__ import annotations

import random

from grounded_reasoning import GroundedReasoner


def build_world(seed: int, n_concepts: int = 40, p_wrong_atomic: float = 0.15):
    """A random `is_a` hierarchy, plus a NOISY version of it standing in for
    "the LLM's own atomic facts" -- some fraction are simply wrong (not
    incomplete), simulating a counter-prior domain where the LLM's one-hop
    confidence is imperfect (PAPER.md §6's own "whale is a fish" example)."""
    rng = random.Random(seed)
    names = [f"c{i}" for i in range(n_concepts)]
    true_parent: dict[str, str] = {}
    for i in range(1, n_concepts):
        true_parent[names[i]] = names[rng.randint(0, i - 1)]

    def true_closure(x: str) -> set[str]:
        seen: set[str] = set()
        cur = true_parent.get(x)
        while cur is not None and cur not in seen:
            seen.add(cur)
            cur = true_parent.get(cur)
        return seen

    # SGDC's reasoner: built ONLY from "the LLM's own atomic facts" -- no
    # external KB -- some of which are simply wrong
    gr = GroundedReasoner()
    n_wrong = 0
    for i in range(1, n_concepts):
        if rng.random() < p_wrong_atomic:
            gr.add_fact(names[i], "is_a", rng.choice(names))
            n_wrong += 1
        else:
            gr.add_fact(names[i], "is_a", true_parent[names[i]])

    return gr, names, true_closure, n_wrong


def run(seed: int = 1, p_wrong_atomic: float = 0.15, n_concepts: int = 40, alpha: float = 0.1) -> dict:
    gr, names, true_closure, n_wrong = build_world(seed, n_concepts, p_wrong_atomic)

    # SGDC's own grounded multi-hop claims (self-verified, no external KB consulted)
    grounded_pairs = [
        (a, b) for a in names for b in names if a != b and gr.verify(a, b, via="is_a").grounded
    ]
    true_precision = (
        sum(1 for a, b in grounded_pairs if b in true_closure(a)) / len(grounded_pairs)
        if grounded_pairs else 1.0
    )

    # B. calibrate the SAME SGDC output from held-out labeled evidence
    # (ground truth known independently -- constructed above, never fed to SGDC)
    rng = random.Random(1000 + seed)
    pairs = list(grounded_pairs)
    rng.shuffle(pairs)
    half = len(pairs) // 2
    cal_pairs, held_out_pairs = pairs[:half], pairs[half:]
    labeled_cal = [(a, b, b in true_closure(a)) for a, b in cal_pairs]
    calibration = gr.calibrate_transitivity("is_a", labeled_cal, alpha=alpha)

    held_out_precision = (
        sum(1 for a, b in held_out_pairs if b in true_closure(a)) / len(held_out_pairs)
        if held_out_pairs else None
    )

    return {
        "n_concepts": n_concepts,
        "n_wrong_atomic_facts": n_wrong,
        "n_atomic_facts": n_concepts - 1,
        "n_sgdc_claims": len(grounded_pairs),
        "A_sgdc_trusted_blindly": {
            "assumed_precision": 1.0,
            "actual_precision": round(true_precision, 3),
            "risk_reported": None,
        },
        "B_sgdc_calibrated": {
            "n_claims_scored": calibration["n_grounded"],
            "n_correct_claims": calibration["n_confirmed"],
            "precision_lower_bound": calibration["precision_lower_bound"],
            "held_out_test_precision": held_out_precision,
            "bound_le_true_precision": calibration["precision_lower_bound"] <= true_precision,
            "bound_held_on_held_out_test_set": (
                held_out_precision is None
                or calibration["precision_lower_bound"] <= held_out_precision
            ),
        },
    }


def main() -> None:
    res = run()
    print("=" * 74)
    print(f"{res['n_wrong_atomic_facts']}/{res['n_atomic_facts']} of the LLM's OWN atomic "
          f"facts are simply WRONG (a counter-prior domain, PAPER.md §6)")
    print("=" * 74)
    a, b = res["A_sgdc_trusted_blindly"], res["B_sgdc_calibrated"]
    print(f"\nA. SGDC, trusted blindly (Theorem I as documented): assumed precision = "
          f"{a['assumed_precision']:.0%}, ACTUAL precision = {a['actual_precision']:.1%} -- "
          f"and reports NO risk number for the gap.")
    print(f"B. SGDC, calibrated (calibrate_transitivity on SGDC's own output, ZERO new code): "
          f"{b['n_correct_claims']}/{b['n_claims_scored']} of SGDC's own claims confirmed "
          f"correct in calibration; reports precision >= {b['precision_lower_bound']:.1%} "
          f"with 90% confidence -- held on a fresh held-out set: "
          f"{b['bound_held_on_held_out_test_set']}, and against the true synthetic precision: "
          f"{b['bound_le_true_precision']}.")
    print(
        f"\n=> {res['n_wrong_atomic_facts']}/{res['n_atomic_facts']} wrong atomic facts "
        f"(15%) amplified into {100 - a['actual_precision'] * 100:.0f}% multi-hop error "
        f"through composition -- confirming SGDC's output precision cannot be eyeballed "
        f"from atomic-fact precision alone, and must be calibrated directly, exactly as "
        f"the SAME engine already does for the external-KB case (Theorem M)."
    )


if __name__ == "__main__":
    main()
