"""
Demo: measuring trust instead of assuming it (Theorem M + Theorem N).

Two modeling assumptions the hard guard cannot verify from the graph alone:

  1. Is `via=rel` genuinely transitive in the real world, or did the graph
     just happen to look that way on the facts supplied? (Theorem M)
  2. Does `normalize=` ever accidentally merge two DIFFERENT real-world
     entities, silently risking precision? (Theorem N)

Rather than a binary "trust it" / "reject it" choice, calibrate_transitivity()
and calibrate_normalization() turn each assumption into a measured
Clopper-Pearson lower confidence bound from held-out, independently-labeled
evidence -- the SAME calibration engine, applied to two different questions.

Fully offline, synthetic ground truth (so the true answer is known and the
bound can be checked against it) -- no API key needed.

Run: python examples/calibration_demo.py
"""
from __future__ import annotations

import random

from grounded_reasoning import GroundedReasoner


def transitivity_demo() -> None:
    print("=" * 74)
    print("1. Theorem M -- calibrating a 'mostly, not perfectly, transitive' relation")
    print("=" * 74)
    print(
        "\n'trusts' is NOT logically transitive (A trusts B, B trusts C does not\n"
        "imply A trusts C) -- but suppose in THIS domain it holds 85% of the time.\n"
    )

    rng = random.Random(0)
    true_precision = 0.85
    n_people = 60
    people = [f"person{i}" for i in range(n_people)]

    gr = GroundedReasoner()
    # a chain of "trusts" edges -- the graph itself doesn't know how often
    # the composed claim is REALLY true in the world; only held-out labels can say
    for i in range(1, n_people):
        gr.add_fact(people[rng.randint(0, i - 1)], "trusts", people[i])

    # held-out evidence: for pairs the graph marks grounded=True via "trusts",
    # is the composed claim ACTUALLY true, independently of the graph? (simulated
    # here at the KNOWN true_precision, standing in for human-verified labels)
    grounded_pairs = [
        (a, b) for a in people for b in people
        if a != b and gr.verify(a, b, via="trusts").grounded
    ]
    rng.shuffle(grounded_pairs)
    half = len(grounded_pairs) // 2
    cal_pairs, held_out_pairs = grounded_pairs[:half], grounded_pairs[half:]
    labeled_cal = [(a, b, rng.random() < true_precision) for a, b in cal_pairs]

    result = gr.calibrate_transitivity("trusts", labeled_cal, alpha=0.1)
    bound = result["precision_lower_bound"]
    print(f"Calibrated from {result['n_grounded']} held-out labeled pairs "
          f"({result['n_confirmed']} confirmed true).")
    print(f"Clopper-Pearson bound: precision >= {bound:.1%} with 90% confidence.")
    print(f"(True precision in this synthetic world: {true_precision:.0%} -- "
          f"bound <= true precision: {bound <= true_precision})")

    held_out_labels = [rng.random() < true_precision for _ in held_out_pairs]
    held_out_precision = sum(held_out_labels) / len(held_out_labels)
    print(f"A FRESH held-out test sample measured {held_out_precision:.1%} precision "
          f"-- bound holds: {bound <= held_out_precision}")


def normalization_demo() -> None:
    print("\n" + "=" * 74)
    print("2. Theorem N -- calibrating a fuzzy entity resolver's over-merge risk")
    print("=" * 74)
    print(
        "\nA casefold+strip normalizer fixes inconsistent LLM-extraction spelling\n"
        "('Bob Chen' vs 'bob chen'), but can ALSO accidentally collide two\n"
        "genuinely different real-world people who happen to share a name.\n"
    )

    def fuzzy_normalize(s: str) -> str:
        return " ".join(s.split()).casefold()

    gr = GroundedReasoner(normalize=fuzzy_normalize)
    gr.add_facts([
        ("Alice", "parent", "Bob Chen"),   # Bob Chen (real parent of Alice)
        ("bob chen", "parent", "Carol"),   # SAME Bob Chen, case variant -- a correct merge
        ("BOB CHEN", "parent", "Dave"),    # a DIFFERENT, unrelated "Bob Chen" -- accidental collision
    ])

    v_correct = gr.verify("Alice", "Carol", via="parent")
    v_wrong = gr.verify("Alice", "Dave", via="parent")
    print(f"verify('Alice','Carol'): grounded={v_correct.grounded} "
          f"(correct -- same Bob Chen, real path)")
    print(f"verify('Alice','Dave'):  grounded={v_wrong.grounded} "
          f"(a FALSE POSITIVE in reality -- the two 'Bob Chen's got merged; "
          f"this is Theorem N's exact failure mode)")

    # held-out, independently-labeled (a, b, is_same_entity) triples -- known
    # independently of the normalizer (e.g. a human checked employee records)
    labeled = [
        ("Bob Chen", "bob chen", True),    # same entity -- a correct merge
        ("Bob Chen", "BOB CHEN", False),   # different entity -- the collision
        ("bob chen", "BOB CHEN", False),   # different entity -- also collides
    ]
    result = gr.calibrate_normalization(labeled, alpha=0.1)
    print(f"\nOf {len(labeled)} labeled pairs, the normalizer actually MERGED "
          f"{result['n_grounded']} of them ({result['n_confirmed']} correctly).")
    print(f"Clopper-Pearson bound: merge precision >= "
          f"{result['precision_lower_bound']:.1%} with 90% confidence -- "
          f"a real number instead of blind trust in the resolver.")


if __name__ == "__main__":
    transitivity_demo()
    normalization_demo()
    print("\n" + "=" * 74)
    print("=> Same calibration engine (Clopper-Pearson), two different blind spots "
          "measured instead of assumed away.")
