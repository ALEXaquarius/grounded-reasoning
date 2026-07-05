"""
A/B/C comparison for Theorem N: exact-string matching (safe, limited recall)
versus a fuzzy entity-resolver used blindly versus the same resolver with
`calibrate_normalization` measuring its actual over-merge risk.

Setup: a population of true entities, each with several inconsistent surface
forms (an LLM-extraction-style scenario). A simple fuzzy resolver (casefold +
strip + collapse repeated whitespace) correctly unifies the vast majority of
aliases, but ALSO accidentally collides a small number of genuinely different
entities that happen to normalize to the same key (e.g. two people whose names
differ only in case/whitespace of otherwise-distinct spellings) -- a realistic
failure mode for any non-oracle normalizer.

  A. NO NORMALIZATION: never over-merges (precision stays exactly 1.0, Theorem
     G unmodified) but misses every path fragmented across inconsistent
     surface forms -- recall loss, no risk either way.
  B. FUZZY, TRUSTED BLINDLY: recovers the fragmented paths, but the accidental
     collisions silently risk precision -- and nothing reports this risk.
  C. FUZZY, CALIBRATED (calibrate_normalization): the SAME resolver as B, but
     its over-merge rate is measured from held-out labeled pairs instead of
     assumed away.

Run: python -m grounded_reasoning.experiments.normalization_calibration_eval
(fully offline -- synthetic ground truth, no LLM call.)
"""
from __future__ import annotations

import random

from grounded_reasoning.agent import GroundedReasoner


def _fuzzy_normalize(s: str) -> str:
    return " ".join(s.split()).casefold()


def build_world(seed: int, n_entities: int = 120, collision_rate: float = 0.06):
    """n_entities true entities, each with 1-3 whitespace/case-varied aliases.
    A `collision_rate` fraction of entities are deliberately given a SECOND
    identity's alias spelling too (so the fuzzy resolver's key collides with
    a genuinely different entity) -- the realistic source of over-merge error."""
    rng = random.Random(seed)
    entities = [f"Entity{i}" for i in range(n_entities)]

    def variants(e):
        # sorted, not list(a set literal): set iteration order is hash-seed-
        # dependent, and rng.sample() below picks by POSITION, so an unsorted
        # list here would make the SPECIFIC aliases chosen for each entity
        # depend on PYTHONHASHSEED even with a fixed rng seed.
        return sorted({e, e.upper(), "  " + e + "  ", e.replace("Entity", "entity ")})

    aliases = {e: rng.sample(variants(e), rng.randint(1, 3)) for e in entities}

    # force some accidental collisions: pick pairs of DISTINCT entities and
    # give one of them an alias that fuzzy-normalizes to the same key as the other
    n_collisions = max(1, int(n_entities * collision_rate))
    collided_pairs = []
    pool = entities[:]
    rng.shuffle(pool)
    for i in range(0, 2 * n_collisions, 2):
        e1, e2 = pool[i], pool[i + 1]
        colliding_key = _fuzzy_normalize(rng.choice(aliases[e1]))
        # give e2 a NEW alias that normalizes to e1's key but is guaranteed to
        # differ as a RAW string from every alias already in use (a tab is
        # whitespace to _fuzzy_normalize's .split() but appears in none of the
        # variants() forms) -- otherwise the injected alias could coincide
        # character-for-character with one of e1's own variants (e.g. via
        # .title() reconstructing the exact base name) and create a collision
        # at the raw-string level, contaminating the "no normalization" control.
        colliding_alias = colliding_key + "\t"
        assert colliding_alias not in aliases[e1] and colliding_alias not in aliases[e2]
        aliases[e2].append(colliding_alias)
        collided_pairs.append((e1, e2))

    true_edges = [(entities[rng.randint(0, i - 1)], entities[i]) for i in range(1, n_entities)]
    adj: dict[str, set[str]] = {}
    for a, b in true_edges:
        adj.setdefault(a, set()).add(b)

    def true_closure(x):
        seen, frontier = set(), list(adj.get(x, ()))
        while frontier:
            nf = []
            for u in frontier:
                if u not in seen:
                    seen.add(u)
                    nf += list(adj.get(u, ()))
            frontier = nf
        return seen

    facts = [
        (rng.choice(aliases[a]), "parent", rng.choice(aliases[b])) for a, b in true_edges
    ]
    return entities, facts, true_closure, collided_pairs, aliases


def run(seed: int = 0, alpha: float = 0.1, n_entities: int = 120) -> dict:
    entities, facts, true_closure, collided_pairs, aliases = build_world(seed, n_entities)

    gr_none = GroundedReasoner()
    gr_none.add_facts(facts)
    gr_fuzzy = GroundedReasoner(normalize=_fuzzy_normalize)
    gr_fuzzy.add_facts(facts)

    n_checked = fp_none = fp_fuzzy = recall_gain = n_true = 0
    for a in entities:
        for b in entities:
            if a == b:
                continue
            n_checked += 1
            truly_related = b in true_closure(a)
            n_true += truly_related
            v_none = gr_none.verify(a, b, via="parent").grounded
            v_fuzzy = gr_fuzzy.verify(a, b, via="parent").grounded
            fp_none += v_none and not truly_related
            fp_fuzzy += v_fuzzy and not truly_related
            recall_gain += v_fuzzy and not v_none

    # C. calibrate the SAME fuzzy resolver from held-out labeled (alias-pair,
    # same-entity?) data -- ground truth known independently (constructed above)
    rng = random.Random(1000 + seed)
    labeled = []
    for e, als in aliases.items():
        if len(als) >= 2:
            labeled.append((als[0], als[1], True))  # two aliases of the SAME entity
    for e1, e2 in collided_pairs:
        labeled.append((aliases[e1][0], aliases[e2][-1], False))  # the injected collision
    rng.shuffle(labeled)
    half = len(labeled) // 2
    cal, held_out = labeled[:half], labeled[half:]
    calibration = gr_fuzzy.calibrate_normalization(cal, alpha=alpha)

    held_out_gt = {(a, b): t for a, b, t in held_out}
    held_out_merged = [(a, b) for a, b, _ in held_out if _fuzzy_normalize(a) == _fuzzy_normalize(b)]
    held_out_precision = (
        sum(held_out_gt[p] for p in held_out_merged) / len(held_out_merged)
        if held_out_merged else None
    )

    return {
        "n_entities": n_entities,
        "n_true_pairs": n_true,
        "n_injected_collisions": len(collided_pairs),
        "A_no_normalization": {"false_positives": fp_none, "recall_relative_to_fuzzy": -recall_gain},
        "B_fuzzy_trusted_blindly": {"false_positives": fp_fuzzy, "risk_reported": None},
        "C_fuzzy_calibrated": {
            "n_merges_scored": calibration["n_grounded"],
            "n_correct_merges": calibration["n_confirmed"],
            "merge_precision_lower_bound": calibration["precision_lower_bound"],
            "held_out_test_precision": held_out_precision,
            "bound_held": (
                held_out_precision is None or calibration["precision_lower_bound"] <= held_out_precision
            ),
        },
    }


def main() -> None:
    res = run()
    print("=" * 74)
    print(f"{res['n_entities']} true entities, {res['n_injected_collisions']} deliberately "
          f"injected alias collisions (simulating a realistic fuzzy-matcher error)")
    print("=" * 74)
    a, b, c = res["A_no_normalization"], res["B_fuzzy_trusted_blindly"], res["C_fuzzy_calibrated"]
    print(f"\nA. No normalization: {a['false_positives']} false positives (always 0), "
          f"but misses paths fuzzy-matching would have recovered.")
    print(f"B. Fuzzy, trusted blindly: {b['false_positives']} false positives -- "
          f"and reports NO risk number for them.")
    print(f"C. Fuzzy, calibrated: {c['n_correct_merges']}/{c['n_merges_scored']} merges "
          f"confirmed correct in calibration; reports merge precision >= "
          f"{c['merge_precision_lower_bound']:.0%} with 90% confidence -- "
          f"held on a fresh held-out set: {c['bound_held']}.")
    print("\n=> Same resolver as B, but C tells you exactly how much to trust it "
          "instead of leaving the risk unmeasured.")


if __name__ == "__main__":
    main()
