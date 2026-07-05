"""
Demo: Self-Grounded Deductive Consistency (SGDC, Theorem I) -- NO external
knowledge base needed.

Every other demo in this repo supplies an external KB (a fact list, a
database) for the guard to verify claims against. SGDC removes that
requirement entirely: it uses ONLY the LLM's own confident one-hop
assertions -- facts the model already stated with high confidence -- builds
the operator closure from THOSE, and rejects any of the model's OWN
multi-hop conclusions that fall outside its own closure. Self-contradiction
is the hallucination signal, not disagreement with an external source.

The idea (PAPER.md §6): LLMs are reliably accurate on atomic (1-hop) facts
but hallucinate on composition. So take the model at its word for the atomic
facts, and use grounded composition to catch it fabricating on the multi-hop
conclusion -- 0 external knowledge, 0 extra tokens.

This demo simulates what a real LLM call would return (see
grounded_reasoning/experiments/self_grounded_eval.py for the live-DeepSeek
version) so it runs fully offline, with no API key.

Run: python examples/self_grounded_demo.py
"""
from __future__ import annotations

from grounded_reasoning import GroundedReasoner

# --- Step 1: the LLM's OWN confident atomic (1-hop) facts, taken at face value ---
# (this IS the "knowledge base" -- but it's the model's own output, not an
# external database the guard was handed)
LLM_ATOMIC_FACTS = [
    ("sparrow", "is_a", "bird"),
    ("bird", "is_a", "animal"),
    ("animal", "is_a", "organism"),
    ("robin", "is_a", "bird"),
]

# --- Step 2: the LLM's OWN multi-hop conclusion, asked separately ---
# "List everything a sparrow transitively is-a, deduced from your own facts above."
# A real LLM composing several hops sometimes fabricates -- here it claims
# "organism" and "animal" (both correct) but also "plant" (not entailed by
# ANY of its own atomic facts above -- a pure self-contradiction).
LLM_MULTI_HOP_CLAIM = {"animal", "organism", "plant"}


def main() -> None:
    gr = GroundedReasoner()
    gr.add_facts(LLM_ATOMIC_FACTS)  # built from the model's OWN atomic facts, nothing else

    print("=" * 74)
    print("The model's OWN atomic facts (taken at face value, no external KB):")
    for s, r, o in LLM_ATOMIC_FACTS:
        print(f"  {s} --{r}--> {o}")
    print("=" * 74)

    print(f"\nThe model's OWN multi-hop claim: sparrow is_a {sorted(LLM_MULTI_HOP_CLAIM)}")
    print("Self-verifying that claim against the model's OWN facts (no external source):\n")

    # sorted: set iteration order is hash-seed-dependent for strings, and each
    # target here determines a printed line's position -- sorting keeps output
    # (and the ordered assertion below) reproducible across PYTHONHASHSEED.
    claims = [("sparrow", target, "is_a") for target in sorted(LLM_MULTI_HOP_CLAIM)]
    kept, dropped = [], []
    for (subj, obj, rel), verdict in gr.filter_claims(claims):
        (kept if verdict.grounded else dropped).append(obj)
        status = "KEPT (self-consistent)" if verdict.grounded else "REJECTED (self-contradiction)"
        proof = " -> ".join(verdict.proof) if verdict.proof else "(no path in the model's own facts)"
        print(f"  {status:32} sparrow is_a {obj:10} | proof: {proof}")

    print(f"\nSGDC precision on the model's OWN multi-hop claim: "
          f"{len(kept)}/{len(kept) + len(dropped)} kept, {len(dropped)} rejected "
          f"({sorted(dropped)}) -- using ONLY the model's own atomic facts.")

    # --- Step 3: the spectral contradiction certificate (free consequence, Theorem H) ---
    # is_a should be a strict hierarchy (acyclic). If the model's OWN atomic facts
    # ever assert a cycle, that alone certifies self-contradiction -- no multi-hop
    # claim or external KB needed at all.
    print("\n" + "=" * 74)
    print("Bonus: detecting the model contradicting ITSELF in its own atomic facts.")
    print("=" * 74)
    gr2 = GroundedReasoner()
    gr2.add_facts(LLM_ATOMIC_FACTS + [("organism", "is_a", "sparrow")])  # the model asserts a cycle
    cycle = gr2.contradictions("is_a")
    if cycle:
        print(f"Contradiction certificate: {' -> '.join(cycle[0])} -> {cycle[0][0]} "
              f"is a cycle in a relation that must be acyclic -- the model contradicted "
              f"itself, detected at 0 tokens, 0 external knowledge.")
    else:
        print("No contradiction found (unexpected for this input).")

    assert kept == ["animal", "organism"]
    assert dropped == ["plant"]
    assert cycle, "expected the injected cycle to be detected"


if __name__ == "__main__":
    main()
