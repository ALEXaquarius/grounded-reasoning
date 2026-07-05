"""
Demo: GroundedReasoner as a post-processing guard for a RAG/agent pipeline.

Scenario: a small medical knowledge base (drug -treats-> symptom -symptom_of->
condition, the kind of thing an LLM might extract from documents), plus a
batch of "a drug treats a condition" claims an LLM proposed by composing two
DIFFERENT relations together. Some claims are TRUE compositions of the KB;
others are FABRICATED (wrong endpoint, wrong relation, or wrong order).
filter_claims() dispatches a heterogeneous `via=[...]` claim to verify_path()
and keeps only the ones with a real evidence path -- at 0 model tokens, with
a proof for every one it keeps.

This is the pattern described in docs/integration.md #4 (the post-processing
guard) combined with PAPER.md §5.3.4's heterogeneous relation-path
verification, made runnable. Fully offline -- no API key needed.

Run: python examples/rag_pipeline_demo.py
"""
from __future__ import annotations

from grounded_reasoning import GroundedReasoner

# --- The knowledge base (facts an LLM extracted from documents earlier) ---
KB_FACTS = [
    ("aspirin", "treats", "headache"),
    ("aspirin", "treats", "fever"),
    ("ibuprofen", "treats", "inflammation"),
    ("inflammation", "symptom_of", "arthritis"),
    ("headache", "symptom_of", "migraine"),
    ("fever", "symptom_of", "flu"),
    ("flu", "caused_by", "influenza_virus"),
]

CHAIN = ["treats", "symptom_of"]  # "drug treats a symptom that is a symptom_of a condition"

# --- Claims an LLM proposed after "reasoning" over the KB above ---
# (subject, object, is_this_claim_actually_true)
LLM_CLAIMS = [
    ("aspirin", "migraine", True),          # aspirin-treats->headache-symptom_of->migraine: real
    ("aspirin", "flu", True),                # aspirin-treats->fever-symptom_of->flu: real
    ("ibuprofen", "arthritis", True),        # ibuprofen-treats->inflammation-symptom_of->arthritis: real
    ("aspirin", "influenza_virus", False),   # FABRICATED: that's 3 hops through "caused_by", not this chain
    ("ibuprofen", "migraine", False),        # FABRICATED: ibuprofen's chain never reaches migraine
    ("migraine", "aspirin", False),          # FABRICATED: right entities, backwards direction
]


def main() -> None:
    gr = GroundedReasoner()
    gr.add_facts(KB_FACTS)

    print("=" * 74)
    print("Knowledge base (facts, as if extracted by an LLM from documents):")
    for s, r, o in KB_FACTS:
        print(f"  {s} --{r}--> {o}")
    print("\nClaim pattern under test: drug --treats--> symptom --symptom_of--> condition")
    print("=" * 74)

    claims = [(s, o, CHAIN) for s, o, _ in LLM_CLAIMS]
    truth = {(s, o): gold for s, o, gold in LLM_CLAIMS}

    print("\nLLM's proposed claims, filtered through the guard (0 tokens):\n")
    kept_wrong = dropped_true = tp = blocked_fake = 0
    for claim, verdict in gr.filter_claims(claims):
        s, o = claim[0], claim[1]
        gold = truth[(s, o)]
        status = "KEPT   " if verdict.grounded else "BLOCKED"
        correct = "OK" if verdict.grounded == gold else "WRONG"
        if verdict.grounded and gold:
            tp += 1
        elif verdict.grounded and not gold:
            kept_wrong += 1
        elif not verdict.grounded and gold:
            dropped_true += 1
        else:
            blocked_fake += 1
        proof = " -> ".join(verdict.proof) if verdict.proof else "(no path)"
        print(f"  {status} [{correct}] {s} treats-a-symptom-of {o}")
        print(f"           proof: {proof}")

    n_true = sum(1 for *_, gold in LLM_CLAIMS if gold)
    n_false = len(LLM_CLAIMS) - n_true
    print("\n" + "=" * 74)
    print(
        f"Of {n_true} genuinely true claims: kept {tp}/{n_true}.\n"
        f"Of {n_false} fabricated claims: blocked {blocked_fake}/{n_false}, "
        f"{kept_wrong} leaked through (should be 0 -- Theorem G, extended to "
        f"heterogeneous chains per PAPER.md §5.3.4).\n"
        f"0 LLM tokens spent on this filtering step: it's local operator algebra, "
        f"not a second model call."
    )
    assert kept_wrong == 0 and dropped_true == 0, "Theorem G violated -- this must never happen"


if __name__ == "__main__":
    main()
