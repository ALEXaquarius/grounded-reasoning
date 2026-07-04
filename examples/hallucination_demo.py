"""
Demo: does an LLM FABRICATE answers on multi-hop relational inference, and how
does the grounded system compare?

The test passage below describes a DEEP (9-step) management/ownership chain,
with easily-confused names and the links told OUT OF ORDER. We ask the LLM
multi-hop inference questions (including REVERSED-DIRECTION questions — exactly
where LLMs tend to fabricate), then compare against the grounded system
(relation algebra, 0 tokens, with a proof).

Typical result: the LLM gets the forward chain right but FABRICATES on the
reversed-direction questions; the grounded system scores 8/8.

Last measured live against `deepseek-chat` on this exact English text: the LLM
answered 7/8 correctly (1 fabrication, on a reversed-direction question); the
grounded system scored 8/8, 0 tokens, with a proof for every accepted claim
(see README.md's "Evidence on real LLMs" table). An earlier version of this
scenario ran in Vietnamese instead — composition/reachability logic is
language-agnostic by construction (entities and relations are opaque Unicode
strings; see tests/test_agent.py::TestMultilingual), but an LLM's tendency to
fabricate on a specific prompt is not guaranteed to be identical across
languages, so the English run above was re-measured from scratch rather than
assumed to match the original Vietnamese numbers.

Run:  DEEPSEEK_API_KEY=... python examples/hallucination_demo.py
      (or LLM_PROVIDER=groq/openai/... — see grounded_reasoning.LLMClient)
"""
from __future__ import annotations

import os
import time

from grounded_reasoning import GroundedReasoner, LLMClient

# --- The world (ground truth, used for scoring) ---
CHAIN = ["Grant", "Owens", "Reid", "Foster", "Hale", "Brooks", "Doyle", "Kerr", "Vance", "Wells"]
OWN = ["Grant", "Meridian Group", "Solvex Corp", "Eastview Branch", "Baxter Plant", "Warehouse K9"]
FACTS = (
    [(CHAIN[i], "manages", CHAIN[i + 1]) for i in range(len(CHAIN) - 1)]
    + [(OWN[i], "owns", OWN[i + 1]) for i in range(len(OWN) - 1)]
)

PASSAGE = """
At Meridian Group, the chain of command is fairly tangled, and people often get it
wrong when retelling it. Doyle is Kerr's direct manager. Before that, Grant — the
chairman — directly manages only one person: Owens. Foster is under Reid's
authority. The manager of Wells (the youngest employee) is Vance. Hale, in turn, is
Brooks's direct manager, while Brooks manages Doyle. Don't forget: Owens directly
manages Reid, and Kerr is Vance's direct manager. The remaining link: Foster manages
Hale. Altogether this forms one long chain from the chairman down to the most
junior employee — though the links are told out of order here.

Ownership is a separate matter: Grant owns Meridian Group. Meridian Group owns
Solvex Corp. Solvex Corp owns the Eastview branch. The Eastview branch owns the
Baxter plant, and the Baxter plant owns the warehouse coded Warehouse K9. Several
rivals deliberately spread rumors reversing these relationships to cause confusion.
""".strip()

# (subject, object, via, question text, correct answer)
QUESTIONS = [
    ("Grant", "Wells", "manages", "Does Grant manage Wells (indirectly)?", True),
    ("Hale", "Wells", "manages", "Does Hale manage Wells (indirectly)?", True),
    # reversed direction — fabrication trap
    ("Wells", "Grant", "manages", "Does Wells manage Grant (indirectly)?", False),
    ("Kerr", "Hale", "manages", "Does Kerr manage Hale (indirectly)?", False),
    ("Vance", "Brooks", "manages", "Does Vance manage Brooks (indirectly)?", False),
    ("Owens", "Kerr", "manages", "Does Owens manage Kerr (indirectly)?", True),
    ("Grant", "Warehouse K9", "owns", "Does Grant own Warehouse K9 (indirectly)?", True),
    ("Warehouse K9", "Grant", "owns", "Does Warehouse K9 own Grant (indirectly)?", False),
]


def _ask(client: LLMClient, question: str, tries: int = 6) -> tuple[bool, str]:
    prompt = (
        f"Read the passage below carefully and rely ONLY on it:\n\n{PASSAGE}\n\n"
        f"Question: {question}\n"
        "Answer the FIRST LINE with exactly one word: YES or NO. Then give a short explanation."
    )
    for i in range(tries):
        try:
            ans = client.ask(prompt, temperature=0.0)
            break
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    first = ans.strip().split("\n")[0].upper()
    yes = "YES" in first and "NO" not in first
    return yes, first[:46]


def main() -> None:
    client = LLMClient(provider=os.environ.get("LLM_PROVIDER", "deepseek"))
    gr = GroundedReasoner()
    gr.add_facts(FACTS)

    print("=" * 74)
    print(f"LLM ({client.model}) free-form reasoning  vs  GROUNDED SYSTEM (0 tokens, with proof)")
    print("=" * 74)
    llm_ok = gr_ok = halluc = 0
    for subj, obj, via, desc, gold in QUESTIONS:
        llm_yes, llm_first = _ask(client, desc)
        v = gr.verify(subj, obj, via=via)
        a, b = (llm_yes == gold), (v.grounded == gold)
        llm_ok += a
        gr_ok += b
        halluc += 0 if a else 1
        proof = "→".join(v.proof) if v.proof else "— (no path)"
        print(f"\nQ: {desc}")
        print(f"   correct={'YES' if gold else 'NO':5} | "
              f"LLM={'YES' if llm_yes else 'NO':5} {'OK' if a else 'FABRICATED/WRONG'} | "
              f"guard={'YES' if v.grounded else 'NO':5} {'OK' if b else 'X'}")
        print(f"   guard's proof: {proof}")
    n = len(QUESTIONS)
    print("\n" + "=" * 74)
    print(f"LLM correct {llm_ok}/{n} (fabricated/wrong {halluc})  |  "
          f"Grounded system correct {gr_ok}/{n}, 0 tokens, with proof")
    print(f"LLM tokens: {client.total_tokens}")


if __name__ == "__main__":
    main()
