"""
Horn inference (forward-chaining) — GENERALIZES the transitive-closure guard to
full Horn logic.

The relational composition guard (Theorem G) is the special case of Horn logic
with a SINGLE rule edge(a,b) AND edge(b,c) -> edge(a,c). Forward-chaining computes
the LEAST MODEL of a Horn program: sound (every derived fact has a proof) +
complete (every derivable fact is derived) + terminating. This is a 0-token
verifier for LLM inference expressed as general rules (conjunctive bodies) — a
path toward ProofWriter/EntailmentBank-style tasks.

Honestly: this is classical Datalog semantics (not new); the value is packaging
it as a guaranteed verification layer for LLM output, unified with the operator
algebra.
"""
from __future__ import annotations

Rule = tuple[frozenset, object]  # (body: a set of literals, head: a literal)


def forward_chain(facts: set, rules: list[Rule]) -> set:
    """Least model: close `facts` under `rules` to a fixpoint (O(#rules * rounds))."""
    derived = set(facts)
    changed = True
    while changed:
        changed = False
        for body, head in rules:
            if head not in derived and body <= derived:
                derived.add(head)
                changed = True
    return derived


def entails(facts: set, rules: list[Rule], goal) -> bool:
    """Whether goal is derivable (goal is in the least model)."""
    return goal in forward_chain(facts, rules)


def explain(facts: set, rules: list[Rule], goal) -> list[Rule] | None:
    """
    A grounded proof of goal: the sequence of rules that fired en route to goal
    (or None). This is the GROUNDED property — the guard accepts a claim iff a
    proof exists.
    """
    derived = set(facts)
    proof: list[Rule] = []
    if goal in derived:
        return proof
    changed = True
    while changed:
        changed = False
        for body, head in rules:
            if head not in derived and body <= derived:
                derived.add(head)
                proof.append((body, head))
                if head == goal:
                    return proof
                changed = True
    return None
