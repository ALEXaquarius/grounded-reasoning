"""
A tool for AGENTS (function-calling) — verifies a relational claim BEFORE it is
asserted.

Stateless & JSON-friendly: the agent passes `facts` + a claim, and gets back a
verdict + proof path. Catches multi-hop relational hallucination at 0 LLM tokens,
with a precision guarantee (Theorem G).

Use with Anthropic/OpenAI function-calling:
    tools=[TOOL_SPEC]; ... when the model calls the tool -> run_tool(tool_input)
Or call directly: verify_relation(facts, subject, relation, object).
"""
from __future__ import annotations

from grounded_reasoning.agent.verifier import GroundedReasoner

TOOL_SPEC = {
    "name": "verify_relation",
    "description": (
        "Verify whether a relational claim is grounded (provable by chaining the "
        "given atomic facts) BEFORE asserting it. Use this to catch hallucinated "
        "multi-hop relations (e.g. ancestor/prerequisite/cause chains). Returns "
        "grounded=false with proof=null when the claim cannot be derived, and a "
        "proof path when it can. Costs zero model tokens and never accepts an "
        "ungrounded claim."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "description": "Known atomic (one-hop) facts as [subject, relation, object] triples.",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "subject": {"type": "string", "description": "Start entity of the claim."},
            "relation": {
                "type": "string",
                "description": "Relation to verify transitively (e.g. 'parent' to test ancestor).",
            },
            "object": {"type": "string", "description": "End entity of the claim."},
        },
        "required": ["facts", "subject", "relation", "object"],
    },
}


def openai_tool_spec() -> dict:
    """TOOL_SPEC in OpenAI/DeepSeek/Groq (function-calling) format."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_SPEC["name"],
            "description": TOOL_SPEC["description"],
            "parameters": TOOL_SPEC["input_schema"],
        },
    }


def verify_relation(facts, subject: str, relation: str, object: str) -> dict:  # noqa: A002
    """
    Verify `subject --relation*--> object` from `facts`. Returns a JSON dict:
    {grounded, proof, confidence, relation}.

    Tolerant of LLM input: SKIPS any fact not shaped like [s, r, o] (instead of
    crashing the agent), reports the number skipped under `skipped_facts`, and
    strips incidental leading/trailing whitespace from every entity string —
    an LLM emitting " Bob" in one place and "Bob" in another is an extraction
    artifact, not a claim that they're different entities, and left unhandled
    it silently breaks a real proof path (see GroundedReasoner's docstring).
    Differences in case are left alone (case CAN be semantically meaningful),
    so callers with case-insensitive domains should build their own
    `GroundedReasoner(normalize=...)` instead of using this stateless helper.
    """
    gr = GroundedReasoner()
    clean, skipped = [], 0
    for t in facts or []:
        if isinstance(t, (list, tuple)) and len(t) == 3 and all(x is not None for x in t):
            clean.append([str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()])
        else:
            skipped += 1
    gr.add_facts(clean)
    out = gr.verify(str(subject).strip(), str(object).strip(), via=str(relation).strip()).as_dict()
    if skipped:
        out["skipped_facts"] = skipped
    return out


def run_tool(tool_input: dict) -> dict:
    """Dispatcher for function-calling: takes the tool's JSON input, returns a verdict dict."""
    return verify_relation(
        tool_input["facts"],
        tool_input["subject"],
        tool_input["relation"],
        tool_input["object"],
    )
