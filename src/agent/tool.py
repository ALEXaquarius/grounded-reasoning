"""
Tool cho AGENT (function-calling) — kiểm chứng claim quan hệ TRƯỚC khi khẳng định.

Stateless & JSON-friendly: agent truyền `facts` + claim, nhận verdict + đường đi bằng
chứng. Bắt ảo giác quan hệ nhiều bước với 0 token LLM, precision đảm bảo (Định lý G).

Dùng với Anthropic/OpenAI function-calling:
    tools=[TOOL_SPEC]; ... khi model gọi tool → run_tool(tool_input)
Hoặc gọi trực tiếp: verify_relation(facts, subject, relation, object).
"""
from __future__ import annotations

from src.agent.verifier import GroundedReasoner

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
    """TOOL_SPEC ở định dạng OpenAI/DeepSeek/Groq (function-calling)."""
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
    Kiểm chứng `subject --relation*--> object` từ `facts`. Trả dict JSON:
    {grounded, proof, confidence, relation}.

    Khoan dung với đầu vào LLM: BỎ QUA fact không đúng dạng [s, r, o] (thay vì làm
    treo agent), và báo số fact bị bỏ ở khóa `skipped_facts`.
    """
    gr = GroundedReasoner()
    clean, skipped = [], 0
    for t in facts or []:
        if isinstance(t, (list, tuple)) and len(t) == 3 and all(x is not None for x in t):
            clean.append([str(t[0]), str(t[1]), str(t[2])])
        else:
            skipped += 1
    gr.add_facts(clean)
    out = gr.verify(str(subject), str(object), via=str(relation)).as_dict()
    if skipped:
        out["skipped_facts"] = skipped
    return out


def run_tool(tool_input: dict) -> dict:
    """Bộ điều phối cho function-calling: nhận input JSON của tool → verdict dict."""
    return verify_relation(
        tool_input["facts"],
        tool_input["subject"],
        tool_input["relation"],
        tool_input["object"],
    )
