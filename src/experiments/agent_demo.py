"""
DEMO FUNCTION-CALLING THẬT: agent tự KIỂM CHỨNG suy luận quan hệ trước khi trả lời.

Kịch bản: agent biết vài fact họ hàng 1-bước; người dùng hỏi câu quan hệ NHIỀU BƯỚC
(dễ ảo giác). Agent được cấp tool `verify_relation` (backed bởi GroundedReasoner) và
buộc gọi tool trước khi khẳng định ⟹ claim grounded thì xác nhận, ảo giác thì từ chối.

Đa-provider (OpenAI-compatible): DeepSeek/OpenAI/Groq/OpenRouter/Together/Mistral/Ollama.
    python -m src.experiments.agent_demo                    # DeepSeek (mặc định)
    LLM_PROVIDER=groq python -m src.experiments.agent_demo  # provider khác
"""
from __future__ import annotations

import json
import os

from src.agent import GroundedReasoner

# KB: chuỗi họ hàng 1-bước (agent "biết"); KHÔNG có đường Ann→Frank.
FACTS = [
    ("Ann", "parent", "Bill"),
    ("Bill", "parent", "Carol"),
    ("Carol", "parent", "Dan"),
    ("Eve", "parent", "Frank"),   # cụm rời — Ann KHÔNG là tổ tiên Frank
]

# Tool backed bởi KB: model chỉ truyền claim, đồ thị nằm trong tool (mẫu thực tế).
_GR = GroundedReasoner()
_GR.add_facts(FACTS)

TOOL = {
    "type": "function",
    "function": {
        "name": "verify_relation",
        "description": (
            "Check whether 'subject is a <relation>-ancestor of object' is provable "
            "by chaining the known family facts. Returns grounded=false (with "
            "proof=null) for claims that cannot be derived. Call this BEFORE "
            "asserting any multi-step relationship."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "relation": {"type": "string", "description": "base relation, e.g. 'parent'"},
                "object": {"type": "string"},
            },
            "required": ["subject", "relation", "object"],
        },
    },
}


def _handle(args: dict) -> dict:
    return _GR.verify(args["subject"], args["object"], via=args.get("relation", "parent")).as_dict()


def ask_agent(client, question: str, max_turns: int = 5, verbose: bool = True) -> str:
    facts_str = "; ".join(f"{s} is a {r} of {o}" for s, r, o in FACTS)
    messages = [
        {"role": "system", "content": (
            "You answer family-relationship questions. You may ONLY rely on facts "
            "proven by the verify_relation tool. Before asserting any multi-step "
            "relationship, CALL verify_relation. If grounded is false, say it does "
            "not hold. Known base facts: " + facts_str)},
        {"role": "user", "content": question},
    ]
    for _ in range(max_turns):
        msg = client.chat(messages, tools=[TOOL], temperature=0.0)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            if verbose:
                print(f"  Q: {question}\n  A: {msg.get('content','').strip()}\n")
            return msg.get("content", "")
        for call in calls:
            args = json.loads(call["function"]["arguments"] or "{}")
            result = _handle(args)
            if verbose:
                print(f"  ↳ tool verify_relation({args}) → grounded={result['grounded']}")
            messages.append({
                "role": "tool", "tool_call_id": call["id"],
                "content": json.dumps(result),
            })
    return "(hết lượt)"


def run(provider: str | None = None, verbose: bool = True):
    from src.reasoning.llm_client import LLMClient

    provider = provider or os.environ.get("LLM_PROVIDER", "deepseek")
    client = LLMClient(provider=provider)
    if verbose:
        print(f"Provider: {provider} | model: {client.model}\n")
    q1 = "Is Ann an ancestor of Dan?"        # grounded (Ann→Bill→Carol→Dan)
    q2 = "Is Ann an ancestor of Frank?"      # ảo giác-bait (không có đường)
    a1 = ask_agent(client, q1, verbose=verbose)
    a2 = ask_agent(client, q2, verbose=verbose)
    if verbose:
        print(f"Tokens: {client.total_tokens}")
    return {"q1": a1, "q2": a2, "tokens": client.total_tokens}


if __name__ == "__main__":
    run()
