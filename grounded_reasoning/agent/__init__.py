"""
Agent/LLM integration layer for grounded reasoning.

- GroundedReasoner : facade — load facts, verify multi-hop relational claims.
- verify_relation  : a stateless function-calling tool (0 tokens, returns a proof).
- TOOL_SPEC        : tool schema (Anthropic/OpenAI) for registering with a model.
"""
from grounded_reasoning.agent.tool import TOOL_SPEC, openai_tool_spec, run_tool, verify_relation
from grounded_reasoning.agent.verifier import GroundedReasoner, Verdict

__all__ = [
    "GroundedReasoner",
    "Verdict",
    "verify_relation",
    "run_tool",
    "TOOL_SPEC",
    "openai_tool_spec",
]
