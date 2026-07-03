"""
Lớp tích hợp AGENT/LLM cho suy diễn grounded.

- GroundedReasoner : facade — nạp fact, kiểm chứng claim quan hệ nhiều bước.
- verify_relation  : tool stateless cho function-calling (0 token, có bằng chứng).
- TOOL_SPEC        : schema tool (Anthropic/OpenAI) để đăng ký với model.
"""
from src.agent.tool import TOOL_SPEC, openai_tool_spec, run_tool, verify_relation
from src.agent.verifier import GroundedReasoner, Verdict

__all__ = [
    "GroundedReasoner",
    "Verdict",
    "verify_relation",
    "run_tool",
    "TOOL_SPEC",
    "openai_tool_spec",
]
