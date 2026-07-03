"""
grounded-reasoning — a relation-reasoning verifier for LLMs and agents.

Public API (clean import surface):

    from grounded_reasoning import GroundedReasoner, verify_relation, TOOL_SPEC

- GroundedReasoner : load facts, then verify / filter_claims / contradictions.
- verify_relation  : a stateless function-calling tool (0 tokens, returns a proof).
- TOOL_SPEC / openai_tool_spec : tool schemas (Anthropic / OpenAI).
- ConformalReasoner : distribution-free coverage guarantee >=1-alpha under a noisy graph.
- LLMClient : a multi-provider (OpenAI-compatible) client for demos/experiments.
"""
from src.agent import (
    TOOL_SPEC,
    GroundedReasoner,
    Verdict,
    openai_tool_spec,
    run_tool,
    verify_relation,
)
from src.reasoning.conformal_reasoning import ConformalReasoner, conformal_threshold
from src.reasoning.llm_client import LLMClient

__version__ = "0.1.1"

__all__ = [
    "GroundedReasoner",
    "Verdict",
    "verify_relation",
    "run_tool",
    "TOOL_SPEC",
    "openai_tool_spec",
    "ConformalReasoner",
    "conformal_threshold",
    "LLMClient",
    "__version__",
]
