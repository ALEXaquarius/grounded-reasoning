"""
grounded-reasoning — a relation-reasoning verifier for LLMs and agents.

Public API (clean import surface):

    from grounded_reasoning import GroundedReasoner, verify_relation, TOOL_SPEC

- GroundedReasoner : load facts, then verify / filter_claims / contradictions.
- verify_relation  : a stateless function-calling tool (0 tokens, returns a proof).
- TOOL_SPEC / openai_tool_spec : tool schemas (Anthropic / OpenAI).
- ConformalReasoner : distribution-free coverage guarantee >=1-alpha under a noisy graph.
- AdaptiveConformalReasoner : like ConformalReasoner, but tracks a DRIFTING noise
  level over a stream of confirmed-true examples instead of one frozen threshold.
- LLMClient : a multi-provider (OpenAI-compatible) client for demos/experiments.
"""
from grounded_reasoning._version import __version__
from grounded_reasoning.agent import (
    TOOL_SPEC,
    GroundedReasoner,
    Verdict,
    openai_tool_spec,
    run_tool,
    verify_relation,
)
from grounded_reasoning.reasoning.conformal_reasoning import (
    AdaptiveConformalReasoner,
    ConformalReasoner,
    conformal_threshold,
)
from grounded_reasoning.reasoning.llm_client import LLMClient

__all__ = [
    "GroundedReasoner",
    "Verdict",
    "verify_relation",
    "run_tool",
    "TOOL_SPEC",
    "openai_tool_spec",
    "ConformalReasoner",
    "AdaptiveConformalReasoner",
    "conformal_threshold",
    "LLMClient",
    "__version__",
]
