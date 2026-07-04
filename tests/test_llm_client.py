"""
OFFLINE tests for the multi-provider LLM client + agent_demo's tool-calling loop.
No network calls: uses a fake client.
"""
import json

import pytest

from grounded_reasoning.reasoning.llm_client import PROVIDERS, LLMClient


class TestLLMClient:
    def test_provider_presets_exist(self):
        for p in ("deepseek", "openai", "groq", "openrouter", "together", "mistral", "ollama"):
            assert p in PROVIDERS

    def test_ollama_needs_no_key(self):
        c = LLMClient(provider="ollama")           # API key is not required
        assert "11434" in c.url and c.model == "llama3.2"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            LLMClient(provider="does-not-exist")

    def test_custom_endpoint(self):
        c = LLMClient(base_url="http://x/y", api_key_env="NOPE", model="m")
        assert c.url == "http://x/y" and c.model == "m"


class _FakeClient:
    """Fake LLM: turn 1 calls the tool, turn 2 gives the final answer."""

    def __init__(self):
        self.turn = 0
        self.total_tokens = 0
        self.model = "fake"

    def chat(self, messages, tools=None, temperature=0.0):
        self.turn += 1
        if self.turn == 1:
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "verify_relation",
                             "arguments": json.dumps({"subject": "Ann", "relation": "parent", "object": "Dan"})},
            }]}
        # verify the tool result was appended to messages
        assert any(m.get("role") == "tool" for m in messages)
        return {"role": "assistant", "content": "Yes, grounded."}


def test_agent_demo_tool_loop_offline():
    from grounded_reasoning.experiments.agent_demo import ask_agent

    out = ask_agent(_FakeClient(), "Is Ann an ancestor of Dan?", verbose=False)
    assert "grounded" in out.lower()
