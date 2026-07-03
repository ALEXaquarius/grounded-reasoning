"""
Regression tests from fuzzing `llm_client.py` (1500 __init__ runs + 500 malformed
mock API response runs) — found 2 real bugs, now fixed; locked down here to prevent
recurrence. No network calls (mocks `_opener.open`), no real API key needed.

Bug 1 — `LLMClient.__init__` only checked `base_url is None` to catch the "unknown
provider + no base_url" error, so `base_url=""` (an empty string, falsy but not
None) SLIPPED PAST the guard, creating a client with `url=''` — it didn't fail
immediately, only breaking later with a confusing message when the API was actually
called. Fix: use `not base_url` (catches both None and the empty string).

Bug 2 — `_post` did not check `data["choices"]` before letting `ask()`/`chat()`
access `data["choices"][0]`. When the API returns valid JSON but `choices` is
empty/missing (rate limiting, content filter, provider error, ...) ⟹ a confusing
`IndexError`/`KeyError` with no original response context for debugging. Fix: raise
a clear RuntimeError including the response.
"""
import json
import os

import pytest

from src.reasoning.llm_client import LLMClient


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _client_with_mock_response(payload_or_bytes):
    os.environ["DEEPSEEK_API_KEY"] = "fake-key"
    client = LLMClient(provider="deepseek")
    body = payload_or_bytes if isinstance(payload_or_bytes, bytes) else json.dumps(payload_or_bytes).encode()
    client._opener.open = lambda req, timeout=None: _FakeResponse(body)
    return client


class TestUnknownProviderEmptyBaseUrlRegression:
    def test_exact_fuzz_repro_empty_string_base_url(self):
        with pytest.raises(ValueError, match="unknown provider"):
            LLMClient(provider="unknown_xyz", base_url="")

    def test_none_base_url_still_raises(self):
        with pytest.raises(ValueError):
            LLMClient(provider="unknown_xyz")   # base_url defaults to None

    def test_known_provider_with_empty_base_url_ignores_it(self):
        # valid provider + base_url="" -> "" is ignored (falsy), preset provider is used
        os.environ["DEEPSEEK_API_KEY"] = "fake"
        client = LLMClient(provider="deepseek", base_url="")
        assert client.url == "https://api.deepseek.com/v1/chat/completions"

    def test_real_base_url_still_bypasses_key_requirement(self):
        # old behavior (real base_url -> key optional) is not regressed
        client = LLMClient(base_url="http://localhost:9999/v1/chat", api_key_env="NOPE_XYZ")
        assert client.url == "http://localhost:9999/v1/chat"


class TestMissingChoicesRegression:
    def test_exact_fuzz_repro_empty_choices_list(self):
        client = _client_with_mock_response({"choices": []})
        with pytest.raises(RuntimeError, match="choices"):
            client.ask("hi")

    def test_error_response_no_choices_key(self):
        client = _client_with_mock_response({"error": {"message": "rate limited"}})
        with pytest.raises(RuntimeError, match="choices"):
            client.ask("hi")

    def test_empty_object_response(self):
        client = _client_with_mock_response({})
        with pytest.raises(RuntimeError, match="choices"):
            client.chat([{"role": "user", "content": "hi"}])

    def test_valid_response_still_works(self):
        client = _client_with_mock_response({
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })
        assert client.ask("hi") == "hello"
        assert client.total_tokens == 4
