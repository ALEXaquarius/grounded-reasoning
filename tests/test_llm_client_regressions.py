"""
Test hồi quy từ fuzz `llm_client.py` (1500 lượt __init__ + 500 lượt mock response
API méo mó) — tìm ra 2 bug thật, đã sửa; khóa lại để không tái phát. Không gọi mạng
(mock `_opener.open`), không cần API key thật.

Bug 1 — `LLMClient.__init__` chỉ kiểm tra `base_url is None` để bắt lỗi "provider lạ
+ không có base_url", nên `base_url=""` (chuỗi rỗng, falsy nhưng không phải None)
LỌT QUA guard, tạo ra client với `url=''` — không lỗi ngay mà chỉ vỡ khi thực sự gọi
API với thông báo khó hiểu. Sửa: dùng `not base_url` (bắt cả None lẫn chuỗi rỗng).

Bug 2 — `_post` không kiểm tra `data["choices"]` trước khi để `ask()`/`chat()`
truy cập `data["choices"][0]`. Khi API trả JSON hợp lệ nhưng `choices` rỗng/thiếu
(rate limit, content filter, lỗi provider…) ⟹ `IndexError`/`KeyError` khó hiểu,
không có ngữ cảnh response gốc để debug. Sửa: raise RuntimeError rõ ràng kèm response.
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
            LLMClient(provider="unknown_xyz")   # base_url mặc định None

    def test_known_provider_with_empty_base_url_ignores_it(self):
        # provider hợp lệ + base_url="" -> "" bị bỏ qua (falsy), dùng preset provider
        os.environ["DEEPSEEK_API_KEY"] = "fake"
        client = LLMClient(provider="deepseek", base_url="")
        assert client.url == "https://api.deepseek.com/v1/chat/completions"

    def test_real_base_url_still_bypasses_key_requirement(self):
        # hành vi cũ (base_url thật -> key tuỳ chọn) không bị hồi quy
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
