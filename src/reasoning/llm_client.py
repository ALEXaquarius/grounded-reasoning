"""
Client LLM đa-provider (OpenAI-compatible) — CHỈ đọc key từ biến môi trường,
KHÔNG BAO GIỜ hardcode key vào mã nguồn.

Hỗ trợ mọi endpoint OpenAI-compatible: DeepSeek, OpenAI, Groq, OpenRouter, Together,
Mistral, Ollama (local)… — chỉ khác base_url + biến môi trường key + tên model. Nhờ
đó tích hợp ĐƠN GIẢN: đổi provider mà không đổi mã.

    LLMClient()                         # mặc định DeepSeek
    LLMClient(provider="groq")          # đọc GROQ_API_KEY
    LLMClient(provider="ollama")        # local, không cần key
    LLMClient(base_url=..., api_key_env="MY_KEY", model="...")   # tuỳ biến

`.ask(prompt)` → text.  `.chat(messages, tools=...)` → message dict (cho tool-calling).
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.request

# provider -> (base_url, biến-môi-trường-key, model-mặc-định, key-bắt-buộc)
PROVIDERS: dict[str, tuple[str, str, str, bool]] = {
    "deepseek": ("https://api.deepseek.com/v1/chat/completions", "DEEPSEEK_API_KEY", "deepseek-chat", True),
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY", "gpt-4o-mini", True),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "llama-3.3-70b-versatile", True),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", "openai/gpt-4o-mini", True),
    "together": ("https://api.together.xyz/v1/chat/completions", "TOGETHER_API_KEY", "meta-llama/Llama-3.3-70B-Instruct-Turbo", True),
    "mistral": ("https://api.mistral.ai/v1/chat/completions", "MISTRAL_API_KEY", "mistral-small-latest", True),
    "ollama": ("http://localhost:11434/v1/chat/completions", "OLLAMA_API_KEY", "llama3.2", False),
}


class LLMClient:
    """Client OpenAI-compatible đa-provider (đọc key từ env)."""

    def __init__(
        self,
        provider: str = "deepseek",
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if provider not in PROVIDERS and not base_url:
            raise ValueError(f"provider '{provider}' không rõ; cấp base_url + api_key_env.")
        p_url, p_env, p_model, p_required = PROVIDERS.get(
            provider, (base_url, api_key_env or "", model or "", False)
        )
        self.url = base_url or p_url
        self.model = model or p_model
        env = api_key_env or p_env
        self.key = os.environ.get(env, "") if env else ""
        # base_url tuỳ biến ⟹ key tuỳ chọn (không ép theo preset).
        required = p_required and not base_url
        if required and not self.key:
            raise RuntimeError(f"Chưa đặt {env} trong môi trường (.env).")
        self.timeout = timeout
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0

        # Tôn trọng proxy egress + CA bundle của môi trường (giống curl).
        cafile = (
            os.environ.get("SSL_CERT_FILE")
            or os.environ.get("REQUESTS_CA_BUNDLE")
            or os.environ.get("CURL_CA_BUNDLE")
        )
        ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
        handlers: list[urllib.request.BaseHandler] = [urllib.request.HTTPSHandler(context=ctx)]
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"https": proxy, "http": proxy}))
        self._opener = urllib.request.build_opener(*handlers)

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers)
        with self._opener.open(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        if not data.get("choices"):
            # API trả JSON hợp lệ nhưng không có completion nào (rate limit, content
            # filter, lỗi provider…) — báo lỗi RÕ kèm response gốc, thay vì để
            # data["choices"][0] ném IndexError/KeyError khó hiểu ở nơi gọi.
            raise RuntimeError(f"API không trả về 'choices': {data}")
        usage = data.get("usage", {})
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        self.n_calls += 1
        return data

    def ask(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        data = self._post({"model": self.model, "messages": msgs, "temperature": temperature})
        return data["choices"][0]["message"]["content"]

    def chat(self, messages: list[dict], tools: list | None = None,
             temperature: float = 0.0) -> dict:
        """Một lượt chat (hỗ trợ tool-calling). Trả về message dict của assistant."""
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        data = self._post(payload)
        return data["choices"][0]["message"]

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens


class DeepSeekClient(LLMClient):
    """Tương thích ngược: LLMClient chốt provider DeepSeek."""

    def __init__(self, model: str = "deepseek-chat", timeout: float = 60.0) -> None:
        super().__init__(provider="deepseek", model=model, timeout=timeout)
