# Tích hợp vào Agent / LLM

Read in English: **[integration.md](integration.md)**

`grounded-reasoning` là một **bộ kiểm chứng suy luận quan hệ**: agent/LLM tự kiểm
tra một claim nhiều bước (ancestor, prerequisite, cause-chain, part-of…) **trước khi
khẳng định**. Bắt ảo giác quan hệ với **0 token model** và **precision đảm bảo**
(chấp nhận ⟺ tồn tại đường đi grounded — Định lý G).

Nó **bổ trợ**, không thay thế LLM. Miền khớp: agent knowledge-graph, ontology,
nhân-quả, prerequisite/planning, multi-hop QA trên triple đã trích.

---

## 1. Thư viện (Python)

```python
from grounded_reasoning import GroundedReasoner

gr = GroundedReasoner()
gr.add_facts([("alice", "parent", "bob"),
              ("bob",   "parent", "carol"),
              ("carol", "parent", "dave")])

v = gr.verify("alice", "dave", via="parent")   # claim: alice là tổ tiên của dave?
print(v.grounded)      # True
print(v.proof)         # ['alice', 'bob', 'carol', 'dave']  — bằng chứng
print(v.confidence)    # 0.216 — niềm tin (giảm theo độ sâu)

gr.verify("alice", "zed", via="parent").grounded   # False — ảo giác bị chặn
```

- `verify(a, b, via=None)` — `via=None`: đường đi bất kỳ quan hệ; `via=rel`: bắc cầu
  của riêng `rel`.
- `filter_claims([(a, b, rel), ...])` — lọc một LÔ claim LLM, giữ grounded.
- `contradictions(rel)` — nếu `rel` đáng ra acyclic mà có chu trình ⟹ trả chu trình
  mâu thuẫn (miễn phí, dùng phổ).

---

## 2. Function-calling (Anthropic / OpenAI)

```python
from grounded_reasoning import TOOL_SPEC, run_tool

# đăng ký tool với model
response = client.messages.create(
    model="claude-...",
    tools=[TOOL_SPEC],           # {"name":"verify_relation", "input_schema": {...}}
    messages=[...],
)

# khi model gọi tool:
for block in response.content:
    if block.type == "tool_use" and block.name == "verify_relation":
        result = run_tool(block.input)     # {"grounded":..., "proof":..., ...}
        # gửi result trở lại model như tool_result
```

Tool `verify_relation(facts, subject, relation, object)` là **stateless & JSON-only**
— agent truyền các fact 1-bước đã biết + claim, nhận verdict. Model dùng nó để tự
kiểm tra bước suy luận, thay vì tin mù vào chuỗi hợp thành do chính nó sinh (nơi LLM
hay ảo giác nhất).

- Anthropic: dùng `TOOL_SPEC` (đã đúng định dạng `input_schema`).
- OpenAI/DeepSeek/Groq/…: dùng `openai_tool_spec()` (định dạng `function.parameters`).

**Đa-provider** — `LLMClient` nói OpenAI-compatible với mọi endpoint, đổi provider
KHÔNG đổi mã (key đọc từ env):

```python
from grounded_reasoning import LLMClient
LLMClient()                      # DeepSeek (DEEPSEEK_API_KEY)
LLMClient(provider="openai")     # OPENAI_API_KEY
LLMClient(provider="groq")       # GROQ_API_KEY
LLMClient(provider="openrouter") # OPENROUTER_API_KEY
LLMClient(provider="together")   # TOGETHER_API_KEY
LLMClient(provider="mistral")    # MISTRAL_API_KEY
LLMClient(provider="ollama")     # local, không cần key
LLMClient(base_url="...", api_key_env="MY_KEY", model="...")   # tuỳ biến
```

**Demo function-calling thật** (agent tự kiểm chứng, chặn ảo giác):
`python -m grounded_reasoning.experiments.agent_demo` (hoặc `LLM_PROVIDER=groq python -m ...`).

**Đa ngôn ngữ:** entity/relation là chuỗi Unicode opaque ⟹ chạy với MỌI ngôn ngữ
không cần cấu hình — `("Anh","cha","Bảo")`, `("父","是","祖父")`, `("أب","والد","جد")`
đều verify + trả proof đúng chữ gốc.

---

## 3. MCP server (cho Claude / agent tương thích MCP)

```bash
pip install "grounded-reasoning[mcp]"
grounded-reasoning-mcp                    # stdio MCP server (hoặc: python -m grounded_reasoning.agent.mcp_server)
```

Server phơi đúng một tool `verify_relation`. Cấu hình client MCP trỏ tới lệnh trên;
agent sẽ thấy và gọi tool như mọi MCP tool khác.

Không cần cài cố định — mọi MCP client (Claude Desktop, Cursor, ...) có thể chạy
thẳng từ PyPI qua `uvx`:

```json
{
  "mcpServers": {
    "grounded-reasoning": {
      "command": "uvx",
      "args": ["--from", "grounded-reasoning[mcp]", "grounded-reasoning-mcp"]
    }
  }
}
```

---

## 4. Mẫu dùng: guard hậu-xử-lý cho pipeline RAG/agent

```python
# LLM sinh các claim quan hệ (có thể ảo giác ở bước hợp thành):
llm_claims = [("aspirin", "headache", "treats"),
              ("aspirin", "cancer",   "treats")]   # ← bịa

gr = GroundedReasoner()
gr.add_facts(known_medical_triples)                # KB hoặc atomic facts LLM tự trích

grounded = [c for c, v in gr.filter_claims(llm_claims) if v.grounded]
# chỉ những claim có đường đi bằng chứng mới được xuất ra người dùng
```

**Khi đồ thị NHIỄU** (ví dụ atomic facts do LLM trích từ văn bản thô), dùng
`ConformalReasoner` (`grounded_reasoning.reasoning.conformal_reasoning`) để có bảo đảm PHỦ
phân-phối-tự-do ≥ 1−α thay cho precision cứng — xem `PAPER.md` §7.1 (tiếng Anh).

---

## Giới hạn (đọc trước khi tích hợp)

- Cần quan hệ ở dạng **typed triple** (subject, relation, object). Nếu phải trích từ
  văn bản tự do bằng LLM, bước trích có thể sai — guard chỉ đảm bảo *trong phạm vi
  đồ thị được cấp* (dùng conformal để làm mềm khi nhiễu).
- Chỉ nhắm **ảo giác quan hệ/bắc cầu**, KHÔNG nhắm ảo giác sự kiện tự do.
- Là lớp **kiểm chứng**, không phải bộ suy luận mở thay LLM.
