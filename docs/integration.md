# Integrating with an Agent / LLM

Đọc bằng tiếng Việt: **[integration.vi.md](integration.vi.md)**

`grounded-reasoning` is a **relation-reasoning verifier**: an agent/LLM checks a
multi-hop claim (ancestor, prerequisite, cause-chain, part-of…) **before asserting
it**. It catches relational hallucinations with **zero model tokens** and a
**guaranteed precision** (accepts a claim iff a grounded proof path exists —
Theorem G).

It **complements** an LLM, it doesn't replace it. Good fit: agent knowledge-graphs,
ontologies, causal reasoning, prerequisite/planning, multi-hop QA over already-
extracted triples.

---

## 1. Library (Python)

```python
from grounded_reasoning import GroundedReasoner

gr = GroundedReasoner()
gr.add_facts([("alice", "parent", "bob"),
              ("bob",   "parent", "carol"),
              ("carol", "parent", "dave")])

v = gr.verify("alice", "dave", via="parent")   # claim: is alice an ancestor of dave?
print(v.grounded)      # True
print(v.proof)         # ['alice', 'bob', 'carol', 'dave']  — the proof
print(v.confidence)    # 0.216 — confidence (decreases with depth)

gr.verify("alice", "zed", via="parent").grounded   # False — hallucination blocked
```

- `verify(a, b, via=None)` — `via=None`: any-relation path; `via=rel`: transitive
  closure of a specific `rel`.
- `filter_claims([(a, b, rel), ...])` — filter a BATCH of LLM claims, keep only grounded ones.
- `contradictions(rel)` — if `rel` should be acyclic but has a cycle ⟹ returns the
  contradictory cycle (free, uses the spectrum).

---

## 2. Function-calling (Anthropic / OpenAI)

```python
from grounded_reasoning import TOOL_SPEC, run_tool

# register the tool with the model
response = client.messages.create(
    model="claude-...",
    tools=[TOOL_SPEC],           # {"name":"verify_relation", "input_schema": {...}}
    messages=[...],
)

# when the model calls the tool:
for block in response.content:
    if block.type == "tool_use" and block.name == "verify_relation":
        result = run_tool(block.input)     # {"grounded":..., "proof":..., ...}
        # send result back to the model as a tool_result
```

The `verify_relation(facts, subject, relation, object)` tool is **stateless &
JSON-only** — the agent passes the known one-hop facts plus the claim, and gets back
a verdict. The model uses it to check its own reasoning step, instead of blindly
trusting a composed chain it generated itself (exactly where LLMs hallucinate most).

- Anthropic: use `TOOL_SPEC` (already in `input_schema` format).
- OpenAI/DeepSeek/Groq/…: use `openai_tool_spec()` (`function.parameters` format).

**Multi-provider** — `LLMClient` speaks OpenAI-compatible to any endpoint; switching
providers requires **no code change** (the key is read from an env var):

```python
from grounded_reasoning import LLMClient
LLMClient()                      # DeepSeek (DEEPSEEK_API_KEY)
LLMClient(provider="openai")     # OPENAI_API_KEY
LLMClient(provider="groq")       # GROQ_API_KEY
LLMClient(provider="openrouter") # OPENROUTER_API_KEY
LLMClient(provider="together")   # TOGETHER_API_KEY
LLMClient(provider="mistral")    # MISTRAL_API_KEY
LLMClient(provider="ollama")     # local, no key needed
LLMClient(base_url="...", api_key_env="MY_KEY", model="...")   # custom
```

**Real function-calling demo** (the agent verifies itself, blocks hallucination):
`python -m grounded_reasoning.experiments.agent_demo` (or `LLM_PROVIDER=groq python -m ...`).

**Multilingual:** entities/relations are opaque Unicode strings ⟹ works in ANY
language with zero configuration — `("Anh","cha","Bảo")`, `("父","是","祖父")`,
`("أب","والد","جد")` all verify correctly and return proofs in the original script.

---

## 3. MCP server (for Claude / any MCP-compatible agent)

```bash
pip install "grounded-reasoning[mcp]"
grounded-reasoning-mcp                    # stdio MCP server (or: python -m grounded_reasoning.agent.mcp_server)
```

The server exposes exactly one tool, `verify_relation`. Point your MCP client
configuration at the command above; the agent will see and call the tool like any
other MCP tool.

No permanent install needed — any MCP client (Claude Desktop, Cursor, ...) can
launch it straight from PyPI via `uvx`:

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

## 4. Usage pattern: a post-processing guard for a RAG/agent pipeline

```python
# LLM generates relational claims (may hallucinate at the composition step):
llm_claims = [("aspirin", "headache", "treats"),
              ("aspirin", "cancer",   "treats")]   # ← fabricated

gr = GroundedReasoner()
gr.add_facts(known_medical_triples)                # a KB, or atomic facts the LLM itself extracted

grounded = [c for c, v in gr.filter_claims(llm_claims) if v.grounded]
# only claims with an evidence path are surfaced to the user
```

**When the graph is NOISY** (e.g. atomic facts extracted by an LLM from raw text), use
`ConformalReasoner` (`grounded_reasoning.reasoning.conformal_reasoning`) for a distribution-free
**coverage** guarantee ≥ 1−α instead of hard precision — see `PAPER.md` §7.1.

---

## Limitations (read before integrating)

- Requires relations as **typed triples** (subject, relation, object). If you have to
  extract them from free text with an LLM, that extraction step can be wrong — the
  guard only guarantees correctness *within the graph it was given* (use conformal to
  soften this under noise).
- Targets **relational/transitive hallucination** specifically, NOT free-standing
  factual hallucination.
- It is a **verification layer**, not an open-ended reasoning engine that replaces the LLM.
