# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## 0.1.1 — MCP Registry metadata

No code changes. Adds `server.json` and the `mcp-name` ownership marker in the
README so the MCP server can be listed on the official MCP Registry
(`io.github.alexaquarius/grounded-reasoning`), and updates install docs for the
PyPI release.

## 0.1.0 — Initial public release

The first public release of `grounded-reasoning`: a relation-algebra verifier for
LLM/agent multi-hop reasoning, extracted from a broader private research project into
a standalone, focused package.

### Added

- **Public API** (`grounded_reasoning`): `GroundedReasoner`, `verify_relation`,
  `TOOL_SPEC` / `openai_tool_spec`, `ConformalReasoner`, `LLMClient`.
- **Reasoning core** (`src/reasoning/`): fuzzy compositional inference, relation
  operator algebra, relation spectrum/Katz resolvent, composition-table learning,
  conformal reasoning, Horn forward-chaining, provider-agnostic LLM client.
- **Agent integration** (`src/agent/`): a `HallucinationGuard`-backed verifier, a
  stateless function-calling tool (Anthropic/OpenAI schemas), and an MCP server.
- **Seven theorems (F–L)** with numerical verification (`src/theory/theorems.py`),
  each backed by a dedicated test in `tests/`.
- **Real-LLM experiments** validated on DeepSeek and the public CLUTRR benchmark:
  hallucination guard, self-grounded deductive consistency (SGDC), natural-language
  ontology boundary, token-cost comparison, and end-to-end conformal reasoning over
  an LLM-extracted graph.
- Full research paper ([PAPER.md](PAPER.md)) with proofs and reproduction instructions
  for every claim in the README.
- Test suite (offline-locked; no API key required to run `pytest tests/`), including
  dedicated fuzz-regression suites for the graph/closure algorithms and the LLM
  client's error handling.

### Notes

- This package is a focused extraction from a larger private research repository
  that also explored an (ultimately negative-result) embedding-free retrieval
  direction. That retrieval research is not included here — see
  [README.md — Origin story](README.md#origin-story) for context.
- Every module under `src/reasoning/`, `src/agent/`, and `src/theory/` is held to
  strict lint rules as the "published library" surface; `src/experiments/` holds
  research/reproduction scripts with a looser style (see `pyproject.toml`).
