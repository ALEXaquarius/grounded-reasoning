# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## 0.1.3 — Fix a real packaging bug; hardening from an independent code review

### Fixed

- **Namespace collision on install (the important one).** The published wheel
  shipped a top-level `src` package — `import grounded_reasoning` broke with
  `ModuleNotFoundError: No module named 'src.agent'` in any environment where
  another package (or simply the user's own project) already had a `src/`
  directory on `sys.path`, since that `src/` resolved first and shadowed the
  library's own. All internal modules moved from `src/{agent,reasoning,theory,
  experiments}` to `grounded_reasoning/{agent,reasoning,theory,experiments}` —
  the wheel now ships exactly one top-level package. Every import, the
  `grounded-reasoning-mcp` entry point, and every doc/example path were updated
  to match. Verified by reproducing the original collision (installing
  alongside a project with its own `src/__init__.py`) and confirming it no
  longer occurs.

### Changed

- **Single-sourced the version.** `grounded_reasoning/_version.py` is now the
  only place the version is written by hand for the Python package;
  `pyproject.toml` reads it via `[tool.setuptools.dynamic]`, and
  `tests/test_public_api.py` asserts `__version__` matches the installed
  distribution's metadata instead of pinning a literal that needed updating on
  every release. (`CITATION.cff` and `server.json` still need a manual bump —
  they're read by non-Python tooling — but that is now 2 files instead of 4.)
- `GroundedReasoner.verify`/`filter_claims` now cache the diffusion inference
  per source node (invalidated on `add_fact`), so filtering N claims that share
  a subject no longer recomputes the same graph-wide inference N times.
- CI no longer double-runs the full test matrix on every PR commit (it
  triggered on both `push` and `pull_request`); `push` is now restricted to
  `main`.
- `tests/test_theorems.py` runs each theorem's Monte-Carlo verification once
  per session (module-scoped fixture) instead of once for the umbrella test
  plus once again for each individual test.
- `tests/test_llm_client_regressions.py` and `test_spectrum_regressions.py` no
  longer assert on exact exception-message wording (which already broke once,
  purely from translating the message to English with no behavior change);
  they assert on exception type, and — for the "missing choices" regression —
  that the raw API response is embedded in the error for debugging, which is
  the actual behavior the regression test exists to protect.
- `.github/workflows/publish-mcp-registry.yml` pins `mcp-publisher` to a
  specific release instead of `releases/latest` (this job holds
  `id-token: write`), and fails fast if `server.json`'s version doesn't match
  the package version, instead of silently publishing a stale one.
- Removed `requirements.txt` (duplicated and had already drifted from
  `pyproject.toml`, which is the real dependency source).

## 0.1.2 — Fix MCP Registry ownership marker casing

No code changes. The `mcp-name` marker in the README (which becomes the PyPI
package description used for the registry's ownership check) now matches the
registry namespace casing exactly: `io.github.ALEXaquarius/grounded-reasoning`.
The marker published with 0.1.1 used lowercase and the check is case-sensitive.

## 0.1.1 — MCP Registry metadata

No code changes. Adds `server.json` and the `mcp-name` ownership marker in the
README so the MCP server can be listed on the official MCP Registry
(`io.github.ALEXaquarius/grounded-reasoning`), and updates install docs for the
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
