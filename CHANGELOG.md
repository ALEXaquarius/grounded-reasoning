# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## Unreleased — heterogeneous relation-path verification and calibration

Not yet version-bumped or published — pending confirmation before cutting a
release (multiple small releases in quick succession is something to avoid
going forward; batching is preferred where it doesn't block real fixes).

### Added

- **`GroundedReasoner.verify_path(subject, obj, via=[rel1, rel2, ...])`** —
  verifies a claim through an exact sequence of possibly *different* relation
  types (e.g. `["parent", "employer"]`), generalizing `verify(via=rel)`
  (a single relation's repeated closure). Not new math:
  `OperatorRelationAlgebra.follow` already composes heterogeneous relation
  chains exactly (Theorem G's own numerical verification exercises mixed
  chains); this exposes that capability at the facade and adds proof-path
  reconstruction, which `follow` (a pure reachability check) does not provide.
  Checked against independent ground-truth BFS across 8,000 (subject,
  relation-chain, object) triples with zero mismatches, every returned proof
  path independently confirmed to consist of real edges in the declared order.
- **`GroundedReasoner.calibrate_path(via, labeled_pairs, alpha=0.1)`** —
  calibrates a fixed heterogeneous path pattern with the identical
  Clopper-Pearson machinery as `calibrate_transitivity` (Theorem M);
  documented as an honest generalization (PAPER.md §5.3.4), not a new theorem
  letter, since nothing in Theorem M's argument was specific to a single
  relation's closure. Practical caveat documented directly: calibration is
  per exact path pattern — evidence for one sequence says nothing about a
  different one.
- **`grounded_reasoning/experiments/heterogeneous_path_calibration_eval.py`**
  — a fully offline demo (synthetic "financially dependent on" claim composed
  from `parent` then `employer`), checked against the *known* true precision
  (88% empirical coverage over 200 trials against a 90% target) as well as a
  noisier, realistic comparison against a finite held-out test sample.

## 0.1.6 — Theorem N: closing the other boundary (entity normalization)

0.1.4 fixed the entity-identity gap (§5.3.1) with an opt-in `normalize=` hook,
but never characterized exactly when it's safe. Theorem N does, and — mirroring
0.1.5's treatment of transitivity — reuses the same Clopper-Pearson machinery
to make that safety measurable instead of assumed.

### Added

- **Theorem N (Normalization Precision Isolation)**, numerically verified
  (`theorem_normalization_precision_isolation`, `tests/test_theorems.py`, 60
  random trials): a normalizer that never merges two genuinely distinct
  entities preserves precision=1.0 *exactly* and never regresses recall;
  over-merging is the *only* way precision can break, and when it does, the
  false-positive's proof path always passes through the over-merged key.
- **`GroundedReasoner.calibrate_normalization(labeled_pairs, alpha=0.1)`** —
  the measured counterpart: calibrates a Clopper-Pearson lower confidence
  bound on how many of a normalizer's actual merges are correct, from
  held-out `(a, b, is_same_entity)` triples. Reuses the identical calibration
  engine built for Theorem M (`transitivity_calibration.py`), applied to a
  different empirical question. Raises `ValueError` if `normalize=` wasn't
  set — there's nothing to calibrate for the default (always-safe,
  exact-string) case.
- **`grounded_reasoning/experiments/normalization_calibration_eval.py`** — a
  fully offline A/B/C comparison (synthetic ground truth): no normalization
  (0 false positives, always, but misses fragmented paths) vs. a fuzzy
  resolver trusted blindly (several false positives, no risk reported) vs.
  the same resolver calibrated (merge precision reported and held on a
  held-out test set across every seed tested).
- README.md/README.vi.md and PAPER.md §5.3.3 document the theorem, proof, and
  A/B/C comparison.

## 0.1.5 — Theorem M: a measured alternative to the binary transitivity guard

0.1.4 closed the transitivity gap (§5.3.1) with a binary allowlist,
`transitive_relations={...}`: declare a relation transitive, or the guard
rejects it outright. Sound, but coarse — it can't say *how much* to trust a
relation that's mostly-but-not-perfectly transitive, only accept-all or
reject-all.

### Added

- **`GroundedReasoner.calibrate_transitivity(relation, labeled_pairs, alpha=0.1)`**
  — a measured alternative (Theorem M): calibrates a Clopper-Pearson exact
  lower confidence bound on "a graph-grounded composed claim for `relation`
  is actually true," from held-out (subject, object, ground_truth) triples
  where ground truth is known independently of the graph. Deliberately
  bypasses the `transitive_relations` allowlist gate — that gate blocks a
  *blind* assumption; this method measures the assumption instead of making
  it, so it shouldn't be blocked by the other (binary) guard.
- **`grounded_reasoning/reasoning/transitivity_calibration.py`** —
  `clopper_pearson_lower(k, n, alpha)`, the exact one-sided binomial
  confidence bound (Clopper & Pearson, 1934), computed by bisection on the
  binomial survival function (no scipy dependency; cross-checked to
  `scipy.stats.beta.ppf` to machine precision in development).
- **Theorem M**, numerically verified (`theorem_transitivity_calibration`,
  `tests/test_theorems.py`): 93.6% empirical coverage across 3,000 random
  true precisions against a 90% target; degenerate evidence (0 or all
  confirmations) behaves sanely.
- **`grounded_reasoning/experiments/transitivity_calibration_eval.py`** — a
  fully offline A/B comparison (synthetic ground truth) of the binary
  allowlist against the calibrated bound on a "trusts"-like relation that's
  85% (not 100%) transitive: the binary mechanism either silently trusts
  everything (17% silently wrong, no risk reported) or rejects everything
  (loses all 85% correct claims); the calibrated bound reports "≥80% true,
  90% confidence" and that bound held on a held-out test set.
- README.md/README.vi.md and PAPER.md §5.3.2 document the new theorem, proof,
  and A/B comparison.

## 0.1.4 — Two opt-in guards for boundaries the algebra can't see itself

Raised in an external review of the algebra (entity identity, transitivity as
a modeling assumption), reproduced with failing tests first, then fixed —
both opt-in and off by default, so this release changes no existing behavior
unless you opt in.

### Added

- **`GroundedReasoner(normalize=...)`** — an optional entity-canonicalization
  hook (e.g. `lambda s: s.strip().casefold()`) applied to every subject/object
  before it becomes a graph key. Closes a real failure mode: an LLM extraction
  that's inconsistent about one entity's surface form (`"Bob"` vs `"bob"`)
  silently splits it into two graph nodes, breaking an otherwise-true proof
  path — the guard then correctly, per its own contract, rejects a claim that
  is actually true, because identity resolution failed one layer above the
  algebra. Proof paths and `contradictions()` cycles still report each
  entity's original first-seen spelling, never the normalized form.
- **`GroundedReasoner(transitive_relations={...})`** — an optional allowlist.
  When set, `verify(..., via=rel)` raises `ValueError` for any `rel` not in
  the set, instead of silently returning a confident `grounded=True` for a
  relation that may not actually be transitive in reality (Theorem G
  guarantees "a path exists under the closure of `via`," not "`via` composes
  transitively in the world" — the algebra cannot tell the two apart from
  data alone). Converts a silent modeling assumption into an explicit,
  checked one.
- `verify_relation`/`run_tool` (the stateless agent tool) now strip incidental
  leading/trailing whitespace from every entity string by default (not
  opt-in — this is unambiguously safe and matches the function's existing
  "tolerant of LLM input" contract). Case is left untouched by default; use
  `GroundedReasoner(normalize=...)` directly for case-insensitive domains.

### Changed

- README.md/README.vi.md and PAPER.md §5.3.1 document both boundaries
  explicitly, with the exact failure-mode reproductions
  (`tests/test_agent.py::TestEntityNormalization`,
  `::TestTransitiveRelationsGuard`) referenced by name.

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
