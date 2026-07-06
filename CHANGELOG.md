# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- **`identify_suspect_edges` / `prune_edges` / `identify_and_prune_edges` /
  `identify_suspect_edges_propagated` / `masked_infer`**
  (`grounded_reasoning/reasoning/edge_pruning.py`) — held-out-evidence edge
  pruning: identifies and removes specific spurious edges from a noisy
  relation graph using held-out labeled evidence, instead of only
  recalibrating a threshold around them. A simple decision rule, not a
  statistical guarantee — no false-discovery-rate bound, verified
  empirically instead. Cuts false-positive rate substantially and
  consistently across 5 synthetic noise regimes (e.g. 77% → 49% under
  dropout-dominant noise), while the wrongly-removed rate at the
  recommended configuration (`identify_frac=0.85, min_evidence=2`, the
  default of `identify_and_prune_edges`) stays at 1.5–4.2% (95% upper
  bound 2.6–8.9%) — down from 13–32% at the naive default split.
  Checked against real DeepSeek hallucinations, not just simulated noise
  (n_seeds=6, 6 batches, 5126 labeled pairs / 5763 unique edges, 71%
  hallucinated, correctly namespaced so pooled query batches never
  collide — an earlier pass had a node-tagging bug that silently merged
  distinct query DAGs and spuriously inflated per-edge evidence counts,
  now fixed): on data where each candidate edge is backed by exactly one
  labeled encounter (no repeated queries — the realistic single-verification
  case), the count-based rules (`min_evidence≥2`, and
  `identify_suspect_edges_propagated`'s hub-magnet refinement) never fire
  at all, and lowering to `min_evidence=1` alone makes downstream FPR
  *worse* than no pruning (63.0% → 70.7%, traced to the diffusion engine's
  row-normalization concentrating probability onto a source's surviving
  edges once its others are pruned). `masked_infer` — scoring that
  normalizes by each source's pre-prune degree instead, so removal only
  ever removes confidence mass rather than redistributing it — recovers a
  real improvement on that same data (63.0% → 54.0%, beats raw in 12/15
  splits), with no regression on the synthetic benchmark. See
  `edge_pruning_eval.py`, `edge_pruning_llm_eval.py`,
  `tests/test_edge_pruning.py`, and PAPER.md §7.1's remark.

## 0.1.7 — Heterogeneous paths, deeper fuzzing, SGDC calibration, and two conformal extensions

### Added

- **`AdaptiveConformalReasoner`** (Adaptive Conformal Inference, Gibbs & Candès
  2021 — classical, not new): both `ConformalReasoner` and its Mondrian
  extension assume calibration and test data are exchangeable, which breaks
  if the noise level *drifts* over time (e.g. later document batches
  extracted more/less cleanly). ACI instead updates its threshold from a
  stream of confirmed-true examples, with no stationarity assumption needed
  for its own guarantee. Verified over 15 trials: a stream shifting from
  `p_drop=0.05` to `p_drop=0.45` partway through collapses a frozen
  threshold's coverage from 88.6% to 47.6%, while ACI recovers to 89.6% and
  stays there in every trial — `grounded_reasoning/experiments/drift_conformal_eval.py`,
  `tests/test_drift_conformal_eval.py`. This followed 6 other hypotheses
  tried and falsified/found neutral in the same exploration (hop-distance
  grouping, bootstrap-stability scoring, finer redundancy buckets,
  contribution-concentration grouping, cross-view corroboration) — recorded
  honestly rather than hidden.
- **`ConformalReasoner.calibrate(..., group_fn=...)`** (Mondrian /
  group-conditional conformal prediction — classical, not new): calibrates a
  separate threshold per group of an available-at-test-time partitioning
  function instead of one global threshold, still satisfying the same
  coverage guarantee within each group. `redundancy_group` (new:
  `FuzzyInferenceEngine.path_multiplicity`) groups a pair by whether it has
  more than one walk in the extracted graph; verified over 60 seeds/scenario
  to cut FPR from 98.7% to 80.8% when dropped edges dominate the noise
  (matching real LLM-extraction noise), with no benefit when spurious added
  edges dominate instead (disclosed, not hidden) —
  `grounded_reasoning/experiments/redundancy_conformal_eval.py`,
  `tests/test_redundancy_conformal_eval.py`. A different grouping (hop-
  distance) was tried first and numerically falsified before shipping.
- **`GroundedReasoner.calibrate_transitivity` extended to SGDC, with zero new
  code**: SGDC's Theorem I precision=1.0 guarantee is conditional on the
  LLM's own atomic facts being sound. `calibrate_transitivity` doesn't care
  whether a reasoner's facts came from an external KB or the model's own
  atomic self-assertions, so it already calibrates SGDC's real output
  precision directly. Verified: with 15% of a synthetic domain's atomic facts
  wrong, SGDC's real precision fell to ~74% (not the naively-expected ~85% —
  a single wrong atomic edge composes into several downstream claims,
  amplifying its damage), and the calibrated bound stayed below that in
  98.3% of trials — `grounded_reasoning/experiments/self_grounded_calibration_eval.py`,
  documented as a Remark in PAPER.md §6 (not a new theorem).
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
- **`grounded_reasoning/experiments/guard_llm_stress_eval.py`** — a harder,
  live-DeepSeek hallucination-guard stress test (48-person tree, sibling/
  spouse distractor facts, shuffled prose, T=0.7, guaranteed-empty trap
  questions). Measured: raw precision as low as 4.6% (2124 fabricated names,
  86/90 trap questions fabricated); guarded precision 100%, 0 leaked, 0
  correct answers dropped.
- **4 new runnable, fully offline demos** in `examples/` (previously only 2
  existed): `self_grounded_demo.py` (SGDC with no external KB), `rag_pipeline_demo.py`
  (`filter_claims` as a RAG post-processing guard over heterogeneous
  `via=[...]` claims), `calibration_demo.py` (Theorem M + N side by side),
  `conformal_demo.py` (coverage-vs-noise tradeoff, now also demonstrating
  `redundancy_group` grouping and `AdaptiveConformalReasoner`).
- A wider fuzz-testing pass (`tests/test_fuzz_regressions.py::TestWiderFuzzProperty`)
  cross-checking `verify(via=None)`, `verify_path`, `contradictions`,
  `normalize=`, `filter_claims` dispatch, and `transitive_relations=` against
  independent baselines over thousands of random graphs — 0 correctness
  failures found. Surfaced one real, previously-undocumented property (not a
  bug): `Verdict.confidence` is `Sum_k alpha^k*(P^k)[a,b]`, a sum across
  hop-counts, not itself a probability, so a strong self-loop/cycle can push
  it above 1.0 — clarified in `Verdict.confidence`'s docstring and locked
  with a regression test.

### Fixed

- **Hash-seed-dependent test/demo nondeterminism**: three places iterated a
  Python `set` of *strings* (`edges` in `conformal_llm_eval.py`'s `render()`,
  `gold` in `tests/test_conformal_llm.py`, and a `variants()` set in
  `normalization_calibration_eval.py`) while consuming an RNG stream one draw
  per item. Set iteration order is randomized per-process for `str`/`bytes`
  keys (`PYTHONHASHSEED`) — unlike `int` keys, which hash deterministically —
  so which RNG draw landed on which item, and hence the numeric/textual
  outcome, silently varied by interpreter hash seed even with every
  `random.Random(seed)` fixed. This is the confirmed root cause of the
  intermittent, previously-unreproducible coverage-assertion failures seen
  in `test_conformal_llm.py` and `test_normalization_calibration_eval.py`.
  Fixed by wrapping each set in `sorted()` before iterating/sampling.
  Verified: all three call paths now produce byte-identical output across 5+
  distinct `PYTHONHASHSEED` values (previously varied on every one). Audited
  every other `set`-of-strings iteration in the codebase (theorems, other
  experiments, other tests) for the same pattern — none found.
- **2 doc/code drift gaps** found in a careful documentation audit: the
  `Verdict` repr example in README.md/README.vi.md omitted the real
  `confidence`/`relation` fields; the Quickstart's runnable-command list
  omitted `guard_cost_eval.py` and `nl_ontology_eval.py` despite both
  needing a real LLM key and backing claims elsewhere in the same README.
- **1 miscategorization in PAPER.md's "Reproducing this work"**:
  `inference_eval.py` (a fully offline synthetic simulation, no LLM call
  anywhere in the file) was listed under "Real-LLM experiments"; moved to
  the offline-only list, and two experiments that existed but were missing
  from either list (`guard_llm_stress_eval.py`, `heterogeneous_path_calibration_eval.py`)
  were added to the correct one.
- **A degenerate-precision false alarm in `guard_llm_stress_eval.py`**:
  `max_queries_per_trial` previously let an all-trap subsample through (a
  trap query's ground truth is empty by construction, so an all-trap sample
  has zero true positives), misreporting `tp/(tp+fp)` as 0.0 and reading as
  "GUARD LEAK" even though the guard had correctly filtered every single
  hallucination. Fixed by capping trap queries at half the per-trial budget
  and reporting precision as `None` (undefined, not "bad") rather than 0.0
  when `tp+fp==0`; the pass/fail check now keys off `guard_leaked==0` directly.
- **A measurement artifact in `examples/conformal_demo.py`**: its FPR metric
  included pairs with confidence exactly 0 (never genuinely "at risk" of
  acceptance except in the degenerate `tau<=0` case), which made the demo's
  own "efficiency degrades with noise" narrative an artifact of pure edge-
  dropping (which can never fabricate a false positive) rather than a real
  phenomenon. Fixed by filtering to genuine candidates and adding a fixed
  spurious-edge rate, matching `theorem_conformal_reasoning`'s own convention.

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

### Fixed

- **Hash-seed-dependent test/demo nondeterminism**: three places iterated a
  Python `set` of *strings* (`edges` in `conformal_llm_eval.py`'s `render()`,
  `gold` in `tests/test_conformal_llm.py`, and a `variants()` set in
  `normalization_calibration_eval.py`) while consuming an RNG stream one draw
  per item. Set iteration order is randomized per-process for `str`/`bytes`
  keys (`PYTHONHASHSEED`) — unlike `int` keys, which hash deterministically —
  so which RNG draw landed on which item, and hence the numeric/textual
  outcome, silently varied by interpreter hash seed even with every
  `random.Random(seed)` fixed. This is the confirmed root cause of the
  intermittent, previously-unreproducible coverage-assertion failures seen
  in `test_conformal_llm.py` and `test_normalization_calibration_eval.py`.
  Fixed by wrapping each set in `sorted()` before iterating/sampling.
  Verified: all three call paths now produce byte-identical output across 5+
  distinct `PYTHONHASHSEED` values (previously varied on every one). Audited
  every other `set`-of-strings iteration in the codebase (theorems, other
  experiments, other tests) for the same pattern — none found.

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
