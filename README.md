# grounded-reasoning — Grounded, Guaranteed Reasoning for LLMs & Agents

[![CI](https://github.com/ALEXaquarius/grounded-reasoning/actions/workflows/ci.yml/badge.svg)](https://github.com/ALEXaquarius/grounded-reasoning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/grounded-reasoning.svg)](https://pypi.org/project/grounded-reasoning/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALEXaquarius/grounded-reasoning/blob/main/examples/quickstart.ipynb)

> **TL;DR.** LLMs hallucinate on multi-hop relational reasoning. This is a
> **relation-algebra verifier** an agent calls to check a claim *before* asserting it:
> **zero model tokens**, **precision-guaranteed** (accepts a claim iff a grounded proof
> path exists), language-agnostic, and provider-agnostic. Plugs in as a **library**, a
> **function-calling tool**, or an **MCP server**. Validated on **real LLMs** (DeepSeek
> et al.) and the public **CLUTRR** benchmark. See [docs/integration.md](docs/integration.md).

📄 Full paper: **[PAPER.md](PAPER.md)** · Integration guide: **[docs/integration.md](docs/integration.md)** · Try it in 30 seconds: **[quickstart notebook](https://colab.research.google.com/github/ALEXaquarius/grounded-reasoning/blob/main/examples/quickstart.ipynb)**

Đọc bằng tiếng Việt: **[README.vi.md](README.vi.md)**

---

## Why this exists

LLMs are solid on one-hop facts but **collapse on composition** — chaining several
correct facts into a multi-step conclusion. On CLUTRR (kinship reasoning), DeepSeek's
accuracy **falls off with depth**, while a grounded operator-composition solver holds
**~100% flat — at zero tokens**:

```
acc
100% ●─────●─────●─────●─────●─────●─────●   ● Grounded solver (algebra, 0 tokens)
 90% |
 80% ○
 70% |  ╲
 60% |   ╲
 50% |    ╲
 40% |     ○           ○                     ○ DeepSeek (LLM)
 30% |      ╲         ╱ ╲
 20% |       ○─────○     ╲
 10% |                    ○─────○
  0% +──┴─────┴─────┴─────┴─────┴─────┴─────┴─
      hop 2    3     4     5     6     7     8   (composition steps)

     hop:      2     3     4     5     6     7     8
     DeepSeek: 83%   42%   25%   25%   42%   17%   8%
     Solver:   100%  100%  100%  100%  100%  100%  100%
```

*(CLUTRR/v1 gen_train234_test2to10, clean-chain, n=12/hop; full test set n=635: solver
covers 99.5%, accuracy 99.2%. `grounded_reasoning/experiments/clutrr_eval.py`.)*

---

## What it is / is NOT (honestly)

**Is:** a guaranteed reasoning-verification layer built on relation operator algebra.
- **Precision = 1.0, guaranteed** (Theorem G) — accepts a claim only if a grounded proof path exists.
- **Zero extra tokens** — local matrix multiplication, no LLM call. Compare to
  "have the LLM self-verify," which costs +110% tokens for 34% precision.
- **Two-sided guarantee** (Theorem I) — precision *and* recall both have tight bounds.
- **No external KB required** (SGDC) — uses the LLM's own internal consistency.
  Precision=1.0 is conditional on the LLM's own atomic facts being sound; that
  assumption can be measured too — `calibrate_transitivity` doesn't care
  whether facts came from an external KB or the model's own assertions, so it
  already calibrates SGDC's real output precision with zero new code (see
  [`self_grounded_calibration_eval.py`](grounded_reasoning/experiments/self_grounded_calibration_eval.py),
  PAPER.md §6's remark).

**Is not:** an "unprecedented breakthrough." The Katz index, the Neumann series,
graph reachability, and neuro-symbolic grounding are all classical math and
technique. The contribution here is unification, a measured guarantee, and
benchmark numbers — not a new primitive. The guard needs a relation graph
(supplied, or extracted from LLM facts); flexibility is bounded (see
[PAPER §5](PAPER.md)).

### Two sharp edges the algebra itself can't see (and how to guard them)

Raised in review, reproduced, and fixed with an opt-in guard each — not swept
under the rug:

- **Entity identity is exact-string by default.** If an LLM extraction is
  inconsistent about one entity's surface form (`"Bob"` vs `"bob"`), the graph
  treats them as two nodes and a real path silently breaks — the guard then
  (correctly, per its own contract) rejects a claim that is actually true.
  Fix (binary): `GroundedReasoner(normalize=lambda s: s.strip().casefold())`
  folds surface-form variants together before they become graph keys; proofs
  still display each entity's original first-seen spelling. **Theorem N**
  characterizes exactly when this is safe: precision stays *exactly* 1.0 as
  long as `normalize` never merges two genuinely distinct entities — that's
  the *only* way it can go wrong, so it's exactly what
  `gr.calibrate_normalization(labeled_pairs)` measures from held-out evidence,
  reusing the same Clopper-Pearson machinery as Theorem M.
- **Theorem G doesn't know if `via` is transitive in reality.** It guarantees
  "a path exists under the closure of `via`," not "`via` actually composes in
  the world." Compose a relation that's only partially/conditionally
  transitive (`"trusts"`: A trusts B, B trusts C, does not imply A trusts C)
  and you get a confident, mathematically correct `grounded=True` that answers
  a different question than the one you meant to ask. Fix (binary):
  `GroundedReasoner(transitive_relations={"parent", "is_a", ...})` makes the
  guard raise `ValueError` for any undeclared relation, turning a silent
  modeling assumption into an explicit, checked one. Fix (measured — **Theorem
  M**): `gr.calibrate_transitivity(rel, labeled_pairs)` replaces the binary
  declare-or-reject with an actual number — a Clopper-Pearson lower confidence
  bound on "a graph-grounded claim for `rel` is really true," computed from
  held-out labeled pairs. Where the binary guard can only guess or block
  outright, the calibrated bound tells you *how much* to trust it.

Both opt-in guards are off by default (identical behavior to previous
releases). Reproductions: `tests/test_agent.py::TestEntityNormalization`,
`::TestTransitiveRelationsGuard`, `::TestTransitivityCalibration`,
`::TestNormalizationCalibration`; the A/B comparisons:
[`transitivity_calibration_eval.py`](grounded_reasoning/experiments/transitivity_calibration_eval.py),
[`normalization_calibration_eval.py`](grounded_reasoning/experiments/normalization_calibration_eval.py).

**Heterogeneous relation chains.** `verify(via=rel)` composes ONE relation with
itself; `gr.verify_path(subject, obj, via=["parent","employer"])` composes an
exact sequence of *different* relations (e.g. a derived "financially dependent
on" claim) — not new math (`OperatorRelationAlgebra.follow` already composes
mixed-relation chains exactly per Theorem G, this just exposes it at the
facade with proof-path reconstruction) — and `gr.calibrate_path(via,
labeled_pairs)` calibrates that fixed pattern with the same Clopper-Pearson
engine as `calibrate_transitivity` (see PAPER.md §5.3.4). Checked against
independent ground-truth BFS across 8,000 triples with zero mismatches:
`tests/test_agent.py::TestHeterogeneousPathVerification`,
[`heterogeneous_path_calibration_eval.py`](grounded_reasoning/experiments/heterogeneous_path_calibration_eval.py).

### How this differs from the usual fixes

| Approach | Extra tokens | Guarantee | Needs an external KB |
|---|---|---|---|
| LLM self-verification (2nd call) | +110% | none (measured 34% precision) | no |
| Self-consistency / majority vote | multiplies with sample count | none, statistical only | no |
| RAG / external KG grounding | varies | only as good as retrieval | yes |
| **This guard** | **+0** | **precision = 1.0** (Theorem G) | no |
| **This guard, self-grounded (SGDC)** | **+0** | precision = 1.0 given sound atomic facts (Theorem I) | no |
| **This guard, conformal** | **+0** | coverage ≥ 1−α, distribution-free (Theorem K) | no |

---

## Three theorems, one operator (F = G = H)

The reasoning core rests on a single unification (numerically verified, zero error):

| View | Theorem | Content |
|------|---------|---------|
| Fuzzy diffusion inference | **F** | conf(a→b) = Σ αᵏ(Pᵏ)[a,b], calibrated + grounded |
| Relation operator algebra | **G** | composition = operator product, transitive closure = Σ powers |
| Spectral analysis (Katz) | **H** | `engine.infer` = resolvent (I−αP)⁻¹−I (matches **0.0** error) |

⟹ fuzzy inference **is** spectral analysis of the relation operator. `grounded_reasoning/reasoning/`.

Six further theorems extend this core: **I** (two-sided precision/recall guarantee
for a self-grounded, no-external-KB variant), **J** (closure-learning completeness,
validated on CLUTRR), **K** (conformal reasoning — distribution-free coverage under a
*noisy* relation graph, including one extracted by an LLM from raw text), **L**
(Horn forward-chaining, generalizing transitive closure to conjunctive rules),
**M** (empirical transitivity calibration — a Clopper-Pearson confidence bound
replacing a blind transitivity assumption with a measured one), and **N**
(normalization precision isolation — precision=1.0 breaks only via an
over-merge, and *only* that is what needs calibrating). All
nine are stated, proved, and numerically verified in [PAPER.md](PAPER.md).

---

## Evidence on real LLMs (DeepSeek)

| Experiment | Result |
|------------|--------|
| Hallucination guard (kinship) | precision **33% → 100%**, catches 92/92 (two seeds), 0 false rejects |
| Hallucination guard, harder stress test (48-person tree, sibling/spouse distractor facts, shuffled prose, T=0.7, guaranteed-empty trap questions) | raw DeepSeek precision **4.6%** (2124 fabricated names, 86/90 trap questions answered with a fabrication); guarded precision **100%**, 0 leaked, 0 correct answers dropped — [`guard_llm_stress_eval.py`](grounded_reasoning/experiments/guard_llm_stress_eval.py) |
| Guard token cost | **+0 tokens** (vs. LLM self-verify: +110% tokens, 34% precision) |
| SGDC (self-grounded, no external KB) | precision **78% → 100%** from internal consistency alone |
| Dense, anti-commonsense ontology | precision **31% → 100%**, catches 106/106, 0 false rejects — [`nl_ontology_eval.run_dense`](grounded_reasoning/experiments/nl_ontology_eval.py) |
| CLUTRR (public benchmark) | solver **~100% at every hop** vs. DeepSeek 83%→8% |
| Hard passage (9-step chain, 8 questions) | DeepSeek **fabricates 1/8** (wrong direction); grounded system **8/8**, with proofs — [`examples/hallucination_demo.py`](examples/hallucination_demo.py) |

---

## Guaranteed reasoning over a graph an LLM extracted from raw text

The guard/solver needs a **clean** graph. But if you let an **LLM extract** relations
from natural-language text, the graph is **noisy** (missing/spurious edges).
**Conformal Reasoning** (Theorem K) fixes exactly that: use operator confidence as a
score, calibrate a threshold ⟹ **distribution-free coverage ≥ 1−α**, even on a noisy
graph.

End-to-end demo: **DeepSeek extracts an "is a" graph from text** → conformal runs on
that extracted graph (ground truth is used only for scoring):

| Text | LLM extraction (P / R) | Coverage (target ≥90%) | Efficiency (FPR) |
|------|------------------------:|----------------------------:|------------------:|
| Easy | 100% / 99.7% | **91.3%** | 0.0 |
| Hard (nested clauses + near-miss distractors) | 99.5% / **68.5%** | **93.0%** | 0.77 |

> The LLM's extraction **drops 31% of the edges** (a genuinely noisy graph) →
> **the coverage guarantee still holds** (93% ≥ 90%), only efficiency degrades.
> *Validity always holds; efficiency scales with graph quality.*

⟹ A path to guaranteed reasoning over **natural-language relations** — where the hard
guard can't reach. `grounded_reasoning/experiments/conformal_llm_eval.py`.

---

## Self-verification with NO external knowledge base (SGDC)

The guard above needs *some* relation graph handed to it. Self-Grounded
Deductive Consistency (Theorem I) removes even that: it exploits the fact
that LLMs are reliably accurate on **atomic (1-hop) facts** but hallucinate on
**composition**. Take the model's own confident 1-hop facts, build the
operator closure from *those*, then reject any of the model's own multi-hop
conclusions that fall outside its *own* closure — self-contradiction is the
hallucination signal, not disagreement with an external source.

```python
from grounded_reasoning import GroundedReasoner

# the LLM's OWN atomic facts (no external KB) -- taken at face value
gr = GroundedReasoner()
gr.add_facts([("sparrow", "is_a", "bird"), ("bird", "is_a", "animal")])

# the LLM's OWN multi-hop conclusion, self-verified against ITS OWN facts above
gr.verify("sparrow", "animal", via="is_a")   # grounded=True: self-consistent
gr.verify("sparrow", "plant",  via="is_a")   # grounded=False: self-contradiction, blocked
```

| | precision | recall |
|---|---:|---:|
| Raw multi-hop (LLM) | 78% | 87% |
| **SGDC (self-grounded, zero external knowledge)** | **100%** | 72% |
| Ceiling: filtering with an external graph | 100% | 87% |

The honest cost is recall (72% vs. 87%): self-closure is conservative. And
Theorem I's precision=1.0 is *conditional* — it holds if the model's own
atomic facts are sound; in a counter-prior domain (e.g. "a whale is a fish"),
atomic precision itself can drop, and recall suffers with it (PAPER.md §6
records this honestly rather than hiding it).

**That assumption can be measured too, with zero new code.**
`gr.calibrate_transitivity(rel, labeled_pairs)` (Theorem M) doesn't care
whether `gr`'s facts came from an external KB or the model's own atomic
self-assertions — so calling it on a reasoner built purely from an LLM's own
facts calibrates SGDC's *actual* output precision directly, from held-out
evidence, instead of assuming atomic soundness. In a synthetic domain with
15% of the atomic facts deliberately wrong, SGDC's real precision fell to
~74% (**not** the naively-expected ~85% — a single wrong atomic edge
composes into several downstream claims, amplifying its damage), and the
calibrated bound correctly stayed below that in 98.3% of trials —
[`self_grounded_calibration_eval.py`](grounded_reasoning/experiments/self_grounded_calibration_eval.py),
PAPER.md §6's remark.

Runnable: [`examples/self_grounded_demo.py`](examples/self_grounded_demo.py)
(offline) · live on DeepSeek:
`grounded_reasoning/experiments/self_grounded_eval.py`.

---

## Quickstart

```bash
pip install grounded-reasoning

# or, for development (tests + lint):
git clone https://github.com/ALEXaquarius/grounded-reasoning
cd grounded-reasoning && pip install -e ".[dev]"
pytest tests/                       # every theorem + offline-locked logic, no network needed

# Use it right now (no LLM/network needed):
python -c "from grounded_reasoning import GroundedReasoner as G; r=G(); r.add_facts([('a','p','b'),('b','p','c')]); print(r.verify('a','c',via='p'))"

# Real-LLM experiments (need a key — read from an env var, NEVER hardcoded):
export DEEPSEEK_API_KEY=sk-...        # bring your own; .env is gitignored
python -m grounded_reasoning.experiments.guard_llm_eval        # hallucination guard
python -m grounded_reasoning.experiments.guard_llm_stress_eval # harder: distractors + traps + high temperature
python -m grounded_reasoning.experiments.self_grounded_eval    # SGDC
python -m grounded_reasoning.experiments.clutrr_eval           # public CLUTRR benchmark
python -m grounded_reasoning.experiments.conformal_llm_eval    # end-to-end conformal (LLM-extracted graph)
python -m grounded_reasoning.experiments.guard_cost_eval       # token cost: guard vs. LLM self-verify
python -m grounded_reasoning.experiments.nl_ontology_eval      # dense anti-commonsense ontology (add run_dense() for the 106/106 result)
```

---

## Integrating with an Agent / LLM (`grounded_reasoning/agent/`)

A **relation-reasoning verifier** for agents: check a multi-hop claim **before
asserting it** — zero model tokens, precision guaranteed (accepts iff a grounded proof
path exists).

```python
from grounded_reasoning import GroundedReasoner
gr = GroundedReasoner()
gr.add_facts([("alice","parent","bob"),("bob","parent","carol")])
gr.verify("alice","carol", via="parent")   # Verdict(grounded=True, proof=['alice','bob','carol'], confidence=0.36, relation='parent')
gr.verify("alice","zed",   via="parent")   # Verdict(grounded=False, proof=None, confidence=0.0, relation='parent')  ← hallucination blocked
```

Three integration paths (details: [docs/integration.md](docs/integration.md)):
- **Library**: `GroundedReasoner.verify / filter_claims / contradictions`.
- **Function-calling**: `TOOL_SPEC` (Anthropic) / `openai_tool_spec()` (OpenAI) + `run_tool` — a stateless `verify_relation` tool.
- **MCP server**: `python -m grounded_reasoning.agent.mcp_server` — plugs into Claude or any MCP-compatible agent.

**Multi-provider** (not just DeepSeek): `LLMClient(provider=...)` for DeepSeek / OpenAI /
Groq / OpenRouter / Together / Mistral / Ollama (local) — all OpenAI-compatible, switch
providers without changing code. **Multilingual**: entities/relations are opaque
Unicode strings ⟹ works with any language (`cha`, `父`, `والد`…) with zero configuration.

A real function-calling demo (agent verifies itself, blocks hallucination):
`python -m grounded_reasoning.experiments.agent_demo`. When the graph is **noisy** (relations
extracted by an LLM from text), use `ConformalReasoner` for a **coverage ≥1−α**
guarantee instead of hard precision.

---

## Source map

| Path | Content |
|------|---------|
| `grounded_reasoning/` | Public package — `GroundedReasoner`, `verify_relation`, `TOOL_SPEC`, `ConformalReasoner`, `LLMClient` |
| `grounded_reasoning/agent/{verifier,tool,mcp_server}.py` | Public API implementation — HallucinationGuard, function-calling tool, MCP server |
| `grounded_reasoning/reasoning/abstract_inference.py` | FuzzyInferenceEngine, TypedInferenceEngine, HallucinationGuard (Theorem F) |
| `grounded_reasoning/reasoning/operator_algebra.py` | Relation operator algebra (Theorem G) |
| `grounded_reasoning/reasoning/relation_spectrum.py` | Spectrum, nilpotency, Katz resolvent (Theorem H) |
| `grounded_reasoning/reasoning/conformal_reasoning.py` | Conformal — coverage guarantee under noise (Theorem K) |
| `grounded_reasoning/reasoning/composition_algebra.py` | Composition-table learning, validated on CLUTRR (Theorem J) |
| `grounded_reasoning/reasoning/horn.py` | Horn forward-chaining, least-model semantics (Theorem L) |
| `grounded_reasoning/reasoning/transitivity_calibration.py` | Clopper-Pearson calibration — reused for both the transitivity assumption (Theorem M) and the normalization over-merge risk (Theorem N) |
| `grounded_reasoning/reasoning/llm_client.py` | Provider-agnostic LLM client (key read from an env var) |
| `grounded_reasoning/theory/theorems.py` | **Nine theorems (F–N)** with numerical verification |
| `grounded_reasoning/experiments/{guard_llm,guard_llm_stress,self_grounded,self_grounded_calibration,nl_ontology,guard_cost,clutrr,conformal_llm,inference,transitivity_calibration,normalization_calibration,heterogeneous_path_calibration}_eval.py` | Real-LLM and benchmark experiments backing every claim above |
| `examples/hallucination_demo.py` | End-to-end function-calling demo (real LLM, needs a key) |
| `examples/self_grounded_demo.py` | SGDC (Theorem I): self-verify a model's own multi-hop claim with NO external KB (offline) |
| `examples/rag_pipeline_demo.py` | `filter_claims` as a RAG/agent post-processing guard, heterogeneous claims (offline) |
| `examples/calibration_demo.py` | Theorem M + N side by side: measuring transitivity and normalization trust instead of assuming it (offline) |
| `examples/conformal_demo.py` | Coverage guarantee vs. noise tradeoff, clean vs. noisy graph side by side (offline) |
| `examples/quickstart.ipynb` | Runnable tour of the library (offline, Colab-ready) |

---

## Origin story

This project began as an attempt to invent an embedding-free retrieval algorithm that
could compete with dense/RAG retrieval. That research question reached a rigorous,
fully honest **negative** conclusion (ties BM25, loses significantly to dense
embeddings — with a proof of why). The same mathematical toolkit — operator algebra,
spectral analysis — turned out to have real, measurable value on a different problem:
**guaranteeing** multi-hop relational reasoning. This repository ships only that
validated, tested reasoning system; the full retrieval research trail (including every
failed attempt, honestly recorded) lives in a separate research repository and is not
part of this package. See [PAPER.md §1](PAPER.md) for the full framing.

---

## Contributing & Community

- How to contribute + research principles: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) · Security: [SECURITY.md](SECURITY.md)
- Version history: [CHANGELOG.md](CHANGELOG.md) · Citation: [CITATION.cff](CITATION.cff)
- License: **MIT** ([LICENSE](LICENSE))

---

*Principle: proof before code, formal definitions, falsifiability, and honest
reporting of negative results — see [CONTRIBUTING.md](CONTRIBUTING.md).*

<!-- mcp-name: io.github.ALEXaquarius/grounded-reasoning -->
