# Grounded, Guaranteed Reasoning for LLMs and Agents

**Abstract.** Large language models are solid at recalling one-hop facts but
degrade sharply on **multi-hop relational reasoning** — composing several facts
into a conclusion. We present a **relation-algebra verifier** that checks a
multi-hop claim *before* it is asserted, at **zero model tokens**, with a
**precision guarantee**: it accepts a claim if and only if a grounded proof path
exists in a supplied relation graph. The core result is a theoretical
unification — three independently-motivated formulations of relational
inference (fuzzy diffusion, operator algebra, spectral/Katz analysis) are
proven to be **the same operator**, matching to zero numerical error. Built on
this, a **HallucinationGuard** turns a real LLM's (DeepSeek) multi-hop
precision from **33% to 100%** at **+0 tokens**, versus an LLM-self-verification
baseline that costs **+110% tokens** for only 34% precision. We further prove a
**two-sided precision/recall guarantee** for a self-grounded variant that needs
no external knowledge base (Theorem I), and a **conformal** extension giving
**distribution-free coverage ≥ 1−α** even when the relation graph itself is
noisy — including end-to-end on a graph extracted by an LLM from raw text
(Theorem K). The system is validated on the public **CLUTRR** benchmark, where
the grounded solver holds ~100% accuracy flat across 2–10 composition hops
while DeepSeek's accuracy falls from 83% to 8%. Nine theorems are stated,
proved, and numerically verified — including a pair of Clopper-Pearson
calibrations (Theorems M and N) that replace two blind modeling assumptions
(transitivity, and entity-normalization safety) with measured confidence
bounds; every claim distinguishes what is genuinely new (a unification, a
measured guarantee, a described failure boundary) from what is classical
mathematics applied honestly (Katz index, Neumann series, conformal
prediction, Clopper-Pearson intervals, Horn logic).

---

## 1. Introduction

LLMs hallucinate most on **composition**: chaining several individually-correct
facts into a multi-step conclusion. This is a well-documented weakness, and
existing mitigations — asking the model to self-check, sampling multiple
chains-of-thought and voting, or grounding in an external knowledge base — are
either expensive (extra tokens), unguaranteed (still hallucinate at a measured
rate), or require infrastructure the caller may not have (a curated KB).

This project takes a narrower, more tractable position: if the individual
relations involved can be represented as a graph (subject, relation, object),
then checking whether a multi-hop *claim* is entailed by that graph is a
problem with an **exact, cheap, and classical** solution — linear algebra over
a boolean/fuzzy relation operator. The contribution is not a new primitive; it
is (a) proving that three natural ways of setting this up are **the same
operator**, (b) turning that into a hallucination guard with a **measured
guarantee and zero token cost** on a real commercial LLM, and (c) honestly
describing the boundary — where the guard needs a graph it cannot supply
itself.

> **Origin note.** This project began as an attempt to invent an
> embedding-free retrieval algorithm competitive with dense/RAG retrieval. That
> line of work reached a rigorous, fully negative conclusion (it ties BM25 but
> loses significantly to dense embeddings, with an information-theoretic
> argument for why). The pivot to relational reasoning below is where the same
> mathematical toolkit (operator algebra, spectral analysis) turned out to have
> real, measurable value. The full retrieval research trail — including every
> failed attempt — is kept in a separate research repository and is not part
> of this package; this repository ships only the validated, tested reasoning
> system.

Contributions:

1. **The F=G=H unification** — fuzzy diffusion inference, relation operator
   algebra, and the Katz spectral resolvent are proven to be the same
   operator, matching to zero numerical error.
2. **HallucinationGuard** — verifies LLM claims with a precision = 1.0
   guarantee (Theorem G), at **+0 tokens**; evaluated on **real DeepSeek**
   (33% → 100% precision).
3. **A falsifiable boundary** — a description, backed by experiments, of where
   LLM multi-hop reasoning holds versus collapses.
4. **Nine theorems (F–N)**, each with numerical verification
   (`grounded_reasoning/theory/theorems.py`, exercised by `tests/`), including **Theorem I**
   — a two-sided precision *and* recall guarantee for a self-grounded variant
   that needs no external knowledge base — **Theorem K**, a conformal
   extension with a distribution-free coverage guarantee under a noisy,
   LLM-extracted graph — and **Theorems M/N**, a pair of Clopper-Pearson
   calibrations replacing two blind modeling assumptions (transitivity,
   entity-normalization safety) with measured confidence bounds.

The letter labels (F–L) are kept as-is from the project's internal numbering,
where A–E were unrelated retrieval theorems not included in this repository.

---

## 2. Related work

- **Neuro-symbolic grounding / knowledge-graph verification** (Garcez et al.
  and the broader neuro-symbolic literature): verifying model output against
  symbolic knowledge is an established line of work — this guard belongs to
  it. The contribution here is a specific unification and a measured
  guarantee, not a new paradigm.
- **Katz index** (Katz, 1953) and the **Neumann series / resolvent**
  (I−αA)⁻¹: classical operator analysis for multi-hop graph proximity. Used
  here as one of three equivalent views of the same inference operator.
- **Conformal prediction** (Vovk et al.): distribution-free coverage
  guarantees. Applied here to an operator-confidence score over a relation
  graph, including under LLM-induced extraction noise.
- **Horn logic / Datalog forward-chaining**: classical least-model semantics,
  generalizing the transitive-closure guard to conjunctive rule bodies.
- **Clopper-Pearson intervals** (Clopper & Pearson, 1934): the classical exact
  confidence interval for a binomial proportion. Applied here to calibrate
  confidence in two modeling assumptions from held-out labeled pairs: the
  transitivity assumption itself (§5.3.2, Theorem M) and the safety of an
  entity-normalization function (§5.3.3, Theorem N).
- **CLUTRR** (Sinha et al., EMNLP 2019): a public kinship-reasoning benchmark
  used here for a genuinely comparable, third-party evaluation.
- **Self-consistency** (Wang et al.) and **Chain-of-Verification**: sampling
  or self-checking approaches to reducing hallucination — statistical, token-
  costly, and unguaranteed, contrasted directly with this system in §6.

None of these tools are claimed as new. The contribution is unifying three of
them into one operator, and measuring the resulting guarantee's cost and
limits on a real LLM.

---

## 3. The reasoning engine — three views of one operator

**FuzzyInferenceEngine** (`grounded_reasoning/reasoning/abstract_inference.py`) represents
relational knowledge as a graph and computes confidence by diffusion:
`conf(a→b) = Σ_{k=1}^{K} α^k (P^k)[a,b]`.

**Theorem F (Grounded Fuzzy Inference).** This engine has three properties:
1. **Calibrated**: confidence decreases monotonically with inference depth
   (1→6 hops: 0.60 → 0.047 in the reference experiment).
2. **Deep chaining**: infers ≥4-hop relations that no 1-hop matching
   procedure can reach.
3. **Grounded**: zero false inferences across 3,036 trial pairs — it infers a
   relation only when a real path exists, never fabricating one.

### 3.1 Multi-hop inference — a Pareto point (`grounded_reasoning/experiments/inference_eval.py`)

Three strategies compared on deep-recall (≥4 hops) versus hallucination rate:

| Strategy | Deep-recall (≥4 hops) | Hallucination rate |
|----------|------------------------:|---------------------:|
| A. One-hop matching (≈ embedding similarity) | 0.00% | 0.00% |
| B. **Fuzzy diffusion (this engine)** | **100.00%** | **0.00%** |
| C. Guesser (≈ an overconfident LLM) | 100.00% | 68.43% |

Fuzzy diffusion is the only strategy that reaches both axes simultaneously —
maximal deep inference *and* zero hallucination, a Pareto point neither
alternative reaches.

### 3.2 Operator algebra — Theorem G (`grounded_reasoning/reasoning/operator_algebra.py`)

Each relation `r` maps to a boolean operator `R_r` on ℝⁿ (concepts as basis
vectors): `R_r[i,j] = 1 ⟺ j --r--> i`. Inference becomes exact linear algebra:

| Inference operation | Operator |
|----------------------|--------------------------|
| Composition `r∘s` (e.g. grandparent = parent∘parent) | product `R_r · R_s` |
| Transitive closure `r*` (e.g. ancestor) | `Σ_k R_r^k` |
| Inverse `r⁻¹` (backward inference) | transpose `R_rᵀ` |
| Analogy A:B::C:? | apply the operator inferred from (A→B) to C |

**Theorem G (Operator-Compositional Equivalence)** — numerically verified,
zero mismatch on random graphs (`test_operator_equivalence_exact`): operator
product equals set-based composition; the sum of powers equals BFS
reachability; transpose equals the reversed-edge relation. Because matrix
multiplication is associative, an arbitrarily long inference chain collapses
into a single operator: a derived relation (grandparent, ancestor, ...) is an
element of the algebra generated by the base relations. This is grounded,
guaranteed reasoning — no learning, no embeddings.

### 3.3 Spectral analysis — Theorem H (`grounded_reasoning/reasoning/relation_spectrum.py`)

For the adjacency matrix `A` of a relation, the inference structure is a
spectral invariant:

| Relation structure | Spectral signature |
|----------------------|-----------------------|
| A real hierarchy (parent, part-of) — acyclic | `A` nilpotent ⟺ ρ(A)=0 ⟺ closure halts after ≤n steps |
| Has a cycle | ρ(A) ≥ 1; vertex `i` on a cycle ⟺ `(Σ A^k)[i,i] > 0` |
| Transitive belief (Katz) | `Σ_{k≥1} α^k P^k = (I−αP)⁻¹ − I`, converges iff `α·ρ(P) < 1` |

**Theorem H (Spectral Structure)** — numerically verified, zero mismatch:
acyclic ⟺ nilpotent ⟺ ρ=0; cycles detected exactly. The key result:

> `FuzzyInferenceEngine.infer` matches the truncated resolvent `(I−αP)⁻¹−I`
> to **0.0** numerical error (`test_relation_spectrum_engine_equals_resolvent`).

This unifies the reasoning axis: fuzzy inference (F), operator algebra (G),
and spectral analysis (H) are three views of **the same operator**. Transitive
belief is the Neumann series of the relation operator; acyclicity, cycles, and
propagation all read off from its spectrum.

---

## 4. Evidence on real LLMs (DeepSeek)

### 4.1 Blocking hallucination — `grounded_reasoning/experiments/guard_llm_eval.py`

DeepSeek is given only one-hop facts (e.g. `parent`) and asked 48 multi-hop
questions (grandparent, great-grandparent, ancestor, including empty-answer
trap questions); the operator algebra (Theorem G) serves as ground truth and
guard, across two independent seeds:

| | Raw LLM | After guard |
|---|---:|---:|
| seed 0 | precision 35.1% (50 fabricated names) | **100%** — catches 50/50, 0 leaked, 0 wrongly dropped |
| seed 1 | precision 38.2% (42 fabricated names) | **100%** — catches 42/42, 0 leaked, 0 wrongly dropped |

The guard turns precision to 100% **without lowering recall** — it is a pure
filter (Theorem G guarantees it never drops an answer with a real path).
Locked offline by `tests/test_guard_llm.py` (a mock LLM, no network/key
needed).

### 4.2 The hallucination boundary — `grounded_reasoning/experiments/nl_ontology_eval.py`

Extending the guard to three real natural-language relations (is-a, causes,
part-of), every one of which is acyclic ⟹ a nilpotent operator (Theorem H,
verified experimentally):

| Scenario | LLM precision | Hallucinations | After guard |
|----------|---------------:|------------------:|--------------|
| Short NL chains, real words (30 questions, with counter-prior traps) | **100%** | 0 | 100% (zero-cost) |
| Dense DAG, abstract concepts (8 questions, large closures) | **32.9%** | 94 | **100%** (catches 94/94, drops 0) |

The LLM does not hallucinate on explicit bridging over short, real chains —
the guard is a zero-cost safety net there. But on transitive closure over a
dense graph of abstract concepts, the LLM collapses (over-claims ~21 concepts
when the truth is 5–9); the guard restores precision to 100% without dropping
a single correct answer. The guard's value is proportional to reasoning
difficulty: harmless when the LLM is right, decisive when it collapses.

(A fresh live reproduction of this same dense-DAG scenario — different
session, different random draw — measured 30.7% precision, 106 hallucinations,
guard 106/106; README.md cites that run under "Dense, anti-commonsense
ontology." Both numbers are the same script, `nl_ontology_eval.run_dense`,
sampled independently; the qualitative collapse-then-restore pattern is what's
load-bearing, not the exact percentage.)

### 4.3 Token cost — `grounded_reasoning/experiments/guard_cost_eval.py`

Measured live on DeepSeek (dense DAG, 6 questions), two ways to fix the same
hallucination:

| Hallucination fix | Extra LLM tokens | Other cost | Precision |
|---------------------|--------------------:|-------------|-----------:|
| **Guard (local algebra)** | **+0** | +7.24ms CPU (O(n²·K)) | **100%** (guaranteed) |
| LLM self-check (2nd call) | **+2253 (+110%)** | one more API round trip | 34% (unguaranteed) |

The guard runs on local matrix multiplication (operator closure) and never
calls the LLM — an offline-locked invariant (`test_guard_uses_zero_llm_tokens`).
Self-verification costs 110% more tokens and is still unguaranteed.

### 4.4 A public benchmark — CLUTRR (`grounded_reasoning/experiments/clutrr_eval.py`)

For genuinely comparable numbers, the system is evaluated on **CLUTRR**
(Sinha et al., EMNLP 2019) — kinship relational reasoning by composition,
mapping directly onto the operator algebra (Theorem G). A grounded, zero-token
solver learns a binary composition table from TRAIN `proof_state` data (never
touching test labels) via closure to a fixpoint (98 rules, 0 conflicts, 2
iterations), then folds along the query path.

On the clean-chain test subset (n=12/hop), DeepSeek vs. the solver by number
of composition steps:

| hop | DeepSeek accuracy | Grounded solver (0 tokens) |
|----:|---------------------:|------------------------------:|
| 2 | 83% | **100%** (12/12) |
| 3 | 42% | **100%** (12/12) |
| 4 | 25% | **100%** (12/12) |
| 5 | 25% | **100%** (12/12) |
| 6 | 42% | **100%** (12/12) |
| 7 | 17% | **100%** (12/12) |
| 8 | 8% | **100%** (12/12) |

On the full clean-chain test set (n=635), the solver covers 99.5% of queries
at 99.2% accuracy, **flat across every hop 2–10**, while DeepSeek falls from
83% to 8% with depth — matching the broader CLUTRR literature's finding that
multi-hop composition is a genuine LLM weakness. The rule table is learned
from TRAIN only, so it is technically external knowledge (this is the guard,
Theorem G, not the self-grounded variant of §6). Locked offline
(`tests/test_clutrr_solver.py`).

The residual ~0.8% is not a reachability failure — coverage (whether the
learned table produces an answer at all) is separate from, and higher than,
accuracy (whether that answer matches the test set's label), and the theorem
below only guarantees the former. A fresh reproduction this session (n=10/hop,
a different sample than the n=12/hop table above) surfaced a concrete instance:
full 10/10 coverage at hop 3 but only 9/10 label matches. CLUTRR's relation
labels are English kinship terms with real semantic overlap (e.g. in-law vs.
blood relations can share a label in some story templates); a table learned
from TRAIN can compose to a technically-reachable but differently-labeled
answer. This is a property of learning from natural-language-labeled data, not
a counterexample to Theorem J, which is stated over coverage.

---

## 5. Positioning, novelty, and limitations (honestly)

**5.1 This is not an unprecedented breakthrough.** Every component is
classical: verifying model output against symbolic knowledge is established
neuro-symbolic practice; the Katz index, the Neumann series, graph
reachability, and operator spectra are decades-old mathematics; LLM multi-hop
hallucination is a widely documented weakness. No claim is made to have
invented any of these tools.

**5.2 The real contribution** is narrow but numerically solid:
1. The **F=G=H unification** — proving, to zero error, that fuzzy diffusion
   inference, relation operator algebra, and the Katz spectral resolvent are
   the same operator.
2. A **measured guarantee and cost** on a real LLM: precision = 1.0 at +0
   tokens, versus +110% tokens for 34% precision with self-verification.
3. A **falsifiable boundary**: solid on short explicit chains, collapsing on
   transitive closure over dense abstract graphs.

**5.3 The core limitation** — bounded flexibility. The guard needs a grounded
relation graph supplied in advance; it **verifies**, it does not extract
relations from free text itself. If the graph must be extracted by an LLM,
that extraction step can itself be wrong — the guard only guarantees
correctness *within the graph it was given* (§6 and Theorem K address
extraction noise directly). This is not a self-contained, open-ended reasoner
replacing the LLM; it is a grounded verification layer, powerful and cheap
when a graph exists, and inert without one.

**5.3.1 Two boundaries the algebra cannot see, raised in external review and
reproduced.** Both are properties of the graph-construction layer sitting
*above* the algebra, not counterexamples to Theorem G, but both silently
change what question is actually being answered if left unguarded:

- *Entity identity.* `follow`/`verify` key entities by exact string equality
  (`OperatorRelationAlgebra._id`). If an upstream LLM extraction is
  inconsistent about one entity's surface form (`"Bob"` vs `"bob"`), the two
  spellings become two distinct graph nodes, and a real path silently breaks
  — the guard then correctly, per Theorem G, rejects a claim that is true in
  the world, because identity resolution failed one layer up, not because the
  matrix product is wrong. Reproduced in
  `tests/test_agent.py::TestEntityNormalization::test_without_normalize_a_case_mismatch_breaks_a_true_path`.
  Fix shipped as an opt-in constructor hook, `GroundedReasoner(normalize=...)`,
  that canonicalizes entity strings before they become graph keys while still
  reporting each entity's original first-seen spelling in `proof`.
- *Transitivity is a modeling assumption, not a derivable property.* Theorem
  G's guarantee is "a path exists under the closure of `via`" — it is silent
  on whether `via` composes transitively in reality. Composing a relation
  that is only partially or conditionally transitive in the world (e.g.
  "trusts": *A trusts B* and *B trusts C* does not imply *A trusts C*)
  still returns a confident, mathematically correct `grounded=True` that
  answers a different question than the one intended. No graph-theoretic test
  over the *supplied* facts can distinguish "genuinely transitive relation,
  sparse sample" from "non-transitive relation" without ground truth outside
  the data. Reproduced in
  `tests/test_agent.py::TestTransitiveRelationsGuard::test_default_is_fully_permissive_backward_compatible`.
  Fix shipped as an opt-in constructor allowlist,
  `GroundedReasoner(transitive_relations={...})`, that raises `ValueError` for
  any undeclared `via`, converting a silent assumption into an explicit,
  checked one at the call site.

Both fixes are opt-in (default `None`, identical behavior to prior releases)
because neither failure mode is fixable *in general* — case sensitivity and
non-transitive relations both have legitimate, intended uses; the library
cannot make that domain judgment call for the caller, only offer the guard.

**5.3.2 A measured alternative to the binary transitivity guard — Theorem M.**
`transitive_relations={...}` (§5.3.1) closes the transitivity gap with a
binary declare-or-reject allowlist. That is sound but coarse: it cannot say
*how much* to trust a relation that's mostly-but-not-perfectly transitive, and
gives the caller only two options — silently trust every grounded claim, or
discard all of them.

> **Theorem M** (Empirical Transitivity Calibration, numerically verified,
> `tests/test_theorems.py::test_transitivity_calibration_coverage`,
> `grounded_reasoning/reasoning/transitivity_calibration.py`):
> Let `p` be the true (unknown) probability that a pair the graph marks
> `grounded=True` via relation `rel` is actually true, estimated from `k`
> confirmed-true pairs out of `n` i.i.d./exchangeable held-out calibration
> pairs (ground truth known independently of the graph). The Clopper-Pearson
> lower bound `p_L(k, n, α)` — the value solving `P(Binomial(n, p_L) ≥ k) = α`,
> computed here by bisection on the binomial survival function rather than
> the closed-form beta quantile (cross-checked to `scipy.stats.beta.ppf` to
> machine precision in development; no scipy dependency shipped) — satisfies
> `P(p_L ≤ p) ≥ 1−α` for every true `p ∈ [0,1]`.

Proof: `p_L` is defined exactly by the point where the observed evidence `k`
becomes the α-quantile outcome under the null `p = p_L`; this is the classical
Clopper & Pearson (1934) exact interval, whose coverage argument requires only
that the `n` calibration draws are i.i.d./exchangeable Bernoulli(`p`) trials —
no assumption on `p` itself (the same "distribution-free" flavor as
conformal prediction, §7.1, though this is a distinct classical tool applied
to a distinct question here). Not new math; the contribution is pairing it
with this system's transitivity gap, exactly as §7.1 pairs conformal
prediction with the operator confidence score.

Verified: for 3,000 random true precisions `p ~ U(0.05, 0.99)` and random
calibration draws (`n=40`, `α=0.1`), empirical coverage was 93.6% (≥ 90%
target); degenerate evidence behaves sanely (`k=0` → bound `0.0`, never
overconfident from nothing; `k=n` → bound `<1.0`, never claims certainty).

**A/B comparison** (`grounded_reasoning/experiments/transitivity_calibration_eval.py`,
fully offline, synthetic ground truth so the true precision is known for
scoring): a "trusts"-like relation with true composed-claim precision
`q=0.85` (mostly, not perfectly, transitive).

| | Claims kept | Precision | Risk reported |
|---|---:|---:|---|
| A. binary, declared transitive | all | 83% (actual) | **none** |
| A. binary, undeclared | 0 | — | loses every one of the 85% that were true |
| **B. calibrated** (`calibrate_transitivity`) | — | ≥ **80%**, 90% confidence | **yes, and it held** on a fresh held-out test set |

The calibrated bound gives an honest, checkable number in exactly the
situation where the binary mechanism can only guess blindly or reject
everything — the same qualitative move as trading the hard guard (§5.3) for
conformal coverage (§7.1) under a noisy graph, applied here to a noisy
*assumption* instead of a noisy *graph*.

**5.3.3 Closing the other boundary — Theorem N.** §5.3.1's first boundary
(entity identity) was fixed with an opt-in `normalize=` hook, but — unlike
transitivity — the precise conditions under which that hook preserves
Theorem G's guarantee had not been characterized. It can be, exactly:

> **Theorem N** (Normalization Precision Isolation, numerically verified over
> 60 random trials, `tests/test_theorems.py::test_normalization_precision_isolation`,
> `grounded_reasoning/theory/theorems.py`): let entities `E` each have one or
> more raw surface-form aliases (`alias: Alias → E`, not necessarily
> injective), and let a normalizer `σ` compose with `alias` to give
> `τ = σ ∘ alias : E → Keys`.
> 1. **Safety preserves precision exactly.** If `τ` is injective (`σ` never
>    maps two *distinct* true entities to the same key — it only ever merges
>    aliases of the *same* entity), precision remains exactly 1.0, identical
>    to no normalization at all.
> 2. **Safety implies recall is monotonically non-decreasing** relative to no
>    normalization: every true path found without normalization is still
>    found with it, and previously-fragmented aliases of the same entity may
>    now connect facts that raw string matching couldn't.
> 3. **Over-merging is the *only* way precision can break**, and when it does,
>    the false-positive's witnessing proof path necessarily passes through
>    the over-merged key — the failure is localized, not scattered.

Proof: (1)+(2) an injective `τ` makes the keyed graph isomorphic, as a
reachability structure, to the graph on the true entities `E` with
inconsistent-alias edges unified onto their single true endpoint — unifying
aliases of the *same* node can only merge previously-split adjacency lists
(weakly increasing reachability), and a graph isomorphism cannot manufacture
a path between two nodes that had none. (3) if `τ` merges `e1 ≠ e2` into key
`k`, anything reachable from `e1`'s aliases becomes reachable from `e2`'s
aliases through `k` and vice versa; a false positive requires exactly this,
so the returned proof necessarily contains `k`. QED (graph reachability under
a quotient map — classical; the contribution is diagnosing precisely where
`normalize=` can and cannot break Theorem G, mirroring §5.3.2's treatment of
the *other* boundary).

**The measured counterpart** — `gr.calibrate_normalization(labeled_pairs)` —
reuses Theorem M's identical Clopper-Pearson machinery, applied to a
different empirical question: not "is this composed claim true" but "does
this normalizer's merge decision agree with independently-known ground
truth." From held-out `(a, b, is_same_entity)` triples, it tallies the pairs
`normalize` actually merges and computes an exact lower confidence bound on
how many of those merges are correct — the *only* quantity Theorem N says can
possibly be at risk.

**A/B/C comparison** (`grounded_reasoning/experiments/normalization_calibration_eval.py`,
fully offline, synthetic ground truth): 120 entities with realistic
inconsistent aliasing, plus a small number of deliberately injected
accidental collisions (a realistic fuzzy-resolver failure mode) —

| | False positives | Risk reported |
|---|---:|---|
| A. No normalization | 0 (always) | — (but misses fragmented paths) |
| B. Fuzzy resolver, trusted blindly | several | **none** |
| **C. Same resolver, calibrated** | — | **yes** — merge precision ≥ 70–85% (90% confidence), held on a fresh held-out set across every seed tested |

Same resolver as B; C is the only one that tells you how much to trust it.

**5.3.4 Remark: both calibrations generalize to heterogeneous relation paths
for free.** Nothing in Theorem M's Clopper-Pearson argument (§5.3.2) is
specific to a *single* relation's closure — it only needs i.i.d./exchangeable
labeled pairs for some fixed graph-derived boolean predicate. `verify_path`
exposes exactly this at the facade: a claim composed through an exact,
possibly heterogeneous, sequence of *different* relations (e.g.
`["parent","employer"]` for a derived "financially dependent on" claim) —
not new math, since `OperatorRelationAlgebra.follow` already composes such
chains exactly (Theorem G's own numerical verification exercises mixed
chains like `[r1, r2, r1]`); `verify_path` adds proof-path reconstruction,
which `follow` (a pure reachability check) does not provide. `calibrate_path`
then calibrates that fixed heterogeneous pattern with the identical
Clopper-Pearson machinery as `calibrate_transitivity`.

This is deliberately **not** given a new theorem letter: doing so would
imply new mathematical content where there is none — the correctness of
`verify_path` is Theorem G restated for a list of relations instead of one
repeated relation, and the calibration is Theorem M's own already-general
argument applied to a different predicate. The honest framing is that F–N
already cover this; what's new in this remark is the *exposed capability*
(`verify_path`/`calibrate_path` at the facade) and one practical caveat the
implementation does not hide: **calibration is per exact path pattern** —
evidence for `["parent","employer"]` says nothing about `["parent","parent"]`
or `["employer","parent"]`; each distinct sequence relied upon needs its own
held-out calibration set.

Verified: `verify_path` checked against independent ground-truth BFS across
8,000 (subject, relation-chain, object) triples with zero mismatches, and
every returned proof path independently confirmed to consist of real edges
in the declared order (`tests/test_agent.py::TestHeterogeneousPathVerification`).
`calibrate_path`'s coverage checked the same way as Theorem M
(`tests/test_heterogeneous_path_calibration_eval.py`), compared against the
*true* synthetic precision (88% empirical coverage over 200 trials against a
90% target) — a separate, deliberately noisier comparison against a finite
held-out test sample (not the true parameter, since a real deployment
wouldn't know it) is reported alongside for realism but is not itself a test
of the theorem, since it compounds two independent sources of sampling
error.

**5.4 Directions to falsify or extend.** (i) Extract a grounded graph from a
trustworthy non-LLM source and measure the guard end-to-end. (ii) Test on
relations with cycles (ρ ≥ 1), where closure requires `α < 1/ρ`. (iii) Search
for a counter-example where the guard wrongly drops a correct answer
(theoretical prediction: impossible, since `explain ⟺ reachable` — falsifiable
if found).

---

## 6. Self-Grounded Deductive Consistency (SGDC)

This section removes the "needs an external graph" limitation of §5.3.

**The idea.** LLMs are reliably accurate on atomic (1-hop) facts but
hallucinate on composition (§4.2). SGDC exploits this asymmetry: take the
LLM's own confident 1-hop facts, build the operator, compute a certified
closure, then reject any of the LLM's multi-hop conclusions that fall outside
its *own* closure — self-contradiction as the hallucination signal. No
external knowledge and no extra tokens (`grounded_reasoning/experiments/self_grounded_eval.py`).

Results on DeepSeek (a real, clean taxonomy):

| | precision | recall |
|---|---:|---:|
| Atomic facts (LLM) | **100%** | — |
| Raw multi-hop (LLM) | 78% | 87% |
| **SGDC (self-grounded, zero external knowledge)** | **100%** | 72% |
| Ceiling: filtering with an external graph | 100% | 87% |

SGDC lifts precision from 78% to **100% using zero external knowledge** — only
the LLM's own internal consistency. The honest cost is recall (72% vs. 87%):
self-closure is somewhat conservative.

**A spectral contradiction certificate.** If a relation should be a partial
order (is-a, causes) and the LLM's asserted graph has ρ(A) > 0, that is a
certificate that the LLM asserted a contradictory cycle, localizable via
`cycle_members` — at zero tokens (`tests/test_self_grounded.py`).

**Positioning versus existing techniques.** *Self-consistency* (sampling
multiple chains-of-thought and voting) is statistical, token-costly, and
unguaranteed. *Chain-of-Verification* still hallucinates when the LLM checks
itself in words (§4.3 measures 34%). *External KG-grounding* needs a
knowledge base; SGDC needs none.

**The survival condition (falsifiable, and its failure is recorded).** SGDC
works only when atomic-fact precision exceeds multi-hop precision. In a
counter-prior world (e.g. "a whale is a fish"), the LLM's atomic-fact
precision itself falls to 70%, the condition fails, and SGDC's recall drops
badly (58%). SGDC is not a panacea — it requires a domain where the LLM's
atomic knowledge is trustworthy.

### Theorem I — a two-sided guarantee (precision and recall)

Let `S := cl_T(x)` be the true answer set for source `x`; `ρ_M` the raw
multi-hop recall; `c = |cl_A ∩ S| / |S|` the closure recall of the LLM's
atomic facts; `ρ_S` the SGDC recall.

> **Theorem I** (numerically verified, 300 trials/seed,
> `tests/test_theorems.py::test_sgdc_recall_bound_two_sided`):
> 1. **Precision = 1.0** if the atomic facts are sound (`A_llm ⊆ T`),
>    regardless of multi-hop hallucination. (`cl_A ⊆ cl_T ⟹ kept ⊆ S ⟹` zero
>    false positives.)
> 2. **Recall (Fréchet bound)**: `ρ_S ≥ max(0, ρ_M + c − 1)`, with no
>    assumptions needed (`|U∩V| ≥ |U|+|V|−|S|` with `U=M∩S`, `V=cl_A∩S`). The
>    bound is tight (worst-case slack ≈ 0).
> 3. **Recall (tight case)**: if the atomic facts cover enough (`cl_A ⊇ S`),
>    then `ρ_S = ρ_M` — SGDC retains every raw true positive.
> 4. **Corollary (domination)**: if the LLM's atomic facts are sound *and*
>    complete, then precision = 1 and recall = `ρ_M` — SGDC strictly
>    dominates the raw output.

SGDC's quality is bounded by a quantity that can be *measured* before trusting
it: precision is locked at 1 by atomic soundness, and recall is lost by
exactly the atomic incompleteness (1−c), no more.

**Remark: the survival condition generalizes to a measured bound for free,
same as §5.3.4.** Theorem I's precision=1.0 guarantee is *conditional* on
atomic soundness (`A_llm ⊆ T`) — a binary assumption the "counter-prior
world" paragraph above shows can silently fail. `calibrate_transitivity`
(Theorem M) does not care whether a `GroundedReasoner`'s facts came from an
external KB or from an LLM's own atomic self-assertions — its argument only
needs i.i.d./exchangeable held-out labeled pairs for *some* fixed
graph-derived boolean predicate, exactly as already noted in §5.3.4. Calling
it on a reasoner built purely from an LLM's own atomic facts — held-out
labeled pairs for SGDC's own `grounded=True` claims — measures SGDC's actual
precision directly, with **zero new code, zero new theorem**: this already
works today.

This is worth stating explicitly because the natural-seeming shortcut —
estimate SGDC's output precision from the atomic layer's own precision,
without recalibrating the output — is *wrong*, not merely imprecise: in a
synthetic domain with a random `is_a` hierarchy (40 concepts) and 15% of the
atomic facts replaced with an incorrect parent (simulating a counter-prior
domain), SGDC's own multi-hop output precision fell to ~74–75%, not the
naively-expected ~85% — a single wrong atomic edge can be *composed into*
several downstream multi-hop claims, amplifying its damage. Calibrating the
final SGDC output directly (rather than propagating a bound through the
atomic layer, which would require new and considerably harder math) sidesteps
this amplification entirely and is checked, numerically, against a
synthetic ground truth where the true precision is known
(`grounded_reasoning/experiments/self_grounded_calibration_eval.py`,
`tests/test_self_grounded_calibration_eval.py`): the calibrated bound stayed
≤ the true precision in 98.3% (118/120) of trials spanning three atomic-error
rates (5%, 15%, 30%, 40 seeds each) — consistent with the α=0.1 target, same
aggregate-rate check used throughout §5.3.2's own verification.

---

## 7. Further directions (verdict: keep / discard)

*Four further directions were tested, each with a theorem and numerical
verification, with an honest keep/discard verdict.*

### 7.1 Conformal Reasoning — Theorem K — **keep** (the largest step)

Attacks the core limitation of §5.3 directly. Uses operator confidence
`conf(a→b)` as a conformal score (Vovk et al.), calibrating a threshold `τ` on
a calibration set:

| noise (p_drop) | coverage (target ≥ 0.90) | FPR (efficiency) |
|------------------:|---------------------------:|---------------------:|
| 0.0 | 0.896 | 0.52 |
| 0.3 | 0.902 | 1.00 |

Coverage stays ≥ 1−α at every noise level (validity, distribution-free) — a
guarantee the hard guard cannot give on a dirty graph. The price is that
efficiency (false-positive rate) degrades with noise. This turns "guaranteed
only when clean" into "coverage always valid, efficiency scales with graph
quality." Conformal prediction is not a new invention here — the contribution
is pairing it with an operator score and characterizing the tradeoff.

**End-to-end demo on a real LLM** (`grounded_reasoning/experiments/conformal_llm_eval.py`).
DeepSeek extracts an "is-a" graph from natural text (a genuinely noisy
source); conformal calibration runs on that LLM-extracted graph; ground truth
is used only for scoring. Easy vs. hard text (nested clauses and near-miss
distractor sentences):

| Text | LLM extraction (P / R) | Coverage (target ≥90%) | FPR (efficiency) |
|------|---------------------------:|----------------------------:|---------------------:|
| Easy | 100% / 99.7% | **91.3%** | 0.0 |
| Hard | 99.5% / **68.5%** | **93.0%** | 0.77 |

When extraction drops 31% of the edges — a genuinely dirty, LLM-produced
graph — the coverage guarantee ≥ 1−α still holds (93%); only efficiency
degrades. This is a path to guaranteed reasoning over natural-language
relations, exactly where the hard guard is helpless. Offline-locked
(`tests/test_conformal_llm.py`).

**Remark: efficiency can be improved further under dropout-dominant noise, at
no cost to validity — one attempt worked, a different one was falsified
first.** Mondrian (group-conditional) conformal prediction (Vovk et al. —
classical, not new) calibrates a SEPARATE threshold per group of an
available-at-test-time partitioning function instead of one global
threshold: the identical split-conformal exchangeability argument, applied
*within* each group, gives that group its own ≥ 1−α coverage, so the marginal
guarantee is unaffected — grouping can only change *efficiency*, never
*validity*. `ConformalReasoner.calibrate(..., group_fn=...)` exposes this
directly (`grounded_reasoning/reasoning/conformal_reasoning.py`).

The first partitioning function tried — hop-distance — was **numerically
falsified before being written up or shipped**: at every noise level tested,
it made the false-positive rate *worse*, not better (splitting the
calibration set costs more in per-group calibration looseness than
hop-distance-correlated noise recovers). This is recorded honestly, not
hidden, per this project's own stated principle of logging negative results.

The one that survived falsification: `redundancy_group` groups a pair by
whether it has more than one walk in the (possibly noisy) extracted graph
(`FuzzyInferenceEngine.path_multiplicity`, computable with no ground truth).
Motivation: `conf(a→b) = Σ_k α^k(P^k)[a,b]` *sums* over every walk, so a
multiply-connected pair gets a mechanically higher, more separable score —
and, more directly relevant here, survives a single random edge being
*dropped* far more often than a singly-connected pair does (Menger's
theorem: deleting one edge cannot disconnect a 2-edge-connected pair).
Verified (`grounded_reasoning/experiments/redundancy_conformal_eval.py`,
`tests/test_redundancy_conformal_eval.py`, 60 seeds/scenario):

| Noise regime | Global FPR | Grouped FPR | Coverage (both) |
|---|---:|---:|---|
| Dropout-dominant (p_drop=0.2, p_add=0.3) | 98.7% | **80.8%** | 89.9% / 91.1% |
| Spurious-dominant (p_drop=0.0, p_add=0.3) | 56.7% | 57.3% | 90.7% / 91.2% |

Coverage holds ≥ ~90% in both regimes (the classical guarantee, expected, not
the finding). Efficiency improves substantially — an ~18-point FPR reduction
— specifically when dropped edges dominate the noise, which is exactly the
noise mode `conformal_llm_eval.py` documents for real LLM extraction (missed
relations, not fabricated ones). It gives essentially no benefit when
spurious *added* edges dominate instead (FPR *marginally worse*, 57.3% vs.
56.7%) — an honest limitation stated plainly rather than glossed over: this
is a targeted efficiency improvement for a specific, common noise mode, not
a universal one.

**Remark: a different, orthogonal weakness — a DRIFTING noise level, not a
HETEROGENEOUS one — needs a different classical tool.** Both the base
conformal reasoner and its Mondrian extension above assume calibration and
test data are exchangeable (drawn from the same distribution). That
assumption is violated, not just stressed, if the noise level *changes over
time* — e.g. an LLM-extraction pipeline processing many document batches
where later batches are cleaner or noisier than the batch used to calibrate.
A frozen threshold then carries no guarantee once conditions shift, and
nothing in either conformal variant above detects or corrects this.

Adaptive Conformal Inference (ACI; Gibbs & Candès, 2021) — again classical,
not new — replaces the frozen threshold with one that updates from a stream
of confirmed-true examples: after each one, adjust an internal quantile
level `α_t ← α_t + γ(α − err_t)` (`err_t = 0` if the example cleared the
current threshold, else `1`), then recompute the threshold from a bounded
window of recent scores at level `α_t`. ACI's own guarantee needs no
stationarity or exchangeability assumption at all: the long-run average
miscoverage rate converges to `α` for *any* score sequence, including
adversarial drift — a strictly different (and, for this failure mode,
strictly stronger) property than split-conformal's.

Verified (`grounded_reasoning/experiments/drift_conformal_eval.py`,
`tests/test_drift_conformal_eval.py`, 15 trials): a stream of query batches
over random relation graphs, extracted cleanly (`p_drop=0.05`) for the first
half of the stream, then noisily (`p_drop=0.45`) for the second half —

| | Pre-shift coverage | Post-shift coverage (target ≥90%) |
|---|---:|---:|
| Frozen threshold (calibrated once, start of stream) | 88.6% | **47.6%** |
| Adaptive (ACI) | 89.9% | **89.6%** |

The frozen threshold collapses to less than half its target coverage the
moment the noise level shifts — silently, with nothing in the frozen-
threshold API signaling that its guarantee no longer holds. ACI recovers to
within its own target band and stays there, in every one of the 15 trials
tested. `AdaptiveConformalReasoner` (`grounded_reasoning/reasoning/conformal_reasoning.py`)
exposes this as a drop-in alternative to `ConformalReasoner` for exactly this
scenario — a different failure mode than Mondrian grouping addresses
(heterogeneity within one static population), solved by a different
classical tool, paired with this system's operator confidence the same way
split-conformal itself was.

**Remark: identifying and removing the specific spurious edges responsible
for false positives beats calibrating a threshold around them — but only
by REMOVING them, not by grouping around them.** Every mechanism above
(base conformal, Mondrian grouping, ACI) works *around* a noisy graph — none
of them touch the graph's own structure. A natural question: can held-out
labeled evidence instead identify *which specific edges* are spurious, and
remove them outright?

`identify_suspect_edges` (`grounded_reasoning/reasoning/edge_pruning.py`)
answers this with a simple decision rule — **not** a statistical guarantee
like the Clopper-Pearson bounds elsewhere in this project: from held-out
labeled `(subject, object, is_true)` triples, an edge that appears on the
shortest proof path of at least one FALSE-labeled claim, and on NO
TRUE-labeled claim's path, is removed from the graph entirely.

The *first* way this "suspect edge" signal was tried was as a Mondrian
`group_fn` (reusing the exact machinery above) rather than outright removal
— and it was numerically **falsified**: at every noise level tested, FPR got
*worse*, not better. The reason is structural, not incidental: Mondrian must
still guarantee coverage *within* whatever group a claim falls into, and the
"suspect" group is disproportionately false claims routed through a bad
edge — satisfying coverage for the few true claims that also happen to
cross it forces that group's own threshold down, admitting more of the
group's false claims than the global threshold did. Outright removal has no
such constraint: the few true claims that depended on the edge lose that
specific path (a real, disclosed recall cost), but every false claim that
depended on it loses its only support too.

Verified (`grounded_reasoning/experiments/edge_pruning_eval.py`,
`tests/test_edge_pruning.py`, 60 seeds/scenario across 5 noise
regimes) — the strongest single efficiency result found in this line of
exploration:

| Noise regime | Raw FPR | Cleaned FPR | Coverage (raw / cleaned) |
|---|---:|---:|---|
| Dropout-dominant (p_drop=0.2, p_add=0.3) | 77.0% | **49.2%** | 91.0% / 90.0% |
| Spurious-dominant (p_drop=0.0, p_add=0.3) | 58.7% | **15.7%** | 91.0% / 92.0% |
| Heavy dropout (p_drop=0.3, p_add=0.3) | 81.6% | **58.9%** | 91.5% / 91.8% |
| Light spurious (p_drop=0.2, p_add=0.1) | 85.8% | **51.4%** | 91.8% / 92.7% |
| Heavy spurious (p_drop=0.2, p_add=0.5) | 68.4% | **52.8%** | 90.5% / 91.1% |

Unlike `redundancy_group` (which helps only under dropout-dominant noise and
gives essentially nothing under spurious-dominant noise), removal helps
substantially in *every* regime tested — including the one redundancy
grouping could not touch. The two are complementary, not competing: this
targets specific corrupted edges from labeled evidence; redundancy grouping
targets structural heterogeneity among otherwise-legitimate claims.

**The tradeoffs, stated plainly and MEASURED across every regime — not just
disclosed as a possibility, not just measured in one scenario (this is a
decision rule, not a proof).** Unlike every calibration method elsewhere in
this project, this one carries no probabilistic bound. (1) There is no
false-discovery-rate guarantee, and the wrongly-removed rate is not
negligible. Measured directly (a genuinely correct edge is one that exists
in the true generating graph), pooling all blocked edges across 60 seeds
per regime, with the default rule (`identify_frac=0.5`, `min_evidence=1`):

| Noise regime | Wrongly-blocked rate (pooled) |
|---|---:|
| Dropout-dominant | 19.3% |
| Spurious-dominant | 14.0% |
| Heavy dropout | 21.5% |
| Light spurious | 32.2% |
| Heavy spurious | 13.2% |

**Mitigation, found by a Pareto sweep, measured the same way across every
regime.** Sweeping `identify_frac` in [0.5, 0.9] and `min_evidence` in
[1, 3] shows `identify_frac=0.85, min_evidence=2` dominates the earlier
0.8/2 choice at every regime tested (strictly lower wrongly-blocked rate,
small extra cleaned-FPR cost); `identify_frac=0.9` was tried and
**rejected** — the reserved evaluation split shrinks enough that the
conformal-calibration split inside it becomes unreliable, and cleaned FPR
degrades sharply back toward the raw baseline. Reported as the pooled rate
together with a one-sided 95% Wilson upper confidence bound (a large-*n*
asymptotic approximation; `clopper_pearson_lower` elsewhere in this project
overflows at these pooled sample sizes, in the hundreds to low thousands —
see `wilson_upper_bound` in `edge_pruning_eval.py`):

| Noise regime | Wrongly-blocked rate (pooled) | 95% upper bound |
|---|---:|---:|
| Dropout-dominant | 2.9% | 4.6% |
| Spurious-dominant | 1.5% | 2.6% |
| Heavy dropout | 2.9% | 4.6% |
| Light spurious | 3.1% | 6.6% |
| Heavy spurious | 2.9% | 4.2% |

Light-spurious noise is both the worst case and the noisiest estimate — it
blocks the fewest edges of any regime, so its confidence bound is widest.
Even so, its upper bound stays under 7%, against a raw baseline of 32.2%
(unmitigated) — a real, quantified, bounded residual risk, not eliminated,
but precisely characterized rather than left as "could in principle
happen." The cost: cleaned FPR rises somewhat (e.g. ~49% to ~59% in the
dropout-dominant regime, still far below the 77% raw baseline), and the
reserved evaluation set shrinks further than at 0.8. See
`run_mitigation_comparison` in `edge_pruning_eval.py` and
`test_larger_identify_split_and_min_evidence_cut_wrongly_blocked_rate` /
`test_wilson_upper_bound_basic_properties`.

**Two further directions were tried and did NOT beat this** — reported
because a decision this consequential should show its negative results,
not just its positive one. (a) *Stability selection* (Meinshausen &
Bühlmann, 2010): bootstrap-resample the identification half repeatedly and
require an edge to be flagged as suspect in most resamples, rather than
once. This gave essentially the same wrongly-blocked rate as
`min_evidence` alone — unsurprising in hindsight: resampling a fixed,
already-scarce identification half cannot manufacture true-claim evidence
an edge never received in the first place; it only helps when the
identification pool itself is large enough to vary meaningfully across
resamples. (b) *A formal per-edge hypothesis test*: for each candidate edge
with `f` false-claim encounters and zero true-claim encounters, treat
`p0^f` (where `p0` is the overall false-label rate in the identification
sample) as a one-sided p-value under the null "this edge behaves like an
average edge," then apply Benjamini-Hochberg to control the expected false
discovery rate across all candidates at a target level `q`. Despite being
the more principled-looking construction, it was numerically **worse**
than the simple rule at matching nominal targets (e.g. targeting `q=0.05`
still gave a pooled wrongly-blocked rate of 7–16% across regimes) — the
null's independence assumption fails here, because a genuinely good edge
can be swept into disproportionately many false-claim encounters simply by
sharing a path with a genuinely bad edge, not because it is bad itself.
Neither direction is shipped; both are recorded so the choice of the
simpler rule is verified, not an oversight.

**Verdict: the residual risk, bounded and quantified this way, is small
enough relative to the FPR reduction retained in every regime to keep the
feature** — with `identify_frac=0.85, min_evidence=2` as the default of a
new convenience entry point, `identify_and_prune_edges` (splits
`labeled_pairs`, identifies, prunes, and returns the untouched reserved
share for independent evaluation, in one call). The lower-level
`identify_suspect_edges` keeps its original `min_evidence=1` default —
unchanged, so existing callers of that primitive see no behavior change —
while the new wrapper makes the measured-safest configuration the path of
least resistance for anyone starting fresh. (2) It costs real recall
regardless of configuration — any true claim
depending solely on a removed edge loses that path (visible above as the
small coverage shifts between raw and cleaned); (3) it edits the graph in
place, a one-way structural change, unlike calibration (which only adjusts
a threshold and leaves the graph untouched) — if the query distribution
later differs from the held-out sample used to prune, a removed edge might
have been needed after all.

**Scope check against a REAL LLM (DeepSeek), not just simulated noise** —
`edge_pruning_llm_eval.py`: on a densely-hallucinated multi-hop-shortcut
scenario (an LLM's own claimed transitive conclusions treated as direct
edges — 65–73% of them hallucinated, real DeepSeek output across 3
independent trials), the blocking decision itself stayed accurate (3–4%
wrongly blocked, matching the synthetic benchmark), but the downstream
effect on cleaned FPR was **inconsistent** — 2 of 3 trials improved, 1
regressed (62.9%→84.3% raw→cleaned) — unlike the consistent 5-of-5-regime
win measured synthetically. Traced to the same row-normalized-diffusion
effect noted for reinforcement-style edge weighting elsewhere in this
project's exploration: a few hub nodes carrying many hallucinated
shortcuts interact with `FuzzyInferenceEngine`'s `P = D^-1 W` diffusion
differently than the synthetic benchmark's sparse, locally-random noise —
removing some of a node's edges can concentrate transition probability
onto whichever false edges remain. **This mitigation's measured benefit is
therefore scoped to locally-random 1-hop noise at moderate density; a
dense, hub-heavy hallucination pattern needs its own validation before
relying on it** — reported here rather than folded into a single
reassuring average.

### 7.2 Theorem J (Closure-Learning Completeness) — **keep**

Turns the CLUTRR result of §4.4 into a theorem: closure learning is (i)
always **sound** (0 conflicts; an answer implies correctness); (ii)
**complete** when the composition table covers `P_A × A` — 100% coverage at
every chain length; (iii) **refutes** the naive hypothesis "the generating set
is enough" (it generates the whole group, but coverage stays ≈0 if a test atom
lies outside `A`). Verified on the dihedral group D₆, showing a phase
transition over training chain length.

### 7.3 Horn forward-chaining — Theorem L — **keep, low novelty**

Generalizes the transitive-closure guard to full Horn logic (conjunctive rule
bodies): the least model is sound and supported (grounded); relational
transitivity is a single Horn rule. Opens a path toward ProofWriter/
EntailmentBank-style tasks. This is classical Datalog — useful as a
general-purpose verification layer, not claimed as a new contribution.

### 7.4 A spectral atomic-soundness certificate — already included

Detecting and localizing contradictions in facts an LLM asserts about itself
(a cycle in a relation that should be acyclic, ρ > 0) is already covered by
Theorem H and exercised in `tests/test_self_grounded.py` — kept as a free
consequence of §3.3, not counted as a separate direction.

**Verdict summary:** all four directions are kept (K and J are genuine
contributions; L and the spectral certificate are useful generalizations).
The most promising direction is **Conformal Reasoning (K)**: a viable path to
running the system with a *guarantee* over noisy, natural-language relations —
exactly where the hard guard is helpless.

---

## 8. Conclusion

Within the scope "the relation graph is supplied, or can be extracted well
enough for a coverage guarantee to hold," this project builds a grounded
relational-reasoning layer with three unified theorems (F=G=H) and a
mathematically guaranteed hallucination guard (precision = 1.0) at zero token
cost, verified on a real commercial LLM and on the public CLUTRR benchmark.
This is not an unprecedented breakthrough but a rigorous, measured, and
honestly limited neuro-symbolic contribution: it verifies LLM reasoning rather
than replacing it, within the scope of a supplied or conformally-calibrated
graph. The value is concrete: it exactly compensates for the LLM's
hallucination weakness at no extra token cost — a complement, not an
omnipotent oracle.

---

## Reproducing this work

- Engine: `grounded_reasoning/reasoning/{abstract_inference,operator_algebra,relation_spectrum}.py`
- LLM client (key read from an environment variable): `grounded_reasoning/reasoning/llm_client.py`
- Real-LLM experiments: `grounded_reasoning/experiments/{guard_llm_eval,guard_llm_stress_eval,nl_ontology_eval,guard_cost_eval,clutrr_eval,conformal_llm_eval,self_grounded_eval}.py`
- Offline-only experiments (synthetic ground truth, no LLM call): `grounded_reasoning/experiments/{inference_eval,transitivity_calibration_eval,normalization_calibration_eval,heterogeneous_path_calibration_eval,self_grounded_calibration_eval,redundancy_conformal_eval,drift_conformal_eval,edge_pruning_eval}.py`
- Nine theorems (F–N): `grounded_reasoning/theory/theorems.py`, exercised by `tests/test_theorems.py`
- Full test suite: `pytest tests/` (no API key required — every LLM-dependent
  invariant has a deterministic offline lock). Real-LLM experiments need
  `DEEPSEEK_API_KEY` (or another supported provider) to run live.

## References

Katz (1953); Kondor & Lafferty (2002, diffusion kernels); Vovk, Gammerman &
Shafer (conformal prediction); Sinha et al. (2019, CLUTRR, EMNLP); Wang et al.
(self-consistency); Garcez et al. (neuro-symbolic grounding); classical Horn
logic / Datalog forward-chaining; Neumann series / resolvent (classical
operator analysis); Clopper & Pearson (1934, exact confidence intervals for a
binomial proportion).
