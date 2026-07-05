"""
GroundedReasoner — the integration facade for AGENTS/LLMs.

A thin facade wrapping the inference core (operator algebra + diffusion + guard)
into an API an agent can use directly: load facts, VERIFY a multi-hop relational
claim BEFORE asserting it, and get back a proof path + confidence. Catches
relational hallucination at **0 LLM tokens** with a **precision guarantee**
(Theorem G: accepted iff a grounded path exists).

Example:
    gr = GroundedReasoner()
    for s, r, o in facts:            # one-hop facts (agent-supplied or LLM-extracted)
        gr.add_fact(s, r, o)
    v = gr.verify("alice", "carol", via="parent")   # a multi-hop claim
    if not v.grounded:               # hallucination -> blocked
        ...
    print(v.proof)                   # ['alice','bob','carol'] — the proof

Two boundaries of the guarantee that the algebra itself cannot see, both with
an opt-in guard below:

  1. Entities are matched by exact string equality. An LLM extraction that is
     inconsistent about a single entity's surface form ("Bob" vs "bob") is
     treated as TWO nodes, silently breaking a real path — the guard then
     (correctly, per its own contract) rejects a claim that is actually true
     in the world, because the graph-construction layer, not the algebra,
     failed to resolve identity. Pass `normalize=` to fold surface-form
     variants together before they become graph keys.
  2. Theorem G's guarantee is scoped to "a path exists under the transitive
     closure of `via`" — it says nothing about whether `via` is transitive
     in reality. Composing a relation that is only partially or conditionally
     transitive (e.g. "trusts": A trusts B and B trusts C does not imply A
     trusts C) will still return a confident, mathematically correct
     `grounded=True`, silently swapping the claim actually being verified.
     The algebra cannot infer this from data — it is the caller's modeling
     choice. Pass `transitive_relations=` to make that choice an explicit,
     checked contract instead of a silent assumption.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine
from grounded_reasoning.reasoning.operator_algebra import OperatorRelationAlgebra


@dataclass
class Verdict:
    """The result of verifying a relational claim."""

    grounded: bool
    proof: list[str] | None = None      # the proof path (None if not grounded)
    confidence: float = 0.0             # diffused confidence (decreases with depth)
    relation: str | None = None

    def as_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "proof": self.proof,
            "confidence": round(self.confidence, 6),
            "relation": self.relation,
        }


class GroundedReasoner:
    """
    A typed relation graph plus grounded verification operations for an agent.

    Parameters
    ----------
    walk_len : maximum inference depth used for the confidence score.
    alpha    : per-step discount in (0,1).
    normalize : optional entity-canonicalization hook, e.g.
        `lambda s: s.strip().casefold()`. Applied to every subject/object
        before it becomes a graph key, so surface-form variants of the same
        real-world entity ("Bob" / "bob" / " Bob") collapse to one node
        instead of silently breaking a path. Off by default (None = exact
        string equality, the previous behavior) because case CAN be
        meaningful (e.g. distinguishing two differently-cased entities is a
        legitimate use of opaque string identity) — this is a per-domain
        judgment call the library cannot make for you. Proof paths and
        contradiction cycles are still reported using each entity's
        first-seen original spelling, never the normalized form.
    transitive_relations : optional allowlist of relation names that are
        known-transitive in the caller's domain. When set, `verify(...,
        via=rel)` raises ValueError for any `rel` not in the set, instead of
        silently returning a confident but semantically meaningless
        "grounded" verdict for a relation that isn't actually transitive
        (see the module docstring). Off by default (None = every relation
        name is accepted, the previous behavior).
    """

    def __init__(
        self,
        walk_len: int = 8,
        alpha: float = 0.6,
        normalize: Callable[[str], str] | None = None,
        transitive_relations: set[str] | None = None,
    ) -> None:
        self._alg = OperatorRelationAlgebra()
        self._eng = FuzzyInferenceEngine(walk_len=walk_len, alpha=alpha)
        self._typed: dict[str, dict[str, set[str]]] = {}  # rel -> a -> {b}
        self._relations: set[str] = set()
        self._infer_cache: dict[str, dict[str, float]] = {}  # source -> {target: confidence}
        self._normalize = normalize
        self._transitive_relations = transitive_relations
        self._display: dict[str, str] = {}  # normalized key -> first-seen original spelling

    def _norm(self, entity: str) -> str:
        if self._normalize is None:
            return entity
        key = self._normalize(entity)
        self._display.setdefault(key, entity)
        return key

    def _to_display(self, path: list[str] | None) -> list[str] | None:
        if path is None or self._normalize is None:
            return path
        return [self._display.get(node, node) for node in path]

    def _check_transitive(self, relation: str) -> None:
        if self._transitive_relations is not None and relation not in self._transitive_relations:
            raise ValueError(
                f"via={relation!r} was not declared in transitive_relations="
                f"{sorted(self._transitive_relations)!r}. The precision guarantee "
                f"(Theorem G) only holds if a relation genuinely composes "
                f"transitively in the real world (a --rel--> b --rel--> c implies "
                f"a --rel--> c) — the algebra cannot verify this from data, it's a "
                f"modeling assumption. If {relation!r} is transitive in your "
                f"domain, add it to transitive_relations=... at construction."
            )

    # -- building the graph ---------------------------------------------------
    def add_fact(self, subject: str, relation: str, obj: str) -> None:
        """Add a one-hop fact (subject --relation--> obj)."""
        subject, obj = self._norm(subject), self._norm(obj)
        self._alg.add(subject, relation, obj)
        self._eng.add_relation(subject, obj)                 # the any-relation graph
        self._typed.setdefault(relation, {}).setdefault(subject, set()).add(obj)
        self._relations.add(relation)
        self._infer_cache.clear()  # the graph changed; any cached diffusion is stale

    def _confidence(self, subject: str, obj: str) -> float:
        """Cached diffusion confidence subject -> obj (recomputed once per source per graph state)."""
        if subject not in self._infer_cache:
            self._infer_cache[subject] = self._eng.infer(subject)
        return self._infer_cache[subject].get(obj, 0.0)

    def add_facts(self, triples) -> None:
        for i, t in enumerate(triples):
            if len(t) != 3:
                raise ValueError(
                    f"fact #{i} must be (subject, relation, object), got: {t!r}"
                )
            self.add_fact(*t)

    # -- verification -----------------------------------------------------------
    def verify(self, subject: str, obj: str, via: str | None = None) -> Verdict:
        """
        Verify a claim: is subject transitively related to obj?

        via=None  -> via ANY relation path (does a grounded path exist?).
        via=rel   -> via the transitive closure OF relation rel (subject --rel*--> obj?).
                     Raises ValueError if this reasoner has transitive_relations=
                     set and `rel` isn't in it (see __init__).

        Accepted iff a real path exists, hence NEVER accepts a hallucination (Theorem G)
        — for a `via` relation that is genuinely transitive; see the module docstring.
        """
        if via is not None:
            self._check_transitive(via)
        subject, obj = self._norm(subject), self._norm(obj)
        if via is None:
            path = self._eng.explain(subject, obj)
            conf = self._confidence(subject, obj)
            return Verdict(path is not None, self._to_display(path), conf, None)
        # BFS on the per-relation graph: O(V+E), avoiding a dense matrix (scales to large graphs).
        path = self._path_via(subject, obj, via)
        reachable = path is not None
        conf = self._confidence(subject, obj) if reachable else 0.0
        return Verdict(reachable, self._to_display(path), conf, via)

    def verify_path(self, subject: str, obj: str, via: list[str]) -> Verdict:
        """
        Verify a claim via an EXACT SEQUENCE of (possibly different) relation
        types: subject --via[0]--> x1 --via[1]--> x2 --...--> via[-1]--> obj.

        Generalizes `verify(via=rel)` (a repeated closure of ONE relation) to a
        fixed chain of arbitrary, possibly heterogeneous relations — e.g.
        `via=["parent", "employer"]` for "subject's parent's employer". This is
        NOT a new guarantee: `OperatorRelationAlgebra.follow` already composes
        heterogeneous relation chains exactly (Theorem G's own verification
        checks chains like `[r1, r2, r1]`) — this method exposes that existing
        capability at the facade, adding proof-path reconstruction, which
        `follow` (a pure reachability check) does not provide.

        IMPORTANT: even with a single relation, this is NOT the same as
        `verify(via=rel)`. `verify_path(a, b, via=["parent"])` requires EXACTLY
        one `parent` hop; `verify(a, b, via="parent")` accepts ANY number of
        hops (the transitive closure). `verify_path(a, b, via=["parent"]*3)`
        requires EXACTLY three hops, not "three or more".

        Does not consult `transitive_relations=` (that gate's semantics are
        about repeatedly composing ONE relation with itself, which does not
        apply to a fixed heterogeneous sequence used once each).

        Raises ValueError if `via` is empty.
        """
        if not via:
            raise ValueError("verify_path needs a non-empty via=[relation, ...] sequence")
        subject_n, obj_n = self._norm(subject), self._norm(obj)
        frontier: dict[str, list[str]] = {subject_n: [subject_n]}
        for rel in via:
            adj = self._typed.get(rel, {})
            next_frontier: dict[str, list[str]] = {}
            for node, path in frontier.items():
                for nxt in adj.get(node, ()):
                    if nxt not in next_frontier:
                        next_frontier[nxt] = path + [nxt]
            frontier = next_frontier
            if not frontier:
                break
        path = frontier.get(obj_n)
        reachable = path is not None
        conf = self._confidence(subject_n, obj_n) if reachable else 0.0
        return Verdict(reachable, self._to_display(path), conf, "->".join(via))

    def calibrate_path(
        self, via: list[str], labeled_pairs: list[tuple[str, str, bool]], alpha: float = 0.1
    ) -> dict:
        """
        Calibrate confidence for a FIXED heterogeneous path pattern (Theorem M
        generalized — see PAPER.md §5.3.4's remark; not a new theorem, since
        nothing in Clopper-Pearson's argument is specific to a single
        relation's closure). From held-out (subject, object,
        is_actually_true) triples, tallies pairs `verify_path` marks
        `grounded=True` for the exact sequence `via` and computes an exact
        lower confidence bound on how many are actually true.

        Practical caveat this method does not hide: calibration is PER PATH
        PATTERN — evidence for `["parent","employer"]` says nothing about
        `["parent","parent"]`. Each distinct pattern you rely on needs its own
        held-out calibration set.
        """
        from grounded_reasoning.reasoning.transitivity_calibration import calibrate_transitivity

        ground_truth = {(a, b): truth for a, b, truth in labeled_pairs}
        grounded_pairs = [
            (a, b) for a, b, _ in labeled_pairs if self.verify_path(a, b, via).grounded
        ]
        return calibrate_transitivity(grounded_pairs, ground_truth, alpha=alpha)

    def filter_claims(self, claims) -> list[tuple[tuple, Verdict]]:
        """
        Filter a BATCH of LLM claims (subject, obj[, via]) — keep the grounded
        ones, block hallucinations. Returns [(claim, Verdict)]. 0 tokens,
        precision guaranteed.

        `via` may be a single relation name or `None` (dispatched to `verify`,
        a closure), or a list of relation names (dispatched to `verify_path`,
        an exact heterogeneous sequence) — so a batch can freely mix claims of
        both kinds.
        """
        out = []
        for c in claims:
            subj, obj = c[0], c[1]
            via = c[2] if len(c) > 2 else None
            if isinstance(via, list):
                out.append((c, self.verify_path(subj, obj, via)))
            else:
                out.append((c, self.verify(subj, obj, via=via)))
        return out

    def calibrate_transitivity(
        self, relation: str, labeled_pairs: list[tuple[str, str, bool]], alpha: float = 0.1
    ) -> dict:
        """
        A measured alternative to `transitive_relations=` (Theorem M): instead of
        blindly declaring `relation` transitive, calibrate a confidence bound from
        held-out evidence.

        `labeled_pairs`: (subject, object, is_actually_true) triples where
        `is_actually_true` is known INDEPENDENTLY of this graph (e.g. human-
        verified) — NOT derived from `verify()`. Pairs the graph does not mark
        `grounded=True` via `relation` are excluded (the bound is about grounded
        claims specifically). These must be held-out calibration data, not the
        pairs you intend to trust the returned bound for.

        Returns {n_grounded, n_confirmed, precision_lower_bound, alpha}:
        with confidence >= 1-alpha, at least `precision_lower_bound` of this
        graph's future grounded=True claims for `relation` are actually true
        (Clopper-Pearson exact interval — classical statistics, not new;
        `grounded_reasoning/reasoning/transitivity_calibration.py`).

        Deliberately bypasses `transitive_relations=`'s allowlist (unlike
        `verify()`): that gate exists to block a BLIND assumption, but here the
        assumption is exactly what's being measured, not made.
        """
        from grounded_reasoning.reasoning.transitivity_calibration import calibrate_transitivity

        ground_truth = {(a, b): truth for a, b, truth in labeled_pairs}
        grounded_pairs = [
            (a, b) for a, b, _ in labeled_pairs
            if self._path_via(self._norm(a), self._norm(b), relation) is not None
        ]
        return calibrate_transitivity(grounded_pairs, ground_truth, alpha=alpha)

    def calibrate_normalization(
        self, labeled_pairs: list[tuple[str, str, bool]], alpha: float = 0.1
    ) -> dict:
        """
        Calibrate confidence that THIS reasoner's `normalize=` never merges two
        distinct real-world entities (Theorem N). Precision=1.0 (Theorem G) is
        preserved EXACTLY as long as `normalize` only merges true aliases of the
        same entity — the only way it can break is an over-merge, so that is
        exactly what this calibrates, from held-out evidence rather than trust.

        `labeled_pairs`: (a, b, is_same_entity) triples where `is_same_entity`
        is known INDEPENDENTLY of `normalize` (e.g. human-verified) — NOT
        derived from it. Only pairs `normalize` actually MERGES (`_norm(a) ==
        _norm(b)`, and `a != b` as raw strings) are scored; these must be
        held-out data, not the pairs you intend to trust the bound for.

        Returns {n_grounded, n_confirmed, precision_lower_bound, alpha} (same
        shape as `calibrate_transitivity`, reusing the identical Clopper-Pearson
        machinery — n_grounded/n_confirmed here count MERGES, not grounded
        claims): with confidence >= 1-alpha, at least `precision_lower_bound`
        of this normalizer's merges are correct (Theorem N;
        `grounded_reasoning/reasoning/transitivity_calibration.py`).

        Raises ValueError if `normalize=` was not set at construction — there's
        nothing to calibrate for the default (exact-string, always-safe) case.
        """
        if self._normalize is None:
            raise ValueError(
                "calibrate_normalization requires normalize= to be set at construction "
                "(the default, exact-string matching, never over-merges — nothing to calibrate)"
            )
        from grounded_reasoning.reasoning.transitivity_calibration import calibrate_transitivity

        ground_truth = {(a, b): truth for a, b, truth in labeled_pairs}
        merged_pairs = [
            (a, b) for a, b, _ in labeled_pairs if a != b and self._norm(a) == self._norm(b)
        ]
        return calibrate_transitivity(merged_pairs, ground_truth, alpha=alpha)

    # -- soundness / contradictions ----------------------------------------------
    def contradictions(self, relation: str) -> list[list[str]]:
        """
        Detect CONTRADICTIONS: if `relation` should be a partial order (acyclic)
        but has a cycle, returns the concepts on ONE cycle (0 tokens). Empty means
        consistent.

        Uses DFS back-edge detection — O(V+E), avoiding eigenvalues (which would
        cost O(n^3)). (Equivalent to the spectral rho>0 test proved in Theorem H.)
        """
        adj = self._typed.get(relation, {})
        if not adj:
            return []
        color: dict[str, int] = {}   # 0=unvisited, 1=on stack, 2=done
        parent: dict[str, str] = {}
        for root in list(adj.keys()):
            if color.get(root, 0) != 0:
                continue
            stack = [(root, iter(adj.get(root, ())))]
            color[root] = 1
            while stack:
                node, it = stack[-1]
                for nxt in it:
                    c = color.get(nxt, 0)
                    if c == 0:
                        color[nxt] = 1
                        parent[nxt] = node
                        stack.append((nxt, iter(adj.get(nxt, ()))))
                        break
                    if c == 1:  # back-edge node->nxt means a cycle exists
                        cyc = [node]
                        x = node
                        while x != nxt and x in parent:
                            x = parent[x]
                            cyc.append(x)
                        return [self._to_display(list(reversed(cyc)))]
                else:
                    color[node] = 2
                    stack.pop()
        return []

    # -- internal ------------------------------------------------------------
    def _path_via(self, subject: str, obj: str, rel: str) -> list[str] | None:
        """
        Shortest BFS path subject→obj via >=1 rel-step. Checks `v==obj` BEFORE
        gating on `visited`, so that a path back to subject ITSELF can be detected
        (a self-loop or cycle when obj==subject) — subject is still seeded into
        `visited` (with NO entry in `prev`) so it is never expanded again, keeping
        the search finite when obj != subject.

        Reconstructs the path walking BACKWARD from `u` (the source endpoint of
        the edge that just matched) via `prev` — not backward from `v` — because
        when obj==subject, `v` equals subject from the very first element, so it
        cannot be used as the loop's stopping condition.
        """
        adj = self._typed.get(rel, {})
        prev: dict[str, str] = {}
        visited = {subject}
        frontier = [subject]
        while frontier:
            nf = []
            for u in frontier:
                for v in adj.get(u, ()):
                    if v == obj:
                        chain = [u]
                        while chain[-1] in prev:
                            chain.append(prev[chain[-1]])
                        chain.reverse()      # subject → … → u
                        chain.append(v)      # subject → … → u → obj
                        return chain
                    if v not in visited:
                        visited.add(v)
                        prev[v] = u
                        nf.append(v)
            frontier = nf
        return None
