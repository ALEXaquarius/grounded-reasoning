"""
Abstract Fuzzy Inference — an ABSTRACT INFERENCE algorithm over a relation graph.

Reframing (retrieval -> REASONING): language/concepts are ABSTRACT — exact
matching is not required, but INDIRECT relations must be inferable. This is a
capability that:
  - Embeddings do NOT have (they measure ONE-hop similarity only, no chaining).
  - LLMs have, but with HALLUCINATION (inferring relations that don't exist).

This inference engine (based on spectral diffusion — the Multi-Hop Bridging
theorem) has 3 properties confirmed by numerical verification
(theorem_fuzzy_inference):

  1. FUZZY CALIBRATED: confidence decreases MONOTONICALLY with inference depth
     (alpha^k) — inferring farther means being less certain. It doesn't need to be
     "exact," it needs the ORDERING to be correct.
  2. DEEP CHAINING: infers N-hop relations (heart->pulse->...->cardiac) that
     1-hop matching is completely blind to.
  3. GROUNDED (no hallucination): infers a relation ONLY when a real path exists;
     disconnected components -> confidence = 0 (the no-false-bridge theorem).
     Inference is GUARANTEED.

Inference confidence: conf(a→b) = Sum_{k=1}^{K} alpha^k * (P^k)[a,b], P = D^-1 W
(the weighted sum of probability of reaching b from a in <= K steps; longer
chains are discounted by alpha).

Unlike retrieval: this does NOT target nDCG, it targets EXPLAINABLE inference
capability (returns the path) with no fabrication — a completely different axis,
where this mathematics is strong.
"""

from __future__ import annotations


class FuzzyInferenceEngine:
    """
    A directed relation graph + fuzzy transitive inference via diffusion.

    Parameters
    ----------
    walk_len : maximum inference depth K.
    alpha    : per-step discount in (0,1) — longer chains are less certain.
    """

    def __init__(self, walk_len: int = 8, alpha: float = 0.6) -> None:
        self.walk_len = walk_len
        self.alpha = alpha
        self._edges: dict[str, dict[str, float]] = {}

    def add_relation(self, a: str, b: str, weight: float = 1.0) -> None:
        """Add a directed, weighted relation a → b."""
        self._edges.setdefault(a, {})
        self._edges[a][b] = self._edges[a].get(b, 0.0) + weight

    def _transition(self) -> dict[str, list[tuple[str, float]]]:
        adj: dict[str, list[tuple[str, float]]] = {}
        for u, nbrs in self._edges.items():
            deg = sum(nbrs.values()) or 1.0
            adj[u] = [(v, w / deg) for v, w in nbrs.items()]
        return adj

    def infer(self, source: str) -> dict[str, float]:
        """
        Infer from `source`: returns {concept: confidence} for every concept that
        can be inferred (within <= walk_len steps). Confidence decreases with depth.
        """
        adj = self._transition()
        x = {source: 1.0}
        out: dict[str, float] = {}
        coef = 1.0
        for _ in range(self.walk_len):
            nx: dict[str, float] = {}
            for u, xu in x.items():
                for v, p in adj.get(u, ()):
                    nx[v] = nx.get(v, 0.0) + xu * p
            x = nx
            coef *= self.alpha
            for v, xv in x.items():
                if xv > 0:
                    out[v] = out.get(v, 0.0) + coef * xv
        return out

    def confidence(self, a: str, b: str) -> float:
        """Inference confidence a → b (0 if no path exists)."""
        return self.infer(a).get(b, 0.0)

    def explain(self, a: str, b: str) -> list[str] | None:
        """
        EXPLAIN the inference: returns the shortest a → b path (BFS) via >=1 STEP,
        or None if not inferable. This is the GROUNDED property — inference comes
        with evidence, never fabricated. EVEN when a==b: only grounded if a real
        cycle/self-loop actually returns to a (self-identity is NOT assumed via a
        trivial "empty path" of 0 steps) — this exactly matches the Guard
        Soundness theorem (explain(a,b) != None iff b is reachable from a).
        """
        prev: dict[str, str] = {}
        visited = {a}
        frontier = [a]
        while frontier:
            nf = []
            for u in frontier:
                for v in self._edges.get(u, {}):
                    if v == b:
                        chain = [u]
                        while chain[-1] in prev:
                            chain.append(prev[chain[-1]])
                        chain.reverse()      # a → … → u
                        chain.append(v)      # a → … → u → b
                        return chain
                    if v not in visited:
                        visited.add(v)
                        prev[v] = u
                        nf.append(v)
            frontier = nf
        return None


class TypedInferenceEngine:
    """
    COMPOSITIONAL and ANALOGICAL inference over a TYPED relation graph.

    - follow(src, [r1,r2,...]): follows a CHAIN of relations (compose) ->
      grandparent = parent o parent, great-grandparent = parent o parent o parent.
    - analogy(a, b, c): A:B::C:? — infers the relation type from (a,b), then
      applies it to c.

    This is ABSTRACT inference: NEW relations (grandparent) are derived from BASE
    relations (parent) by composition — a compositional capability vector
    similarity lacks.
    """

    def __init__(self) -> None:
        self._e: dict[tuple[str, str], set[str]] = {}  # (a, rel) -> {b,...}

    def add(self, a: str, rel: str, b: str) -> None:
        self._e.setdefault((a, rel), set()).add(b)

    def follow(self, src: str, rels: list[str]) -> set[str]:
        """Compositional inference: follows a chain of relations."""
        cur = {src}
        for r in rels:
            nxt: set[str] = set()
            for x in cur:
                nxt |= self._e.get((x, r), set())
            cur = nxt
        return cur

    def relations_between(self, a: str, b: str) -> list[str]:
        return [r for (x, r), ys in self._e.items() if x == a and b in ys]

    def analogy(self, a: str, b: str, c: str) -> set[str]:
        """A:B::C:? — infers the relation type (a→b), then applies it to c."""
        out: set[str] = set()
        for r in self.relations_between(a, b):
            out |= self._e.get((c, r), set())
        return out


class HallucinationGuard:
    """
    Wraps an LLM to BLOCK HALLUCINATION: the LLM proposes a relation, and the
    guaranteed inference engine verifies it via a PATH. A claim is accepted iff a
    grounded path exists.

    Guard Soundness theorem: on the graph, explain(a,b) != None iff b is reachable
    from a (exact BFS). Hence the guard NEVER accepts a claim with no path
    (precision=1.0 on transitive facts) — the LLM cannot "cheat" via hallucination.
    """

    def __init__(self, engine: FuzzyInferenceEngine) -> None:
        self.engine = engine

    def verify(self, a: str, b: str) -> tuple[bool, list[str] | None]:
        """Returns (accepted?, proof path | None)."""
        path = self.engine.explain(a, b)
        return (path is not None, path)
