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
"""
from __future__ import annotations

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
    """

    def __init__(self, walk_len: int = 8, alpha: float = 0.6) -> None:
        self._alg = OperatorRelationAlgebra()
        self._eng = FuzzyInferenceEngine(walk_len=walk_len, alpha=alpha)
        self._typed: dict[str, dict[str, set[str]]] = {}  # rel -> a -> {b}
        self._relations: set[str] = set()
        self._infer_cache: dict[str, dict[str, float]] = {}  # source -> {target: confidence}

    # -- building the graph ---------------------------------------------------
    def add_fact(self, subject: str, relation: str, obj: str) -> None:
        """Add a one-hop fact (subject --relation--> obj)."""
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

        Accepted iff a real path exists, hence NEVER accepts a hallucination (Theorem G).
        """
        if via is None:
            path = self._eng.explain(subject, obj)
            conf = self._confidence(subject, obj)
            return Verdict(path is not None, path, conf, None)
        # BFS on the per-relation graph: O(V+E), avoiding a dense matrix (scales to large graphs).
        path = self._path_via(subject, obj, via)
        reachable = path is not None
        conf = self._confidence(subject, obj) if reachable else 0.0
        return Verdict(reachable, path, conf, via)

    def filter_claims(self, claims) -> list[tuple[tuple, Verdict]]:
        """
        Filter a BATCH of LLM claims (subject, obj[, via]) — keep the grounded
        ones, block hallucinations. Returns [(claim, Verdict)]. 0 tokens,
        precision guaranteed.
        """
        out = []
        for c in claims:
            subj, obj = c[0], c[1]
            via = c[2] if len(c) > 2 else None
            out.append((c, self.verify(subj, obj, via=via)))
        return out

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
                        return [list(reversed(cyc))]
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
