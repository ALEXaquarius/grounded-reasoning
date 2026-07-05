"""
Tests for the AGENT integration layer (GroundedReasoner + verify_relation tool). Offline.
"""
from grounded_reasoning.agent import (
    GroundedReasoner,
    TOOL_SPEC,
    openai_tool_spec,
    run_tool,
    verify_relation,
)


def _kin():
    gr = GroundedReasoner()
    gr.add_facts([("alice", "parent", "bob"), ("bob", "parent", "carol"),
                  ("carol", "parent", "dave")])
    return gr


class TestGroundedReasoner:
    def test_verify_via_grounded_with_proof(self):
        v = _kin().verify("alice", "dave", via="parent")
        assert v.grounded
        assert v.proof == ["alice", "bob", "carol", "dave"]
        assert v.confidence > 0

    def test_verify_blocks_hallucination(self):
        v = _kin().verify("alice", "zed", via="parent")
        assert not v.grounded and v.proof is None and v.confidence == 0.0

    def test_verify_any_relation_path(self):
        gr = GroundedReasoner()
        gr.add_facts([("a", "r1", "b"), ("b", "r2", "c")])
        assert gr.verify("a", "c").grounded          # any-relation path
        assert not gr.verify("c", "a").grounded       # directed

    def test_verify_via_is_relation_specific(self):
        gr = GroundedReasoner()
        gr.add_facts([("a", "parent", "b"), ("b", "owns", "c")])
        # a→c is NOT grounded via 'parent' (step 2 is 'owns')
        assert not gr.verify("a", "c", via="parent").grounded
        assert gr.verify("a", "c").grounded           # but grounded via any-relation path

    def test_filter_claims_batch(self):
        gr = _kin()
        res = gr.filter_claims([("alice", "dave", "parent"), ("alice", "zed", "parent")])
        assert res[0][1].grounded and not res[1][1].grounded

    def test_contradiction_detection(self):
        gr = GroundedReasoner()
        gr.add_facts([("cat", "isa", "mammal"), ("mammal", "isa", "animal"),
                      ("animal", "isa", "cat")])
        cyc = gr.contradictions("isa")
        assert cyc and set(cyc[0]) == {"cat", "mammal", "animal"}
        # consistent graph → no contradiction
        gr2 = GroundedReasoner()
        gr2.add_facts([("cat", "isa", "mammal"), ("mammal", "isa", "animal")])
        assert gr2.contradictions("isa") == []


class TestTool:
    def test_verify_relation_stateless(self):
        r = verify_relation([["a", "p", "b"], ["b", "p", "c"]], "a", "p", "c")
        assert r["grounded"] and r["proof"] == ["a", "b", "c"]
        r2 = verify_relation([["a", "p", "b"]], "a", "p", "z")
        assert not r2["grounded"] and r2["proof"] is None

    def test_run_tool_dispatch(self):
        out = run_tool({"facts": [["x", "p", "y"], ["y", "p", "z"]],
                        "subject": "x", "relation": "p", "object": "z"})
        assert out["grounded"]

    def test_tool_spec_shape(self):
        assert TOOL_SPEC["name"] == "verify_relation"
        props = TOOL_SPEC["input_schema"]["properties"]
        assert set(props) == {"facts", "subject", "relation", "object"}

    def test_openai_tool_spec_format(self):
        spec = openai_tool_spec()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == "verify_relation"
        assert spec["function"]["parameters"] == TOOL_SPEC["input_schema"]

    def test_stateless_tool_strips_incidental_whitespace(self):
        # a realistic LLM-extraction artifact: inconsistent whitespace around
        # the SAME entity across different facts/claim, an easy way to
        # silently break an otherwise-true proof path.
        r = verify_relation(
            [["a ", "p", " b"], [" b", "p", "c "]], "a", "p", " c",
        )
        assert r["grounded"] and r["proof"] == ["a", "b", "c"]


class TestMultilingual:
    def test_unicode_entities_and_relations(self):
        # Vietnamese / Chinese / Arabic — entities are opaque strings ⟹ language-agnostic
        gr = GroundedReasoner()
        gr.add_facts([("Anh", "cha", "Bảo"), ("Bảo", "cha", "Cường")])
        v = gr.verify("Anh", "Cường", via="cha")
        assert v.grounded and v.proof == ["Anh", "Bảo", "Cường"]

        gr2 = GroundedReasoner()
        gr2.add_facts([("父", "是", "祖父"), ("祖父", "是", "曾祖父")])
        assert gr2.verify("父", "曾祖父", via="是").grounded

        assert verify_relation([["أب", "والد", "جد"]], "أب", "والد", "جد")["grounded"]


class TestEntityNormalization:
    """The 'Bob' vs 'bob' failure mode: a real path silently broken by an
    LLM extraction that's inconsistent about one entity's surface form."""

    def test_without_normalize_a_case_mismatch_breaks_a_true_path(self):
        # documents the CURRENT boundary (default behavior), not a bug: exact
        # string equality is the historical/backward-compatible default.
        gr = GroundedReasoner()
        gr.add_facts([("Alice", "parent", "Bob"), ("bob", "parent", "Carol")])
        v = gr.verify("Alice", "Carol", via="parent")
        assert not v.grounded and v.proof is None

    def test_normalize_hook_fixes_the_case_mismatch(self):
        gr = GroundedReasoner(normalize=lambda s: s.strip().casefold())
        gr.add_facts([("Alice", "parent", "Bob"), ("bob", "parent", "Carol")])
        v = gr.verify("Alice", "Carol", via="parent")
        assert v.grounded
        assert v.proof == ["Alice", "Bob", "Carol"]  # original spelling, not "bob"

    def test_normalize_hook_also_fixes_whitespace_variants(self):
        gr = GroundedReasoner(normalize=lambda s: s.strip())
        gr.add_facts([("Alice", "parent", "Bob "), (" Bob", "parent", "Carol")])
        v = gr.verify("Alice", "Carol", via="parent")
        assert v.grounded
        # display uses each entity's literal FIRST-SEEN spelling (untouched,
        # including the trailing space) -- normalization affects graph
        # identity, not cosmetic cleanup of the reported proof
        assert v.proof == ["Alice", "Bob ", "Carol"]

    def test_normalize_preserves_first_seen_spelling_in_contradictions(self):
        gr = GroundedReasoner(normalize=lambda s: s.casefold())
        gr.add_facts([("Cat", "is_a", "Mammal"), ("Mammal", "is_a", "Animal"),
                      ("animal", "is_a", "cat")])  # closes a cycle only once normalized
        cyc = gr.contradictions("is_a")
        assert len(cyc) == 1
        assert set(cyc[0]) == {"Cat", "Mammal", "Animal"}  # original spellings, not lowercased

    def test_no_normalize_is_fully_backward_compatible(self):
        # default (normalize=None) behavior is byte-for-byte the pre-existing one
        gr = GroundedReasoner()
        gr.add_facts([("alice", "parent", "bob"), ("bob", "parent", "carol")])
        v = gr.verify("alice", "carol", via="parent")
        assert v.grounded and v.proof == ["alice", "bob", "carol"]


class TestTransitiveRelationsGuard:
    """Theorem G's guarantee only holds for a `via` relation that is genuinely
    transitive in reality; the algebra can't verify that from data alone."""

    def test_default_is_fully_permissive_backward_compatible(self):
        gr = GroundedReasoner()  # transitive_relations=None (default)
        gr.add_facts([("Alice", "trusts", "Bob"), ("Bob", "trusts", "Mallory")])
        v = gr.verify("Alice", "Mallory", via="trusts")
        assert v.grounded  # unchanged legacy behavior: no relation is rejected

    def test_undeclared_relation_raises(self):
        gr = GroundedReasoner(transitive_relations={"parent", "is_a"})
        gr.add_facts([("Alice", "trusts", "Bob"), ("Bob", "trusts", "Mallory")])
        try:
            gr.verify("Alice", "Mallory", via="trusts")
            assert False, "expected ValueError for an undeclared relation"
        except ValueError as e:
            assert "trusts" in str(e) and "transitive_relations" in str(e)

    def test_declared_relation_works_normally(self):
        gr = GroundedReasoner(transitive_relations={"parent"})
        gr.add_facts([("alice", "parent", "bob"), ("bob", "parent", "carol")])
        v = gr.verify("alice", "carol", via="parent")
        assert v.grounded and v.proof == ["alice", "bob", "carol"]

    def test_any_relation_path_via_none_is_unaffected_by_the_allowlist(self):
        # the allowlist only gates the EXPLICIT via=rel form; via=None ("any
        # relation path exists") has different semantics, documented separately.
        gr = GroundedReasoner(transitive_relations={"parent"})
        gr.add_facts([("alice", "trusts", "bob")])
        v = gr.verify("alice", "bob")  # via=None
        assert v.grounded


class TestTransitivityCalibration:
    """Theorem M: a measured confidence bound instead of a binary declaration."""

    def _trust_graph(self):
        gr = GroundedReasoner()
        # A -> B -> C for several disjoint triples; "trusts" composes for some,
        # not for others (ground truth supplied independently below).
        gr.add_facts([
            ("A1", "trusts", "B1"), ("B1", "trusts", "C1"),
            ("A2", "trusts", "B2"), ("B2", "trusts", "C2"),
            ("A3", "trusts", "B3"), ("B3", "trusts", "C3"),
            ("A4", "trusts", "B4"), ("B4", "trusts", "C4"),
        ])
        return gr

    def test_all_confirmed_gives_a_high_but_not_perfect_bound(self):
        gr = self._trust_graph()
        labeled = [("A1", "C1", True), ("A2", "C2", True),
                   ("A3", "C3", True), ("A4", "C4", True)]
        res = gr.calibrate_transitivity("trusts", labeled, alpha=0.1)
        assert res["n_grounded"] == 4
        assert res["n_confirmed"] == 4
        assert 0.0 < res["precision_lower_bound"] < 1.0  # never claims certainty from n=4

    def test_some_violations_lower_the_bound(self):
        gr = self._trust_graph()
        labeled = [("A1", "C1", True), ("A2", "C2", False),
                   ("A3", "C3", True), ("A4", "C4", False)]
        res = gr.calibrate_transitivity("trusts", labeled, alpha=0.1)
        assert res["n_grounded"] == 4 and res["n_confirmed"] == 2
        assert res["precision_lower_bound"] < 0.5

    def test_ungrounded_pairs_are_excluded_from_the_count(self):
        gr = self._trust_graph()
        # ("A1", "C2", ...) has no path -> excluded, not counted as a violation
        labeled = [("A1", "C1", True), ("A1", "C2", False)]
        res = gr.calibrate_transitivity("trusts", labeled, alpha=0.1)
        assert res["n_grounded"] == 1 and res["n_confirmed"] == 1

    def test_bypasses_the_transitive_relations_allowlist(self):
        # calibrate_transitivity measures the assumption; it must not be
        # blocked by the OTHER (binary) guard meant for blind declarations.
        gr = GroundedReasoner(transitive_relations={"parent"})  # "trusts" NOT declared
        gr.add_facts([("A1", "trusts", "B1"), ("B1", "trusts", "C1")])
        res = gr.calibrate_transitivity("trusts", [("A1", "C1", True)], alpha=0.1)
        assert res["n_grounded"] == 1 and res["n_confirmed"] == 1

    def test_respects_entity_normalization(self):
        gr = GroundedReasoner(normalize=lambda s: s.strip().casefold())
        gr.add_facts([("A1", "trusts", "B1"), ("b1", "trusts", "C1")])
        res = gr.calibrate_transitivity("trusts", [("A1", "C1", True)], alpha=0.1)
        assert res["n_grounded"] == 1
