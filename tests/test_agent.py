"""
Tests for the AGENT integration layer (GroundedReasoner + verify_relation tool). Offline.
"""
from src.agent import (
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
