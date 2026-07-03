"""
Test for the `grounded_reasoning` PUBLIC API — ensures clean imports for external users.
"""
import grounded_reasoning as grx


def test_public_exports_present():
    for name in ("GroundedReasoner", "Verdict", "verify_relation", "run_tool",
                 "TOOL_SPEC", "openai_tool_spec", "ConformalReasoner",
                 "conformal_threshold", "LLMClient"):
        assert hasattr(grx, name), name
    assert grx.__version__ == "0.1.0"


def test_public_facade_works():
    gr = grx.GroundedReasoner()
    gr.add_facts([("a", "p", "b"), ("b", "p", "c")])
    v = gr.verify("a", "c", via="p")
    assert v.grounded and v.proof == ["a", "b", "c"]
    assert grx.verify_relation([["a", "p", "b"]], "a", "p", "z")["grounded"] is False
