"""
Test for the `grounded_reasoning` PUBLIC API — ensures clean imports for external users.
"""
import importlib.metadata

import grounded_reasoning as grx


def test_public_exports_present():
    for name in ("GroundedReasoner", "Verdict", "verify_relation", "run_tool",
                 "TOOL_SPEC", "openai_tool_spec", "ConformalReasoner",
                 "conformal_threshold", "LLMClient"):
        assert hasattr(grx, name), name
    # __version__ must match what the installed distribution's metadata reports —
    # catches drift between grounded_reasoning/_version.py and pyproject.toml's
    # dynamic version wiring, without pinning a literal that needs bumping here too.
    assert grx.__version__ == importlib.metadata.version("grounded-reasoning")


def test_public_facade_works():
    gr = grx.GroundedReasoner()
    gr.add_facts([("a", "p", "b"), ("b", "p", "c")])
    v = gr.verify("a", "c", via="p")
    assert v.grounded and v.proof == ["a", "b", "c"]
    assert grx.verify_relation([["a", "p", "b"]], "a", "p", "z")["grounded"] is False
