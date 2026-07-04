"""
OFFLINE test for the examples/hallucination_demo example: the grounded system must
be 100% correct on the example's world (no LLM calls). Locks down the demo's logic
and world data.
"""
import importlib.util
import pathlib

from grounded_reasoning import GroundedReasoner

_PATH = pathlib.Path(__file__).resolve().parent.parent / "examples" / "hallucination_demo.py"
_spec = importlib.util.spec_from_file_location("hallucination_demo", _PATH)
_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_demo)


def test_grounded_perfect_on_demo_world():
    gr = GroundedReasoner()
    gr.add_facts(_demo.FACTS)
    for subj, obj, via, _desc, gold in _demo.QUESTIONS:
        v = gr.verify(subj, obj, via=via)
        assert v.grounded == gold, (subj, obj, via)
        # a 'YES' answer always carries a proof path; a 'NO' answer never fabricates one
        assert (v.proof is not None) == gold


def test_deep_chain_and_reverse_trap():
    gr = GroundedReasoner()
    gr.add_facts(_demo.FACTS)
    # 9-hop deep chain
    assert gr.verify("Grant", "Wells", via="manages").proof == _demo.CHAIN
    # reversed direction ⟹ not grounded (a spot where LLMs tend to fabricate)
    assert not gr.verify("Kerr", "Hale", via="manages").grounded
    assert not gr.verify("Warehouse K9", "Grant", via="owns").grounded
