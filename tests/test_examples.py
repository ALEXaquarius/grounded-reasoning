"""
Test OFFLINE cho ví dụ examples/hallucination_demo: hệ grounded phải đúng 100% trên
thế giới của ví dụ (không gọi LLM). Khóa logic + thế giới của demo.
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
        # 'CÓ' luôn kèm đường đi; 'KHÔNG' không bịa đường
        assert (v.proof is not None) == gold


def test_deep_chain_and_reverse_trap():
    gr = GroundedReasoner()
    gr.add_facts(_demo.FACTS)
    # chuỗi sâu 9 bước
    assert gr.verify("Tùng", "Vũ", via="quản lý").proof == _demo.CHAIN
    # ngược hướng ⟹ không grounded (điểm LLM hay bịa)
    assert not gr.verify("Toàn", "Thành", via="quản lý").grounded
    assert not gr.verify("Kho K9", "Tùng", via="sở hữu").grounded
