# -*- coding: utf-8 -*-
"""
Ví dụ: LLM có BỊA khi suy diễn quan hệ nhiều bước không, và hệ grounded khác ra sao?

Cho một đoạn văn tự nhiên chứa một chuỗi quản lý/sở hữu SÂU (9 bước), tên dễ nhầm và
các mắt xích kể LỘN XỘN. Hỏi LLM các câu suy diễn nhiều bước (gồm câu NGƯỢC HƯỚNG —
nơi LLM hay bịa), rồi so với hệ grounded (đại số quan hệ, 0 token, có bằng chứng).

Kết quả điển hình: LLM đúng chuỗi thuận nhưng BỊA ở câu ngược hướng; hệ grounded 10/10.

Chạy:  DEEPSEEK_API_KEY=... python examples/hallucination_demo.py
       (hoặc LLM_PROVIDER=groq/openai/... — xem grounded_reasoning.LLMClient)
"""
from __future__ import annotations

import os
import time

from grounded_reasoning import GroundedReasoner, LLMClient

# --- Thế giới (ground truth, để chấm) ---
CHAIN = ["Tùng", "Trung", "Tuấn", "Tân", "Thành", "Thắng", "Thịnh", "Toàn", "Tú", "Vũ"]
OWN = ["Tùng", "Sao Mai", "Việt Long", "Đông Đô", "Ba Vì", "Kho K9"]
FACTS = (
    [(CHAIN[i], "quản lý", CHAIN[i + 1]) for i in range(len(CHAIN) - 1)]
    + [(OWN[i], "sở hữu", OWN[i + 1]) for i in range(len(OWN) - 1)]
)

PASSAGE = """
Ở tập đoàn Sao Mai, sơ đồ quyền lực khá rắc rối và nhiều người kể lại sai. Anh Thịnh là
cấp trên trực tiếp của anh Toàn. Trước đó, ông Tùng — chủ tịch — chỉ trực tiếp quản lý
một người duy nhất là anh Trung. Anh Tân thì nằm dưới quyền anh Tuấn. Người quản lý anh
Vũ (nhân viên trẻ nhất) chính là anh Tú. Anh Thành lại là cấp trên trực tiếp của anh
Thắng, trong khi anh Thắng quản lý anh Thịnh. Đừng quên: anh Trung là người quản lý trực
tiếp anh Tuấn, còn anh Toàn là cấp trên trực tiếp của anh Tú. Mắt xích còn lại: anh Tân
quản lý anh Thành. Như vậy toàn bộ tạo thành một dây chuyền dài từ chủ tịch xuống tận
nhân viên tuyến cuối, dù ở đây các mắt xích được kể lộn xộn.

Về tài sản thì tách bạch: ông Tùng sở hữu tập đoàn Sao Mai. Tập đoàn Sao Mai sở hữu công
ty Việt Long. Công ty Việt Long sở hữu chi nhánh Đông Đô. Chi nhánh Đông Đô sở hữu xưởng
Ba Vì, và xưởng Ba Vì sở hữu kho hàng mang mã Kho K9. Nhiều đối thủ cố tình đồn thổi
đảo ngược các quan hệ này để gây nhiễu thông tin.
""".strip()

# (subject, object, via, mô tả, đáp án đúng)
QUESTIONS = [
    ("Tùng", "Vũ", "quản lý", "Ông Tùng có quản lý (gián tiếp) anh Vũ không?", True),
    ("Thành", "Vũ", "quản lý", "Anh Thành có quản lý (gián tiếp) anh Vũ không?", True),
    ("Vũ", "Tùng", "quản lý", "Anh Vũ có quản lý (gián tiếp) ông Tùng không?", False),
    ("Toàn", "Thành", "quản lý", "Anh Toàn có quản lý (gián tiếp) anh Thành không?", False),
    ("Tú", "Thắng", "quản lý", "Anh Tú có quản lý (gián tiếp) anh Thắng không?", False),
    ("Trung", "Toàn", "quản lý", "Anh Trung có quản lý (gián tiếp) anh Toàn không?", True),
    ("Tùng", "Kho K9", "sở hữu", "Ông Tùng có sở hữu (gián tiếp) Kho K9 không?", True),
    ("Kho K9", "Tùng", "sở hữu", "Kho K9 có sở hữu (gián tiếp) ông Tùng không?", False),
]


def _ask(client: LLMClient, question: str, tries: int = 6) -> tuple[bool, str]:
    prompt = (
        f"Đọc kỹ đoạn văn sau và CHỈ dựa vào nó:\n\n{PASSAGE}\n\n"
        f"Câu hỏi: {question}\n"
        "Trả lời DÒNG ĐẦU đúng một từ: CÓ hoặc KHÔNG. Rồi giải thích ngắn."
    )
    for i in range(tries):
        try:
            ans = client.ask(prompt, temperature=0.0)
            break
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    first = ans.strip().split("\n")[0].upper()
    yes = ("CÓ" in first or "CO" in first) and "KHÔNG" not in first and "KHONG" not in first
    return yes, first[:46]


def main() -> None:
    client = LLMClient(provider=os.environ.get("LLM_PROVIDER", "deepseek"))
    gr = GroundedReasoner()
    gr.add_facts(FACTS)

    print("=" * 74)
    print(f"LLM ({client.model}) suy luận tự do  vs  HỆ GROUNDED (0 token, có bằng chứng)")
    print("=" * 74)
    llm_ok = gr_ok = halluc = 0
    for subj, obj, via, desc, gold in QUESTIONS:
        llm_yes, llm_first = _ask(client, desc)
        v = gr.verify(subj, obj, via=via)
        a, b = (llm_yes == gold), (v.grounded == gold)
        llm_ok += a
        gr_ok += b
        halluc += 0 if a else 1
        proof = "→".join(v.proof) if v.proof else "— (không có đường)"
        print(f"\nQ: {desc}")
        print(f"   đúng={'CÓ' if gold else 'KHÔNG':5} | "
              f"LLM={'CÓ' if llm_yes else 'KHÔNG':5} {'OK' if a else 'BỊA/SAI'} | "
              f"hệ={'CÓ' if v.grounded else 'KHÔNG':5} {'OK' if b else 'X'}")
        print(f"   bằng chứng hệ: {proof}")
    n = len(QUESTIONS)
    print("\n" + "=" * 74)
    print(f"LLM đúng {llm_ok}/{n} (bịa/sai {halluc})  |  "
          f"Hệ grounded đúng {gr_ok}/{n}, 0 token, có bằng chứng")
    print(f"LLM tokens: {client.total_tokens}")


if __name__ == "__main__":
    main()
