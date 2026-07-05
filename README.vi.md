# grounded-reasoning — Suy diễn có bảo đảm cho LLM & Agent

[![CI](https://github.com/ALEXaquarius/grounded-reasoning/actions/workflows/ci.yml/badge.svg)](https://github.com/ALEXaquarius/grounded-reasoning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/grounded-reasoning.svg)](https://pypi.org/project/grounded-reasoning/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALEXaquarius/grounded-reasoning/blob/main/examples/quickstart.ipynb)

Read in English: **[README.md](README.md)**

> **Một dòng.** LLM ảo giác khi suy diễn quan hệ nhiều bước. Đây là một **bộ kiểm
> chứng đại số quan hệ** mà agent gọi để kiểm tra một khẳng định **trước khi** phát
> ra nó: **0 token model**, **bảo đảm precision** (chấp nhận khẳng định khi và chỉ khi
> tồn tại đường chứng minh grounded), không phụ thuộc ngôn ngữ hay nhà cung cấp LLM.
> Dùng được như **thư viện**, **function-calling tool**, hoặc **MCP server**. Đã kiểm
> chứng trên **LLM thật** (DeepSeek…) và benchmark công khai **CLUTRR**.

📄 Bài báo đầy đủ: **[PAPER.md](PAPER.md)** (tiếng Anh) · Hướng dẫn tích hợp: **[docs/integration.vi.md](docs/integration.vi.md)** · Chạy thử trong 30 giây: **[notebook quickstart](https://colab.research.google.com/github/ALEXaquarius/grounded-reasoning/blob/main/examples/quickstart.ipynb)**

---

## Vì sao dự án này tồn tại

LLM vững ở fact 1-bước nhưng **sụp đổ ở hợp thành** — nối nhiều fact đúng thành một kết
luận nhiều bước. Trên CLUTRR (suy luận quan hệ họ hàng), độ chính xác của DeepSeek
**tụt theo độ sâu**, trong khi một solver hợp-thành-toán-tử grounded giữ **~100%
phẳng — với 0 token**:

```
acc
100% ●─────●─────●─────●─────●─────●─────●   ● Solver grounded (đại số, 0 token)
 90% |
 80% ○
 70% |  ╲
 60% |   ╲
 50% |    ╲
 40% |     ○           ○                     ○ DeepSeek (LLM)
 30% |      ╲         ╱ ╲
 20% |       ○─────○     ╲
 10% |                    ○─────○
  0% +──┴─────┴─────┴─────┴─────┴─────┴─────┴─
      hop 2    3     4     5     6     7     8   (số bước hợp thành)

     hop:      2     3     4     5     6     7     8
     DeepSeek: 83%   42%   25%   25%   42%   17%   8%
     Solver:   100%  100%  100%  100%  100%  100%  100%
```

*(CLUTRR/v1 gen_train234_test2to10, clean-chain, n=12/hop; toàn bộ test set n=635:
solver phủ 99.5%, accuracy 99.2%. `grounded_reasoning/experiments/clutrr_eval.py`.)*

---

## Nó là gì / KHÔNG là gì (trung thực)

**LÀ:** một lớp kiểm chứng suy diễn **có bảo đảm**, xây trên đại số toán tử quan hệ.
- **Precision = 1.0 được bảo đảm** (Định lý G): chỉ chấp nhận khẳng định khi tồn
  tại đường chứng minh grounded.
- **0 token phụ trội**: nhân ma trận cục bộ, không gọi LLM. So với "LLM tự kiểm
  tra lại chính nó" — tốn thêm +110% token mà chỉ đạt 34% precision.
- **Bảo đảm hai chiều** (Định lý I): cả precision *và* recall đều có cận chặt.
- **Không cần knowledge-base ngoài** (SGDC): chỉ dùng tính nhất quán nội tại của
  chính LLM.

**KHÔNG LÀ:** một "đột phá chưa từng có". Katz index, chuỗi Neumann, reachability, và
neuro-symbolic grounding đều là **toán học/kỹ thuật cổ điển**. Đóng góp nằm ở **sự
hợp nhất + một bảo đảm được đo đạc + số liệu benchmark**, không phải một nguyên thủy
mới. Bộ bảo vệ **cần một đồ thị quan hệ** (được cung cấp, hoặc trích từ fact do LLM
sinh ra) — tính linh hoạt có giới hạn (xem [PAPER.md §5](PAPER.md)).

### Hai điểm mù mà bản thân đại số không thể tự thấy (và cách chặn)

Được nêu ra khi review, tái hiện lại bằng test, và fix bằng cơ chế opt-in cho
từng điểm — không giấu đi:

- **Định danh entity mặc định là so khớp chuỗi chính xác.** Nếu LLM trích xuất
  không nhất quán về cách viết một entity (`"Bob"` vs `"bob"`), đồ thị coi đó
  là 2 node khác nhau và một đường chứng minh thật bị đứt âm thầm — guard khi
  đó (đúng theo hợp đồng của chính nó) từ chối một khẳng định thực ra là đúng.
  Cách fix (nhị phân): `GroundedReasoner(normalize=lambda s: s.strip().casefold())`
  gộp các biến thể cách viết lại trước khi chúng trở thành khóa đồ thị; proof
  vẫn hiển thị đúng cách viết gốc xuất hiện đầu tiên của mỗi entity. **Định lý
  N** chỉ rõ chính xác khi nào việc này an toàn: precision giữ *đúng* 1.0 chừng
  nào `normalize` không bao giờ gộp nhầm 2 entity thật sự khác nhau — đó là
  **cách duy nhất** nó có thể sai, nên đó chính xác là điều
  `gr.calibrate_normalization(labeled_pairs)` đo được từ dữ liệu giữ riêng ra,
  tái dùng đúng máy Clopper-Pearson của Định lý M.
- **Định lý G không biết `via` có bắc cầu thật trong thực tế hay không.** Nó
  chỉ đảm bảo "tồn tại đường đi dưới closure của `via`", không đảm bảo `via`
  thực sự bắc cầu trong thế giới thật. Áp dụng cho một quan hệ chỉ bắc cầu một
  phần/có điều kiện (`"trusts"`: A tin B, B tin C, không suy ra A tin C) sẽ vẫn
  cho ra `grounded=True` một cách tự tin và đúng về mặt toán học — nhưng trả
  lời một câu hỏi khác với câu bạn thực sự muốn hỏi. Cách fix (nhị phân):
  `GroundedReasoner(transitive_relations={"parent", "is_a", ...})` khiến guard
  raise `ValueError` cho bất kỳ quan hệ nào chưa khai báo, biến một giả định
  ngầm thành một hợp đồng tường minh, được kiểm tra. Cách fix (đo được — **Định
  lý M**): `gr.calibrate_transitivity(rel, labeled_pairs)` thay thế lựa chọn
  nhị phân "khai báo hoặc từ chối" bằng một con số thật — cận tin cậy dưới
  Clopper-Pearson cho "một khẳng định được đồ thị chấp nhận cho `rel` có thực
  sự đúng không", tính từ các cặp đã gán nhãn giữ riêng ra (held-out). Nơi guard
  nhị phân chỉ đoán mù hoặc chặn hoàn toàn, cận đo được cho biết *nên tin đến
  mức nào*.

Cả hai cơ chế opt-in đều tắt mặc định (hành vi giống hệt các bản trước).
Tái hiện tại: `tests/test_agent.py::TestEntityNormalization`,
`::TestTransitiveRelationsGuard`, `::TestTransitivityCalibration`,
`::TestNormalizationCalibration`; so sánh A/B:
[`transitivity_calibration_eval.py`](grounded_reasoning/experiments/transitivity_calibration_eval.py),
[`normalization_calibration_eval.py`](grounded_reasoning/experiments/normalization_calibration_eval.py).

**Chuỗi quan hệ hỗn hợp (heterogeneous).** `verify(via=rel)` hợp thành MỘT quan
hệ với chính nó; `gr.verify_path(subject, obj, via=["parent","employer"])` hợp
thành một chuỗi các quan hệ **khác nhau** theo đúng thứ tự (VD một khẳng định
suy ra "phụ thuộc tài chính vào") — không phải toán mới
(`OperatorRelationAlgebra.follow` đã hợp thành chuỗi quan hệ hỗn hợp chính xác
theo Định lý G, chỗ này chỉ lộ ra ở tầng facade kèm tái tạo proof path) — và
`gr.calibrate_path(via, labeled_pairs)` hiệu chỉnh pattern cố định đó bằng
đúng máy Clopper-Pearson như `calibrate_transitivity` (xem PAPER.md §5.3.4).
Đã kiểm tra đối chiếu BFS độc lập qua 8.000 tổ hợp, không lệch lần nào:
`tests/test_agent.py::TestHeterogeneousPathVerification`,
[`heterogeneous_path_calibration_eval.py`](grounded_reasoning/experiments/heterogeneous_path_calibration_eval.py).

### So với các cách chống ảo giác thường gặp

| Cách tiếp cận | Token phụ trội | Bảo đảm | Cần KB ngoài |
|---|---|---|---|
| LLM tự kiểm tra lại (gọi lần 2) | +110% | không có (đo được 34% precision) | không |
| Self-consistency / vote đa mẫu | nhân theo số mẫu | không có, chỉ thống kê | không |
| RAG / grounding qua KG ngoài | tùy | chỉ tốt bằng chất lượng retrieval | có |
| **Bộ bảo vệ này** | **+0** | **precision = 1.0** (Định lý G) | không |
| **Bộ bảo vệ, self-grounded (SGDC)** | **+0** | precision = 1.0 khi atomic fact sound (Định lý I) | không |
| **Bộ bảo vệ, conformal** | **+0** | coverage ≥ 1−α, phân-phối-tự-do (Định lý K) | không |

---

## Ba định lý, một toán tử (F = G = H)

Lõi suy diễn dựa trên MỘT sự hợp nhất duy nhất (đã kiểm chứng số, sai số 0):

| Góc nhìn | Định lý | Nội dung |
|------|---------|---------|
| Suy diễn khuếch tán mờ | **F** | conf(a→b) = Σ αᵏ(Pᵏ)[a,b], được hiệu chỉnh + grounded |
| Đại số toán tử quan hệ | **G** | hợp thành = tích toán tử, bao đóng bắc cầu = Σ lũy thừa |
| Phân tích phổ (Katz) | **H** | `engine.infer` = giải thức (I−αP)⁻¹−I (khớp sai số **0.0**) |

⟹ suy diễn mờ **chính là** phân tích phổ của toán tử quan hệ. `grounded_reasoning/reasoning/`.

Sáu định lý mở rộng thêm: **I** (bảo đảm hai chiều precision/recall cho biến thể
self-grounded không cần KB ngoài), **J** (tính đầy đủ của học bao đóng, kiểm chứng
trên CLUTRR), **K** (suy diễn conformal — bảo đảm coverage phân phối-tự-do dưới đồ
thị quan hệ NHIỄU, kể cả đồ thị do LLM trích từ văn bản thô), **L** (suy diễn tiến
Horn, tổng quát hóa bao đóng bắc cầu sang luật có nhiều tiền đề), **M** (hiệu
chỉnh thực nghiệm giả định bắc cầu — cận tin cậy Clopper-Pearson thay thế một giả
định bắc cầu mù bằng một giả định đo được), và **N** (cô lập precision khi chuẩn
hóa — precision=1.0 chỉ vỡ khi gộp nhầm, và đó chính xác là điều duy nhất cần đo).
Cả chín định lý được phát biểu, chứng minh, và kiểm chứng số đầy đủ trong
[PAPER.md](PAPER.md).

---

## Bằng chứng trên LLM thật (DeepSeek)

| Thực nghiệm | Kết quả |
|------------|--------|
| Guard chống ảo giác (quan hệ họ hàng) | precision **33% → 100%**, bắt được 92/92 (2 seed), 0 từ chối nhầm |
| Guard chống ảo giác, bài test khó hơn (cây 48 người, sự kiện gây nhiễu anh/em/vợ chồng, văn xuôi xáo trộn, T=0.7, câu hỏi bẫy đáp án rỗng) | DeepSeek thô precision **4.6%** (2124 tên bịa, 86/90 câu hỏi bẫy bị bịa); sau guard precision **100%**, 0 lọt, 0 đáp án đúng bị loại nhầm — [`guard_llm_stress_eval.py`](grounded_reasoning/experiments/guard_llm_stress_eval.py) |
| Chi phí token của guard | **+0 token** (so với LLM tự kiểm: +110% token, 34% precision) |
| SGDC (self-grounded, không KB ngoài) | precision **78% → 100%** chỉ từ tính nhất quán nội tại |
| Ontology dày đặc, phản trực giác | precision **31% → 100%**, bắt được 106/106, 0 từ chối nhầm — [`nl_ontology_eval.run_dense`](grounded_reasoning/experiments/nl_ontology_eval.py) |
| CLUTRR (benchmark công khai) | solver **~100% ở mọi hop** so với DeepSeek 83%→8% |
| Đoạn văn khó (chuỗi 9 bước, 8 câu hỏi) | DeepSeek **bịa 1/8** (sai hướng); hệ grounded **8/8**, có chứng minh — [`examples/hallucination_demo.py`](examples/hallucination_demo.py) |

---

## Điểm nhấn: suy diễn có bảo đảm trên đồ thị do chính LLM trích xuất từ văn bản

Guard/solver cần một đồ thị **sạch**. Nhưng nếu để **LLM tự trích** quan hệ từ văn bản
tự nhiên, đồ thị sẽ **nhiễu** (thiếu/thừa cạnh). **Suy diễn Conformal** (Định lý K)
giải quyết đúng vấn đề này: dùng độ tin cậy của toán tử làm điểm số, hiệu chỉnh một
ngưỡng ⟹ **coverage phân phối-tự-do ≥ 1−α**, ngay cả trên đồ thị nhiễu.

Demo đầu-cuối: **DeepSeek trích đồ thị "is a" từ văn bản** → conformal chạy trên đồ
thị trích xuất đó (ground truth chỉ dùng để chấm điểm):

| Văn bản | Trích xuất LLM (P / R) | Coverage (mục tiêu ≥90%) | Hiệu quả (FPR) |
|------|------------------------:|----------------------------:|------------------:|
| Dễ | 100% / 99.7% | **91.3%** | 0.0 |
| Khó (mệnh đề lồng nhau + câu gây nhiễu gần giống) | 99.5% / **68.5%** | **93.0%** | 0.77 |

> Việc trích xuất của LLM **mất 31% số cạnh** (đồ thị thực sự nhiễu) →
> **bảo đảm coverage vẫn giữ vững** (93% ≥ 90%), chỉ hiệu quả suy giảm.
> *Tính hợp lệ luôn giữ vững; hiệu quả tỉ lệ theo chất lượng đồ thị.*

⟹ Một hướng đi tới suy diễn có bảo đảm trên **quan hệ ngôn ngữ tự nhiên** — nơi guard
cứng không chạm tới được. `grounded_reasoning/experiments/conformal_llm_eval.py`.

---

## Bắt đầu nhanh

```bash
pip install grounded-reasoning

# hoặc, để phát triển (test + lint):
git clone https://github.com/ALEXaquarius/grounded-reasoning
cd grounded-reasoning && pip install -e ".[dev]"
pytest tests/                       # mọi định lý + logic khóa offline, không cần mạng

# Dùng ngay (không cần LLM/mạng):
python -c "from grounded_reasoning import GroundedReasoner as G; r=G(); r.add_facts([('a','p','b'),('b','p','c')]); print(r.verify('a','c',via='p'))"

# Thực nghiệm LLM thật (cần API key — đọc từ biến môi trường, KHÔNG BAO GIỜ hardcode):
export DEEPSEEK_API_KEY=sk-...        # tự mang key của bạn; .env đã gitignore
python -m grounded_reasoning.experiments.guard_llm_eval        # guard chống ảo giác
python -m grounded_reasoning.experiments.guard_llm_stress_eval # khó hơn: nhiễu + bẫy + nhiệt độ cao
python -m grounded_reasoning.experiments.self_grounded_eval    # SGDC
python -m grounded_reasoning.experiments.clutrr_eval           # benchmark công khai CLUTRR
python -m grounded_reasoning.experiments.conformal_llm_eval    # conformal đầu-cuối (đồ thị do LLM trích)
python -m grounded_reasoning.experiments.guard_cost_eval       # chi phí token: guard vs. LLM tự kiểm
python -m grounded_reasoning.experiments.nl_ontology_eval      # ontology dày đặc phản trực giác (dùng run_dense() để ra kết quả 106/106)
```

---

## Tích hợp với Agent / LLM (`grounded_reasoning/agent/`)

Một **bộ kiểm chứng suy diễn quan hệ** cho agent: kiểm tra một khẳng định nhiều bước
**trước khi phát ra nó** — 0 token model, bảo đảm precision (chấp nhận khi và chỉ khi
tồn tại đường chứng minh grounded).

```python
from grounded_reasoning import GroundedReasoner
gr = GroundedReasoner()
gr.add_facts([("alice","parent","bob"),("bob","parent","carol")])
gr.verify("alice","carol", via="parent")   # Verdict(grounded=True, proof=['alice','bob','carol'], confidence=0.36, relation='parent')
gr.verify("alice","zed",   via="parent")   # Verdict(grounded=False, proof=None, confidence=0.0, relation='parent')  ← chặn ảo giác
```

Ba cách tích hợp (chi tiết: [docs/integration.vi.md](docs/integration.vi.md)):
- **Thư viện**: `GroundedReasoner.verify / filter_claims / contradictions`.
- **Function-calling**: `TOOL_SPEC` (Anthropic) / `openai_tool_spec()` (OpenAI) + `run_tool` — tool `verify_relation` stateless.
- **MCP server**: `python -m grounded_reasoning.agent.mcp_server` — cắm vào Claude hoặc bất kỳ agent tương thích MCP nào.

**Đa nhà cung cấp** (không chỉ DeepSeek): `LLMClient(provider=...)` cho DeepSeek /
OpenAI / Groq / OpenRouter / Together / Mistral / Ollama (local) — tất cả tương thích
OpenAI, đổi nhà cung cấp không cần đổi code. **Đa ngôn ngữ**: entity/relation là chuỗi
Unicode "trong suốt" ⟹ hoạt động với bất kỳ ngôn ngữ nào (`cha`, `父`, `والد`…) không
cần cấu hình.

---

## Sơ đồ mã nguồn

| Đường dẫn | Nội dung |
|------|---------|
| `grounded_reasoning/` | Package công khai — `GroundedReasoner`, `verify_relation`, `TOOL_SPEC`, `ConformalReasoner`, `LLMClient` |
| `grounded_reasoning/agent/{verifier,tool,mcp_server}.py` | Triển khai API công khai — HallucinationGuard, tool function-calling, MCP server |
| `grounded_reasoning/reasoning/abstract_inference.py` | FuzzyInferenceEngine, TypedInferenceEngine, HallucinationGuard (Định lý F) |
| `grounded_reasoning/reasoning/operator_algebra.py` | Đại số toán tử quan hệ (Định lý G) |
| `grounded_reasoning/reasoning/relation_spectrum.py` | Phổ, tính lũy linh, giải thức Katz (Định lý H) |
| `grounded_reasoning/reasoning/conformal_reasoning.py` | Conformal — bảo đảm coverage dưới nhiễu (Định lý K) |
| `grounded_reasoning/reasoning/composition_algebra.py` | Học bảng hợp thành, kiểm chứng trên CLUTRR (Định lý J) |
| `grounded_reasoning/reasoning/horn.py` | Suy diễn tiến Horn, ngữ nghĩa least-model (Định lý L) |
| `grounded_reasoning/reasoning/transitivity_calibration.py` | Hiệu chỉnh Clopper-Pearson — tái dùng cho cả giả định bắc cầu (Định lý M) và rủi ro gộp nhầm khi chuẩn hóa (Định lý N) |
| `grounded_reasoning/reasoning/llm_client.py` | LLM client không phụ thuộc nhà cung cấp (key đọc từ biến môi trường) |
| `grounded_reasoning/theory/theorems.py` | **Chín định lý (F–N)** với kiểm chứng số |
| `grounded_reasoning/experiments/*.py` | Thực nghiệm LLM thật và benchmark hậu thuẫn mọi khẳng định trên |
| `examples/hallucination_demo.py` | Demo function-calling đầu-cuối |
| `examples/quickstart.ipynb` | Tour chạy được của thư viện (offline, sẵn sàng cho Colab) |

---

## Câu chuyện khởi nguồn

Dự án bắt đầu từ nỗ lực phát minh một thuật toán retrieval embedding-free có thể cạnh
tranh với dense/RAG retrieval. Hướng nghiên cứu đó đi tới một kết luận **âm tính**
hoàn toàn trung thực (hòa BM25, thua có ý nghĩa so với dense embedding — có chứng
minh vì sao). Cùng bộ công cụ toán học đó — đại số toán tử, phân tích phổ — lại cho
thấy giá trị thật, đo được trên một bài toán khác: **bảo đảm** suy diễn quan hệ nhiều
bước. Repo này chỉ chứa hệ suy diễn đã kiểm chứng, đã test đó; toàn bộ hành trình
nghiên cứu retrieval (kể cả mọi hướng thất bại, ghi chép trung thực) nằm ở một repo
nghiên cứu riêng, không thuộc package này. Xem [PAPER.md §1](PAPER.md) để biết khung
đầy đủ.

---

## Đóng góp & Cộng đồng

- Cách đóng góp + nguyên tắc nghiên cứu: [CONTRIBUTING.md](CONTRIBUTING.md)
- Quy tắc ứng xử: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) · Bảo mật: [SECURITY.md](SECURITY.md)
- Lịch sử phiên bản: [CHANGELOG.md](CHANGELOG.md) · Trích dẫn: [CITATION.cff](CITATION.cff)
- Giấy phép: **MIT** ([LICENSE](LICENSE))

---

*Nguyên tắc: chứng minh trước code, định nghĩa hình thức, falsifiability, và ghi
chép trung thực các kết quả âm tính — xem [CONTRIBUTING.md](CONTRIBUTING.md).*
