"""
Conformal Reasoning — bảo đảm PHỦ phân-phối-tự-do cho suy diễn nhiều bước KỂ CẢ khi
đồ thị quan hệ NHIỄU (cạnh thiếu/thừa).

Guard/SGDC cho precision=1.0 nhưng CHỈ khi đồ thị sạch (sound). Khi quan hệ là ngôn
ngữ nhiễu, không còn bảo đảm cứng. Conformal prediction (Vovk, Gammerman, Shafer)
vá đúng chỗ đó: dùng độ tin cậy toán tử conf(a→b) làm điểm số, hiệu chỉnh ngưỡng τ
trên tập calibration để BẢO ĐẢM P(đáp án đúng được giữ) ≥ 1−α — không giả định phân
phối, đúng với MỌI chất lượng đồ thị.

Ta KHÔNG phát minh conformal; đóng góp = ghép nó với độ tin cậy toán tử (resolvent
Katz, Định lý H) và mô tả đánh đổi: PHỦ luôn hợp lệ, HIỆU QUẢ (kích thước tập/FPR)
suy giảm theo độ nhiễu. Xem kiểm chứng: theorem_conformal_reasoning.
"""
from __future__ import annotations


def conformal_threshold(cal_scores: list[float], alpha: float) -> float:
    """
    Ngưỡng split-conformal: chấp nhận nếu score ≥ τ ⟹ phủ ≥ 1−α (biên, phân-phối-tự-do).
    τ = phần tử nhỏ thứ k của điểm calibration, k = ⌊α·(n+1)⌋ (τ=−∞ nếu k=0).

    Raises
    ------
    ValueError nếu cal_scores rỗng: với 0 điểm hiệu chỉnh, bảo đảm phủ ≥1−α KHÔNG
    THỂ thiết lập được (lập luận exchangeability cần ≥1 mẫu) — thà báo lỗi rõ còn
    hơn âm thầm trả một ngưỡng "trông hợp lý" nhưng không mang bảo đảm gì.
    """
    if not cal_scores:
        raise ValueError("conformal_threshold cần ≥1 điểm calibration để có bảo đảm phủ")
    s = sorted(cal_scores)
    n = len(s)
    k = int(alpha * (n + 1))
    if k < 1:
        return float("-inf")
    return s[min(k, n) - 1]


class ConformalReasoner:
    """
    Bọc một máy suy diễn (có `.infer(x)->{b:conf}`), hiệu chỉnh ngưỡng conformal từ
    ví dụ có nhãn, rồi trả TẬP dự đoán có bảo đảm phủ ≥ 1−α.
    """

    def __init__(self, engine, alpha: float = 0.1) -> None:
        self.engine = engine
        self.alpha = alpha
        self.tau = float("-inf")
        self._cache: dict = {}

    def _conf(self, x, b) -> float:
        if x not in self._cache:
            self._cache[x] = self.engine.infer(x)
        return self._cache[x].get(b, 0.0)

    def calibrate(self, true_pairs: list[tuple]) -> float:
        """Hiệu chỉnh τ trên các cặp (x,b) ĐÚNG (positive class)."""
        scores = [self._conf(x, b) for x, b in true_pairs]
        self.tau = conformal_threshold(scores, self.alpha)
        return self.tau

    def accept(self, x, b) -> bool:
        """Chấp nhận (x,b) nếu độ tin cậy ≥ ngưỡng conformal."""
        return self._conf(x, b) >= self.tau

    def predict_set(self, x, candidates) -> set:
        """Tập dự đoán {b : conf(x→b) ≥ τ} — bảo đảm chứa đáp án đúng với xs ≥ 1−α."""
        return {b for b in candidates if self._conf(x, b) >= self.tau}
