"""
Test hồi quy từ fuzz `composition_algebra.py` (800 lượt, đối chiếu cộng-mod-N thật)
và `conformal_reasoning.py` (300 cấu hình phủ + 500 lượt ConformalReasoner + edge
case) — tìm ra 2 bug crash thật, đã sửa; khóa lại để không tái phát.

Bug 1 — `composition_algebra.fold(())`: chuỗi RỖNG khiến `spans=[]`, vòng lặp
`while len(spans)>1` không chạy, rồi `return spans[0]` crash IndexError trên list
rỗng. Sửa: trả None cho chuỗi rỗng (nhất quán với hợp đồng "None khi không rút gọn
được", không bịa một identity element không tồn tại).

Bug 2 — `conformal_reasoning.conformal_threshold([], alpha≥1/(n+1))`: với 0 điểm
calibration VÀ alpha đủ lớn để `k=int(alpha*(n+1))≥1` (n=0 nên chỉ cần alpha≥1),
biểu thức `s[min(k,n)-1]` = `s[min(k,0)-1]` = `s[-1]` trên list rỗng ⟹ IndexError.
Bug lan tới `ConformalReasoner.calibrate([])` (dùng thật khi true_pairs rỗng, ví
dụ nếu agent quên cấp ví dụ hiệu chỉnh). Sửa: raise ValueError rõ ràng — với 0 điểm
calibration, bảo đảm phủ ≥1−α KHÔNG THỂ thiết lập được về mặt toán học (lập luận
exchangeability cần ≥1 mẫu); thà báo lỗi còn hơn âm thầm trả ngưỡng vô căn cứ.
"""
import pytest

from src.reasoning.composition_algebra import fold, learn_composition
from src.reasoning.conformal_reasoning import ConformalReasoner, conformal_threshold


class TestCompositionEmptySequenceRegression:
    def test_fold_empty_sequence_returns_none(self):
        assert fold((), {}) is None
        assert fold((), {("a", "b"): "c"}) is None  # bảng không rỗng vẫn None

    def test_fold_single_element_unaffected(self):
        assert fold((5,), {}) == 5

    def test_learn_composition_empty_chains(self):
        table, conflicts, iters = learn_composition([])
        assert table == {} and conflicts == 0


class TestConformalEmptyCalibrationRegression:
    def test_exact_fuzz_repro_empty_scores_alpha_one(self):
        with pytest.raises(ValueError):
            conformal_threshold([], 1.0)

    def test_empty_scores_various_alpha_raise(self):
        for alpha in (0.0, 0.5, 1.0, 1.5, -0.1):
            with pytest.raises(ValueError):
                conformal_threshold([], alpha)

    def test_nonempty_still_works_after_fix(self):
        # bất biến cũ (k<1 -> -inf) vẫn giữ nguyên cho n≥1
        assert conformal_threshold([1.0, 2.0, 3.0], 0.0) == float("-inf")
        tau = conformal_threshold([1.0, 2.0, 3.0], 1.0)
        assert tau == 3.0   # alpha=1: chấp nhận chỉ điểm cao nhất

    def test_conformal_reasoner_calibrate_empty_raises(self):
        class _Engine:
            def infer(self, x):
                return {}

        cr = ConformalReasoner(_Engine(), alpha=1.0)
        with pytest.raises(ValueError):
            cr.calibrate([])   # true_pairs rỗng -> không thể hiệu chỉnh

    def test_conformal_reasoner_calibrate_nonempty_still_works(self):
        class _Engine:
            def infer(self, x):
                return {"b": 0.8, "c": 0.2}

        cr = ConformalReasoner(_Engine(), alpha=0.1)
        cr.calibrate([("a", "b")])
        assert cr.accept("a", "b")
