"""
PHỔ của toán tử quan hệ — cấu trúc suy diễn qua trị riêng.

Nối đại số toán tử (Định lý G) với lý thuyết phổ của dự án. Cho một quan hệ với
ma trận kề A (A[i,j]=1 ⟺ i--r-->j), phổ của A tiết lộ cấu trúc suy diễn:

  • ACYCLIC ⟺ NILPOTENT ⟺ bán kính phổ ρ(A)=0.
      Quan hệ phân cấp thật (parent, part-of) có A nilpotent: A^n=0 → đóng kín bắc
      cầu DỪNG sau ≤ n bước (không vòng lặp vô hạn).
  • CHU TRÌNH: khái niệm i nằm trên một chu trình ⟺ (Σ_{k=1}^n A^k)[i,i] > 0
      ⟺ tồn tại trị riêng khác 0. Đây là các LỚP TƯƠNG ĐƯƠNG (mutual reachability).
  • RESOLVENT = DIFFUSION: với P = D⁻¹A (row-stochastic), niềm tin khuếch tán
      Σ_{k≥1} α^k P^k = (I-αP)⁻¹ - I  (chuỗi Neumann / chỉ số Katz), hội tụ ⟺
      α·ρ(P) < 1. ⟹ FuzzyInferenceEngine CHÍNH LÀ resolvent cắt cụt — suy diễn mờ
      là giải tích phổ của toán tử quan hệ.

Đây là suy diễn trừu tượng nhìn qua LĂNG KÍNH PHỔ: acyclicity, chu trình, và niềm
tin bắc cầu đều là bất biến phổ của cùng một toán tử.
"""
from __future__ import annotations

import numpy as np


def spectral_radius(A: np.ndarray) -> float:
    """ρ(A) = max |trị riêng|."""
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def is_nilpotent(A: np.ndarray, tol: float = 1e-9) -> bool:
    """
    A nilpotent (tính CẤU TRÚC của đồ thị quan hệ, KHÔNG phụ thuộc trọng số cạnh)
    ⟺ A^n = 0 khi NHỊ PHÂN HÓA (chỉ dấu hiệu có/không cạnh) ⟺ đồ thị ACYCLIC.

    NHỊ PHÂN HÓA trước khi kiểm tra: nếu dùng trực tiếp ma trận trọng số, một chu
    trình THẬT với trọng số <1 khiến các phần tử A^n suy giảm theo hàm mũ và tụt
    dưới `tol` thuần vì số học (không phải vì đồ thị acyclic) ⟹ false negative.
    Acyclicity là thuộc tính TÔ-PÔ của đồ thị, không phải của trọng số.
    """
    n = A.shape[0]
    if n == 0:
        return True
    B = (A != 0).astype(float)
    return float(np.abs(np.linalg.matrix_power(B, n)).max()) <= tol


def is_acyclic(A: np.ndarray, tol: float = 1e-9) -> bool:
    """Quan hệ không có chu trình (DAG) ⟺ toán tử nilpotent."""
    return is_nilpotent(A, tol)


def cycle_members(A: np.ndarray, tol: float = 1e-9) -> set[int]:
    """
    Chỉ số khái niệm nằm trên ÍT NHẤT một chu trình (self-reachable qua ≥1 bước).
    NHỊ PHÂN HÓA trước khi cộng dồn (cùng lý do với `is_nilpotent`): thuộc tính CẤU
    TRÚC không phụ thuộc trọng số, tránh false-negative khi trọng số cạnh <1 làm
    (Aᵏ)[i,i] suy giảm dưới tol dù có chu trình thật.
    """
    n = A.shape[0]
    B = (A != 0).astype(float)
    acc = np.zeros((n, n))
    P = np.eye(n)
    for _ in range(n):
        P = P @ B
        acc += P
    return {i for i in range(n) if acc[i, i] > tol}


def katz_resolvent(A: np.ndarray, alpha: float) -> np.ndarray:
    """
    Σ_{k≥1} α^k A^k = (I-αA)⁻¹ - I  (chỉ số Katz). Hội tụ ⟺ α·ρ(A) < 1.
    Với A nilpotent (acyclic) tổng HỮU HẠN và đồng nhất chính xác.

    Raises
    ------
    ValueError nếu α·ρ(A) ≥ 1: chuỗi Neumann PHÂN KỲ, nghịch đảo ma trận vẫn có
    thể tính được (không suy biến chính xác) nhưng KHÔNG còn bằng Σ α^k A^k nữa —
    thà báo lỗi rõ còn hơn âm thầm trả một ma trận "trông hợp lý" nhưng vô nghĩa.
    """
    n = A.shape[0]
    if n > 0:
        rho = spectral_radius(A)
        if alpha * rho >= 1.0:
            raise ValueError(
                f"katz_resolvent phân kỳ: α·ρ(A) = {alpha}·{rho} = {alpha * rho:.6g} ≥ 1 "
                "(chuỗi Neumann cần α·ρ(A) < 1 để hội tụ)"
            )
    eye = np.eye(n)
    return np.linalg.inv(eye - alpha * A) - eye


def row_stochastic(A: np.ndarray) -> np.ndarray:
    """P = D⁻¹A: chuẩn hoá mỗi hàng (ma trận chuyển khuếch tán)."""
    deg = A.sum(axis=1)
    P = np.zeros_like(A, dtype=float)
    nz = deg > 0
    P[nz] = A[nz] / deg[nz, None]
    return P


def diffusion_sum(A: np.ndarray, alpha: float, K: int) -> np.ndarray:
    """Σ_{k=1}^K α^k A^k — tổng khuếch tán cắt cụt (giống FuzzyInferenceEngine)."""
    n = A.shape[0]
    S = np.zeros((n, n))
    P = np.eye(n)
    for k in range(1, K + 1):
        P = P @ A
        S += (alpha ** k) * P
    return S
