"""
1D 热传导算例的仿射算子接口（FOMSolver 协议实现）
==================================================
ISEF 2028 ROM 项目 · 任务 #9 ｜ 架构文档 §3.1
负责人：同学 B

仿射分解推导（半离散系统 dT/dt = A(μ)T + b(μ)，μ = {k, h, q_gen}）：

  内部节点（中心差分）:
      dT_i/dt = [k/(ρcp·dx²)] (T_{i-1} - 2T_i + T_{i+1}) + q_gen/(ρcp)
  右端 Robin（-k dT/dx = h(T - t_cool)，一阶后向差分消元）:
      dT_{n-1}/dt = [k/(ρcp·dx²)] (T_{n-2} - T_{n-1})
                    + [h/(ρcp·dx)] (t_cool - T_{n-1}) + q_gen/(ρcp)
      —— 注意 Bi = h·dx/k 中的 k 与热扩散率中的 k 恰好相消，
         因此分解对 {k, h, q_gen} 严格仿射，无任何近似！

  定义 θ₁ = k/(ρcp·dx²)，θ₂ = h/(ρcp·dx)，θ₃ = q_gen/(ρcp)：
      A(μ) = θ₁ A₁ + θ₂ A₂
      b(μ) = θ₃ b₁ + θ₂ b₂
      A₁: 内部 Laplacian + 末行 (T_{n-2} - T_{n-1}) 结构
      A₂: 仅 A₂[-1,-1] = -1
      b₁ = [0, 1, ..., 1]ᵀ（左端 Dirichlet 行置零）
      b₂ = [0, ..., 0, t_cool]ᵀ
"""

import numpy as np
import scipy.sparse as sp

from ..rom.operators import AffineOperators
from .heat_1d import Heat1DParams

# 本算例的可扫描参数键（其余物理量取 Heat1DParams 缺省值）
PARAM_KEYS = ["k", "h_right", "q_gen"]


def affine_operators(base: Heat1DParams | None = None) -> AffineOperators:
    """构造 1D 热传导算例的仿射算子（固定量来自 base）。"""
    p = base or Heat1DParams()
    n = p.n_x
    dx = p.length / (n - 1)
    rho_cp_dx2 = p.rho * p.c_p * dx**2
    rho_cp_dx = p.rho * p.c_p * dx

    # --- A₁：内部 Laplacian + 末行扩散结构（行 0 清零，Dirichlet 由 mean 携带）---
    main = np.full(n, -2.0)
    off = np.ones(n - 1)
    A1 = sp.diags([off, main, off], [-1, 0, 1], format="lil")
    A1[0, 0] = 0.0
    A1[0, 1] = 0.0
    A1[-1, -1] = -1.0   # 末行覆盖为 (T_{n-2} - T_{n-1})
    # A₁[-1,-2] 保持 diags 的 1.0 ✓

    # --- A₂：仅末行对流耗散 ---
    A2 = sp.lil_matrix((n, n))
    A2[-1, -1] = -1.0

    # --- b₁：均匀内热源（行 0 置零）；b₂：末行对流常数项 ---
    b1 = np.ones(n)
    b1[0] = 0.0
    b2 = np.zeros(n)
    b2[-1] = p.t_cool

    def theta_A(params: dict) -> list[float]:
        return [params["k"] / rho_cp_dx2, params["h_right"] / rho_cp_dx]

    def theta_b(params: dict) -> list[float]:
        return [params["q_gen"] / (p.rho * p.c_p), params["h_right"] / rho_cp_dx]

    return AffineOperators(A_terms=[A1.tocsr(), A2.tocsr()],
                           b_terms=[b1, b2],
                           theta_A=theta_A, theta_b=theta_b)


def params_from_heat1d(p: Heat1DParams) -> dict:
    """从 Heat1DParams 提取可扫描参数 dict（与 PARAM_KEYS 对齐）。"""
    return {"k": p.k, "h_right": p.h_right, "q_gen": p.q_gen}
