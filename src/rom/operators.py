"""
仿射参数算子（pROM 核心抽象）
==============================
ISEF 2028 ROM 项目 · 任务 #9 ｜ 升级方案①：pROM 仿射参数分解

核心思想：若算子对参数 μ 有仿射依赖

    A(μ) = Σ_q θ_q(μ) A_q,      b(μ) = Σ_q ϑ_q(μ) b_q

则降阶算子可离线一次性投影：

    A_r(μ) = Σ_q θ_q(μ) (Vᵀ A_q V),   f_r(μ) = 随参数组合

在线阶段对每个新参数点 **零投影成本** 装配小系统——这是 500 组
参数扫描（任务 #19）加速比的数学基础，也是 SIAM/AIP 评委的"硬菜"。

1D 热传导的具体分解见 src/fom/heat_1d_affine.py（含推导注释）。
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class AffineOperators:
    """仿射参数依赖的全阶算子。

    属性:
        A_terms: 与参数无关的稀疏矩阵列表 [A_1, ..., A_Qa]
        b_terms: 与参数无关的向量列表 [b_1, ..., b_Qb]
        theta_A: 参数 dict → 系数列表 [θ_1(μ), ..., θ_Qa(μ)]
        theta_b: 参数 dict → 系数列表 [ϑ_1(μ), ..., ϑ_Qb(μ)]
    """

    A_terms: list[sp.csr_matrix]
    b_terms: list[np.ndarray]
    theta_A: Callable[[dict], list[float]]
    theta_b: Callable[[dict], list[float]]

    def assemble(self, params: dict) -> tuple[sp.csr_matrix, np.ndarray]:
        """按参数组合出全阶算子 A(μ), b(μ)（主要用于验证与对照）。"""
        A = sum(th * Aq for th, Aq in zip(self.theta_A(params), self.A_terms))
        b = sum(th * bq for th, bq in zip(self.theta_b(params), self.b_terms))
        return A.tocsr(), np.asarray(b, dtype=float)
