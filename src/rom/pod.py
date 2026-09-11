"""
POD 分解器（核心框架模块）
==========================
ISEF 2028 ROM 项目 · 任务 #9 ｜ 架构文档 §3.3
负责人：同学 B（实现）｜ rank 选取策略后续可接入同学 A 的自适应准则

数学依据见 docs/math_notes_B.md §1：
  POD 基 = 中心化快照矩阵的左奇异向量 = 快照协方差矩阵主特征向量
  Eckart-Young：||S - S_r||_F = sqrt(Σ_{i>r} σ_i²)（最优 rank-r 逼近）
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PODBasis:
    """POD 基及其能量信息。

    属性:
        mean: 快照均值 (n_x,)（lifting，Galerkin 展开的锚点）
        V: POD 基矩阵 (n_x, n_modes)，列正交 V^T V = I
        sigma: 全部奇异值 (n_modes,)
        energy_ratio: 累积能量比 (n_modes,)
    """

    mean: np.ndarray
    V: np.ndarray
    sigma: np.ndarray
    energy_ratio: np.ndarray

    def rank_for_energy(self, threshold: float = 0.999) -> int:
        """达到给定累积能量所需的最小模态数。"""
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold 须在 (0, 1] 内，得到 {threshold}")
        return int(np.searchsorted(self.energy_ratio, threshold) + 1)

    def truncate(self, r: int) -> "PODBasis":
        """截取前 r 个模态。"""
        if r > self.V.shape[1]:
            raise ValueError(f"r={r} 超过可用模态数 {self.V.shape[1]}")
        return PODBasis(mean=self.mean, V=self.V[:, :r],
                        sigma=self.sigma[:r], energy_ratio=self.energy_ratio)


class PODDecomposer:
    """从快照矩阵构建 POD 基。"""

    def __init__(self, center: bool = True):
        """
        参数:
            center: 是否减去均值（lifting）。边界条件非齐次时必须 True，
                    使模态在 Dirichlet 边界上近似为零（见数学笔记 §4.1）。
        """
        self.center = center

    def fit(self, S: np.ndarray) -> PODBasis:
        """对快照矩阵 S (n_x, n_t) 做 SVD，返回 PODBasis。"""
        if S.ndim != 2 or min(S.shape) < 2:
            raise ValueError(f"快照矩阵须为二维且每维 ≥2，得到 {S.shape}")
        mean = S.mean(axis=1) if self.center else np.zeros(S.shape[0])
        U, sigma, _ = np.linalg.svd(S - mean[:, None], full_matrices=False)
        energy = sigma**2
        total = energy.sum()
        energy_ratio = np.cumsum(energy) / total if total > 0 else np.ones_like(energy)
        return PODBasis(mean=mean, V=U, sigma=sigma, energy_ratio=energy_ratio)

    @staticmethod
    def reconstruction_error(basis: PODBasis, S: np.ndarray, r: int) -> float:
        """rank-r 重构的相对 Frobenius 误差（应等于 sqrt(1-能量比_r），Eckart-Young）。"""
        Sc = S - basis.mean[:, None]

        Sr = basis.V[:, :r] @ (basis.V[:, :r].T @ Sc)
        return float(np.linalg.norm(Sc - Sr) / np.linalg.norm(Sc))
