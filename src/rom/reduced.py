"""
降阶模型装配 + 降阶求解器（核心框架模块）
==========================================
ISEF 2028 ROM 项目 · 任务 #9 ｜ 架构文档 §3.4
负责人：同学 B（实现）｜ 投影推导见 docs/math_notes_B.md §4

Galerkin 投影（lifting 形式）：
    T(t) ≈ T̄ + V a(t)  →  ȧ = VᵀAV a + Vᵀ(AT̄ + b)

仿射情形下离线一次性投影各 A_q / b_q：
    A_r(μ) = Σ_q θ_q(μ) A_r,q        A_r,q = Vᵀ A_q V
    f_r(μ) = Σ_q θ_q(μ) VᵀA_qT̄ + Σ_q ϑ_q(μ) Vᵀb_q
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operators import AffineOperators
from .pod import PODBasis


@dataclass
class ReducedModel:
    """装配完成的降阶模型（离线阶段的产出）。"""

    mean: np.ndarray            # lifting 锚点 (n_x,)
    V: np.ndarray               # 截断 POD 基 (n_x, r)
    Ar_terms: list[np.ndarray]  # Vᵀ A_q V，每个 (r, r)
    fr_A_terms: list[np.ndarray]  # Vᵀ A_q T̄，每个 (r,)
    fr_b_terms: list[np.ndarray]  # Vᵀ b_q，每个 (r,)
    theta_A: Callable[[dict], list[float]]
    theta_b: Callable[[dict], list[float]]

    @property
    def r(self) -> int:
        return self.V.shape[1]

    def assemble_reduced(self, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """在线零投影成本装配 A_r(μ), f_r(μ)。"""
        A_r = sum(th * Ar for th, Ar in zip(self.theta_A(params), self.Ar_terms))
        f_r = (sum(th * fr for th, fr in zip(self.theta_A(params), self.fr_A_terms))
               + sum(th * fr for th, fr in zip(self.theta_b(params), self.fr_b_terms)))
        return np.asarray(A_r), np.asarray(f_r)


class ReducedModelBuilder:
    """离线装配：POD 基 × 仿射算子 → ReducedModel。"""

    def build(self, basis: PODBasis, ops: AffineOperators, r: int) -> ReducedModel:
        if r > basis.V.shape[1]:
            raise ValueError(f"r={r} 超过基的模态数 {basis.V.shape[1]}")
        V = basis.V[:, :r]
        Ar_terms = [np.asarray(V.T @ Aq @ V) for Aq in ops.A_terms]
        fr_A_terms = [np.asarray(V.T @ (Aq @ basis.mean)) for Aq in ops.A_terms]
        fr_b_terms = [np.asarray(V.T @ bq) for bq in ops.b_terms]
        return ReducedModel(mean=basis.mean, V=V,
                            Ar_terms=Ar_terms, fr_A_terms=fr_A_terms,
                            fr_b_terms=fr_b_terms,
                            theta_A=ops.theta_A, theta_b=ops.theta_b)


# ----------------------------------------------------------------------
# 降阶求解器
# ----------------------------------------------------------------------
@dataclass
class ROMResult:
    """降阶仿真结果。"""

    t: np.ndarray               # 时间网格 (n_t,)
    a: np.ndarray               # 降阶系数轨迹 (r, n_t)
    snapshots: np.ndarray       # 重构全维温度场 (n_x, n_t)
    alerts: list[dict]          # 误差监控告警记录（任务 #14/#15 埋点）


# 监控回调协议（架构文档 §3.5 ErrorMonitor 的最小实现）：
#   monitor(step: int, a: np.ndarray, residual: np.ndarray) -> str | None
# 返回非 None 字符串即记为一条告警；None 表示正常。
MonitorFn = Callable[[int, np.ndarray, np.ndarray], "str | None"]


class ROMSolver:
    """降阶系统时间积分（向后欧拉，与全阶求解器一致）。"""

    def __init__(self, dt: float, t_end: float):
        if dt <= 0 or t_end <= 0:
            raise ValueError("dt 与 t_end 必须为正")
        self.dt = float(dt)
        self.t_end = float(t_end)

    def solve(self, model: ReducedModel, params: dict, T0: np.ndarray,
              monitor: "MonitorFn | None" = None,
              full_ops: AffineOperators | None = None) -> ROMResult:
        """积分降阶系统。

        参数:
            model: 降阶模型
            params: 参数 dict（经仿射系数在线装配）
            T0: 初始全维场 (n_x,)（投影得 a0）
            monitor: 可选误差监控回调（同学 A 的插件挂载点）
            full_ops: 监控需要算全阶残差时传入（r = b - A T - Ṫ 的子空间外分量）
        """
        n_x, r = model.V.shape
        n_t = int(round(self.t_end / self.dt)) + 1
        t = np.linspace(0.0, self.t_end, n_t)

        A_r, f_r = model.assemble_reduced(params)
        lhs = np.eye(r) - self.dt * A_r
        lu = spla.factorized(sp.csc_matrix(lhs))

        a = np.empty((r, n_t))
        a[:, 0] = model.V.T @ (T0 - model.mean)
        alerts: list[dict] = []

        for n in range(1, n_t):
            a[:, n] = lu(a[:, n - 1] + self.dt * f_r)

            if monitor is not None and full_ops is not None:
                T_rom = model.mean + model.V @ a[:, n]
                T_prev = model.mean + model.V @ a[:, n - 1]
                A_full, b_full = full_ops.assemble(params)
                # 残差 Ṫ - A T - b 在基空间外的分量（数学笔记 §4.3）
                res = (T_rom - T_prev) / self.dt - A_full @ T_rom - b_full
                res_out = res - model.V @ (model.V.T @ res)
                msg = monitor(n, a[:, n], res_out)
                if msg is not None:
                    alerts.append({"step": n, "t": t[n], "message": msg,
                                   "residual_out_norm": float(np.linalg.norm(res_out))})

        snapshots = model.mean[:, None] + model.V @ a
        return ROMResult(t=t, a=a, snapshots=snapshots, alerts=alerts)
