"""
误差预算仪表盘（任务 #26 / 方向②主攻）
==========================================
ISEF 2028 ROM 项目
负责人：同学 B（工程实现）｜ 各组分上界严格推导：同学 A（任务 #14）

方向②核心交付：每个参数查询点输出一张"误差预算证书"——

    E_total ≤ E_proj + E_time + E_param     (E_aff = 0 已恒等验证)

组分状态：
    E_proj   ✅ 离线可算：谱尾 sqrt(Σ_{i>r} σ_i²)（Eckart-Young）
    E_time   ✅ 在线可算：Richardson 外推（dt 与 dt/2 双解差，一阶格式）
    E_param  ⚠️ 当前为无认证残差指示器 ||(I-VVᵀ)r||；
              认证版 Δ(μ)=||r||/β_LB 待同学 A 任务 #14 接入
              （接口已预留：certified_bound_fn 参数）
    E_aff    ✅ = 0（test_affine_matches_direct 恒等验证）

权衡曲线（#26 原定义）：r vs E_proj vs 在线耗时/加速比。
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .pod import PODBasis
from .reduced import ReducedModel, ROMSolver


def projection_error(basis: PODBasis, r: int) -> float:
    """E_proj：基截断误差（相对谱尾，Eckart-Young）。

    等于 best rank-r 重构的相对 Frobenius 误差 sqrt(Σ_{i>r}σ_i² / Σσ_i²)。
    """
    if r < 1 or r > basis.V.shape[1]:
        raise ValueError(f"r={r} 超出 [1, {basis.V.shape[1]}]")
    if r == basis.V.shape[1]:
        return 0.0
    return float(np.sqrt(1.0 - basis.energy_ratio[r - 1]))


def time_error_estimate(model: ReducedModel, params: dict, T0: np.ndarray,
                        dt: float, t_end: float) -> tuple[float, np.ndarray]:
    """E_time：Richardson 外推（向后欧拉一阶 → err(dt) ≈ ||u(dt) - u(dt/2)||）。

    返回 (相对时间误差估计, dt 解的末时刻场)。
    相对基准：dt/2 细解的涨落范数。
    """
    rom_coarse = ROMSolver(dt=dt, t_end=t_end).solve(model, params, T0)
    rom_fine = ROMSolver(dt=dt / 2, t_end=t_end).solve(model, params, T0)
    u_c = rom_coarse.snapshots[:, -1]
    u_f = rom_fine.snapshots[:, -1]
    den = np.linalg.norm(u_f - model.mean)
    est = np.linalg.norm(u_c - u_f) / den
    return float(est), u_c


def residual_indicator(model: ReducedModel, params: dict, T0: np.ndarray,
                       dt: float, t_end: float, full_ops) -> dict:
    """E_param 的无认证指示器：基空间外残差 ||(I-VVᵀ)r|| 的轨迹统计。

    注意：这不是严格上界——认证版 Δ(μ)=||r||/β_LB 待任务 #14 接入。
    """
    norms: list[float] = []

    def collect(step, a, res_out):
        norms.append(float(np.linalg.norm(res_out)))
        return None  # 不告警，只收集

    ROMSolver(dt=dt, t_end=t_end).solve(model, params, T0,
                                        monitor=collect, full_ops=full_ops)
    norms = np.array(norms)
    return {"max": float(norms.max()), "mean": float(norms.mean()),
            "final": float(norms[-1]), "certified": False}


# ----------------------------------------------------------------------
# 误差预算证书
# ----------------------------------------------------------------------
@dataclass
class BudgetCertificate:
    """单参数查询点的误差预算证书（方向②核心交付物）。"""

    params: dict
    r: int
    e_proj: float
    e_time: float
    param_indicator: dict
    e_aff: float = 0.0
    certified_bound: float | None = None     # #14 接入后填
    true_error: float | None = None          # 有全阶对照时填（调试用）

    @property
    def total_bound(self) -> float:
        """当前总上界（E_param 用认证版则含，否则为部分上界）。"""
        e_param = self.certified_bound if self.certified_bound is not None else 0.0
        return self.e_proj + self.e_time + e_param + self.e_aff

    def to_dict(self) -> dict:
        return {
            "params": self.params, "r": self.r,
            "E_proj": self.e_proj, "E_time": self.e_time,
            "E_aff": self.e_aff,
            "E_param_indicator (uncertified)": self.param_indicator,
            "certified_bound": self.certified_bound,
            "total_bound": self.total_bound,
            "true_error (debug)": self.true_error,
            "note": "E_param 认证版待任务 #14；total_bound 暂不含无认证指示器",
        }

    def summary(self) -> str:
        lines = [
            "=" * 56,
            f"误差预算证书  r={self.r}  params={self.params}",
            "-" * 56,
            f"  E_proj  (基截断, Eckart-Young) : {self.e_proj:.3e}",
            f"  E_time  (Richardson 外推)      : {self.e_time:.3e}",
            f"  E_aff   (仿射近似, 已证=0)      : {self.e_aff:.3e}",
            f"  E_param (残差指示器, 无认证)    : max {self.param_indicator['max']:.3e} [K/s]",
        ]
        if self.certified_bound is not None:
            lines.append(f"  E_param (认证上界, #14)        : {self.certified_bound:.3e}")
        lines.append(f"  总上界（当前已认证部分）        : {self.total_bound:.3e}")
        if self.true_error is not None:
            eta = (self.total_bound / self.true_error) if self.true_error > 0 else float("inf")
            lines.append(f"  [调试] 真实误差 = {self.true_error:.3e}, "
                         f"effectivity η = {eta:.1f}")
        lines.append("=" * 56)
        return "\n".join(lines)


def budget_certificate(model: ReducedModel, basis: PODBasis, params: dict,
                       T0: np.ndarray, dt: float, t_end: float, full_ops,
                       certified_bound_fn: Callable | None = None,
                       true_field: np.ndarray | None = None) -> BudgetCertificate:
    """生成单点误差预算证书。

    参数:
        certified_bound_fn: 认证误差界函数 fn(model, params) -> float，
                            由同学 A 任务 #14 提供后接入
        true_field: 调试用的全阶对照场（末时刻）
    """
    e_proj = projection_error(basis, model.r)
    e_time, u_rom = time_error_estimate(model, params, T0, dt, t_end)
    indicator = residual_indicator(model, params, T0, dt, t_end, full_ops)
    cert = certified_bound_fn(model, params) if certified_bound_fn else None

    true_err = None
    if true_field is not None:
        den = np.linalg.norm(true_field - model.mean)
        true_err = float(np.linalg.norm(u_rom - true_field) / den)

    return BudgetCertificate(params=params, r=model.r, e_proj=e_proj,
                             e_time=e_time, param_indicator=indicator,
                             certified_bound=cert, true_error=true_err)


# ----------------------------------------------------------------------
# 权衡曲线（任务 #26 原定义：r vs 预估误差 vs 加速比）
# ----------------------------------------------------------------------
@dataclass
class TradeoffCurve:
    r_list: list[int]
    e_proj: list[float]
    rom_time_ms: list[float]
    true_error: list[float] | None = field(default=None)

    def plot(self, save_path: str, fom_time_s: float | None = None) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        ax = axes[0]
        ax.semilogy(self.r_list, self.e_proj, "o-", label="E_proj (a priori)")
        if self.true_error:
            ax.semilogy(self.r_list, self.true_error, "s--",
                        label="true error (FOM ref)")
        ax.set_xlabel("ROM dimension r")
        ax.set_ylabel("relative error")
        ax.set_title("Accuracy vs dimension")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(self.r_list, self.rom_time_ms, "o-", color="tab:green")
        ax.set_xlabel("ROM dimension r")
        ax.set_ylabel("online solve time [ms]")
        if fom_time_s:
            ax2 = ax.twinx()
            ax2.plot(self.r_list,
                     [fom_time_s * 1000 / t for t in self.rom_time_ms],
                     "s--", color="tab:red", label="speedup")
            ax2.set_ylabel("speedup vs FOM", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title("Cost vs dimension")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)


def tradeoff_curve(basis: PODBasis, ops, builder, params: dict,
                   T0: np.ndarray, dt: float, t_end: float,
                   r_list: list[int] | None = None,
                   true_snapshots_fn: Callable[[], np.ndarray] | None = None,
                   ) -> TradeoffCurve:
    """扫描 ROM 维度，输出精度-成本权衡数据。"""
    import time

    if r_list is None:
        r_list = list(range(1, min(8, basis.V.shape[1]) + 1))
    curve = TradeoffCurve(r_list=r_list, e_proj=[], rom_time_ms=[],
                          true_error=[] if true_snapshots_fn else None)
    S_true = true_snapshots_fn() if true_snapshots_fn else None
    den = np.linalg.norm(S_true - basis.mean[:, None]) if S_true is not None else None

    for r in r_list:
        model = builder.build(basis, ops, r)
        curve.e_proj.append(projection_error(basis, r))
        t0 = time.perf_counter()
        rom = ROMSolver(dt=dt, t_end=t_end).solve(model, params, T0)
        curve.rom_time_ms.append((time.perf_counter() - t0) * 1000)
        if S_true is not None:
            curve.true_error.append(
                float(np.linalg.norm(rom.snapshots - S_true) / den))
    return curve
