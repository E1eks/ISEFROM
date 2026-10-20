"""
认证后验误差界插件接口（任务 #14 挂载点 · 接口已冻结 2026-10-20）
================================================================
ISEF 2028 ROM 项目 ｜ 接口冻结：同学 B ｜ 严格实现：同学 A（任务 #14）

这份模块定义 A 的认证误差界 Δ(μ)=‖r‖/β_LB 插入框架的**唯一入口**。
A 只需要：
    1. 写一个满足 CertifiedBoundFn 签名的函数
       （残差轨迹用 residual_out_trajectory 现成拿到，β_LB 用 SCM 或解析下界）
    2. 在 ESTIMATOR_REGISTRY 注册一行
之后流水线（#22）、误差预算证书（#26）、贪心采样（#10）全部自动生效。

数学契约（memo_A_certified_error_bound.md §1-2）：
    ‖T(μ) - V a(μ)‖ ≤ Δ(μ)  必须恒成立  ⇔  effectivity η = Δ/真实误差 ≥ 1
    η < 1 即实现有 bug（自我诊断器，failure log #02 的教训）

约定：
    - delta 为**相对**误差界，参考范数 = 末时刻场相对快照均值的涨落范数
      ‖u(·,t_end) - mean‖₂（与 pipeline 的 rel_error、budget 的 true_error 同约定）
    - certified=True 仅当 Δ 是严格上界；占位/启发式实现必须 False，
      未认证的 delta 不会计入 BudgetCertificate.total_bound
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse.linalg as spla

from .operators import AffineOperators
from .reduced import ReducedModel, ROMResult, ROMSolver, solve_steady


@dataclass
class BoundContext:
    """调用估计器时框架提供的全部上下文。"""

    full_ops: AffineOperators       # 全阶仿射算子（残差/β 计算用）
    kind: str                       # "time"（抛物）| "steady"（椭圆）
    T0: np.ndarray | None = None    # 初始全维场（抛物必给）
    dt: float | None = None         # 时间步长（抛物必给）
    t_end: float | None = None      # 终止时刻（抛物必给）


@dataclass
class BoundResult:
    """估计器返回值（A 的实现必须填全前四项）。"""

    delta: float                    # Δ(μ)，相对误差界（参考范数见模块 docstring）
    beta_lb: float                  # 稳定性常数下界 β_LB(μ)
    residual_norm: float            # 残差范数（抛物：max_t‖r_out‖；椭圆：‖r‖）
    certified: bool                 # True=严格上界；False=占位/启发式
    method: str                     # 实现标识（写进报告，如 "scm" / "placeholder"）
    per_step: np.ndarray | None = field(default=None, repr=False)  # 可选轨迹

    def to_dict(self) -> dict:
        return {
            "delta": self.delta, "beta_lb": self.beta_lb,
            "residual_norm": self.residual_norm,
            "certified": self.certified, "method": self.method,
            "n_steps": None if self.per_step is None else len(self.per_step),
        }


# 冻结签名：fn(model, params, ctx) -> BoundResult
CertifiedBoundFn = Callable[[ReducedModel, dict, BoundContext], BoundResult]


# ----------------------------------------------------------------------
# 框架侧工具（A 直接复用，不要重造）
# ----------------------------------------------------------------------
def residual_out_trajectory(model: ReducedModel, params: dict,
                            T0: np.ndarray, dt: float, t_end: float,
                            full_ops: AffineOperators,
                            ) -> tuple[np.ndarray, ROMResult]:
    """抛物型：每步基空间外残差范数 ‖(I-VVᵀ)r‖₂ 的轨迹（数学笔记 §4.3）。

    返回 (范数轨迹 (n_t-1,), ROM 结果)。末时刻场可取 rom.snapshots[:, -1]。
    """
    norms: list[float] = []

    def collect(step, a, res_out):
        norms.append(float(np.linalg.norm(res_out)))
        return None

    rom = ROMSolver(dt=dt, t_end=t_end).solve(model, params, T0,
                                              monitor=collect,
                                              full_ops=full_ops)
    return np.array(norms), rom


def steady_residual(model: ReducedModel, params: dict,
                    full_ops: AffineOperators,
                    ) -> tuple[float, np.ndarray]:
    """椭圆型：稳态残差 ‖A u_r + b‖₂（方程 A u + b = 0，架构 §3.4 约定）。

    返回 (残差范数, 降阶解 u_r)。
    """
    u_rom = solve_steady(model, params)
    A_full, b_full = full_ops.assemble(params)
    res = A_full @ u_rom + b_full
    return float(np.linalg.norm(res)), u_rom


def spectral_beta_lb(ops: AffineOperators, params: dict) -> float:
    """数值谱下界 β = λ_min⁺(K_sym)，K_sym = (-A + (-A)ᵀ)/2（占位用）。

    细节（placeholder 够用，严格版是 A 的任务）：
    - 用**对称部分**——coercivity 常数的定义即 λ_min(对称部分)，
      且本框架算例因 Dirichlet 行清零而非对称，直接 eigvalsh 会读错三角
    - 取**最小正**特征值——Dirichlet 清零行与自由节点的耦合会给对称部分
      引入一个伪负特征值（数值实验：-0.031 vs 真 β≈1.8e-4），属边界
      约束的数学人造物，非物理不稳定性；严格处理应在约束空间上定义
      coercivity（留给 A 的 #14 推导）
    - 对**离散** FOM 系统是精确常数；对连续问题非严格（离散 β ≥ 连续 β）。
      A 的 SCM 版将提供廉价且连续意义下严格的下界
    """
    A_full, _ = ops.assemble(params)
    K = (-A_full).tocsc()
    n = K.shape[0]
    Ksym = (K + K.T) * 0.5
    if n <= 1024:
        lam_all = np.linalg.eigvalsh(Ksym.toarray())
    else:
        # 两端各取若干，覆盖最小正特征值
        lam_all = np.concatenate([
            spla.eigsh(Ksym, k=4, sigma=0.0, which="LM",
                       return_eigenvectors=False),
            spla.eigsh(Ksym, k=1, which="LA", return_eigenvectors=False)])
    lam_max = float(lam_all.max())
    positive = lam_all[lam_all > lam_max * 1e-12]
    if positive.size == 0:
        raise ValueError("对称部分无正特征值——-A 不正定，谱占位下界不适用")
    return float(positive.min())


# ----------------------------------------------------------------------
# 占位实现（certified=False，仅验证管线通畅；A 的 #14 严格版替换之）
# ----------------------------------------------------------------------
def placeholder_bound(model: ReducedModel, params: dict,
                      ctx: BoundContext) -> BoundResult:
    """Δ̃ = ‖r‖/β_spectral——公式简化的占位界，**不是严格上界**。

    抛物：max_t‖r_out(t)‖/β（严格界需时间积分 + Gronwall，memo §1）。
    椭圆：‖r‖/β 对离散系统其实严格（e=-A⁻¹r，‖A⁻¹‖=1/λ_min(-A)），
    但仍标 certified=False——是否采纳由 A review 推导后决定（分工铁律）。
    """
    beta = spectral_beta_lb(ctx.full_ops, params)
    if ctx.kind == "steady":
        res_norm, u_rom = steady_residual(model, params, ctx.full_ops)
        per_step = None
    else:
        if ctx.T0 is None or ctx.dt is None or ctx.t_end is None:
            raise ValueError("抛物型 BoundContext 缺 T0/dt/t_end")
        norms, rom = residual_out_trajectory(model, params, ctx.T0,
                                             ctx.dt, ctx.t_end, ctx.full_ops)
        res_norm = float(norms.max())
        per_step = norms / beta
        u_rom = rom.snapshots[:, -1]
    ref = np.linalg.norm(u_rom - model.mean)
    delta = float(res_norm / beta / max(ref, 1e-14))
    return BoundResult(delta=delta, beta_lb=beta, residual_norm=res_norm,
                       certified=False, method="placeholder", per_step=per_step)


# 估计器注册表：A 的严格实现注册 "scm" 等键后，配置里写名字即用
ESTIMATOR_REGISTRY: dict[str, CertifiedBoundFn] = {
    "placeholder": placeholder_bound,
}
