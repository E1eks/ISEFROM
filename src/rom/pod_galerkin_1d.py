"""
POD-Galerkin 最小 Demo（任务 #4）
==================================
ISEF 2028 POD-Galerkin ROM 项目
负责人：同学 B（工程实现）｜数学推导需同学 A 共同 review

目的：
  1. 复现 POD-Galerkin 离线-在线范式最小流程：
     快照 → SVD → POD 基 → Galerkin 投影 → 降阶 ODE → 重构
  2. 【重点】故意制造失效并记录：
     - 实验 F1: 截断模态过少
     - 实验 F2: 参数外推（q_gen 超出快照工况）
     - 实验 F3: 参数外推（h 超出快照工况，边界层结构改变）

数学框架（同学 A 需能手推的部分）：
  全阶系统（连续形式，M=I）:   dT/dt = A T + b
   lifting + 展开:            T(t) ≈ T_mean + V a(t),  V ∈ R^{n×r}
  Galerkin 投影（V^T 左乘）:   da/dt = A_r a + f_r
    其中  A_r = V^T A V,  f_r = V^T (A T_mean + b)
  POD 基物理含义：V 的列是快照协方差矩阵的主特征向量，
    即按"平均能量"排序的最优正交模态；奇异值平方 = 该模态携带的能量。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.fom.heat_1d import Heat1DParams, FOMResult, solve_fom  # noqa: E402

BASELINE_NPY = Path(__file__).resolve().parents[2] / "snapshots_1d_baseline.npy"


# ----------------------------------------------------------------------
# 连续时间全阶算子（与 heat_1d.py 的隐式离散严格一致，仅改写为 dT/dt 形式）
# ----------------------------------------------------------------------
def build_continuous_operators(p: Heat1DParams) -> tuple[sp.csr_matrix, np.ndarray]:
    """构造 dT/dt = A T + b 的稀疏 A 与常向量 b。"""
    n = p.n_x
    dx = p.length / (n - 1)
    c = p.alpha / dx**2           # alpha/dx^2
    s = p.q_gen / (p.rho * p.c_p)  # 源项 [K/s]
    bi = p.h_right * dx / p.k      # 右端 Biot 数

    main = np.full(n, -2.0 * c)
    off = np.full(n - 1, c)
    A = sp.diags([off, main, off], [-1, 0, 1], format="lil")
    b = np.full(n, s)

    # 左端 Dirichlet: dT/dt = 0（行清零，边界值由 lifting/mean 携带）
    A[0, 0] = 0.0
    A[0, 1] = 0.0
    b[0] = 0.0

    # 右端 Robin（与隐式离散同构）:
    #   dT[-1]/dt = c * (T[-2] - (1+Bi) T[-1]) + c * Bi * t_cool + s
    A[-1, -1] = -(1.0 + bi) * c
    A[-1, -2] = c
    b[-1] = s + c * bi * p.t_cool

    return A.tocsr(), b


# ----------------------------------------------------------------------
# POD 分解
# ----------------------------------------------------------------------
@dataclass
class PODBasis:
    mean: np.ndarray        # 快照均值 (n_x,)
    V: np.ndarray           # POD 基 (n_x, r)
    sigma: np.ndarray       # 全部奇异值 (min(n_x, n_t),)
    energy_ratio: np.ndarray  # 累积能量比

    @staticmethod
    def from_snapshots(S: np.ndarray) -> "PODBasis":
        mean = S.mean(axis=1)
        U, sigma, _ = np.linalg.svd(S - mean[:, None], full_matrices=False)
        energy = sigma**2
        return PODBasis(mean=mean, V=U, sigma=sigma,
                        energy_ratio=np.cumsum(energy) / energy.sum())

    def rank_for_energy(self, threshold: float) -> int:
        """达到给定累积能量所需的最小模态数。"""
        return int(np.searchsorted(self.energy_ratio, threshold) + 1)


# ----------------------------------------------------------------------
# Galerkin 投影降阶求解器
# ----------------------------------------------------------------------
def solve_rom(basis: PODBasis, r: int, p: Heat1DParams) -> np.ndarray:
    """在 r 维 POD 基上积分降阶系统（向后欧拉，与 FOM 同一时间步）。"""
    A, b = build_continuous_operators(p)
    V = basis.V[:, :r]

    A_r = np.asarray(V.T @ A @ V)          # (r, r)
    f_r = V.T @ (A @ basis.mean + b)       # (r,)

    n_t = p.n_t
    a = np.empty((r, n_t))
    # 初始条件投影
    a[:, 0] = V.T @ (np.full(p.n_x, p.t_init) - basis.mean)

    lhs = np.eye(r) - p.dt * A_r
    lu = spla.factorized(sp.csc_matrix(lhs))
    for n in range(1, n_t):
        a[:, n] = lu(a[:, n - 1] + p.dt * f_r)

    return basis.mean[:, None] + V @ a      # 重构全维温度场 (n_x, n_t)


def relative_error(S_rom: np.ndarray, S_fom: np.ndarray, mean: np.ndarray) -> float:
    """相对 Frobenius 误差（相对温度涨落，避免被 298 K  baseline 稀释）。"""
    num = np.linalg.norm(S_rom - S_fom)
    den = np.linalg.norm(S_fom - mean[:, None])
    return num / den


# ----------------------------------------------------------------------
# 实验
# ----------------------------------------------------------------------
def run_experiments() -> dict:
    log = {}

    # --- 离线阶段：基线工况快照 + POD ---
    p_base = Heat1DParams()
    fom_base = solve_fom(p_base)
    basis = PODBasis.from_snapshots(fom_base.snapshots)

    r_999 = basis.rank_for_energy(0.999)
    r_9999 = basis.rank_for_energy(0.9999)
    print(f"[离线] 快照矩阵 {fom_base.snapshots.shape}，POD 能量谱：")
    print(f"       sigma[:8] = {np.array2string(basis.sigma[:8], precision=3)}")
    print(f"       99.9% 能量需 r={r_999}，99.99% 需 r={r_9999}（全维 {p_base.n_x}）")

    # --- 在线验证：同参数内插（ROM 应表现良好）---
    rom_interp = solve_rom(basis, r_999, p_base)
    err_interp = relative_error(rom_interp, fom_base.snapshots, basis.mean)
    print(f"\n[实验·内插] r={r_999} 同参数重构: 相对误差 = {err_interp:.3e}")
    log["interp"] = (r_999, err_interp)

    # --- F1: 模态截断过少 ---
    rom_r1 = solve_rom(basis, 1, p_base)
    err_r1 = relative_error(rom_r1, fom_base.snapshots, basis.mean)
    print(f"[实验 F1 ] r=1 强行截断:        相对误差 = {err_r1:.3e}  ← 失败案例①")
    log["F1"] = (1, err_r1)

    # --- F2: 参数外推（产热率 q_gen ×2.4，模拟 3C 高倍率工况）---
    p_hot = Heat1DParams(q_gen=1.2e5)
    fom_hot = solve_fom(p_hot)
    rom_hot = solve_rom(basis, r_999, p_hot)
    err_hot = relative_error(rom_hot, fom_hot.snapshots, basis.mean)
    print(f"[实验 F2 ] q_gen 5e4→1.2e5 外推: 相对误差 = {err_hot:.3e}  ← 失败案例②")
    log["F2"] = (r_999, err_hot)

    # --- F3: 参数外推（换热系数 h ×0.3，边界层结构改变）---
    p_conv = Heat1DParams(h_right=300.0)
    fom_conv = solve_fom(p_conv)
    rom_conv = solve_rom(basis, r_999, p_conv)
    err_conv = relative_error(rom_conv, fom_conv.snapshots, basis.mean)
    print(f"[实验 F3 ] h 1000→300 外推:      相对误差 = {err_conv:.3e}  ← 失败案例③")
    log["F3"] = (r_999, err_conv)

    return dict(basis=basis, fom_base=fom_base,
                cases=dict(interp=rom_interp, F1=rom_r1, F2=(rom_hot, fom_hot),
                           F3=(rom_conv, fom_conv)), log=log)


def plot_experiments(res: dict, save_path: str = "rom_demo.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    basis: PODBasis = res["basis"]
    fom_base: FOMResult = res["fom_base"]
    x_mm = fom_base.x * 1000

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) 奇异值谱
    ax = axes[0, 0]
    ax.semilogy(basis.sigma[:30], "o-", ms=4)
    ax.set_xlabel("mode index")
    ax.set_ylabel("singular value")
    ax.set_title("(a) POD energy spectrum (SVD)")
    ax.grid(alpha=0.3)

    # (b) 前 4 个 POD 模态
    ax = axes[0, 1]
    for i in range(4):
        ax.plot(x_mm, basis.V[:, i], label=f"mode {i + 1}")
    ax.set_xlabel("x [mm]")
    ax.set_title("(b) Leading POD modes")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) 内插验证：末时刻剖面对比
    ax = axes[1, 0]
    ax.plot(x_mm, fom_base.snapshots[:, -1], "k-", lw=2, label="FOM")
    ax.plot(x_mm, res["cases"]["interp"][:, -1], "r--", label=f"ROM r={res['log']['interp'][0]}")
    ax.plot(x_mm, res["cases"]["F1"][:, -1], "b:", label="ROM r=1 (F1)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("T [K]")
    ax.set_title("(c) Interpolation check @ t=600 s")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) 外推失效：末时刻剖面对比
    ax = axes[1, 1]
    rom_hot, fom_hot = res["cases"]["F2"]
    rom_conv, fom_conv = res["cases"]["F3"]
    ax.plot(x_mm, fom_hot.snapshots[:, -1], "k-", lw=2, label="FOM q_gen×2.4")
    ax.plot(x_mm, rom_hot[:, -1], "r--", label="ROM q_gen×2.4 (F2)")
    ax.plot(x_mm, fom_conv.snapshots[:, -1], "k-", lw=2, alpha=0.4, label="FOM h×0.3")
    ax.plot(x_mm, rom_conv[:, -1], "b--", label="ROM h×0.3 (F3)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("T [K]")
    ax.set_title("(d) Extrapolation failures @ t=600 s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\n[输出] 对比图已保存: {save_path}")


def write_failure_log(log: dict, path: str = "logs/failure_log_01.md") -> None:
    """失败案例数据库第一条记录（任务 #18 的种子）。"""
    text = f"""# ROM 失败案例日志 #01 —— POD-Galerkin 最小 Demo（任务 #4）

> 日期：2026-09-11 ｜ 记录人：同学 B
> 算例：一维热传导（抛物型），基线工况 k=1.5, h=1000, q_gen=5e4

## 实验结果汇总

| 实验 | 设置 | 相对误差 | 结论 |
|------|------|----------|------|
| 内插验证 | r={log['interp'][0]}（99.9% 能量），同参数 | {log['interp'][1]:.2e} | ROM 内插精度良好 |
| F1 截断过少 | r=1 | {log['F1'][1]:.2e} | 能量截断不足 → 空间结构失真 |
| F2 源项外推 | q_gen 5e4→1.2e5（×2.4） | {log['F2'][1]:.2e} | 幅度失真 |
| F3 边界外推 | h 1000→300（×0.3） | {log['F3'][1]:.2e} | 边界层结构未被快照覆盖 → 右端温度场整体失真 |

## 观察与教训（科研日志）

1. **纯扩散问题对 POD-Galerkin 相当宽容**：内插工况下 {log['interp'][0]} 个模态即可达到
   {log['interp'][1]:.1%} 误差。ROM 真正的失稳重灾区在对流主导/强非线性问题，
   本项目后续反应-扩散算例（任务 #17）才是试金石。
2. **失效是"安静的"**：F2/F3 中 ROM 给出光滑、看似合理的温度场，但数值错误。
   没有任何内置机制提醒用户结果不可信 —— 这正是本项目核心创新②
   （运行时残差误差监控与告警）的存在意义。
3. **基的外推能力受快照库覆盖范围限制**：F3 改变了边界层结构，
   基线快照中从未出现过的空间模态无法被基表示 —— 这是核心创新①
   （自适应快照采样）要解决的问题。
4. 下一步：与同学 A 讨论残差估计量 ||b - (I - VV^T)(A T_rom)|| 能否在
   不运行全阶的情况下提前捕捉 F2/F3 类失效。

## 复现方式

    python src/rom/pod_galerkin_1d.py
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    print(f"[输出] 失败日志已保存: {path}")


if __name__ == "__main__":
    results = run_experiments()
    plot_experiments(results)
    write_failure_log(results["log"])
