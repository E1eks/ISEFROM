"""
一维热传导全阶仿真 Demo（FOM, Full-Order Model）
================================================
ISEF 2028 POD-Galerkin ROM 项目 · 任务 #3
负责人：同学 B（工程/应用主线）

控制方程（抛物型 PDE，含体积内热源，与电池热模型同一体系）:

    rho * cp * dT/dt = k * d2T/dx2 + q_gen

离散格式（与技术方案 §5.1 一致）:
    - 空间: 二阶中心差分，均匀网格
    - 时间: 一阶隐式（向后欧拉），无条件稳定
    - 边界: 左端 Dirichlet 恒温 / 右端 Robin 对流换热

输出:
    快照矩阵 S (n_x, n_t)，每列是一个时刻的温度场。
    该格式即后续任务 #4（POD-Galerkin 最小 Demo）与
    任务 #9（快照管理器）的标准输入格式。
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class Heat1DParams:
    """一维热传导全阶模型参数。"""

    # --- 几何与网格 ---
    length: float = 0.2        # 计算域长度 L [m]（≈ 电池包截面宽度 200 mm）
    n_x: int = 101             # 空间节点数

    # --- 材料属性（缺省值取电池单元等效值，见技术方案 §3.2）---
    rho: float = 2000.0        # 密度 [kg/m^3]
    c_p: float = 1000.0        # 比热容 [J/(kg·K)]
    k: float = 1.5             # 导热系数 [W/(m·K)]（扫描参数 k_cell: 1.0~3.0）

    # --- 内热源（电池放电产热，简化 Bernardi 模型 §2.2）---
    q_gen: float = 5.0e4       # 体积产热率 [W/m^3]（随放电倍率变化）

    # --- 边界条件 ---
    t_left: float = 298.0      # 左端恒温 [K]（≈ 冷板强制冷却）
    h_right: float = 1000.0    # 右端对流换热系数 [W/(m^2·K)]（扫描参数 h: 500~3000）
    t_cool: float = 298.0      # 右端冷却液温度 [K]

    # --- 时间与初值 ---
    t_init: float = 298.0      # 初始温度 [K]
    dt: float = 1.0            # 时间步长 [s]
    t_end: float = 600.0       # 总仿真时长 [s]（10 min 恒流放电）

    @property
    def alpha(self) -> float:
        """热扩散率 alpha = k / (rho * cp) [m^2/s]。"""
        return self.k / (self.rho * self.c_p)

    @property
    def n_t(self) -> int:
        """时间步数。"""
        return int(round(self.t_end / self.dt)) + 1


@dataclass
class FOMResult:
    """全阶仿真结果。"""

    x: np.ndarray              # 空间网格 (n_x,)
    t: np.ndarray              # 时间网格 (n_t,)
    snapshots: np.ndarray      # 快照矩阵 (n_x, n_t)，每列一个时刻
    params: Heat1DParams = field(default=None)

    @property
    def t_max_history(self) -> np.ndarray:
        """每个时刻的全域最高温度 [K]。"""
        return self.snapshots.max(axis=0)


def build_system_matrices(p: Heat1DParams) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """组装向后欧拉离散的线性系统。

    离散后方程写为:  M * T^{n+1} = M * T^n + dt * (q - K_d * T^{n+1})
    即:              A * T^{n+1} = M * T^n + dt * q

    返回:
        A: (I + dt*alpha/dx^2 * L) 系统矩阵 (n_x, n_x)
        M: 单位阵（质量阵，常系数下即 I）
        b_const: 常数右端项（内热源 + 边界贡献） (n_x,)
    """
    n = p.n_x
    dx = p.length / (n - 1)
    r = p.alpha * p.dt / dx**2  # 无量纲网格傅里叶数（隐式格式无稳定性限制）

    # 二阶中心差分 Laplacian（三对角）
    main = np.full(n, 1.0 + 2.0 * r)
    off = np.full(n - 1, -r)
    A = sp.diags([off, main, off], [-1, 0, 1], format="lil")

    # 常数右端：内热源项 dt * q_gen / (rho * cp)
    b_const = np.full(n, p.dt * p.q_gen / (p.rho * p.c_p))

    # --- 左端 Dirichlet: T[0] = t_left ---
    A[0, 0] = 1.0
    A[0, 1] = 0.0
    b_const[0] = p.t_left

    # --- 右端 Robin 对流: -k dT/dx = h (T - t_cool) ---
    # 一阶后向差分: -k (T[-1] - T[-2]) / dx = h (T[-1] - t_cool)
    # 整理进向后欧拉系统，引入无量纲数 Bi = h * dx / k
    bi = p.h_right * dx / p.k
    A[-1, -1] = 1.0 + r + r * bi
    A[-1, -2] = -r
    b_const[-1] = p.dt * p.q_gen / (p.rho * p.c_p) + r * bi * p.t_cool

    return A.tocsr(), sp.eye(n, format="csr"), b_const


def solve_fom(p: Heat1DParams) -> FOMResult:
    """运行全阶仿真，返回完整快照矩阵。"""
    x = np.linspace(0.0, p.length, p.n_x)
    t = np.linspace(0.0, p.t_end, p.n_t)

    A, _, b_const = build_system_matrices(p)
    # 矩阵常系数 → 只做一次稀疏 LU 分解，全部时间步复用（全阶加速的关键技巧）
    lu = spla.factorized(A.tocsc())

    snapshots = np.empty((p.n_x, p.n_t))
    T = np.full(p.n_x, p.t_init)
    T[0] = p.t_left
    snapshots[:, 0] = T

    for n in range(1, p.n_t):
        rhs = T + b_const
        rhs[0] = p.t_left  # Dirichlet 边界不随上一时刻累积
        T = lu(rhs)
        snapshots[:, n] = T

    return FOMResult(x=x, t=t, snapshots=snapshots, params=p)


def sanity_checks(result: FOMResult) -> None:
    """物理合理性检查：能量来源单调性、稳态上界、边界条件。"""
    p = result.params
    s = result.snapshots

    # 1) 有内热源 + 冷却边界 → 温度应 ≥ 冷却液温度
    assert s.min() >= p.t_cool - 1e-6, f"温度低于冷却液温度: {s.min():.2f} K"

    # 2) 正产热下最高温度应单调不减并趋于稳态
    tmax = result.t_max_history
    assert np.all(np.diff(tmax) >= -1e-9), "最高温度出现非物理下降"
    last = tmax[-min(50, len(tmax)):]
    drift = abs(last[-1] - last[0]) / max(abs(last[-1]), 1e-12)
    print(f"  [检查] 末段 50 步最高温度相对漂移 = {drift:.2e}（趋稳态应 << 1）")

    # 3) 粗估稳态最高温度量级：均匀源 + 一端恒温的解析上界
    #    T_max ~ t_left + q_gen * L^2 / (2k)（两端冷却时更小）
    t_max_est = p.t_left + p.q_gen * p.length**2 / (2.0 * p.k)
    print(f"  [检查] 稳态最高温度数值 = {tmax[-1]:.1f} K，"
          f"解析量级估计 ≈ {t_max_est:.1f} K（右端散热故应低于该上界）")
    assert tmax[-1] < t_max_est * 1.05, "稳态温度超过解析上界，物理不合理"

    # 4) 边界条件验证
    assert abs(s[0, -1] - p.t_left) < 1e-8, "左端 Dirichlet 边界未满足"

    print("  [检查] 全部物理合理性断言通过 ✔")


def plot_result(result: FOMResult, save_path: str = "heat_1d_demo.png") -> None:
    """绘制温度场演化图并保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图：不同时刻的温度分布曲线
    ax = axes[0]
    n_curve = 6
    idx = np.linspace(0, result.snapshots.shape[1] - 1, n_curve, dtype=int)
    for i in idx:
        ax.plot(result.x * 1000, result.snapshots[:, i],
                label=f"t = {result.t[i]:.0f} s")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("T [K]")
    ax.set_title("Temperature profiles (1D heat conduction, FOM)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 右图：最高温度随时间演化
    ax = axes[1]
    ax.plot(result.t, result.t_max_history)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("max T [K]")
    ax.set_title("Peak temperature evolution")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  [输出] 图已保存: {save_path}")


if __name__ == "__main__":
    params = Heat1DParams()
    print("=" * 60)
    print("一维热传导全阶仿真 Demo（任务 #3）")
    print("=" * 60)
    print(f"  网格: n_x={params.n_x}, dx={params.length / (params.n_x - 1) * 1000:.2f} mm")
    print(f"  时间: dt={params.dt} s, t_end={params.t_end} s, 共 {params.n_t} 步")
    print(f"  热扩散率 alpha = {params.alpha:.3e} m^2/s")
    print(f"  网格傅里叶数 Fo = {params.alpha * params.dt / (params.length / (params.n_x - 1))**2:.3f}"
          "（隐式格式，无稳定性限制）")

    result = solve_fom(params)
    print(f"\n  快照矩阵形状: {result.snapshots.shape}  (n_x × n_t)")
    print(f"  初始温度: {result.snapshots[:, 0].mean():.1f} K")
    print(f"  末时刻最高温度: {result.t_max_history[-1]:.2f} K "
          f"(≈ {result.t_max_history[-1] - 273.15:.1f} °C)")
    print(f"  末时刻最低温度: {result.snapshots[:, -1].min():.2f} K")

    print("\n物理合理性检查:")
    sanity_checks(result)

    np.save("snapshots_1d_baseline.npy", result.snapshots)
    print("\n  [输出] 快照矩阵已保存: snapshots_1d_baseline.npy"
          "（任务 #4 POD 分解的直接输入）")

    plot_result(result)
