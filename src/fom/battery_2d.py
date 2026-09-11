"""
电动车电池包 2D 热管理模型（FOM + 仿射算子）
==============================================
ISEF 2028 ROM 项目 · 任务 #11【杀手级应用·核心案例】
负责人：同学 B ｜ 几何与材料参数见 ISEF_杀手级应用_电池热管理技术方案.md §3

控制方程（抛物型，含内热源，多材料）:

    rho(x,y) * cp(x,y) * dT/dt = div( k(x,y) grad T ) + q_gen(x,y)

区域划分（200 mm × 120 mm 截面，网格 100 × 60）:

    y=120 ┌──────────────────────────────┐ ← 上冷却流道（Robin, h 扫描）
          │      填充/TIM (k=1.0)        │
          │  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐   │
          │  │█│ │█│ │█│ │█│ │█│ │█│   │ 6 颗电芯 (k_cell 扫描, q_gen 产热)
          │  └─┘ └─┘ └─┘ └─┘ └─┘ └─┘   │
          ├──────────────────────────────┤ 铝板 (k=200, 厚 5 mm)
          │      填充/TIM (k=1.0)        │
    y=0   └──────────────────────────────┘ ← 下冷却流道（Robin, h 扫描）
     左右边界：自然对流（h_amb=8, t_amb=298 K，固定）

仿射分解（对 mu = {k_cell, h, q_gen} 严格仿射）:
  面导热系数取两端节点 k 的算术平均 → 系数对 k_cell 线性；
  A(mu) = k_cell * A_cell + A_fixed + h * A_conv
  b(mu) = q_gen * b_q + b_amb + h * b_conv
  其中 A_xxx 均已左乘 M^{-1}（M = diag(rho*cp)，常数对角阵）。

注：POD 内积当前用 L2 非加权内积；多材料下严格的 M-加权内积
列为已知局限（论文 §25 素材）。
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..rom.operators import AffineOperators

# 本算例可扫描参数键（C-rate 经 qgen_from_c_rate 换算为 q_gen）
PARAM_KEYS = ["k_cell", "h", "q_gen"]


@dataclass
class BatteryParams:
    """2D 电池包热模型参数。尺寸单位 SI（m, s, K, W）。"""

    # --- 几何（技术方案 §3.1，电芯宽度微调为 28 mm 使 6 颗+间距正好 198 mm）---
    lx: float = 0.200           # 总宽 [m]
    ly: float = 0.120           # 总高 [m]
    nx: int = 100               # x 方向节点数
    ny: int = 60                # y 方向节点数
    n_cells: int = 6            # 电芯数量
    cell_w: float = 0.028       # 电芯宽 [m]
    cell_h: float = 0.100       # 电芯高 [m]
    cell_gap: float = 0.006     # 电芯间距 [m]
    cell_y0: float = 0.010      # 电芯底部 y 坐标 [m]
    plate_t: float = 0.005      # 铝板厚度 [m]（y ∈ [0.005, 0.010]）

    # --- 材料属性（技术方案 §3.2）---
    rho_cell: float = 2000.0
    cp_cell: float = 1000.0
    rho_plate: float = 2700.0
    cp_plate: float = 900.0
    k_plate: float = 200.0
    rho_fill: float = 1200.0    # 填充/TIM 等效
    cp_fill: float = 1200.0
    k_fill: float = 1.0

    # --- 扫描参数（技术方案 §4.1）---
    k_cell: float = 1.5         # 电芯导热系数 [1.0, 3.0]
    h: float = 1000.0           # 冷却液对流换热系数 [500, 3000]
    q_gen: float = 5.0e4        # 体积产热率（由 C-rate 换算）

    # --- 固定边界 ---
    h_amb: float = 8.0          # 左右自然对流
    t_amb: float = 298.0
    t_cool: float = 298.0       # 冷却液温度

    # --- 时间与初值 ---
    t_init: float = 298.0
    dt: float = 1.0
    t_end: float = 600.0

    @property
    def n_t(self) -> int:
        return int(round(self.t_end / self.dt)) + 1

    @property
    def n_nodes(self) -> int:
        return self.nx * self.ny


def qgen_from_c_rate(c_rate: float, q_1c: float = 3.0e4) -> float:
    """简化 Bernardi 产热模型（技术方案 §2.2）：
    焦耳热主导 q_gen ∝ I² ∝ C²；熵热项忽略。
    q_1c 缺省 3e4 W/m^3（18650 量级文献值）。"""
    return q_1c * c_rate**2


# ----------------------------------------------------------------------
# 几何与区域
# ----------------------------------------------------------------------
# 区域编码
FILL, CELL, PLATE = 0, 1, 2


def build_region_grid(p: BatteryParams) -> np.ndarray:
    """返回 (ny, nx) 区域编码数组。"""
    x = np.linspace(0, p.lx, p.nx)
    y = np.linspace(0, p.ly, p.ny)
    X, Y = np.meshgrid(x, y)
    region = np.full((p.ny, p.nx), FILL, dtype=int)

    # 铝板条
    region[(Y >= p.cell_y0 - p.plate_t) & (Y < p.cell_y0)] = PLATE

    # 电芯（居中布局）
    total_w = p.n_cells * p.cell_w + (p.n_cells - 1) * p.cell_gap
    x0 = (p.lx - total_w) / 2.0
    for c in range(p.n_cells):
        xa = x0 + c * (p.cell_w + p.cell_gap)
        xb = xa + p.cell_w
        mask = ((X >= xa) & (X < xb)
                & (Y >= p.cell_y0) & (Y < p.cell_y0 + p.cell_h))
        region[mask] = CELL
    return region


def build_material_fields(p: BatteryParams, region: np.ndarray):
    """返回节点级材料场：k_fixed（非电芯部分的 k）、rho_cp、是否电芯。"""
    is_cell = (region == CELL).ravel()
    k_fixed = np.where(region == CELL, 0.0,
                       np.where(region == PLATE, p.k_plate, p.k_fill)).ravel()
    rho_cp = np.where(region == CELL, p.rho_cell * p.cp_cell,
                      np.where(region == PLATE, p.rho_plate * p.cp_plate,
                               p.rho_fill * p.cp_fill)).ravel()
    return k_fixed, rho_cp, is_cell


# ----------------------------------------------------------------------
# 仿射算子组装
# ----------------------------------------------------------------------
def build_affine_operators(p: BatteryParams) -> AffineOperators:
    """组装 2D 电池热模型的仿射算子（M^{-1} 已烘焙进各项）。

    返回 AffineOperators:
        A(mu) = mu[k_cell] * A_cell + 1 * A_fixed + mu[h] * A_conv
        b(mu) = mu[q_gen] * b_q + 1 * b_amb + mu[h] * b_conv
    """
    nx, ny = p.nx, p.ny
    n = nx * ny
    dx = p.lx / (nx - 1)
    dy = p.ly / (ny - 1)
    region = build_region_grid(p)
    k_fixed, rho_cp, is_cell = build_material_fields(p, region)

    K_cell = sp.lil_matrix((n, n))    # k_cell 乘子的扩散贡献
    K_fixed = sp.lil_matrix((n, n))   # 板/填充扩散 + 左右自然对流
    f_amb = np.zeros(n)               # 自然对流常数项

    def idx(iy: int, ix: int) -> int:
        return iy * nx + ix

    # --- 内部面：算术平均 k_face = (k_i + k_j)/2，按端点归属拆分 ---
    # 电芯节点的 k = mu[k_cell]（仿射变量），非电芯节点 k = k_fixed（常数）
    def add_face(i: int, j: int, d2: float) -> None:
        # 端点 i 贡献 0.5*k_i/d2，端点 j 贡献 0.5*k_j/d2
        for a, b in ((i, j), (j, i)):
            K = K_cell if is_cell[a] else K_fixed
            ka = 1.0 if is_cell[a] else k_fixed[a]  # 电芯项取 1，乘子为 mu[k_cell]
            c = 0.5 * ka / d2
            K[a, a] += c
            K[a, b] -= c
            K[b, b] += c
            K[b, a] -= c

    for iy in range(ny):
        for ix in range(nx):
            i = idx(iy, ix)
            if ix + 1 < nx:
                add_face(i, idx(iy, ix + 1), dx**2)
            if iy + 1 < ny:
                add_face(i, idx(iy + 1, ix), dy**2)

    # --- 左右边界：自然对流（固定，进 K_fixed / f_amb）---
    for iy in range(ny):
        for ix in (0, nx - 1):
            i = idx(iy, ix)
            K_fixed[i, i] += p.h_amb / dx
            f_amb[i] += p.h_amb * p.t_amb / dx

    # --- 上下边界：冷却流道 Robin（仿射 h，进 K_conv / f_conv）---
    K_conv = sp.lil_matrix((n, n))
    f_conv = np.zeros(n)
    for ix in range(nx):
        for iy in (0, ny - 1):
            i = idx(iy, ix)
            K_conv[i, i] += 1.0 / dy        # 乘子 mu[h]
            f_conv[i] += p.t_cool / dy      # 乘子 mu[h]

    # --- 内热源：仅电芯节点 ---
    f_q = is_cell.astype(float)

    # --- M^{-1} 烘焙：A = -M^{-1} K，b = M^{-1} f ---
    m_inv = sp.diags(1.0 / rho_cp)
    A_cell = (-m_inv @ K_cell).tocsr()
    A_fixed = (-m_inv @ K_fixed).tocsr()
    A_conv = (-m_inv @ K_conv).tocsr()
    b_q = m_inv @ f_q
    b_amb = m_inv @ f_amb
    b_conv = m_inv @ f_conv

    def theta_A(mu: dict) -> list[float]:
        return [mu["k_cell"], 1.0, mu["h"]]

    def theta_b(mu: dict) -> list[float]:
        return [mu["q_gen"], 1.0, mu["h"]]

    return AffineOperators(A_terms=[A_cell, A_fixed, A_conv],
                           b_terms=[b_q, b_amb, b_conv],
                           theta_A=theta_A, theta_b=theta_b)


def params_from_battery(p: BatteryParams) -> dict:
    """提取扫描参数 dict（与 PARAM_KEYS 对齐）。"""
    return {"k_cell": p.k_cell, "h": p.h, "q_gen": p.q_gen}


# ----------------------------------------------------------------------
# 全阶求解
# ----------------------------------------------------------------------
@dataclass
class BatteryFOMResult:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    region: np.ndarray          # (ny, nx) 区域编码
    snapshots: np.ndarray       # (n_nodes, n_t)，节点按 iy*nx+ix 排列
    params: BatteryParams

    def field(self, step: int = -1) -> np.ndarray:
        """取某时刻的 2D 温度场 (ny, nx)。"""
        return self.snapshots[:, step].reshape(self.params.ny, self.params.nx)

    @property
    def t_max_history(self) -> np.ndarray:
        return self.snapshots.max(axis=0)

    def t_max_cell_history(self) -> np.ndarray:
        """电芯区域最高温度（工程核心指标）。"""
        cell_nodes = (self.region == CELL).ravel()
        return self.snapshots[cell_nodes].max(axis=0)


def solve_fom(p: BatteryParams) -> BatteryFOMResult:
    """运行 2D 全阶仿真（向后欧拉，矩阵常系数 → 一次 LU 分解复用）。"""
    ops = build_affine_operators(p)
    A, b = ops.assemble(params_from_battery(p))

    n = p.n_nodes
    n_t = p.n_t
    t = np.linspace(0.0, p.t_end, n_t)
    lu = spla.factorized((sp.eye(n, format="csc") - p.dt * A).tocsc())

    snapshots = np.empty((n, n_t))
    T = np.full(n, p.t_init)
    snapshots[:, 0] = T
    for k in range(1, n_t):
        T = lu(T + p.dt * b)
        snapshots[:, k] = T

    x = np.linspace(0, p.lx, p.nx)
    y = np.linspace(0, p.ly, p.ny)
    return BatteryFOMResult(x=x, y=y, t=t, region=build_region_grid(p),
                            snapshots=snapshots, params=p)


def sanity_checks(res: BatteryFOMResult) -> None:
    """物理合理性检查。"""
    p = res.params
    s = res.snapshots

    # 1) 温度下界：正产热 + 边界冷却 → 不低于冷却液/环境温度的较小者
    t_floor = min(p.t_cool, p.t_amb)
    assert s.min() >= t_floor - 1e-6, f"温度低于下界: {s.min():.2f} K"

    # 2) 电芯最高温度单调不减（正产热）
    tmax = res.t_max_cell_history()
    assert np.all(np.diff(tmax) >= -1e-9), "电芯最高温度非物理下降"

    # 3) 电芯温升量级检查（数 K ~ 数十 K 为合理区间）
    rise = tmax[-1] - p.t_init
    print(f"  [检查] 电芯温升 = {rise:.1f} K（数 K~数十 K 为合理区间）")
    assert 0.1 < rise < 100.0, f"电芯温升 {rise:.1f} K 超出合理量级"

    print("  [检查] 全部物理合理性断言通过 OK")


def plot_result(res: BatteryFOMResult, save_path: str = "battery_2d_demo.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    T = res.field(-1) - 273.15
    im = ax.pcolormesh(res.x * 1000, res.y * 1000, T, cmap="hot", shading="auto")
    fig.colorbar(im, ax=ax, label="T [°C]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Battery pack cross-section @ t={res.t[-1]:.0f}s "
                 f"(k_cell={res.params.k_cell}, h={res.params.h:.0f}, "
                 f"q={res.params.q_gen:.1e})")

    ax = axes[1]
    ax.plot(res.t, res.t_max_cell_history() - 273.15, label="max T (cells)")
    ax.plot(res.t, res.snapshots.mean(axis=0) - 273.15, label="mean T", alpha=0.7)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T [°C]")
    ax.set_title("Temperature evolution")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  [输出] 图已保存: {save_path}")


if __name__ == "__main__":
    import time

    params = BatteryParams()
    print("=" * 64)
    print("电池包 2D 热管理全阶仿真（任务 #11）")
    print("=" * 64)
    print(f"  网格: {params.nx}×{params.ny} = {params.n_nodes} 节点, "
          f"dt={params.dt}s, {params.n_t} 步")
    print(f"  参数: k_cell={params.k_cell}, h={params.h}, q_gen={params.q_gen:.1e}"
          f"（≈{np.sqrt(params.q_gen / 3e4):.1f}C）")

    t0 = time.perf_counter()
    res = solve_fom(params)
    dt_solve = time.perf_counter() - t0
    print(f"  求解耗时: {dt_solve:.2f} s（技术方案 §5.2 估计 30-60 s，节点分裂法优于预期）")
    print(f"  末时刻电芯最高温度: {res.t_max_cell_history()[-1] - 273.15:.1f} °C")
    print(f"  末时刻全场温度范围: {res.snapshots[:, -1].min() - 273.15:.1f} ~ "
          f"{res.snapshots[:, -1].max() - 273.15:.1f} °C")

    print("\n物理合理性检查:")
    sanity_checks(res)
    plot_result(res)
