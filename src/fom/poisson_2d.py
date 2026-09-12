"""
二维泊松方程算例（椭圆型，FOMSolver 协议实现）
================================================
ISEF 2028 ROM 项目 · 任务 #12（工程半边）
负责人：同学 B（实现）｜ 数学推导（弱形式）由同学 A 负责（A8）

控制方程（椭圆型，稳态）:

    -div( k(x,y) grad u ) = f_amp * f0(x,y)   in Ω = [0,1]^2
                       u  = 0                  on ∂Ω

几何：内嵌方块高/低导热区（k1），外部基体（k2）——模拟双材料稳态传导，
是电池热模型稳态极限的数学表亲，验证框架对椭圆型 PDE 的扩展性。

仿射分解（对 mu = {k1, k2, f_amp} 严格仿射）:
    面导热取算术平均 → 系数对 k1/k2 线性；
    沿用电池模型的符号约定 A = -K（K 为 SPD 刚度阵）：
    A(mu) = k1 * A_k1 + k2 * A_k2 + A_bc
    b(mu) = f_amp * b_f
    稳态系统: A(mu) u + b(mu) = 0

注：椭圆型无时间维，"快照"= 不同参数点的稳态解（单时刻入快照库）。
降阶稳态求解见 src/rom/reduced.py 的 solve_steady()。
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..rom.operators import AffineOperators

PARAM_KEYS = ["k1", "k2", "f_amp"]


@dataclass
class PoissonParams:
    """2D 泊松算例参数。"""

    n: int = 41                 # 每维节点数
    inner_lo: float = 0.25      # 内嵌区域下界
    inner_hi: float = 0.75      # 内嵌区域上界

    # 扫描参数
    k1: float = 5.0             # 内嵌区域导热系数
    k2: float = 1.0             # 基体导热系数
    f_amp: float = 100.0        # 源幅值

    @property
    def n_nodes(self) -> int:
        return self.n * self.n


def build_region(p: PoissonParams) -> np.ndarray:
    """返回 (n, n) 布尔数组：True = 内嵌区域（k1）。"""
    x = np.linspace(0, 1, p.n)
    X, Y = np.meshgrid(x, x)
    return ((X >= p.inner_lo) & (X <= p.inner_hi)
            & (Y >= p.inner_lo) & (Y <= p.inner_hi))


def build_affine_operators(p: PoissonParams) -> AffineOperators:
    """组装泊松算例的仿射算子。

    A(mu) = mu[k1]*A_k1 + mu[k2]*A_k2 + A_bc
    b(mu) = mu[f_amp]*b_f
    （A = -K 约定；边界行在各扩散项中清零，A_bc 为单位行）
    """
    n = p.n
    nn = n * n
    h = 1.0 / (n - 1)
    inner = build_region(p).ravel()

    K1 = sp.lil_matrix((nn, nn))   # 内嵌区域端点的扩散贡献（乘子 k1）
    K2 = sp.lil_matrix((nn, nn))   # 基体端点的扩散贡献（乘子 k2）

    def idx(iy: int, ix: int) -> int:
        return iy * n + ix

    def add_face(i: int, j: int) -> None:
        c = 0.5 / h**2             # 面 k = (k_a + k_b)/2，按端点归属拆分
        for a, b in ((i, j), (j, i)):
            K = K1 if inner[a] else K2
            K[a, a] += c
            K[a, b] -= c
            K[b, b] += c
            K[b, a] -= c

    for iy in range(n):
        for ix in range(n):
            i = idx(iy, ix)
            if ix + 1 < n:
                add_face(i, idx(iy, ix + 1))
            if iy + 1 < n:
                add_face(i, idx(iy + 1, ix))

    # 边界行清零（Dirichlet 由 A_bc 单位行接管）
    boundary = np.zeros(nn, dtype=bool)
    for iy in range(n):
        for ix in (0, n - 1):
            boundary[idx(iy, ix)] = True
    for ix in range(n):
        for iy in (0, n - 1):
            boundary[idx(iy, ix)] = True
    for K in (K1, K2):
        K = K.tolil()
        K[boundary, :] = 0.0

    # A_bc：边界单位行（u=0 → 行 i: u_i = 0）
    A_bc = sp.lil_matrix((nn, nn))
    for i in np.where(boundary)[0]:
        A_bc[i, i] = -1.0          # A=-K 约定下与"零均值 Dirichlet"一致：
    # A u + b = 0 → -u_i = 0。符号与扩散行一致（A = -K 负定 + 单位行负号）

    # 源项：f0 = sin(pi x) sin(pi y)，边界为零
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)
    b_f = (np.sin(np.pi * X) * np.sin(np.pi * Y)).ravel()
    b_f[boundary] = 0.0

    def theta_A(mu: dict) -> list[float]:
        return [mu["k1"], mu["k2"], 1.0]

    def theta_b(mu: dict) -> list[float]:
        return [mu["f_amp"]]

    return AffineOperators(A_terms=[(-K1).tocsr(), (-K2).tocsr(), A_bc.tocsr()],
                           b_terms=[b_f],
                           theta_A=theta_A, theta_b=theta_b)


def params_from_poisson(p: PoissonParams) -> dict:
    return {"k1": p.k1, "k2": p.k2, "f_amp": p.f_amp}


@dataclass
class PoissonFOMResult:
    x: np.ndarray
    inner: np.ndarray           # (n, n) 内嵌区域掩码
    solution: np.ndarray        # (n_nodes,)
    params: PoissonParams

    def field(self) -> np.ndarray:
        return self.solution.reshape(self.params.n, self.params.n)


def solve_fom(p: PoissonParams) -> PoissonFOMResult:
    """直接稀疏求解稳态系统 A u + b = 0。"""
    ops = build_affine_operators(p)
    A, b = ops.assemble(params_from_poisson(p))
    u = spla.spsolve(A.tocsc(), -b)
    x = np.linspace(0, 1, p.n)
    return PoissonFOMResult(x=x, inner=build_region(p),
                            solution=u, params=p)


def sanity_checks(res: PoissonFOMResult) -> None:
    """物理合理性检查（最大值原理：f≥0 且 u|∂Ω=0 → u≥0）。"""
    assert res.solution.min() > -1e-8, f"违反最大值原理: min={res.solution.min():.2e}"
    # 对称性：几何与源关于 x=y 对称 → 解应近似对称
    U = res.field()
    sym_err = np.abs(U - U.T).max() / np.abs(U).max()
    print(f"  [检查] 对角对称性误差 = {sym_err:.2e}（应 << 1）")
    assert sym_err < 1e-10, "对称性破坏"
    print("  [检查] 全部物理合理性断言通过 OK")


def plot_result(res: PoissonFOMResult, save_path: str = "poisson_2d_demo.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(res.x, res.x, res.field(), cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="u")
    ax.contour(res.x, res.x, res.inner.astype(float), levels=[0.5],
               colors="white", linewidths=1.0, linestyles="dashed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Poisson 2D (k1={res.params.k1}, k2={res.params.k2}, "
                 f"f_amp={res.params.f_amp})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  [输出] 图已保存: {save_path}")


if __name__ == "__main__":
    params = PoissonParams()
    print("=" * 64)
    print("二维泊松方程全阶求解（任务 #12 工程半边）")
    print("=" * 64)
    print(f"  网格: {params.n}×{params.n} = {params.n_nodes} 节点")
    print(f"  参数: k1={params.k1}（内嵌）, k2={params.k2}（基体）, f_amp={params.f_amp}")

    res = solve_fom(params)
    print(f"  解范围: [{res.solution.min():.3f}, {res.solution.max():.3f}]")

    print("\n物理合理性检查:")
    sanity_checks(res)
    plot_result(res)
