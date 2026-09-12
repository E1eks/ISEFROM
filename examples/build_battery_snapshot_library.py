"""
电池热管理 10 组快照库生成（任务 #11 收尾）
=============================================
流程：
  1. 拉丁超立方采样（LHS）10 组参数：k_cell∈[1,3], h∈[500,3000], C-rate∈[0.5,3]
  2. 逐组全阶仿真，登记入 SnapshotManager（含 C-rate 标注）
  3. POD 基诊断（能量谱、rank）
  4. 留一法（LOO）验证：9 组训练 → 盲测第 10 组，报告误差分布
     —— 这是任务 #19/#20（500 组扫描+全阶验证）流程的预演

运行: python examples/build_battery_snapshot_library.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fom.battery_2d import (BatteryParams, build_affine_operators,
                                params_from_battery, qgen_from_c_rate, solve_fom)
from src.rom.pod import PODDecomposer
from src.rom.reduced import ReducedModelBuilder, ROMSolver
from src.rom.snapshot import SnapshotManager

N_SAMPLES = 10
RANGES = {"k_cell": (1.0, 3.0), "h": (500.0, 3000.0), "c_rate": (0.5, 3.0)}
SEED = 42
LIB_PATH = ROOT / "snapshots_battery_10runs.npz"
TIME_SUB = 10  # POD 诊断用时间子采样（每 10 步取 1），完整快照仍全量保存


def lhs_samples(n: int = N_SAMPLES) -> list[dict]:
    """拉丁超立方采样（任务 #11 指定方法），返回参数 dict 列表。"""
    sampler = qmc.LatinHypercube(d=3, seed=SEED)
    u = sampler.random(n)
    lo = np.array([v[0] for v in RANGES.values()])
    hi = np.array([v[1] for v in RANGES.values()])
    pts = lo + u * (hi - lo)
    out = []
    for k_cell, h, c_rate in pts:
        out.append({"k_cell": float(k_cell), "h": float(h),
                    "c_rate": float(c_rate),
                    "q_gen": qgen_from_c_rate(c_rate)})
    return out


def run_library(samples: list[dict]) -> tuple[SnapshotManager, list]:
    """逐组全阶仿真并入库。"""
    mgr = SnapshotManager()
    foms = []
    for i, mu in enumerate(samples):
        p = BatteryParams(k_cell=mu["k_cell"], h=mu["h"], q_gen=mu["q_gen"])
        t0 = time.perf_counter()
        fom = solve_fom(p)
        foms.append(fom)
        params = params_from_battery(p)
        params["c_rate"] = mu["c_rate"]  # 额外标注（theta 不读，仅回溯用）
        mgr.add_run(params, fom.t, fom.snapshots, tag=f"lhs-{i:02d}")
        print(f"  [{i + 1:>2}/{len(samples)}] k_cell={mu['k_cell']:.2f} "
              f"h={mu['h']:6.0f} C={mu['c_rate']:.2f} "
              f"→ 电芯最高 {fom.t_max_cell_history()[-1] - 273.15:5.1f} °C "
              f"({time.perf_counter() - t0:.1f}s)")
    return mgr, foms


def pod_diagnostics(mgr: SnapshotManager) -> None:
    """POD 基能量谱诊断（时间子采样加速 SVD）。"""
    S = np.concatenate([r.snapshots[:, ::TIME_SUB] for r in mgr.runs], axis=1)
    basis = PODDecomposer().fit(S)
    r99 = basis.rank_for_energy(0.99)
    r999 = basis.rank_for_energy(0.999)
    r9999 = basis.rank_for_energy(0.9999)
    print(f"\n[POD 诊断] 拼接快照 {S.shape}（时间 1/{TIME_SUB} 子采样）")
    print(f"  sigma[:8] = {np.array2string(basis.sigma[:8], precision=2)}")
    print(f"  99% → r={r99},  99.9% → r={r999},  99.99% → r={r9999}"
          f"（全维 {S.shape[0]}）")
    print(f"  → 500 组扫描（#19）的 ROM 维度预算可参照 r={r999}~{r9999}")


def loo_validation(mgr: SnapshotManager, foms: list) -> list[float]:
    """留一法：依次盲测每组参数（预演 #19/#20 验证流程）。"""
    print("\n[LOO 验证] 9 组训练 → 盲测留出的 1 组：")
    ops = build_affine_operators(BatteryParams())
    errors = []
    for i in range(len(foms)):
        train = [j for j in range(len(foms)) if j != i]
        S_train = np.concatenate([mgr.runs[j].snapshots[:, ::TIME_SUB] for j in train],
                                 axis=1)
        basis = PODDecomposer().fit(S_train)
        r = basis.rank_for_energy(0.999)
        model = ReducedModelBuilder().build(basis, ops, r)

        p = foms[i].params
        T0 = np.full(p.n_nodes, p.t_init)
        rom = ROMSolver(dt=p.dt, t_end=p.t_end).solve(
            model, params_from_battery(p), T0, full_ops=ops)
        num = np.linalg.norm(rom.snapshots - foms[i].snapshots)
        den = np.linalg.norm(foms[i].snapshots - basis.mean[:, None])
        err = num / den
        errors.append(err)
        print(f"  盲测 lhs-{i:02d} (k={mgr.runs[i].params['k_cell']:.2f}, "
              f"h={mgr.runs[i].params['h']:5.0f}, C={mgr.runs[i].params['c_rate']:.2f})"
              f" → 相对误差 {err:.3%}  (r={r})")
    errors = np.array(errors)
    print(f"\n  LOO 误差: 中位 {np.median(errors):.3%}, "
          f"最大 {errors.max():.3%}（lhs-{int(errors.argmax()):02d}）")
    print(f"  → #19 目标 <3%、#20 告警阈值参考：误差最大的样本即误差监控"
          f"（#14）必须捕捉的对象")
    return errors.tolist()


def plot_summary(samples: list[dict], save_path: str = "battery_lhs_library.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = [s["k_cell"] for s in samples]
    h = [s["h"] for s in samples]
    c = [s["c_rate"] for s in samples]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(k, h, c=c, cmap="viridis", s=80, edgecolors="k")
    for i, (ki, hi) in enumerate(zip(k, h)):
        ax.annotate(str(i), (ki, hi), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    fig.colorbar(sc, label="C-rate")
    ax.set_xlabel("k_cell [W/(m·K)]")
    ax.set_ylabel("h [W/(m²·K)]")
    ax.set_title(f"LHS snapshot library (n={len(samples)}, seed={SEED})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\n[输出] 参数空间分布图: {save_path}")


if __name__ == "__main__":
    print("=" * 64)
    print("电池热管理 10 组 LHS 快照库（任务 #11）")
    print("=" * 64)

    samples = lhs_samples()
    print(f"[LHS] {len(samples)} 组参数（seed={SEED}，可复现）：")
    mgr, foms = run_library(samples)

    mgr.save(LIB_PATH)
    total_mb = sum(r.snapshots.nbytes for r in mgr.runs) / 1e6
    print(f"\n[输出] 快照库已保存: {LIB_PATH.name}（{len(mgr.runs)} runs, "
          f"{total_mb:.0f} MB，参数元数据在同名 .json）")

    pod_diagnostics(mgr)
    loo_validation(mgr, foms)
    plot_summary(samples)
