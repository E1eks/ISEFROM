"""
核心框架 v0.1 端到端示例（任务 #9）
====================================
演示 pROM 离线-在线工作流：
  离线: 2 组参数全阶仿真 → 快照库 → POD 基 → 仿射一次性投影
  在线: 未见过的新参数 → 零投影装配 → 秒级求解 → 对比全阶

运行: python examples/run_framework_v01.py
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fom.heat_1d import Heat1DParams, solve_fom
from src.fom.heat_1d_affine import affine_operators, params_from_heat1d
from src.rom.pod import PODDecomposer
from src.rom.reduced import ReducedModelBuilder, ROMSolver
from src.rom.snapshot import SnapshotManager


def main() -> None:
    print("=" * 64)
    print("核心框架 v0.1 端到端演示：pROM 离线-在线工作流（任务 #9）")
    print("=" * 64)

    # ---------------- 离线阶段 ----------------
    t0 = time.perf_counter()
    mgr = SnapshotManager()
    train_params = [Heat1DParams(k=1.5), Heat1DParams(k=2.5)]
    for p in train_params:
        fom = solve_fom(p)
        mgr.add_run(params_from_heat1d(p), fom.t, fom.snapshots,
                    tag=f"train-k{p.k}")
        print(f"[离线] 全阶仿真 k={p.k}: 快照 {fom.snapshots.shape} 入库")

    basis = PODDecomposer(center=True).fit(mgr.basis_input())
    r = basis.rank_for_energy(0.999)
    print(f"[离线] POD 基: 99.9% 能量 → r={r}（全维 {basis.V.shape[0]}）")

    ops = affine_operators()
    model = ReducedModelBuilder().build(basis, ops, r)
    print(f"[离线] 仿射一次性投影完成: {len(model.Ar_terms)} 个 A_r,q ({r}×{r})"
          f"（耗时 {time.perf_counter() - t0:.2f} s）")

    # ---------------- 在线阶段 ----------------
    p_new = Heat1DParams(k=2.0, q_gen=8.0e4)   # 未见过的新参数点
    mu = params_from_heat1d(p_new)
    T0 = np.full(p_new.n_x, p_new.t_init)
    T0[0] = p_new.t_left

    t0 = time.perf_counter()
    rom = ROMSolver(dt=p_new.dt, t_end=p_new.t_end).solve(model, mu, T0, full_ops=ops)
    t_rom = time.perf_counter() - t0
    print(f"\n[在线] 新参数 {mu} → ROM 求解 {t_rom * 1000:.1f} ms（零投影，仿射装配）")

    t0 = time.perf_counter()
    fom_new = solve_fom(p_new)
    t_fom = time.perf_counter() - t0
    print(f"[对照] 同参数全阶仿真 {t_fom * 1000:.1f} ms"
          f"（1D 小规模下加速比不具代表性；200-600× 加速比的目标场景是"
          f"任务 #11 的 2D 电池模型 6000 节点 × 600 步，届时 ROM 维度仍为 r 量级）")

    num = np.linalg.norm(rom.snapshots - fom_new.snapshots)
    den = np.linalg.norm(fom_new.snapshots - basis.mean[:, None])
    print(f"[精度] 相对误差 = {num / den:.3%}（参数内插点）")

    # ---------------- 快照库持久化 ----------------
    mgr.save(ROOT / "snapshots_library_demo.npz")
    mgr2 = SnapshotManager.load(ROOT / "snapshots_library_demo.npz")
    print(f"[持久化] 快照库保存/恢复 OK（{len(mgr2.runs)} runs，参数表: "
          f"{[(m['tag'], m['k']) for m in mgr2.param_table()]}）")


if __name__ == "__main__":
    main()
