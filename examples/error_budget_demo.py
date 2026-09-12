"""
误差预算仪表盘演示（任务 #26 / 方向②）
==========================================
电池算例：4 组 LHS 离线 → 单点误差预算证书 + 权衡曲线。
运行: python examples/error_budget_demo.py
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fom.battery_2d import (BatteryParams, build_affine_operators,
                                params_from_battery, qgen_from_c_rate, solve_fom)
from src.rom.budget import budget_certificate, tradeoff_curve
from src.rom.pod import PODDecomposer
from src.rom.reduced import ReducedModelBuilder
from src.rom.snapshot import SnapshotManager


def main() -> None:
    print("=" * 64)
    print("误差预算仪表盘演示（任务 #26 / 方向②）")
    print("=" * 64)

    # ---------------- 离线（4 组 LHS，与流水线 demo 同种子）----------------
    from scipy.stats import qmc

    u = qmc.LatinHypercube(d=3, seed=7).random(4)
    lo = np.array([1.0, 500.0, 3e4])
    hi = np.array([3.0, 3000.0, 1.2e5])
    mgr = SnapshotManager()
    for k_cell, h, q_gen in lo + u * (hi - lo):
        p = BatteryParams(k_cell=float(k_cell), h=float(h), q_gen=float(q_gen))
        fom = solve_fom(p)
        mgr.add_run(params_from_battery(p), fom.t, fom.snapshots)
    basis = PODDecomposer().fit(
        np.concatenate([r.snapshots[:, ::10] for r in mgr.runs], axis=1))
    ops = build_affine_operators(BatteryParams())
    builder = ReducedModelBuilder()

    # ---------------- 查询点 ----------------
    p_mid = BatteryParams(k_cell=2.0, h=1500.0, q_gen=qgen_from_c_rate(2.0))
    mu = params_from_battery(p_mid)
    T0 = np.full(p_mid.n_nodes, p_mid.t_init)
    r = basis.rank_for_energy(0.999)
    model = builder.build(basis, ops, r)

    print("\n[1] 误差预算证书（含全阶对照调试用真值）")
    t0 = time.perf_counter()
    fom_mid = solve_fom(p_mid)
    fom_time = time.perf_counter() - t0
    cert = budget_certificate(model, basis, mu, T0, p_mid.dt, p_mid.t_end,
                              ops, true_field=fom_mid.snapshots[:, -1])
    print(cert.summary())

    # ---------------- 权衡曲线 ----------------
    print("\n[2] 权衡曲线（r vs 预估误差 vs 在线成本）")
    curve = tradeoff_curve(
        basis, ops, builder, mu, T0, p_mid.dt, p_mid.t_end,
        r_list=list(range(1, 7)),
        true_snapshots_fn=lambda: fom_mid.snapshots)
    for r_i, ep, tt, te in zip(curve.r_list, curve.e_proj,
                               curve.rom_time_ms, curve.true_error):
        print(f"  r={r_i}: E_proj={ep:.3e}  online={tt:6.1f} ms  "
              f"true_err={te:.3e}")
    curve.plot(str(ROOT / "error_budget_tradeoff.png"),
               fom_time_s=fom_time)
    print(f"\n[输出] 权衡曲线图: error_budget_tradeoff.png")

    import json
    out = ROOT / "outputs" / "budget_cert_demo.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(cert.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[输出] 证书 JSON: {out.name}")


if __name__ == "__main__":
    main()
