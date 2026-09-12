"""
端到端流水线编排器（任务 #22 雏形）
====================================
配置驱动：JSON 配置 → 快照采集（LHS/显式列表）→ POD → 仿射装配 →
在线求解 → 可选全阶验证 → 报告输出。

框架不认识具体 PDE：算例通过 CASE_REGISTRY 注册（架构文档 §2 目标 G1），
新增算例 = 实现 FOMSolver 协议 + 注册一行。

用法:
    from src.pipeline.runner import run_pipeline
    report = run_pipeline(config: dict)
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import qmc

from ..rom.pod import PODDecomposer
from ..rom.reduced import ReducedModelBuilder, ROMSolver, solve_steady
from ..rom.snapshot import SnapshotManager

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CaseSpec:
    """算例注册项（PDE 插件协议，架构文档 §3.1）。"""

    kind: str                       # "time"（抛物）| "steady"（椭圆）
    make_params: Callable[[dict], Any]      # fom_overrides → 参数对象
    operators: Callable[[Any], Any]         # 参数对象 → AffineOperators
    solve_fom: Callable[[Any], Any]         # 参数对象 → FOM 结果
    params_dict: Callable[[Any], dict]      # 参数对象 → 扫描参数 dict
    fom_snapshots: Callable[[Any], tuple]   # FOM 结果 → (t, snapshots)
    make_initial: Callable[[Any], np.ndarray]  # 参数对象 → 初始场（稳态不用）
    param_ranges: dict              # LHS 默认参数域


def _registry() -> dict[str, CaseSpec]:
    """延迟导入算例插件（避免循环依赖）。"""
    from ..fom.battery_2d import (BatteryParams, build_affine_operators as bat_ops,
                                  params_from_battery, solve_fom as bat_solve)
    from ..fom.heat_1d import Heat1DParams, solve_fom as h1_solve
    from ..fom.heat_1d_affine import affine_operators as h1_ops
    from ..fom.heat_1d_affine import params_from_heat1d
    from ..fom.poisson_2d import (PoissonParams, build_affine_operators as poi_ops,
                                  params_from_poisson, solve_fom as poi_solve)

    return {
        "heat_1d": CaseSpec(
            kind="time",
            make_params=lambda ov: Heat1DParams(**ov),
            operators=lambda p: h1_ops(p),
            solve_fom=h1_solve,
            params_dict=params_from_heat1d,
            fom_snapshots=lambda r: (r.t, r.snapshots),
            make_initial=lambda p: np.where(np.arange(p.n_x) == 0,
                                            p.t_left, p.t_init).astype(float),
            param_ranges={"k": (1.0, 3.0), "h_right": (500.0, 3000.0),
                          "q_gen": (3e4, 2.7e5)}),
        "poisson_2d": CaseSpec(
            kind="steady",
            make_params=lambda ov: PoissonParams(**ov),
            operators=poi_ops,
            solve_fom=poi_solve,
            params_dict=params_from_poisson,
            fom_snapshots=lambda r: (np.array([0.0]), r.solution[:, None]),
            make_initial=lambda p: None,
            param_ranges={"k1": (0.5, 5.0), "k2": (0.5, 5.0)}),
        "battery_2d": CaseSpec(
            kind="time",
            make_params=lambda ov: BatteryParams(**ov),
            operators=bat_ops,
            solve_fom=bat_solve,
            params_dict=params_from_battery,
            fom_snapshots=lambda r: (r.t, r.snapshots),
            make_initial=lambda p: np.full(p.n_nodes, p.t_init),
            param_ranges={"k_cell": (1.0, 3.0), "h": (500.0, 3000.0),
                          "q_gen": (7.5e3, 2.7e5)}),
    }


# ----------------------------------------------------------------------
# 采样
# ----------------------------------------------------------------------
def sample_params(spec: CaseSpec, offline_cfg: dict) -> list[dict]:
    """离线采样：explicit 列表 或 LHS。"""
    if "explicit" in offline_cfg:
        return [dict(mu) for mu in offline_cfg["explicit"]]
    n = int(offline_cfg.get("n_samples", 10))
    seed = int(offline_cfg.get("seed", 42))
    ranges = offline_cfg.get("ranges", spec.param_ranges)
    keys = list(ranges.keys())
    u = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    lo = np.array([ranges[k][0] for k in keys])
    hi = np.array([ranges[k][1] for k in keys])
    return [dict(zip(keys, pt)) for pt in lo + u * (hi - lo)]


# ----------------------------------------------------------------------
# 主流水线
# ----------------------------------------------------------------------
def run_pipeline(config: dict) -> dict:
    """执行端到端流水线，返回报告 dict（并写 outputs/ 下 JSON）。"""
    registry = _registry()
    case_name = config["case"]
    if case_name not in registry:
        raise ValueError(f"未知算例 {case_name!r}，已注册: {list(registry)}")
    spec = registry[case_name]
    fom_ov = config.get("fom_overrides", {})
    offline_cfg = config.get("offline", {})
    online_cfg = config.get("online", {})
    report: dict = {"case": case_name, "config": config}

    # ---------------- 离线 ----------------
    t0 = time.perf_counter()
    base_params = spec.make_params(fom_ov)
    ops = spec.operators(base_params)
    mgr = SnapshotManager()
    for mu in sample_params(spec, offline_cfg):
        p = spec.make_params({**fom_ov, **mu})
        fom = spec.solve_fom(p)
        t, S = spec.fom_snapshots(fom)
        mgr.add_run(spec.params_dict(p), t, S, tag=f"offline-{len(mgr.runs)}")
    report["offline"] = {"n_runs": len(mgr.runs),
                         "params": mgr.param_table()}

    sub = int(offline_cfg.get("time_subsample", 1))
    S_pod = np.concatenate([r.snapshots[:, ::sub] for r in mgr.runs], axis=1)
    basis = PODDecomposer().fit(S_pod)
    r = basis.rank_for_energy(float(offline_cfg.get("energy_threshold", 0.999)))
    model = ReducedModelBuilder().build(basis, ops, r)
    report["pod"] = {"r": r, "sigma_head": basis.sigma[:8].tolist(),
                     "energy_ratio_head": basis.energy_ratio[:8].tolist()}
    report["offline"]["wall_time_s"] = time.perf_counter() - t0

    # ---------------- 在线 ----------------
    results = []
    for mu in online_cfg.get("params", []):
        p = spec.make_params({**fom_ov, **mu})
        mu_full = spec.params_dict(p)
        entry: dict = {"params": mu_full}

        t0 = time.perf_counter()
        if spec.kind == "time":
            T0 = spec.make_initial(p)
            rom = ROMSolver(dt=p.dt, t_end=p.t_end).solve(
                model, mu_full, T0, full_ops=ops)
            u_rom, t_rom = rom.snapshots, rom.t
        else:
            u_rom = solve_steady(model, mu_full)[:, None]  # (n,) → (n,1)，防广播
        entry["rom_time_s"] = time.perf_counter() - t0

        if online_cfg.get("validate_against_fom", False):
            t0 = time.perf_counter()
            fom = spec.solve_fom(p)
            _, S_fom = spec.fom_snapshots(fom)
            entry["fom_time_s"] = time.perf_counter() - t0
            num = np.linalg.norm(u_rom - S_fom)
            den = np.linalg.norm(S_fom - basis.mean[:, None])
            entry["rel_error"] = float(num / den)
            entry["speedup"] = entry["fom_time_s"] / max(entry["rom_time_s"], 1e-9)
        results.append(entry)
    report["online"] = results

    # ---------------- 输出 ----------------
    prefix = config.get("output", {}).get("prefix", "pipeline_report")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{prefix}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                   default=float), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report
