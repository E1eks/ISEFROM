"""
2D 电池热模型测试（任务 #11）
==============================
运行: python -m pytest tests/test_battery_2d.py -v
覆盖：仿射装配恒等式、ROM 精度（粗网格快速验证）、物理合理性。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fom.battery_2d import (BatteryParams, build_affine_operators,
                                build_region_grid, params_from_battery,
                                qgen_from_c_rate, solve_fom, CELL, PLATE)
from src.rom.pod import PODDecomposer
from src.rom.reduced import ReducedModelBuilder, ROMSolver
from src.rom.snapshot import SnapshotManager


def direct_operators(p: BatteryParams):
    """参考实现：直接组装（不仿射），用于恒等式验证。"""
    from src.fom.battery_2d import (build_material_fields, FILL)
    import scipy.sparse as sp

    nx, ny = p.nx, p.ny
    dx = p.lx / (nx - 1)
    dy = p.ly / (ny - 1)
    region = build_region_grid(p)
    is_cell = (region == CELL).ravel()
    k_node = np.where(region == CELL, p.k_cell,
                      np.where(region == PLATE, p.k_plate, p.k_fill)).ravel()
    rho_cp = np.where(region == CELL, p.rho_cell * p.cp_cell,
                      np.where(region == PLATE, p.rho_plate * p.cp_plate,
                               p.rho_fill * p.cp_fill)).ravel()
    n = nx * ny
    K = sp.lil_matrix((n, n))
    f = np.zeros(n)

    def idx(iy, ix):
        return iy * nx + ix

    for iy in range(ny):
        for ix in range(nx):
            i = idx(iy, ix)
            for jy, jx, d2 in ((iy, ix + 1, dx**2), (iy, ix - 1, dx**2),
                               (iy + 1, ix, dy**2), (iy - 1, ix, dy**2)):
                if 0 <= jy < ny and 0 <= jx < nx:
                    j = idx(jy, jx)
                    kf = 0.5 * (k_node[i] + k_node[j])
                    K[i, i] += kf / d2
                    K[i, j] -= kf / d2
            if ix in (0, nx - 1):
                K[i, i] += p.h_amb / dx
                f[i] += p.h_amb * p.t_amb / dx
            if iy in (0, ny - 1):
                K[i, i] += p.h / dy
                f[i] += p.h * p.t_cool / dy
            if is_cell[i]:
                f[i] += p.q_gen

    m_inv = sp.diags(1.0 / rho_cp)
    return (-m_inv @ K).tocsr(), m_inv @ f


@pytest.fixture(scope="module")
def coarse():
    """粗网格参数（测试速度优先）。"""
    return BatteryParams(nx=50, ny=30, t_end=60.0)


class TestAffineIdentity:
    @pytest.mark.parametrize("mu", [
        dict(k_cell=1.5, h=1000.0, q_gen=5.0e4),
        dict(k_cell=2.8, h=550.0, q_gen=1.1e5),
    ])
    def test_affine_matches_direct(self, coarse, mu):
        p = BatteryParams(nx=coarse.nx, ny=coarse.ny, **mu)
        ops = build_affine_operators(p)
        A_ref, b_ref = direct_operators(p)
        A_aff, b_aff = ops.assemble(mu)
        assert abs((A_ref - A_aff).max()) < 1e-10
        np.testing.assert_allclose(b_ref, b_aff, atol=1e-10)


class TestBatteryROM:
    def test_rom_accuracy_coarse(self, coarse):
        """粗网格端到端：2 组参数训练 → 未见参数点 ROM 误差 < 5%。"""
        mgr = SnapshotManager()
        for k_cell in (1.2, 2.8):
            p = BatteryParams(nx=coarse.nx, ny=coarse.ny, t_end=coarse.t_end,
                              k_cell=k_cell)
            fom = solve_fom(p)
            mgr.add_run(params_from_battery(p), fom.t, fom.snapshots)

        basis = PODDecomposer().fit(mgr.basis_input())
        ops = build_affine_operators(coarse)
        r = basis.rank_for_energy(0.999)
        model = ReducedModelBuilder().build(basis, ops, r)

        p_mid = BatteryParams(nx=coarse.nx, ny=coarse.ny, t_end=coarse.t_end,
                              k_cell=2.0, q_gen=qgen_from_c_rate(2.0))
        T0 = np.full(p_mid.n_nodes, p_mid.t_init)
        rom = ROMSolver(dt=p_mid.dt, t_end=p_mid.t_end).solve(
            model, params_from_battery(p_mid), T0, full_ops=ops)
        fom_mid = solve_fom(p_mid)
        num = np.linalg.norm(rom.snapshots - fom_mid.snapshots)
        den = np.linalg.norm(fom_mid.snapshots - basis.mean[:, None])
        assert num / den < 0.05


class TestPhysics:
    def test_geometry(self):
        """几何：6 颗电芯 + 铝板条 + 填充区域齐全。"""
        p = BatteryParams()
        region = build_region_grid(p)
        assert (region == CELL).sum() > 0
        assert (region == PLATE).sum() > 0
        # 电芯应分为 6 个连通块（用行列投影粗检）
        cell_cols = (region == CELL).any(axis=0)
        gaps = np.diff(cell_cols.astype(int))
        assert (gaps == 1).sum() == 6   # 6 段电芯列
        assert (gaps == -1).sum() == 6

    def test_temperature_bounds(self, coarse):
        res = solve_fom(coarse)
        assert res.snapshots.min() >= min(coarse.t_cool, coarse.t_amb) - 1e-6
        rise = res.t_max_cell_history()[-1] - coarse.t_init
        assert 0.01 < rise < 100.0

    def test_c_rate_scaling(self):
        assert qgen_from_c_rate(2.0) == pytest.approx(4 * qgen_from_c_rate(1.0))
