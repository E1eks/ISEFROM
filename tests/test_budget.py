"""
误差预算模块测试（任务 #26 / 方向②）
========================================
运行: python -m pytest tests/test_budget.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fom.heat_1d import Heat1DParams, solve_fom
from src.fom.heat_1d_affine import affine_operators, params_from_heat1d
from src.rom.budget import (budget_certificate, projection_error,
                            residual_indicator, time_error_estimate,
                            tradeoff_curve)
from src.rom.pod import PODDecomposer
from src.rom.reduced import ReducedModelBuilder
from src.rom.snapshot import SnapshotManager


@pytest.fixture(scope="module")
def assets():
    p = Heat1DParams()
    mgr = SnapshotManager()
    for k in (1.5, 2.5):
        pk = Heat1DParams(k=k)
        fom = solve_fom(pk)
        mgr.add_run(params_from_heat1d(pk), fom.t, fom.snapshots)
    basis = PODDecomposer().fit(mgr.basis_input())
    ops = affine_operators()
    model = ReducedModelBuilder().build(basis, ops, 2)
    T0 = np.where(np.arange(p.n_x) == 0, p.t_left, p.t_init).astype(float)
    return basis, ops, model, T0


class TestProjectionError:
    def test_matches_eckart_young(self, assets):
        basis = assets[0]
        e = projection_error(basis, 2)
        theory = np.sqrt(1.0 - basis.energy_ratio[1])
        assert e == pytest.approx(theory, rel=1e-12)

    def test_monotone_decreasing(self, assets):
        basis = assets[0]
        errs = [projection_error(basis, r) for r in range(1, 5)]
        assert all(a >= b for a, b in zip(errs, errs[1:]))


class TestTimeError:
    def test_first_order_scaling(self, assets):
        """一阶格式：dt 减半，Richardson 估计应近似减半。"""
        _, ops, model, T0 = assets
        p = Heat1DParams()
        mu = params_from_heat1d(p)
        e1, _ = time_error_estimate(model, mu, T0, p.dt, p.t_end)
        e2, _ = time_error_estimate(model, mu, T0, p.dt / 2, p.t_end)
        assert e1 > 0 and e2 > 0
        assert e1 / e2 == pytest.approx(2.0, rel=0.15)

    def test_budget_certificate_fields(self, assets):
        basis, ops, model, T0 = assets
        p = Heat1DParams()
        cert = budget_certificate(model, basis, params_from_heat1d(p), T0,
                                  p.dt, p.t_end, ops)
        d = cert.to_dict()
        assert d["E_aff"] == 0.0
        assert d["E_proj"] > 0
        assert d["E_time"] > 0
        assert d["E_param_indicator (uncertified)"]["certified"] is False
        # 当前无认证界 → total_bound 只含已认证部分
        assert d["total_bound"] == pytest.approx(d["E_proj"] + d["E_time"])


class TestResidualIndicator:
    def test_collects_full_trajectory(self, assets):
        _, ops, model, T0 = assets
        p = Heat1DParams()
        out = residual_indicator(model, params_from_heat1d(p), T0,
                                 p.dt, p.t_end, ops)
        assert out["max"] >= out["mean"] >= 0
        assert out["max"] > 0


class TestTradeoffCurve:
    def test_curve_data(self, assets):
        basis, ops, _, T0 = assets
        p = Heat1DParams()
        curve = tradeoff_curve(basis, ops, ReducedModelBuilder(),
                               params_from_heat1d(p), T0, p.dt, p.t_end,
                               r_list=[1, 2, 3])
        assert curve.e_proj[0] >= curve.e_proj[1] >= curve.e_proj[2]
        assert all(t > 0 for t in curve.rom_time_ms)
