"""
认证误差界接口测试（任务 #14 挂载点冻结 / 任务 #22 收尾）
==========================================================
运行: python -m pytest tests/test_estimators.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fom.heat_1d import Heat1DParams, solve_fom
from src.fom.heat_1d_affine import affine_operators, params_from_heat1d
from src.pipeline.runner import run_pipeline
from src.rom.budget import budget_certificate
from src.rom.estimators import (BoundContext, BoundResult,
                                placeholder_bound, residual_out_trajectory,
                                spectral_beta_lb)
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
    ctx = BoundContext(full_ops=ops, kind="time", T0=T0, dt=p.dt, t_end=p.t_end)
    return p, ops, model, T0, ctx


class TestContract:
    def test_residual_trajectory_shape(self, assets):
        """残差轨迹：非负、长度 n_t-1、与 ROM 结果一同返回。"""
        p, ops, model, T0, _ = assets
        mu = params_from_heat1d(p)
        norms, rom = residual_out_trajectory(model, mu, T0, p.dt, p.t_end, ops)
        assert len(norms) == len(rom.t) - 1
        assert (norms >= 0).all() and norms.max() > 0

    def test_spectral_beta_positive(self, assets):
        """热传导 -A 对称正定 → 谱下界为正。"""
        p, ops, _, _, _ = assets
        assert spectral_beta_lb(ops, params_from_heat1d(p)) > 0

    def test_placeholder_result_fields(self, assets):
        """占位界走通冻结签名：字段齐全、delta>0、明确标未认证。"""
        p, _, model, _, ctx = assets
        res = placeholder_bound(model, params_from_heat1d(p), ctx)
        assert isinstance(res, BoundResult)
        assert res.delta > 0 and res.beta_lb > 0 and res.residual_norm > 0
        assert res.certified is False and res.method == "placeholder"
        assert res.per_step is not None
        d = res.to_dict()
        assert set(d) >= {"delta", "beta_lb", "residual_norm",
                          "certified", "method", "n_steps"}

    def test_parabolic_ctx_required(self, assets):
        """抛物型缺 T0/dt/t_end 必须报错（接口契约）。"""
        p, ops, model, _, _ = assets
        bad = BoundContext(full_ops=ops, kind="time")
        with pytest.raises(ValueError, match="T0/dt/t_end"):
            placeholder_bound(model, params_from_heat1d(p), bad)


class TestBudgetIntegration:
    @staticmethod
    def _fake_fn(delta, certified):
        def fn(model, params, ctx):
            assert isinstance(ctx, BoundContext)
            return BoundResult(delta=delta, beta_lb=1.0, residual_norm=0.0,
                               certified=certified, method="fake")
        return fn

    def test_certified_bound_counts_in_total(self, assets):
        """certified=True 的 delta 计入 total_bound。"""
        p, ops, model, T0, _ = assets
        cert = budget_certificate(model, _fixture_basis(model),
                                  params_from_heat1d(p), T0, p.dt, p.t_end,
                                  ops, certified_bound_fn=self._fake_fn(0.5, True))
        assert cert.certified_bound == 0.5
        assert cert.total_bound == pytest.approx(
            cert.e_proj + cert.e_time + 0.5)
        assert cert.bound_details["method"] == "fake"

    def test_uncertified_bound_excluded(self, assets):
        """未认证占位界不计入 total_bound，但详情保留（防 failure log #02 重演）。"""
        p, ops, model, T0, _ = assets
        cert = budget_certificate(model, _fixture_basis(model),
                                  params_from_heat1d(p), T0, p.dt, p.t_end,
                                  ops, certified_bound_fn=self._fake_fn(0.5, False))
        assert cert.certified_bound is None
        assert cert.total_bound == pytest.approx(cert.e_proj + cert.e_time)
        assert cert.bound_details["delta"] == 0.5


def _fixture_basis(model):
    """budget_certificate 仅需 basis 算 E_proj；从 model 反推一个最小 PODBasis。"""
    from src.rom.pod import PODBasis
    n_modes = model.V.shape[1]
    return PODBasis(mean=model.mean, V=model.V,
                    sigma=np.ones(n_modes),
                    energy_ratio=np.linspace(0.5, 0.9, n_modes))


class TestPipelineWiring:
    def test_estimator_reachable_via_config(self):
        """配置启用 placeholder → 在线条目带 error_bound + effectivity。"""
        report = run_pipeline({
            "case": "heat_1d",
            "offline": {"n_samples": 4, "seed": 7,
                        "ranges": {"k": [1.0, 3.0], "h_right": [500.0, 3000.0]},
                        "energy_threshold": 0.999},
            "online": {"params": [{"k": 2.0, "h_right": 1500.0, "q_gen": 8e4}],
                       "validate_against_fom": True},
            "error_monitor": {"estimator": "placeholder"},
            "output": {"prefix": "test_pipe_est"},
        })
        bound = report["online"][0]["error_bound"]
        assert bound["certified"] is False
        assert bound["delta"] > 0
        assert bound["effectivity"] > 0
        # 占位界未认证 → 不得出现 η<1 告警（告警只对 certified=True）
        assert "warning" not in bound

    def test_unknown_estimator_rejected(self):
        with pytest.raises(ValueError, match="未知估计器"):
            run_pipeline({"case": "heat_1d", "offline": {"n_samples": 2},
                          "online": {"params": []},
                          "error_monitor": {"estimator": "scm"}})
