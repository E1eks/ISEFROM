"""
核心框架单元测试（任务 #9）
============================
运行: python -m pytest tests/ -v
覆盖：POD 分解器、快照管理器、仿射装配恒等式、ROM 内插精度、pROM 在线装配。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fom.heat_1d import Heat1DParams, solve_fom
from src.fom.heat_1d_affine import affine_operators, params_from_heat1d
from src.rom.pod import PODDecomposer
from src.rom.reduced import ReducedModelBuilder, ROMSolver
from src.rom.snapshot import SnapshotManager
from src.rom.pod_galerkin_1d import build_continuous_operators  # 旧 Demo 的参考实现


# ---------------- fixtures ----------------
@pytest.fixture(scope="module")
def fom_base():
    return solve_fom(Heat1DParams())


@pytest.fixture(scope="module")
def basis(fom_base):
    return PODDecomposer(center=True).fit(fom_base.snapshots)


# ---------------- POD 分解器 ----------------
class TestPOD:
    def test_orthonormality(self, basis):
        V = basis.V[:, :6]
        assert np.abs(V.T @ V - np.eye(6)).max() < 1e-12

    def test_eckart_young(self, basis, fom_base):
        """截断重构误差 == sqrt(被丢弃能量比例)（Eckart-Young）。"""
        r = 2
        err = PODDecomposer.reconstruction_error(basis, fom_base.snapshots, r)
        theory = np.sqrt(1.0 - basis.energy_ratio[r - 1])
        assert err == pytest.approx(theory, rel=1e-10)

    def test_rank_for_energy(self, basis):
        """基线算例：99.9% 能量需 r=2（任务 #4 实测值）。"""
        assert basis.rank_for_energy(0.999) == 2

    def test_bad_input(self):
        with pytest.raises(ValueError):
            PODDecomposer().fit(np.ones(5))


# ---------------- 快照管理器 ----------------
class TestSnapshotManager:
    def test_add_and_concat(self, fom_base):
        mgr = SnapshotManager()
        mgr.add_run(params_from_heat1d(Heat1DParams()), fom_base.t,
                    fom_base.snapshots, tag="baseline")
        mgr.add_run(params_from_heat1d(Heat1DParams(k=2.5)), fom_base.t[::2],
                    fom_base.snapshots[:, ::2], tag="half-time")
        S = mgr.basis_input()
        assert S.shape == (fom_base.snapshots.shape[0],
                           fom_base.snapshots.shape[1] + fom_base.snapshots[:, ::2].shape[1])
        table = mgr.param_table()
        assert table[0]["k"] == pytest.approx(1.5) and table[1]["tag"] == "half-time"

    def test_dimension_guard(self, fom_base):
        mgr = SnapshotManager()
        mgr.add_run({}, fom_base.t, fom_base.snapshots)
        with pytest.raises(ValueError):
            mgr.add_run({}, fom_base.t, fom_base.snapshots[:-1])  # 空间维不一致

    def test_save_load_roundtrip(self, fom_base, tmp_path):
        mgr = SnapshotManager()
        p = params_from_heat1d(Heat1DParams())
        mgr.add_run(p, fom_base.t, fom_base.snapshots, tag="rt")
        f = tmp_path / "lib.npz"
        mgr.save(f)
        mgr2 = SnapshotManager.load(f)
        assert mgr2.runs[0].params == p
        np.testing.assert_array_equal(mgr2.runs[0].snapshots, fom_base.snapshots)


# ---------------- 仿射装配 ----------------
class TestAffine:
    def test_affine_matches_direct(self):
        """仿射装配 A(μ), b(μ) 与直接构造逐元素一致（对多组参数）。"""
        ops = affine_operators()
        for mu in [dict(k=1.5, h_right=1000.0, q_gen=5.0e4),
                   dict(k=2.7, h_right=333.0, q_gen=1.2e5)]:
            p = Heat1DParams(**mu)
            A_ref, b_ref = build_continuous_operators(p)
            A_aff, b_aff = ops.assemble(mu)
            assert abs((A_ref - A_aff).max()) < 1e-12
            np.testing.assert_allclose(b_ref, b_aff, rtol=0, atol=1e-12)


# ---------------- ROM 求解器 ----------------
class TestROM:
    def _build(self, basis, r=2):
        return ReducedModelBuilder().build(basis, affine_operators(), r)

    def test_interpolation_accuracy(self, basis, fom_base):
        """同参数内插：框架版 ROM vs FOM 相对误差 < 1%（复现任务 #4 结果）。"""
        model = self._build(basis)
        p = Heat1DParams()
        T0 = np.full(p.n_x, p.t_init)
        T0[0] = p.t_left
        res = ROMSolver(dt=p.dt, t_end=p.t_end).solve(
            model, params_from_heat1d(p), T0, full_ops=affine_operators())
        num = np.linalg.norm(res.snapshots - fom_base.snapshots)
        den = np.linalg.norm(fom_base.snapshots - basis.mean[:, None])
        assert num / den < 0.01

    def test_prom_online_assembly_exact(self, basis):
        """pROM 恒等式：A_r(μ) 仿射装配 == 对装配后 A(μ) 直接投影。"""
        model = self._build(basis, r=3)
        ops = affine_operators()
        mu = dict(k=2.2, h_right=777.0, q_gen=7.7e4)
        A_r, f_r = model.assemble_reduced(mu)
        A_full, b_full = ops.assemble(mu)
        V = model.V
        np.testing.assert_allclose(A_r, np.asarray(V.T @ A_full @ V), atol=1e-12)
        np.testing.assert_allclose(f_r, V.T @ (A_full @ model.mean + b_full), atol=1e-12)

    def test_prom_unseen_parameter(self, fom_base):
        """pROM 端到端：k=1.5/2.5 训练 → 在线求解未见的 k=2.0（零新投影）。"""
        mgr = SnapshotManager()
        for k in (1.5, 2.5):
            p = Heat1DParams(k=k)
            fom = solve_fom(p)
            mgr.add_run(params_from_heat1d(p), fom.t, fom.snapshots, tag=f"train-k{k}")
        basis = PODDecomposer().fit(mgr.basis_input())
        model = ReducedModelBuilder().build(basis, affine_operators(),
                                            basis.rank_for_energy(0.999))

        p_mid = Heat1DParams(k=2.0)
        T0 = np.full(p_mid.n_x, p_mid.t_init)
        T0[0] = p_mid.t_left
        res = ROMSolver(dt=p_mid.dt, t_end=p_mid.t_end).solve(
            model, params_from_heat1d(p_mid), T0, full_ops=affine_operators())
        fom_mid = solve_fom(p_mid)
        num = np.linalg.norm(res.snapshots - fom_mid.snapshots)
        den = np.linalg.norm(fom_mid.snapshots - basis.mean[:, None])
        assert num / den < 0.05  # 参数内插点，误差应明显小于外推（任务 #4 F3 为 7.2%）

    def test_monitor_hook_fires(self, basis):
        """监控回调埋点：注入恒报警监控，应每个时间步产生告警。"""
        model = self._build(basis)
        p = Heat1DParams()
        T0 = np.full(p.n_x, p.t_init)
        T0[0] = p.t_left

        def always_alert(step, a, res):
            return f"test alert at step {step}"

        res = ROMSolver(dt=p.dt, t_end=p.t_end).solve(
            model, params_from_heat1d(p), T0,
            monitor=always_alert, full_ops=affine_operators())
        assert len(res.alerts) == p.n_t - 1
        assert res.alerts[0]["step"] == 1
