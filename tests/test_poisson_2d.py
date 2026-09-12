"""
泊松 2D 算例测试（任务 #12）
=============================
运行: python -m pytest tests/test_poisson_2d.py -v
覆盖：仿射装配恒等式、最大值原理、ROM 参数插值精度。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fom.poisson_2d import (PoissonParams, build_affine_operators,
                                params_from_poisson, solve_fom)
from src.rom.pod import PODDecomposer
from src.rom.reduced import ReducedModelBuilder, solve_steady
from src.rom.snapshot import SnapshotManager


def direct_operators(p: PoissonParams):
    """参考实现：直接组装 K u = f（K 为 SPD 刚度阵）。"""
    from src.fom.poisson_2d import build_region
    import scipy.sparse as sp

    n = p.n
    nn = n * n
    h = 1.0 / (n - 1)
    inner = build_region(p).ravel()
    k_node = np.where(inner, p.k1, p.k2)
    K = sp.lil_matrix((nn, nn))
    f = np.zeros(nn)

    def idx(iy, ix):
        return iy * n + ix

    boundary = np.zeros(nn, dtype=bool)
    for iy in range(n):
        for ix in range(n):
            i = idx(iy, ix)
            is_b = ix in (0, n - 1) or iy in (0, n - 1)
            if is_b:
                boundary[i] = True
                continue
            for jy, jx in ((iy, ix + 1), (iy, ix - 1), (iy + 1, ix), (iy - 1, ix)):
                j = idx(jy, jx)
                kf = 0.5 * (k_node[i] + k_node[j])
                K[i, i] += kf / h**2
                K[i, j] -= kf / h**2
            f[i] = p.f_amp * np.sin(np.pi * (ix * h)) * np.sin(np.pi * (iy * h))
    for i in np.where(boundary)[0]:
        K[i, i] = 1.0
    return K.tocsr(), f


class TestAffineIdentity:
    @pytest.mark.parametrize("mu", [
        dict(k1=5.0, k2=1.0, f_amp=100.0),
        dict(k1=0.3, k2=8.0, f_amp=250.0),
    ])
    def test_affine_matches_direct(self, mu):
        p = PoissonParams(**mu)
        K_ref, f_ref = direct_operators(p)
        ops = build_affine_operators(p)
        A, b = ops.assemble(mu)   # A = -K 约定
        assert abs((A + K_ref).max()) < 1e-10
        np.testing.assert_allclose(b, f_ref, atol=1e-10)


class TestPhysics:
    def test_maximum_principle(self):
        res = solve_fom(PoissonParams())
        assert res.solution.min() > -1e-8
        assert res.solution.max() > 0

    def test_diagonal_symmetry(self):
        res = solve_fom(PoissonParams())
        U = res.field()
        assert np.abs(U - U.T).max() / U.max() < 1e-10


class TestPoissonROM:
    def test_rom_parametric_accuracy(self):
        """4 组角点训练 → 参数空间中心盲测（框架扩展性验证：换 PDE 不改核心代码）。"""
        mgr = SnapshotManager()
        corners = [(0.5, 0.5), (0.5, 5.0), (5.0, 0.5), (5.0, 5.0)]
        for k1, k2 in corners:
            p = PoissonParams(k1=k1, k2=k2)
            res = solve_fom(p)
            mgr.add_run(params_from_poisson(p), np.array([0.0]),
                        res.solution[:, None], tag=f"k1{k1}-k2{k2}")

        basis = PODDecomposer().fit(mgr.basis_input())
        r = basis.rank_for_energy(0.9999)  # 双材料高对比需 r=3（99.9% 只给 r=2，误差 2.1%）
        model = ReducedModelBuilder().build(basis, build_affine_operators(PoissonParams()), r)

        p_mid = PoissonParams(k1=2.0, k2=1.5, f_amp=300.0)
        u_rom = solve_steady(model, params_from_poisson(p_mid))
        u_fom = solve_fom(p_mid).solution
        err = np.linalg.norm(u_rom - u_fom) / np.linalg.norm(u_fom - basis.mean)
        assert err < 0.02, f"ROM 参数插值误差 {err:.3%} 超 2%"
