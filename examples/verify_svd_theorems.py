"""
数学基础验证实验（配合 docs/math_notes_B.md §1 阅读）
======================================================
用任务 #3 的真实快照数据验证 SVD/POD 的三个核心定理：
  1. Eckart-Young：截断重构误差 = 被丢弃奇异值平方和的根
  2. POD 基正交性：V^T V = I
  3. 第一模态是最优逼近方向（Rayleigh 商极大化）

运行: python examples/verify_svd_theorems.py
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    S = np.load(ROOT / "snapshots_1d_baseline.npy")
    Sc = S - S.mean(axis=1, keepdims=True)
    U, sig, Vt = np.linalg.svd(Sc, full_matrices=False)

    print("=" * 64)
    print("验证 1: Eckart-Young 定理")
    print("  ||S - S_r||_F 应严格等于 sqrt(σ_{r+1}² + ... + σ_n²)")
    print("=" * 64)
    print(f"{'r':>3} | {'实际误差':>14} | {'理论值':>14} | {'累积能量':>10}")
    for r in [1, 2, 3, 5, 10]:
        err_actual = np.linalg.norm(Sc - (U[:, :r] * sig[:r]) @ Vt[:r])
        err_theory = np.sqrt(np.sum(sig[r:] ** 2))
        energy = np.sum(sig[:r] ** 2) / np.sum(sig ** 2)
        print(f"{r:>3} | {err_actual:>14.6e} | {err_theory:>14.6e} | {energy:>10.5%}")

    print()
    print("=" * 64)
    print("验证 2: POD 基正交性  V^T V = I")
    print("=" * 64)
    dev = np.abs(U[:, :4].T @ U[:, :4] - np.eye(4)).max()
    print(f"  前4模态 V^T V 与单位阵的最大偏差 = {dev:.2e}（机器精度级别）")

    print()
    print("=" * 64)
    print("验证 3: 第一模态 = 最优一维逼近方向（Rayleigh 商极大化）")
    print("  σ_1² = max_{||v||=1} ||S_c^T v||²")
    print("=" * 64)
    rng = np.random.default_rng(0)
    rand_dirs = rng.normal(size=(5, Sc.shape[0]))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    rand_energy = [np.sum((Sc.T @ v) ** 2) for v in rand_dirs]
    print(f"  σ_1²（第一模态捕获能量）= {sig[0] ** 2:.4e}")
    print(f"  5 个随机方向捕获能量均值 = {np.mean(rand_energy):.4e}")
    print(f"  第一模态比随机方向好 {sig[0] ** 2 / np.mean(rand_energy):.0f} 倍")
    print()
    print("  物理含义：σ_1 对应的方向是温度场涨落能量最集中的空间分布，")
    print("  即'主导热形态'——这就是 POD 基物理含义的严格数学表述。")


if __name__ == "__main__":
    main()
