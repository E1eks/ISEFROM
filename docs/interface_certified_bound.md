# 认证误差界插件接口（已冻结 2026-10-20）

> 给同学 A：这是你的任务 #14（认证后验误差估计）插入框架的**唯一入口**。
> 代码：`src/rom/estimators.py` ｜ 测试：`tests/test_estimators.py`
> 数学背景：`docs/memo_A_certified_error_bound.md` §1-2（先读这个）

## 你要做的只有两件事

1. **写一个函数**，签名已冻结：

```python
from src.rom.estimators import BoundContext, BoundResult
from src.rom.reduced import ReducedModel

def scm_bound(model: ReducedModel, params: dict, ctx: BoundContext) -> BoundResult:
    # 残差轨迹不要自己算，用现成工具：
    #   抛物: norms, rom = residual_out_trajectory(model, params, ctx.T0, ctx.dt, ctx.t_end, ctx.full_ops)
    #   椭圆: res_norm, u_rom = steady_residual(model, params, ctx.full_ops)
    # β_LB 用 SCM 或解析下界（占位 spectral_beta_lb 的坑见下文）
    ...
    return BoundResult(delta=delta, beta_lb=beta, residual_norm=res_norm,
                       certified=True, method="scm", per_step=delta_traj)
```

2. **注册一行**：`ESTIMATOR_REGISTRY["scm"] = scm_bound`

之后全自动生效：流水线配置 `"error_monitor": {"estimator": "scm"}`、
误差预算证书（#26 `budget_certificate(certified_bound_fn=scm_bound)`）、
贪心采样（你的 #10 直接调同一函数）。

## 契约细则（违反任一条 = bug）

| 项 | 约定 |
|----|------|
| `delta` | **相对**误差界 Δ(μ)，参考范数 = 末时刻场的涨落范数 ‖u(·,t_end) − mean‖₂（与 pipeline 的 `rel_error`、budget 的 `true_error` 同约定，保证 η 可比） |
| `certified` | 严格上界才 `True`。**只有 certified=True 的 delta 才计入 `BudgetCertificate.total_bound`**——未认证的数混进总上界就是 failure log #02（η=0.4 说谎事件）重演 |
| η 自检 | η = Δ/真实误差 **必须恒 ≥ 1**；pipeline 在全阶验证时自动算 effectivity，certified=True 且 η<1 会在报告里打 warning |
| 抛物 ctx | `ctx.T0/dt/t_end` 必有值；椭圆 `kind="steady"` 时三者是 None |
| `beta_lb` | 你用的稳定性常数下界原值填进来（调试可追溯） |

## 占位实现留给你的坑（`spectral_beta_lb` 踩过的，别重踩）

- **算子非对称**：Dirichlet 边界行被清零（边界值由 `mean` 携带），A 整体非对称。
  `eigvalsh` 直接吃 A 会读错三角——必须先取对称部分 (K+Kᵀ)/2。
- **伪负特征值**：清零行与自由节点的耦合使对称部分混进一个假负特征值
  （实测 -0.031，而真 β≈1.8e-4）。这是边界约束的数学人造物，不是物理不稳定。
  严格做法：coercivity 定义在**约束空间**（满足齐次 Dirichlet 的自由节点空间）上——
  你推导时从 weak form 出发自然会得到这个，memo §1 的 X 范数就是这件事。
- **离散 vs 连续**：离散算子的谱常数是连续 coercivity 常数的**上界**方向不安全
  （β_离散 ≥ β_连续，直接拿来当下界会放大 Δ）。SCM 解决的就是这个问题。
- 抛物型的严格 Δ 不是 max_t‖r‖/β——需要时间积分 + Gronwall 结构
  （Rozza/Huynh 2008 §2-3，PDF 在 `docs/Huynh2018_reduced_basis_tutorial.pdf`）。

## 交付验收（#14 的 DoD）

1. `ESTIMATOR_REGISTRY` 里出现你的实现，`certified=True`
2. effectivity 报告：三算例 × 各自参数域内随机 20 点，η 直方图全部 ∈ [1, 10⁴]，中位数目标 < 50
3. 任何点 η<1 → 先查自己的实现，再怀疑框架（框架侧残差计算有单测覆盖）
4. 跑 `python -m pytest tests/ -q` 全绿 + 你的新测试
