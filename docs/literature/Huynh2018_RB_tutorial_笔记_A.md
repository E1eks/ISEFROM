# 文献笔记：Huynh, Pichi, Rozza (2018) RB 教程 §1–3

> 读者：同学 A ｜ 日期：2026-11-17 ｜ 对应任务：A4 / #2
> 文献：*Reduced basis approximation and a posteriori error estimation: applications to elasticity problems in several parametric settings*（arXiv:1801.06553），Rozza/Huynh/Patera 认证 RB 体系的教程版
> 阅读范围：§1 引言、§2 预备（含仿射分解与 truth 逼近）、§3 RB 方法（贪心 + Galerkin + 离线-在线）

---

## 1. 一页总结（自己的话）

**这篇文章做什么**：把"认证降基（certified Reduced Basis）"方法完整演示在线弹性力学参数化问题上。它和我们项目的对应关系几乎是 1:1 的——他们算弹性位移场，我们算电池温度场；骨架完全相同。

**§1 引言的核心论点**：工程设计需要"多查询"（many-query，成百上千组参数）和"实时"（real-time）场景下的 PDE 求解，高精度全阶模型（他们叫 truth approximation，本质就是 FEM 全阶解）太贵。RB 方法的四大支柱正好是我们任务清单的四个创新点：

1. **Galerkin 投影到参数流形的低维空间** → 我们的 POD-Galerkin（任务 #4/#9）
2. **贪心采样选基** → 我们的自适应快照采样（任务 #10）
3. **后验误差估计（rigorous and sharp bounds）** → 我们的认证误差界（任务 #14）
4. **离线-在线分解** → 我们的 pROM 仿射装配（任务 #9）

**§2 预备的关键概念**：

- **统一弱形式**：所有弹性问题（平面应力/平面应变、各向同性/正交各向异性、2D/3D 轴对称）都写成同一个抽象形式 a(u,v;μ) = f(v;μ)。这印证了我们架构文档的设计直觉——框架不该绑定具体 PDE，只面对双线性形式。
- **参考域 + 几何映射**：几何参数化的问题先映射回固定参考域，所有参数依赖被吸收进系数。我们的电池模型没有做几何参数化（流道布局是 Future Work），物理参数直接进系数，更简单。
- **仿射分解（affine decomposition）**：a(w,v;μ) = Σq Θq(μ)·aq(w,v)，矩阵层面 K(μ) = Σq θq(μ)·Kq。**这是整个离线-在线效率的来源**——参数无关的 Kq 离线投影一次存起来，在线只需要加权和。我们的 `src/fom/battery_2d.py` 做的"严格仿射分解"就是这个，而且我们验证了 E_aff = 0（`test_affine_matches_direct`）。
- **truth approximation**：取 FEM 网格 N 足够大，把 FE 解当作"真值"。RB 的误差都是相对 truth 解定义的——这正是我们 pipeline 里 `true_error` 的定义方式。

**§3 RB 方法的三个要点**：

1. **贪心算法**：μ_{n+1} = argmax η(μ)，每一步选**当前估计误差最大**的参数点补快照进基。基空间是嵌套的（hierarchical），快照做 Gram-Schmidt 正交化（我们用 SVD，数值上更稳，本质等价）。
2. **Galerkin 投影**：u_RB = Z·u_N，小刚度阵 K_RB = ZᵀKZ，右手边 F_RB = ZᵀF。重要结论：降阶刚度阵的条件数有界 γ0(μ)/α0(μ)，**与全阶维度 N 无关**——这就是 §7.3 结构保持的论文版本（SPD 投影后仍 SPD）。
3. **离线-在线**：离线算并存储所有参数无关小矩阵 ZᵀKqZ（存储量 O(Qa·N²max)）；在线对新 μ 只做加权和 + 解 N×N 小系统，**复杂度 O(N³)，与全阶维度 N 完全无关**。所以 N 可以取得任意大（全阶算多准都行）而不影响在线成本——这是"200–600× 加速比"的理论根据。

**对我们项目的直接启示**：
- 我们的贪心采样（#10）必须挂在认证误差界（#14）上，论文里 η(μ) 就是误差估计子——B 在 memo §3 里画的循环和这里逐行对应。
- 条件数结论回应了评委可能问的"降阶系统会不会病态"：只要全阶 coercive/SPD，降阶系统条件数有界。
- 我们比他们多走一步：他们的 η 只用于离线选基；我们把同一个 Δ(μ) 搬到**在线运行时**做监控告警（创新点②）。

**没读懂/待跟进**：§2.2 几何映射的 Jacobian 细节跳过了（我们用不上）；compliant output 的"能量内积"选取对 effectivity 的影响要在 #14 实现时回来核对。

## 2. 术语对照（答辩用）

| 论文术语 | 我们项目里的对应 |
|---|---|
| truth approximation | 全阶模型 FOM |
| RB space / snapshot | POD 基 / 快照 |
| greedy algorithm | 自适应采样（#10） |
| a posteriori error estimator η(μ) | 认证误差界 Δ(μ)（#14） |
| affine decomposition | 仿射分解 θq(μ)Aq |
| Offline-Online | 离线-在线范式 |
| coercivity constant α(μ) | 稳定性常数 β（我们热传导是 SPD，α≈最小特征值） |

## 3. 下一步

- §4（后验误差估计的两个成分：残差对偶范数 + coercivity 下界 SCM）是 #14 的实现蓝本，下次读。
- 周日给 B 讲 15 分钟：重点讲"四大支柱 ↔ 四个创新点"的对应表。
