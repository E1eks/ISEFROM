# 同学 A 数学基础笔记（任务 #1 / A3）

> 读者/作者：同学 A ｜ 日期：2026-11-17 ｜ 配套：B 的笔记 `docs/math_notes_B.md`（起点）、进阶补丁 §7
> 分工要求：在 B 的笔记基础上更深——补能量估计、Gronwall 不等式、L2/H1 范数（认证界推导要用）；本文含手推 Galerkin 投影 + 能量估计各一遍（§4、§5）。

---

## 1. SVD 与 POD

任意矩阵 $S_c \in \mathbb{R}^{m\times n}$（中心化快照矩阵，m=101 空间点，n=601 时刻）：

$$S_c = U\Sigma V^T = \sum_{i=1}^{\min(m,n)} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

- $U$：左奇异向量（空间模态），$U^TU=I$
- $\Sigma=\mathrm{diag}(\sigma_i)$，$\sigma_1\ge\sigma_2\ge\dots\ge0$（$\sigma_i^2$ = 第 i 模态携带的方差）
- $V$：右奇异向量（时间系数）

**Eckart–Young 定理**：rank-r 截断是所有秩 r 矩阵中 Frobenius 误差最小的，且 $\|S_c - S_r\|_F = \sqrt{\sum_{i>r}\sigma_i^2}$。→ "99.9% 能量选 rank"的依据。实测（`examples/verify_svd_theorems.py`）：r=1 覆盖 99.883%，r=2 达 99.997%，r=3 达 99.9999%；基正交性偏差 4.4e-16（机器精度）；第一模态捕获能量比随机方向好 169 倍（Rayleigh 商：$\sigma_1^2 = \max_{\|v\|=1}\|S_c^T v\|^2$）。

**POD 基的物理含义（答辩话术）**：POD 基是快照协方差矩阵的主特征方向，按时间平均能量排序的最优正交模态；第一模态是温度涨落最集中的空间形态（我们算例里即"右端高、左端被冷板钳位"的主导热形态）。三者等价：POD 基 = 快照矩阵左奇异向量 = 快照协方差矩阵 $C=\frac{1}{n}S_cS_c^T$ 的特征向量（$\lambda_i=\sigma_i^2/n$）。

## 2. PDE 分类

二阶线性 PDE $a\,\partial_{xx}u + b\,\partial_{xy}u + c\,\partial_{yy}u + \dots = f$，按 $b^2-4ac$ 分类：

| 类型 | 判别式 | 代表 | 传播特性 | 本项目 |
|---|---|---|---|---|
| 椭圆 | <0 | 泊松 $\nabla^2u=f$ | 全域瞬时耦合 | 任务 #12 |
| 抛物 | =0 | 热传导 $\partial_t u=\alpha\nabla^2u+s$ | 扩散、抹平 | 任务 #3/#11 电池热 |
| 双曲 | >0 | 波动 $\partial_{tt}u=c^2\nabla^2u$ | 有限速、保持间断 | 不做（ROM 禁区） |

对 ROM 的意义：抛物型扩散抹平高频 → 奇异值谱衰减极快（我们 $\sigma_1/\sigma_2\approx30$）→ 极少模态即高精度；双曲型保持间断 → Kolmogorov n-宽度衰减慢 → 模态数爆炸 + 吉布斯震荡（局限性章节 #25 的理论解释）。

## 3. 有限差分

Taylor 展开 $T_{i\pm1} = T_i \pm \Delta x T_i' + \frac{\Delta x^2}{2}T_i'' \pm \dots$，两式相加消奇数阶项：

$$T_i'' = \frac{T_{i-1}-2T_i+T_{i+1}}{\Delta x^2} + \mathcal{O}(\Delta x^2)$$

→ `heat_1d.py` 三对角 `main=-2, off=1` 的来源，二阶精度。

时间积分：显式 Euler 要求 $\mathrm{Fo}=\alpha\Delta t/\Delta x^2 \le 1/2$（CFL）；隐式 Euler（我们用的）每步解 $(I-\Delta tA)T^{n+1}=T^n+\Delta t b$，无条件稳定但时间只有一阶。升阶可换 Crank–Nicolson（二阶，突加源项会有弱震荡）。

## 4. 手推：Galerkin 投影（不看笔记默写版）★

全阶系统（半离散，$T\in\mathbb{R}^N$，N=101）：$\dot T = AT + b$

**第 1 步·降阶展开**：$T(t) \approx \bar T + Va(t)$，$V=[u_1,\dots,u_r]\in\mathbb{R}^{N\times r}$（列 = SVD 左奇异向量），$a(t)\in\mathbb{R}^r$。

**第 2 步·代入定义残差**：$R(a) = V\dot a - A(\bar T + Va) - b \neq 0$（N 维残差，r 个自由度，不可能逐点为零）。

**第 3 步·Galerkin 条件**：强制残差与基空间正交：$V^TR=0 \Rightarrow V^TV\dot a = V^TAVa + V^T(A\bar T + b)$。

**第 4 步·用 $V^TV=I_r$**：

$$\boxed{\dot a = \underbrace{(V^TAV)}_{A_r:\ r\times r}\,a + \underbrace{V^T(A\bar T+b)}_{f_r}}$$

N×N → r×r。离线算一次 $A_r, f_r$（贵），在线对每个新参数积分小 ODE（便宜）——离线-在线范式。

**追问准备**：测试空间=试探空间故称 Galerkin（不同则 Petrov–Galerkin）；结构保持——$K$ SPD + $V$ 列满秩 ⇒ $y^T(V^TKV)y = (Vy)^TK(Vy) > 0$ ⇒ $V^TKV$ SPD ⇒ ROM 无条件稳定（实测 r=3 的 $A_r$ 对称部分特征值全负：−1.41e-2, −2.77e-3, −2.00e-4）；会被破坏的情形：基不满秩 / collocation 代替 Galerkin / hyper-reduction 不当。

**误差来源（接 #14）**：残差只是被强制垂直于基空间。$\rho=\|(I-VV^T)(AT_{rom}+b-\dot T_{rom})\|$：$VV^T$ 是到 span(V) 的正交投影（幂等），$I-VV^T$ 提取基表示不了的分量——不需要全阶解、在线可算，这就是运行时误差监控的出发点。

## 5. 手推：能量估计 + Gronwall（B 笔记没有，A3 要求补的部分）★

**设定**：$\partial_t T = \alpha\nabla^2 T + f$（Ω 上），Dirichlet 边界 $T|_{\partial\Omega}=0$，初值 $T_0$。

**第 1 步（乘 T 积分）**：$\int_\Omega T\partial_tT = \alpha\int_\Omega T\nabla^2T + \int_\Omega fT$

**第 2 步（分部积分）**：左边 $=\frac12\frac{d}{dt}\|T\|_{L^2}^2$；右边第一项 $=-\alpha\|\nabla T\|_{L^2}^2$（边界项因 $T|_{\partial\Omega}=0$ 消失）：

$$\frac12\frac{d}{dt}\|T\|_{L^2}^2 + \alpha\|\nabla T\|_{L^2}^2 = (f,T)$$

**第 3 步（Poincaré）**：$\|\nabla T\|_{L^2}\ge C_P\|T\|_{L^2}$（Dirichlet 下成立）⇒ $\frac12\frac{d}{dt}\|T\|^2 + \alpha C_P^2\|T\|^2 \le \|f\|\|T\|$。

**第 4 步（Young）**：$\|f\|\|T\| \le \frac{\alpha C_P^2}{2}\|T\|^2 + \frac{1}{2\alpha C_P^2}\|f\|^2$，代回乘 2：

$$\frac{d}{dt}\|T\|^2 + \alpha C_P^2\|T\|^2 \le \frac{1}{\alpha C_P^2}\|f\|^2$$

**第 5 步（Gronwall 收尾）**：对 $y=\|T\|^2$，$a=-\alpha C_P^2$，$b=\|f\|^2/(\alpha C_P^2)$ 用

$$y'(t)\le ay(t)+b(t) \Rightarrow y(t)\le e^{at}y(0)+\int_0^t e^{a(t-s)}b(s)\,ds$$

得 $\|T(t)\|^2 \le e^{-\alpha C_P^2 t}\|T_0\|^2 + \frac{1}{\alpha C_P^2}\int_0^t e^{-\alpha C_P^2(t-s)}\|f(s)\|^2ds$。

**读法**：f=0 时能量指数衰减（稳定性+耗散率 $\alpha C_P^2$）；f≠0 时解被初值与源项的"折扣累积"控制。

**连续↔离散对应**（review 时用）：$\|\nabla T\|^2\ge0$ ↔ $K$ SPD（注意只在消去边界的内部子矩阵上检查，见 §6 的坑）；Poincaré 常数 ↔ $\lambda_{\min}(K)$；误差方程 $\dot e=Ae+r$ + Gronwall ↔ 认证界 $\Delta(\mu)=\|r\|_{X'}/\beta_{LB}$。

## 6. 范数与正定性（认证界推导要用）

向量范数：$\|x\|_1=\sum|x_i|$，$\|x\|_2=\sqrt{\sum x_i^2}$，$\|x\|_\infty=\max|x_i|$（最坏点=我们的最高温度指标）。
矩阵范数：$\|A\|_F=\sqrt{\sum a_{ij}^2}=\sqrt{\sum\sigma_i^2}$（快照误差度量）；$\|A\|_2=\max_{\|x\|=1}\|Ax\|=\sigma_{\max}(A)$（算子最大放大倍数）。
两条反复用的不等式：Cauchy–Schwarz $|x\cdot y|\le\|x\|\|y\|$；相容性 $\|Ax\|_2\le\|A\|_2\|x\|_2$。

**SPD**：对称 $K$ 正定 ⟺ $x^TKx>0\ \forall x\neq0$ ⟺ 特征值全正。物理上 $T^TKT$ 是能量耗散率。实测 1D 算例内部子矩阵 $\lambda_{\min}=1.89\times10^{-4}>0$。⚠️ **坑**：含边界行的完整 K 不对称（Dirichlet 行置零、Robin 行改写），能量分析必须在消去边界自由度的内部子矩阵上做（接口文档 `interface_certified_bound.md` 里 B 踩过的伪负特征值 -0.031 就是这件事）。SPD 三个礼物：解唯一 / 能量估计可做 / CG 可用。

**L² vs H¹**：$L^2$ 只量函数值大小（$\int u^2$）；$H^1$ 还量光滑性（$\int(u^2+|\nabla u|^2)$）；能量范数 $\|u\|_E^2=a(u,u)$ 是问题的"自然尺子"。认证界 $\Delta(\mu)=\|r\|_{X'}/\beta_{LB}$ 中 X 的选择直接影响 effectivity 紧致度（目标 η∈[1,10]）。

**弱形式（#12 用）**：强形式 $-\nabla\cdot(k\nabla u)=f$ → 乘测试函数 v（$v|_{\partial\Omega}=0$）积分 → 分部积分把一阶导搬到 v 上、边界项消失 → $a(u,v):=\int k\nabla u\cdot\nabla v = \int fv =:(f,v)\ \forall v$。好处：只要一阶导（双材料 k 分段常数也有意义）；边界条件自动分类；FEM 与 POD-Galerkin 的共同母语（子空间 $V_h$ 取分片多项式=FEM，取 POD 基=我们的 ROM；$K_{ij}=a(\phi_j,\phi_i)$）。

## 7. 自测（不假思索标准）

1. SVD 三因子的维度与含义？（§1）
2. "99.9% 能量"选 rank 依据哪个定理？（§1 Eckart–Young）
3. 手推 $\dot T=AT+b \Rightarrow \dot a=A_ra+f_r$，5 分钟不看书。（§4）
4. 抛物 vs 双曲谁对 POD 友好？用奇异值谱解释。（§2）
5. 隐式 Euler 为何允许大步长？代价？时间精度几阶？（§3）
6. $I-VV^T$ 的几何意义？为何能在线监控误差？（§4）
7. 证明：$K$ SPD + $V$ 列满秩 ⇒ $V^TKV$ SPD；何时被破坏？（§4）
8. 能量估计五步里 Poincaré 和 Young 各用在哪一步、为什么需要？（§5）
9. 写出 Gronwall 并解释它在 $\dot e=Ae+r$ 上如何给出 $\Delta(\mu)$ 形状的界。（§5）
10. 为什么能量分析必须在消去边界的内部子矩阵上做？（§6 的坑）
11. 弱形式比强形式"弱"在哪？边界项去哪了？（§6）
12. $X=L^2$ vs $X=H^1$ 时 $\Delta(\mu)$ 的差别，对 effectivity 的影响？（§6）

---

*依据材料：B 的笔记 `math_notes_B.md` + 进阶补丁 §7 + 接口文档 `interface_certified_bound.md`；推导部分（§4/§5）为按 A3 要求独立重推整理。*
