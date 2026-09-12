# 同学 B 数学基础笔记（任务 #1 配套）

> 目标不是变成数学家，而是达到两条铁律标准：
> ① 能手推 Galerkin 投影公式；② 能讲清 POD 基的物理含义；③ 能听懂同学 A 的推导并 review
> 所有结论都用你已跑过的代码/数据验证过，引用处标注了对应脚本。

---

## 1. SVD 与 POD：一张快照矩阵的"主成分"

### 1.1 SVD 定理

任意矩阵 $S_c \in \mathbb{R}^{m \times n}$（我们：中心化快照矩阵，$m=101$ 空间点，$n=601$ 时刻）可分解为：

$$S_c = U \Sigma V^T = \sum_{i=1}^{\min(m,n)} \sigma_i \, \mathbf{u}_i \mathbf{v}_i^T$$

- $U = [\mathbf{u}_1, \dots]$：左奇异向量，$U^T U = I$（**空间模态**，每个 $\mathbf{u}_i$ 是一种温度空间分布）
- $\Sigma = \mathrm{diag}(\sigma_i)$，$\sigma_1 \ge \sigma_2 \ge \dots \ge 0$（**能量**，$\sigma_i^2$ = 第 $i$ 模态携带的方差）
- $V$：右奇异向量（**时间系数**，每个 $\mathbf{v}_i$ 是该模态随时间的演化）

### 1.2 三个已验证的核心事实（examples/verify_svd_theorems.py）

**① Eckart-Young 定理**——最优低秩逼近： rank-$r$ 截断 $S_r = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ 是所有秩 $r$ 矩阵中 Frobenius 误差最小的，且

$$\|S_c - S_r\|_F = \sqrt{\sum_{i>r} \sigma_i^2}$$

你的数据实证（误差 = 理论值，小数点后完全一致）：

| r | 重构误差 $\|S_c - S_r\|_F$ | 累积能量 |
|---|--------------------------|----------|
| 1 | 3.20e+01 | 99.883% |
| 2 | 4.83e+00 | 99.997% |
| 3 | 1.11e+00 | 99.9999% |

→ 这就是"能量阈值选 rank"的依据：选 $r$ 使 $\sum_{i\le r}\sigma_i^2 / \sum \sigma_i^2 \ge 99.9\%$。

**② 基正交性**：$V^T V = I$ 偏差 4.4e-16（机器精度）。Galerkin 投影公式里 $V^T V = I$ 会被反复用到。

**③ 第一模态是最优方向（Rayleigh 商）**：

$$\sigma_1^2 = \max_{\|\mathbf{v}\|=1} \|S_c^T \mathbf{v}\|^2$$

你的数据：第一模态捕获能量 8.72e+05，随机方向仅 5.16e+03——**好 169 倍**。

> **答辩话术（POD 基物理含义）**：POD 基是快照协方差矩阵的主特征方向，即按时间平均能量排序的最优正交模态；第一模态是温度涨落最集中的空间形态（我们的算例里就是"右端高、左端被冷板钳位"的主导热形态），后续模态依次捕捉剩余的小尺度修正。

### 1.3 SVD ⇄ POD 的对应关系

POD 的经典定义：求正交基最大化平均投影能量。结论：**POD 基 = 快照矩阵的左奇异向量 = 快照协方差矩阵 $C = \frac{1}{n}S_c S_c^T$ 的特征向量**，特征值 $\lambda_i = \sigma_i^2 / n$。三者等价，计算上直接用 SVD 最稳。

---

## 2. PDE 分类：为什么我们的算例对 ROM 友好

二阶线性 PDE $a\,\partial_{xx}u + b\,\partial_{xy}u + c\,\partial_{yy}u + \dots = f$，按判别式 $b^2 - 4ac$ 分类：

| 类型 | 判别式 | 代表方程 | 信息传播 | 本项目算例 |
|------|--------|----------|----------|-----------|
| 椭圆型 | $<0$ | 泊松方程 $\nabla^2 u = f$ | 全域瞬时耦合（稳态） | 任务 #12 |
| 抛物型 | $=0$ | 热传导 $\partial_t u = \alpha \nabla^2 u + s$ | 扩散、抹平、无限速 | 任务 #3/#11（电池热）|
| 双曲型 | $>0$ | 波动方程 $\partial_{tt}u = c^2\nabla^2 u$ | 有限速度、保持间断 | **不做**（激波是 ROM 禁区）|

**对 ROM 的意义**（你在任务 #4 已亲身体会）：
- 抛物型扩散抹平高频 → 奇异值谱衰减极快（你的 $\sigma_1/\sigma_2 \approx 30$）→ 极少模态即高精度
- 双曲型保持间断 → Kolmogorov n-宽度衰减慢 → POD 需要的模态数爆炸 + 吉布斯震荡 → 这是局限性章节（任务 #25）的理论解释

---

## 3. 有限差分：从 Taylor 展开到你的代码

### 3.1 中心差分的推导（要求能手推）

对 $T(x)$ 在 $x_i$ 处 Taylor 展开：

$$T_{i\pm1} = T_i \pm \Delta x\, T'_i + \frac{\Delta x^2}{2}T''_i \pm \frac{\Delta x^3}{6}T'''_i + \frac{\Delta x^4}{24}T^{(4)}_i + \dots$$

两式相加消去奇数阶项：

$$T''_i = \frac{T_{i-1} - 2T_i + T_{i+1}}{\Delta x^2} + \underbrace{\mathcal{O}(\Delta x^2)}_{\text{截断误差：} -\frac{\Delta x^2}{12}T^{(4)}_i}$$

→ 这就是 [heat_1d.py](src/fom/heat_1d.py) 里三对角矩阵 `main=-2, off=1` 的来源，二阶精度。

### 3.2 时间积分与稳定性

半离散后 $\frac{dT_i}{dt} = \alpha\,\frac{T_{i-1}-2T_i+T_{i+1}}{\Delta x^2} + s_i$。

| 格式 | 公式 | 稳定性条件 |
|------|------|-----------|
| 显式 Euler | $T^{n+1} = T^n + \Delta t(\cdots)^n$ | $\mathrm{Fo} = \frac{\alpha\Delta t}{\Delta x^2} \le \frac{1}{2}$（CFL 条件）|
| **隐式 Euler（我们用的）** | $T^{n+1} = T^n + \Delta t(\cdots)^{n+1}$ | **无条件稳定** |

- 显式稳定性的直觉：扰动在显式更新中被逐格放大，$\mathrm{Fo} > 1/2$ 时放大因子 $>1$，数值爆炸
- 隐式每步解线性系统 $(I - \Delta t\, A)T^{n+1} = T^n + \Delta t\, b$，代价换稳定
- 你的参数：$\mathrm{Fo} = 0.188$，显式其实也稳定；但电池 2D 网格加密后 $\Delta x$ 减半 → Fo×4，隐式的价值就出来了
- 精度注意：隐式 Euler 时间只有一阶 $\mathcal{O}(\Delta t)$，要升阶可换 Crank-Nicolson（二阶，但对突加源项会有弱震荡——答辩可能问）

---

## 4. Galerkin 投影：必须能手推的核心推导 ⭐

### 4.1 推导（逐步）

全阶系统（半离散，$T \in \mathbb{R}^{N}$，$N=101$）：

$$\dot{\mathbf{T}} = A\mathbf{T} + \mathbf{b}$$

**第 1 步·降阶展开（lifting）**：温度 ≈ 均值 + $r$ 个模态的线性组合（$r=2$，$V = [\mathbf{u}_1, \mathbf{u}_2] \in \mathbb{R}^{N\times r}$）

$$\mathbf{T}(t) \approx \bar{\mathbf{T}} + V\mathbf{a}(t), \qquad \mathbf{a}(t) \in \mathbb{R}^r$$

**第 2 步·代入并定义残差**：

$$\mathbf{R}(\mathbf{a}) = V\dot{\mathbf{a}} - A(\bar{\mathbf{T}} + V\mathbf{a}) - \mathbf{b} \neq 0$$

$N$ 维残差只有 $r$ 个自由度 $\mathbf{a}$ 可调，不可能逐点为零。

**第 3 步·Galerkin 条件**：强制残差与基张成的空间**正交**：

$$V^T \mathbf{R} = 0 \quad\Longrightarrow\quad V^T V \dot{\mathbf{a}} = V^T A V \mathbf{a} + V^T(A\bar{\mathbf{T}} + \mathbf{b})$$

**第 4 步·用正交性 $V^T V = I_r$**：

$$\boxed{\;\dot{\mathbf{a}} = \underbrace{(V^T A V)}_{A_r:\ r\times r}\,\mathbf{a} + \underbrace{V^T(A\bar{\mathbf{T}} + \mathbf{b})}_{\mathbf{f}_r:\ r}\;}$$

$101\times101$ 的系统 → $2\times2$。**离线**阶段算一次 $A_r, \mathbf{f}_r$（贵），**在线**阶段对每个新参数只积分小 ODE（便宜）——这就是离线-在线范式。

### 4.2 与弱形式的关系（评委可能追问）

Galerkin 投影源自变分表述："求 $T_h \in \mathcal{V}_h$ 使残差对所有测试函数 $v \in \mathcal{V}_h$ 正交"。有限元里 $\mathcal{V}_h$ = 分片多项式空间；POD-Galerkin 里 $\mathcal{V}_h = \mathrm{span}\{\mathbf{u}_1,\dots,\mathbf{u}_r\}$。**同一思想，换了基**。测试空间 = 试探空间故称"Galerkin"（若不同则称 Petrov-Galerkin，A 若做稳定化可能涉及）。

### 4.3 误差从哪来（连接创新点②）

残差不等于零，只是被强制垂直于 $\mathcal{V}_h$。定义**投影残差**：

$$\rho = \big\|(I - VV^T)\big(A\mathbf{T}_{rom} + \mathbf{b} - \dot{\mathbf{T}}_{rom}\big)\big\|$$

- $VV^T$ 是到 $\mathrm{span}(V)$ 的正交投影矩阵（$(VV^T)^2 = VV^T$，幂等）
- $I - VV^T$ 提取残差中**基表示不了**的分量——不需要全阶解，在线可算
- 这就是任务 #14 运行时误差估计的出发点，也是失败日志 #01 留给 A 的问题

---

## 5. 自测题（达到"不假思索"才算过）

1. 写出 SVD 分解并说明三个因子各自维度与含义。
2. 为什么"99.9% 能量"能作为 rank 选取准则？依据哪个定理？
3. 手推：从 $\dot{\mathbf{T}} = A\mathbf{T} + \mathbf{b}$ 到 $\dot{\mathbf{a}} = A_r \mathbf{a} + \mathbf{f}_r$（5 分钟内，不看笔记）。
4. 抛物型 vs 双曲型，哪个对 POD 友好？用奇异值谱衰减解释。
5. 隐式 Euler 为什么允许大时间步？代价是什么？时间精度几阶？
6. 解释 $I - VV^T$ 的几何意义，它为什么能做在线误差监控？
7.（铁律题）给评委画一张"快照 → SVD → 基 → 投影 → 小 ODE"的流程图并口述。

**参考答案位置**：§1.3→题1；§1.2①→题2；§4.1→题3；§2→题4；§3.2→题5；§4.3→题6；架构文档§2→题7。

---

## 6. 下一步：任务 #2 论文阅读建议（与 A 分工）

| 篇目 | 为什么读 | 建议读者 |
|------|----------|----------|
| Benner, Gugercin, Willcox (2015) *A Survey of Projection-Based Model Reduction Methods*, SIAM Review | 投影降阶圣经，§3 POD 一节对照本笔记 | 两人都读 |
| Berkooz, Holmes, Lumley (1993) *The Proper Orthogonal Decomposition...*, Annu. Rev. Fluid Mech. | POD 原始文献，物理含义讲得最好 | B 读 |
| Chaturantabut & Sorensen (2010) *Nonlinear MOR via DEIM*, SIAM JSC | 任务 #16 非线性处理的技术源头 | A 主读，B 了解 |
| 电池热管理方向：搜 "POD reduced order battery thermal" 近 5 年综述 | 确认杀手级应用的新颖性缺口 | B 读 |

> 约定：每人读完用自己的话写 1 页总结进 `docs/literature/`，周日互相讲 15 分钟。

---

## 7. 进阶补丁：A 的世界里你会遇到的数学（2026-09-12 补）

> 定位：**听懂级**——能 review A 的推导、发现明显错误、答辩被追问不冷场。
> 验证脚本：`examples/verify_norms_and_spd.py`（全部数据为实测）。

### 7.1 向量范数与矩阵范数

**向量范数**（$\mathbf{x} \in \mathbb{R}^n$）：

$$\|\mathbf{x}\|_1 = \sum_i |x_i|, \qquad \|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}, \qquad \|\mathbf{x}\|_\infty = \max_i |x_i|$$

直觉：2-范数是"长度"（能量型，最常用）；∞-范数是"最坏点"（我们的最高温度指标）；1-范数稀疏性用（本项目基本不用）。

**矩阵范数**（两个就够）：

| 范数 | 定义 | 与奇异值的关系 | 何时用 |
|------|------|---------------|--------|
| Frobenius $\|A\|_F$ | $\sqrt{\sum_{ij} a_{ij}^2}$ | $= \sqrt{\sum_i \sigma_i^2}$（已验证：差 0.00e+00） | 快照/温度场误差度量——我们一直在用 |
| 诱导 2-范数 $\|A\|_2$ | $\max_{\|x\|=1}\|Ax\|$ | $= \sigma_{\max}(A)$（已验证：随机方向从下方逼近 6.41） | 算子"最大放大倍数"，稳定性分析 |

**两条反复用的不等式**：

$$\text{Cauchy-Schwarz: } |\mathbf{x}\cdot\mathbf{y}| \le \|\mathbf{x}\|\,\|\mathbf{y}\|, \qquad \text{相容性: } \|A\mathbf{x}\|_2 \le \|A\|_2\,\|\mathbf{x}\|_2$$

（已验证：200 个随机方向全部 ≤1，等号只在右奇异向量方向取到。）A 推认证界时每一步放缩基本都是这两条。

### 7.2 正定性（SPD）：我们的世界为什么稳定

**定义**：对称矩阵 $K$ 正定 $\iff \mathbf{x}^T K \mathbf{x} > 0\ \forall \mathbf{x} \neq 0 \iff$ 特征值全正。

**物理含义**：热扩散是耗散过程——$\mathbf{T}^T K \mathbf{T}$ 就是能量耗散率，恒正意味着"热量只能摊平不能自发聚集"。

**实测**（验证 4）：我们 1D 算例的刚度阵 $K$，内部子矩阵对称、$\lambda_{\min} = 1.89\times10^{-4} > 0$ → SPD 成立。注意 $\lambda_{\min}$ 很小 → 条件数大 ≈ 物理上热传导慢（$k=1.5$ 的电芯 vs 金属）。

**一个坑（验证脚本里有注释）**：含边界行的完整 $K$ 不对称（Dirichlet 行置零、Robin 行改写），直接取对称部分会看到微小正/伪特征值。**能量分析必须在消去边界自由度的内部子矩阵上做**——A 推能量估计时会反复强调这点，你看他推导时要检查他是不是这么干的。

**SPD 给你的三个免费礼物**：① 解唯一（$K$ 可逆，椭圆算例直接 `spsolve`）② 能量估计可做（§7.4）③ 共轭梯度法可用（大网格时换迭代求解器的理论基础）。

### 7.3 结构保持：$V^T A V$ 继承定性 ⭐（SIAM 评委必问点）

**半页推导**（#16 的"半天活"核心）：设 $K$ SPD，$V$ 列满秩（我们的 POD 基列正交，当然满秩），则对任意 $\mathbf{y} \neq 0$：

$$\mathbf{y}^T (V^T K V) \mathbf{y} = (V\mathbf{y})^T K (V\mathbf{y}) > 0$$

因为 $V\mathbf{y} \neq 0$（列满秩）且 $K$ SPD。所以 $V^T K V$ 也 SPD。**降阶系统继承了全阶的稳定性结构**——这就是"结构保持"。

**实测**（验证 5）：我们 r=3 的 $A_r$ 对称部分特征值 $[-1.41\times10^{-2}, -2.77\times10^{-3}, -2.00\times10^{-4}]$ 全负（$-K$ 约定下等价于 SPD）→ 我们的 ROM 无条件稳定，这不是"数值上没炸"，是有证明的。

**什么时候会坏**（评委追问用）：① 基不正交/列不满秩 ② 配点法（collocation）代替 Galerkin ③ 非线性项 hyper-reduction 处理不当——任务 #17 弱非线性震荡的理论解释就在这里，也是支线 S1 的入口。

### 7.4 能量估计与 Gronwall 不等式（看懂级）

**能量估计第一步**（A 推导的固定开场）：热方程 $\partial_t T = \alpha \nabla^2 T + f$ 两边乘 $T$、对区域积分、分部积分：

$$\frac{1}{2}\frac{d}{dt}\|T\|_{L^2}^2 + \underbrace{\alpha \|\nabla T\|_{L^2}^2}_{\text{耗散项} \ge 0} = (f, T) \le \|f\|\,\|T\|$$

$f = 0$ 时能量单调不增 → **稳定性**。耗散项就是 SPD 性的连续版本。

**Gronwall 不等式**（时间累积的"复利公式"）：

$$y'(t) \le a\,y(t) + b(t) \;\Longrightarrow\; y(t) \le e^{at}\left(y(0) + \int_0^t e^{-as} b(s)\,ds\right)$$

**A 会怎么用**：推 $E_{param}$ 上界时，误差方程 $\dot{\mathbf{e}} = A\mathbf{e} + \mathbf{r}$（误差 = 算子作用 + 残差驱动）形式上和解的存在性估计一样，用 Gronwall 把"每步残差"累积成"全程误差界"——残差 $\|\mathbf{r}\|$ 就是你 monitor 埋点里已经在算的量，整条链就闭环了。

### 7.5 弱形式 10 分钟版（为 #12 review 准备）

**强形式**（#12 泊松）：$-\nabla\cdot(k\nabla u) = f$，要求 $u$ 二阶可导。

**三步转弱形式**：
1. 乘任意测试函数 $v$（要求 $v|_{\partial\Omega} = 0$），对整个区域积分
2. 分部积分（高维 Green 公式），把一阶导"搬"到 $v$ 上，边界项因 $v=0$ 消失
3. 得弱形式：求 $u$（$u|_{\partial\Omega}=0$）使

$$a(u, v) := \int_\Omega k\,\nabla u \cdot \nabla v \,dx = \int_\Omega f v\,dx =: (f, v) \qquad \forall v$$

**三个好处**：① 只要求一阶导（$k$ 分段常数也有意义——双材料算例正是）② 边界条件自动分类（Dirichlet 强加进函数空间，Neumann/Robin 自动出现在边界项）③ 这是 FEM 和 Galerkin 的共同母语：在子空间 $V_h$ 中求 $u_h$ 使 $a(u_h, v) = (f,v)\ \forall v \in V_h$，$V_h$ 取分片多项式就是 FEM，取 POD 基就是我们的 ROM。A 的 #12 推导就是把 $a(\cdot,\cdot)$ 写成矩阵 $K_{ij} = a(\phi_j, \phi_i)$。

### 7.6 $L^2$ vs $H^1$：认证界里的范数选择

| 范数 | 定义 | 直觉 |
|------|------|------|
| $L^2$ | $\|u\|^2 = \int u^2$ | 只看函数值大小 |
| $H^1$ | $\|u\|^2 = \int (u^2 + \|\nabla u\|^2)$ | 还看"平不平滑" |
| 能量范数 | $\|u\|_E^2 = a(u,u)$ | 问题的"自然尺子"（椭圆问题 ≈ $H^1$ 半范数）|

memo §1 里 $\Delta(\mu) = \|r\|_{X'} / \beta_{LB}$ 的 $X$ 选哪个，直接影响界的紧致度（effectivity 能否落在 1~10）——这是 A 做 #14 时要权衡的，你 review 时问一句"你选的什么范数、为什么"就能逼他把这事想清楚。

### 7.7 自测题（追加 4 道，同样"不假思索"标准）

8. $\|A\|_2$ 和 $\|A\|_F$ 各等于什么？为什么我们度量快照误差用 F 范数而分析算子稳定性用 2-范数？
9. 五行内证明：$K$ SPD + $V$ 列满秩 $\Rightarrow V^T K V$ SPD。这个性质什么时候会被破坏？
10. 能量估计"乘 $T$ 再积分"得到的耗散项是什么？它非负为什么等价于稳定性？
11. 弱形式相对强形式"弱"在哪？分部积分时边界项去哪了？

**答案位置**：题8→§7.1；题9→§7.3；题10→§7.4；题11→§7.5。

### 7.8 进阶路线（事件驱动，不用一次补完）

| 主题 | 触发时机 | 深度 | 读物 |
|------|----------|------|------|
| 弱形式 | **本周议程前**（#12 review） | 听懂级 | Ern《Theory and Practice of Finite Elements》第 1-2 章；或 deal.II tutorial step-3 的数学部分（免费） |
| 能量估计 + Gronwall | A 写完方向②初稿时 | 跟着推一遍 | Evans《PDE》§7.1（读"energy estimates"小节即可） |
| Lax-Milgram 定理 | 若 A 的 #14 需要存在性论证 | 知道陈述 | 同上 §6.2 |
| SCM（β_LB 估计） | A 的 #14 实现阶段 | 知道存在 | memo §1 文献 1 |
| n-width / DEIM / 变分积分器 | 支线 S1-S3 | 选读 | Binev 2011、Chaturantabut 2010 |
