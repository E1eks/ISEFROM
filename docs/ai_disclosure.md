# AI 工具使用披露声明（Living Document）

> 版本 v1.0（2026-09-12 起草）｜ 负责人：同学 B 起草，同学 A 会签
> 用途：ISEF Research Plan、论文致谢、展板、SRC 表格中的 AI 使用披露均以此文档为准
> 原则：**透明是唯一的保护色**。本项目遵守 ISEF 关于 AI 工具使用的披露要求；
> 具体披露格式以 2027 赛季 ISEF 官方规则与指导老师意见为准（任务 #6/#7 时核实）。

---

## 1. 使用的 AI 工具及用途

| 工具 | 用途 | 性质 |
|------|------|------|
| GitHub Copilot（VS Code） | 代码脚手架起草、单元测试生成、调试建议、文档模板起草 | 辅助工具 |
| DeepTutor（自部署学习助手） | 数学概念讲解、自测题生成、论文阅读辅助（仅用于学习，不用于项目产出） | 学习辅助 |

**明确未使用 AI 的环节**：核心实验设计（误差分解方案、失效实验矩阵）、
数据分析与结论解读、项目方向决策（"可信任的 ROM"主线定位、方向②选择）、
失败案例的归因判断——以上由两名学生主导完成。

## 2. 学生主体性保障机制

1. **逐行审阅**：AI 起草的全部代码均经学生逐行审阅，理解后方可提交。
2. **测试验证**：所有模块由学生设计并运行物理合理性检查与单元测试
   （18+ 项），AI 输出未通过学生验证的代码不得入库。
3. **代码主权计划**（2026-12 核心研究期开始前完成）：每位成员须能对
   所负责模块做逐行讲解（code walkthrough）；讲不清的部分自行重写，
   直至能独立解释。
4. **互相讲解制度**：每周日 15 分钟互讲（铁律 3），覆盖对方领域的基础
   问题——即 ISEF 分开面试的常态化演练。
5. **答辩底线**：两名成员均须能手推 Galerkin 投影公式、解释 POD 基物理
   含义、画出框架架构图（分工对照表 §三条铁律）。

## 3. 过程记录（可追溯证据）

- Git commit 历史：记录每次 AI 辅助的起草与学生的审阅修改过程
- 科研日志与失败案例数据库（`logs/`）：记录实验设计决策由学生做出的过程
- 本文件随项目进展持续更新，重大变更由两人会签

## 4. 论文/展板中的披露格式（草稿，随官方要求调整）

> *Acknowledgments: The authors used GitHub Copilot for code drafting and
> debugging assistance, and a self-hosted DeepTutor instance for mathematics
> learning support. All experimental design, data analysis, and conclusions
> are the authors' own work. Both authors can explain and reproduce every
> component of the presented system.*

---

> ⚠️ 本文件为学生诚信承诺，非形式文书。任何成员对项目任何部分"讲不清"，
> 即触发代码主权计划的重写条款。—— B（起草）｜ A（待会签）
