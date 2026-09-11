"""
快照管理器（核心框架模块）
==========================
ISEF 2028 ROM 项目 · 任务 #9 ｜ 架构文档 §3.2
负责人：同学 B

职责：快照的采集登记、参数标注、拼接、持久化。
关键设计：参数标注是头等公民——每条快照可回溯到参数点，
          否则同学 A 的自适应采样（AdaptiveSampler 协议）无法工作。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SnapshotRun:
    """一次全阶仿真的登记记录。"""

    params: dict            # 参数点（如 {"k": 1.5, "h_right": 1000, "q_gen": 5e4}）
    t: np.ndarray           # 时间网格 (n_t,)
    snapshots: np.ndarray   # 快照矩阵 (n_x, n_t)
    tag: str = ""           # 备注（如 "baseline", "greedy-iter-3"）


@dataclass
class SnapshotManager:
    """快照库：参数 ↔ 快照的可追溯存储。"""

    runs: list[SnapshotRun] = field(default_factory=list)

    def add_run(self, params: dict, t: np.ndarray, snapshots: np.ndarray,
                tag: str = "") -> int:
        """登记一次全阶仿真，返回 run 索引。"""
        t = np.asarray(t, dtype=float)
        snapshots = np.asarray(snapshots, dtype=float)
        if snapshots.ndim != 2:
            raise ValueError(f"snapshots 须为二维 (n_x, n_t)，得到 {snapshots.shape}")
        if snapshots.shape[1] != t.shape[0]:
            raise ValueError(f"snapshots 列数 {snapshots.shape[1]} 与 t 长度 {t.shape[0]} 不一致")
        if self.runs and snapshots.shape[0] != self.runs[0].snapshots.shape[0]:
            raise ValueError("空间维度与已有快照不一致（同一物理网格才能入库）")
        self.runs.append(SnapshotRun(params=dict(params), t=t, snapshots=snapshots, tag=tag))
        return len(self.runs) - 1

    def basis_input(self) -> np.ndarray:
        """拼接全部 run 的快照，供 POD 分解：(n_x, Σ n_t_i)。

        待定项（架构文档 §3.3）：多参数 run 时长不一时是否按参数点等权，
        待与同学 A 讨论；当前为简单拼接。
        """
        if not self.runs:
            raise RuntimeError("快照库为空")
        return np.concatenate([r.snapshots for r in self.runs], axis=1)

    def param_table(self) -> list[dict]:
        """参数表：[{run, tag, **params}, ...]，供采样策略查询覆盖范围。"""
        return [{"run": i, "tag": r.tag, **r.params} for i, r in enumerate(self.runs)]

    def param_vector(self, keys: list[str]) -> np.ndarray:
        """按给定键顺序返回参数点矩阵 (n_runs, len(keys))，供采样器做空间分析。"""
        return np.array([[r.params[k] for k in keys] for r in self.runs], dtype=float)

    # ---------------- 持久化 ----------------
    def save(self, path: str | Path) -> None:
        """保存为单个 .npz（快照 + 时间网格）+ 同名 .json（参数元数据）。"""
        path = Path(path)
        arrays = {}
        for i, r in enumerate(self.runs):
            arrays[f"snapshots_{i}"] = r.snapshots
            arrays[f"t_{i}"] = r.t
        np.savez_compressed(path, **arrays)
        meta = [{"params": r.params, "tag": r.tag} for r in self.runs]
        path.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SnapshotManager":
        """从 .npz + .json 恢复。"""
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        data = np.load(path)
        mgr = cls()
        for i, m in enumerate(meta):
            mgr.add_run(params=m["params"], t=data[f"t_{i}"],
                        snapshots=data[f"snapshots_{i}"], tag=m.get("tag", ""))
        return mgr
