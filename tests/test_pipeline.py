"""
流水线测试（任务 #22 雏形）
============================
运行: python -m pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import run_pipeline


class TestPipeline:
    def test_heat_1d_end_to_end(self):
        """1D 配置驱动全流程：离线 4 组 → 在线未见参数验证误差 <5%。"""
        report = run_pipeline({
            "case": "heat_1d",
            "offline": {"n_samples": 4, "seed": 7,
                        "ranges": {"k": [1.0, 3.0], "h_right": [500.0, 3000.0]},
                        "energy_threshold": 0.999},
            "online": {"params": [{"k": 2.0, "h_right": 1500.0, "q_gen": 8e4}],
                       "validate_against_fom": True},
            "output": {"prefix": "test_pipe_1d"},
        })
        assert report["offline"]["n_runs"] == 4
        assert report["pod"]["r"] >= 1
        e = report["online"][0]
        assert e["rel_error"] < 0.05
        assert Path(report["report_path"]).exists()

    def test_poisson_end_to_end(self):
        """椭圆算例走同一流水线（稳态路径），误差 <5%。"""
        report = run_pipeline({
            "case": "poisson_2d",
            "offline": {"n_samples": 4, "seed": 3,
                        "ranges": {"k1": [0.5, 5.0], "k2": [0.5, 5.0]},
                        "energy_threshold": 0.9999},
            "online": {"params": [{"k1": 2.0, "k2": 1.5, "f_amp": 200.0}],
                       "validate_against_fom": True},
            "output": {"prefix": "test_pipe_poi"},
        })
        assert report["online"][0]["rel_error"] < 0.05

    def test_unknown_case_rejected(self):
        with pytest.raises(ValueError, match="未知算例"):
            run_pipeline({"case": "navier_stokes", "offline": {"n_samples": 2},
                          "online": {"params": []}})
