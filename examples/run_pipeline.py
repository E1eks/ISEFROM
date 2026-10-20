"""
流水线 CLI 入口（任务 #22 雏形演示）
====================================
用法: python examples/run_pipeline.py configs/demo_heat_1d.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import run_pipeline


def main() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "demo_heat_1d.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    print("=" * 64)
    print(f"端到端流水线（任务 #22 雏形）: {config['case']}")
    print("=" * 64)

    report = run_pipeline(config)

    off = report["offline"]
    pod = report["pod"]
    print(f"[离线] {off['n_runs']} 组快照, 耗时 {off['wall_time_s']:.1f}s; "
          f"POD r={pod['r']}")
    for e in report["online"]:
        line = f"[在线] {e['params']} → {e['rom_time_s'] * 1000:.1f} ms"
        if "rel_error" in e:
            line += f" | 误差 {e['rel_error']:.3%} | 加速比 {e['speedup']:.0f}x"
        if "error_bound" in e:
            b = e["error_bound"]
            tag = "认证" if b["certified"] else "占位未认证"
            line += f" | Δ({b['method']},{tag})={b['delta']:.3e}"
            if "effectivity" in b:
                line += f" η={b['effectivity']:.1f}"
            if "warning" in b:
                line += f" ⚠️{b['warning']}"
        print(line)
    print(f"[输出] 报告: {report['report_path']}")


if __name__ == "__main__":
    main()
