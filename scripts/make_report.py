"""Render runs/results.json -> a markdown pilot report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, nd=3, default="—"):
    if v is None:
        return default
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/results.json")
    ap.add_argument("--out", default="runs/PILOT.md")
    ap.add_argument("--title", default="STP++ pilot")
    a = ap.parse_args()

    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    ok = [r for r in res if "exact_match" in r]

    lines = [f"# {a.title}", "",
             "Single-GPU LoRA pilot. Absolute numbers are low by construction — read the "
             "*differences between arms* at the same budget, not the absolute level.", "",
             "| arm | data fraction | exact match | TF seq | TF token | unique | collision | curv evidence (m=1) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in ok:
        probe = r.get("probe") or {}
        ce = probe.get("m1", {}).get("curvature_evidence")
        lines.append(
            f"| `{r['arm']}` | {r['fraction']:g} | {fmt(r.get('exact_match'))} | "
            f"{fmt(r.get('tf_seq_acc'))} | {fmt(r.get('tf_token_acc'))} | "
            f"{fmt(r.get('unique_ratio'), 2)} | {fmt(r.get('collision_rate'), 2)} | {fmt(ce, 2)} |"
        )

    lines += ["", "## 怎么读", "",
              "- **exact match** 是数据效率曲线的主轴；同分数下用更少数据的 arm 赢。",
              "- **curvature evidence** = `mse_linear / mse_mlp`。>1 说明隐状态轨迹是曲线，"
              "STP 的\"局部直线\"假设只是零阶近似。",
              "- **collision** 是去中心化端点余弦 > 0.99 的样本对占比；STP 声称能防模式坍塌，这里是实测。",
              "- 单 seed、LoRA、300 条测试集 —— 只能当方向性证据，不能当论文数字。"]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
