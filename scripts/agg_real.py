"""Aggregate the real-corpus selection arms (curvature quartiles + difficulty quartiles).

Reads `runs/select_real/results.json` (curvature) and `runs/select_real_nll/results.json`
(NLL), reports mean/sd per arm and a one-sample t against the full-corpus baseline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = [ROOT / "runs" / "select_real" / "results.json",
       ROOT / "runs" / "select_real_nll" / "results.json"]

LABEL = {("select_real", "all"): "full (all blocks)",
         ("select_real", "rand"): "random 25%",
         ("select_real", "hi"): "curv top 25% (bending)",
         ("select_real", "lo"): "curv bottom 25% (straight)",
         ("select_real_nll", "hi"): "loss top 25% (hardest)",
         ("select_real_nll", "lo"): "loss bottom 25% (easiest)"}


def main() -> None:
    rows = []
    for p in SRC:
        if p.exists():
            rows += json.loads(p.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit("no results yet")

    keys: dict[tuple[str, str], list[float]] = {}
    for p in SRC:
        if not p.exists():
            continue
        root = p.parent.name
        for r in json.loads(p.read_text(encoding="utf-8")):
            keys.setdefault((root, r["arm"]), []).append(r["ppl"])

    base = keys[("select_real", "all")]
    mu0 = sum(base) / len(base)

    lines = [f"real corpus: wikitext-103, 6536 blocks × 512 tok = 3.35M tokens; "
             f"test 300 blocks; 120 steps × bs4 = 245,760 tokens per run",
             f"baseline full-corpus ppl = {mu0:.4f} (n={len(base)})", ""]
    lines.append(f"{'arm':<28}{'n':>3}{'mean_ppl':>10}{'sd':>8}{'delta':>9}{'t':>8}")
    for (root, arm), xs in sorted(keys.items()):
        name = LABEL.get((root, arm), f"{root}/{arm}")
        mu = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / max(len(xs) - 1, 1))
        d = mu - mu0
        t = d / (sd / math.sqrt(len(xs))) if sd > 0 else float("nan")
        lines.append(f"{name:<28}{len(xs):>3}{mu:>10.4f}{sd:>8.4f}{d:>+9.4f}{t:>8.2f}")
    out = ROOT / "runs" / "select_real" / "results.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
