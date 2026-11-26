"""Aggregate the geometry-as-selector arms against the `reg` baseline (same seeds)."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "runs" / "pretrain"
SEL = ROOT / "runs" / "select"


def load(root: Path, prefix: str) -> dict[str, dict[int, float]]:
    rows: dict[str, dict[int, float]] = {}
    for p in root.glob(f"{prefix}*_s*/run.json"):
        tag = p.parent.name
        arm, s = tag.rsplit("_s", 1)
        if not s.isdigit():
            continue
        rows.setdefault(arm, {})[int(s)] = json.loads(
            p.read_text(encoding="utf-8"))["eval"]["ppl"]
    return rows


def main() -> None:
    base = load(BASE, "reg")["reg"]
    sel = load(SEL, "sel")

    print("%-10s%3s%10s%8s%9s%10s" % ("arm", "n", "mean_ppl", "sd", "min", "vs reg"))
    for arm in ("selrand", "selhi", "sello", "selmid"):
        if arm not in sel:
            continue
        xs = list(sel[arm].values())
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else float("nan")
        common = sorted(set(sel[arm]) & set(base))
        d = [sel[arm][s] - base[s] for s in common]
        dm = sum(d) / len(d) if d else float("nan")
        print("%-10s%3d%10.4f%8.4f%9.4f%+10.4f" % (arm, len(xs), m, sd, min(xs), dm))

    print()
    crit = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    for arm in ("selhi", "sello", "selmid", "selrand"):
        common = sorted(set(sel.get(arm, {})) & set(base))
        if len(common) < 2:
            continue
        d = [sel[arm][s] - base[s] for s in common]
        m = sum(d) / len(d)
        sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        t = m / (sd / math.sqrt(len(d))) if sd else float("inf")
        c = crit.get(len(d) - 1, 2.776)
        bm = sum(base[s] for s in common) / len(common)
        print("%-10s vs reg  seeds=%s  dPPL=%+.4f (%+.1f%%)  t=%+.2f  crit=%.2f  %s"
              % (arm, common, m, 100 * m / bm, t, c, "显著" if abs(t) >= c else "不显著"))


if __name__ == "__main__":
    main()
