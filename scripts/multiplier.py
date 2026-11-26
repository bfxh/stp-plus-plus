"""Equivalent-token multiplier: how much `reg` data does the selection arm substitute for?

Uses the three `reg` budgets (120 / 240 / 480 steps, same bs=4×512 tokens) and
log-linear interpolation of ppl against tokens seen.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "runs" / "pretrain"          # 120 steps
B240 = ROOT / "runs" / "budget240"
B480 = ROOT / "runs" / "budget480"
SEL = ROOT / "runs" / "select"

TOK_PER_STEP = 4 * 512


def ppl(path: Path) -> float:
    return json.loads((path / "run.json").read_text(encoding="utf-8"))["eval"]["ppl"]


def interp_steps(target: float, pts: list[tuple[float, float]]) -> float | None:
    """log-linear interpolation of tokens vs ppl; None if target is outside the range."""
    pts = sorted(pts)
    if not (min(p for _, p in pts) <= target <= max(p for _, p in pts)):
        return None
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if p0 >= target >= p1:
            f = (math.log(p0) - math.log(target)) / (math.log(p0) - math.log(p1))
            return t0 * (t1 / t0) ** f
    return None


def main() -> None:
    rows = []
    for seed in (82, 23, 37):
        p120 = ppl(BASE / f"reg_s{seed}")
        p240 = ppl(B240 / f"reg_s{seed}")
        p480 = ppl(B480 / f"reg_s{seed}")
        hi = ppl(SEL / f"selhi_s{seed}")
        hi_tok = interp_steps(hi, [(120 * TOK_PER_STEP, p120),
                                   (240 * TOK_PER_STEP, p240),
                                   (480 * TOK_PER_STEP, p480)])
        mult = (hi_tok / (120 * TOK_PER_STEP)) if hi_tok else None
        rows.append((seed, p120, p240, p480, hi, mult))
        ms = f"{mult:.2f}×" if mult else "超出曲线"
        print(f"seed {seed}: reg@{120}/{240}/{480} = {p120:.4f}/{p240:.4f}/{p480:.4f}  "
              f"selhi={hi:.4f}  → 等效 ≈ {ms}")

    ok = [r[5] for r in rows if r[5]]
    if ok:
        print(f"\nmean multiplier = {sum(ok)/len(ok):.2f}×  (n={len(ok)})")


if __name__ == "__main__":
    main()
