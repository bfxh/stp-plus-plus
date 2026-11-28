"""Fit ppl(N) = a + b*N^-c on a set of run dirs (tokens seen = steps*bs*block).

No scipy: grid over c, closed-form (a, b) by least squares for each c.
Usable for any corpus, so the small-corpus and big-corpus shapes can be compared
on one code path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load(dirs):
    pts = []
    for d in dirs:
        r = json.loads((Path(d) / "run.json").read_text())
        c = r["config"]
        n = r["steps"] * c["bs"] * c["block"]
        pts.append((n, r["eval"]["ppl"], r["steps"]))
    return sorted(pts)


def fit(n, y):
    best = None
    for cc in np.linspace(0.02, 1.5, 14801):
        X = np.column_stack([np.ones_like(n), n ** (-cc)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = float(np.sum((X @ coef - y) ** 2))
        if best is None or resid < best[0]:
            best = (resid, cc, coef[0], coef[1])
    return best


def main() -> None:
    dirs = sys.argv[1:]
    pts = load(dirs)
    n = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    rmse, cc, a, b = fit(n, y)
    print(f"points={len(pts)}  floor a={a:.4f}  b={b:.1f}  c={cc:.4f}  rmse={np.sqrt(rmse/len(pts)):.4f}")
    print(f"{'steps':>7}{'tokens':>12}{'ppl':>9}{'fit':>9}{'resid':>9}")
    for (tok, ppl, steps) in pts:
        f = a + b * tok ** (-cc)
        print(f"{steps:>7}{tok/1e6:>11.2f}M{ppl:>9.3f}{f:>9.3f}{ppl-f:>+9.3f}")
    last = max(p[0] for p in pts)
    for mult in (1.5, 2, 4):
        print(f"  extrapolate {mult}x tokens ({last*mult/1e6:.1f}M): {a + b*(last*mult)**(-cc):.3f}")


if __name__ == "__main__":
    main()
