"""Fit the two-axis (data + rank) loss surface and solve for the optimal rank.

Model (docs/SCALING.md §26):

    L(N, r) = L_inf + A * N^-alpha + B * r^-beta + c * r / N

  N = tokens seen = steps * bs * block
  r = LoRA rank
  the last term is the capacity/data overfit penalty: adapter parameters scale
  with r, so the "tokens per adapter parameter" ratio degrades as r grows.

Given a handful of (N, r, ppl) points, alpha and beta are not identifiable by
closed form, so we grid-search them and solve [L_inf, A, B, c] by least
squares for each (alpha, beta).  Then:

    r*(N) = (beta * B * N / c)^(1/(1+beta))

which is the prediction the r128@3840 arm is meant to falsify.

Usage:
    python scripts/fit_rank_law.py runs/big/r16_s82 runs/big/r32_s82 ...
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_points(dirs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = []
    for d in dirs:
        p = Path(d)
        rj = p / "run.json"
        if not rj.exists():
            print(f"[skip] {p.name}: no run.json (still running?)")
            continue
        j = json.loads(rj.read_text(encoding="utf-8"))
        cfg = j["config"]
        steps = int(j["steps"])
        bs = int(cfg["bs"])
        block = int(cfg["block"])
        n_tok = steps * bs * block
        r = int(cfg["lora_r"])
        ppl = float(j["eval"]["ppl"])
        rows.append((n_tok, r, ppl, p.name, steps))
    rows.sort(key=lambda t: (t[2], t[0]))
    N = np.array([t[0] for t in rows], dtype=np.float64)
    R = np.array([t[1] for t in rows], dtype=np.float64)
    L = np.array([t[2] for t in rows], dtype=np.float64)
    names = [f"{t[3]} (N={t[0]/1e6:.2f}M, r={int(t[1])}, {t[4]}st)" for t in rows]
    return N, R, L, names


def fit(N, R, L, alpha_grid, beta_grid):
    best = None
    for alpha in alpha_grid:
        for beta in beta_grid:
            X = np.stack([np.ones_like(N), N ** (-alpha), R ** (-beta), R / N], axis=1)
            coef, *_ = np.linalg.lstsq(X, L, rcond=None)
            pred = X @ coef
            rmse = float(np.sqrt(np.mean((pred - L) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, alpha, beta, coef, pred)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--beta", type=float, default=None,
                    help="pin the rank exponent. A free 4-param fit on <10 points is "
                         "ill-conditioned (beta runs to the grid edge); pin it from a "
                         "fixed-N rank scan, then fit alpha / c.")
    ap.add_argument("--alpha", type=float, default=None)
    a = ap.parse_args()

    N, R, L, names = load_points(a.dirs)
    if len(N) < 4:
        print(f"[warn] only {len(N)} points; the 4-parameter surface needs >=4 (ideally 6+)")
    print(f"\n{len(N)} points:")
    for nm, ppl in zip(names, L):
        print(f"  {nm:52s} ppl={ppl:.3f}")

    if len(N) < 4:
        return

    rmse, alpha, beta, coef, pred = fit(
        N, R, L,
        alpha_grid=np.array([a.alpha]) if a.alpha is not None else np.arange(0.5, 1.35, 0.01),
        beta_grid=np.array([a.beta]) if a.beta is not None else np.arange(0.2, 1.6, 0.01),
    )
    L_inf, A, B, c = coef
    print(f"\nfit: L_inf={L_inf:.4f}  A={A:.4g}  alpha={alpha:.3f}  "
          f"B={B:.4f}  beta={beta:.3f}  c={c:.4g}   RMSE={rmse:.4f}")
    print(f"     L(N,r) = {L_inf:.3f} + {A:.3g}*N^-{alpha:.2f} + {B:.3f}*r^-{beta:.2f} + {c:.3g}*r/N")

    print("\nresiduals:")
    for nm, ppl, pr in zip(names, L, pred):
        print(f"  {nm:52s} ppl={ppl:.3f} pred={pr:.3f} resid={ppl-pr:+.4f}")

    # optimal rank per data size, and the loss at that optimum
    print("\nrank optimum  r*(N) = (beta*B*N/c)^(1/(1+beta)):")
    for n_tok in [1 * 4 * 512 * s for s in (960, 1920, 3840, 5760, 7680, 11520)]:
        r_star = (beta * B * n_tok / c) ** (1.0 / (1.0 + beta)) if c > 0 and B > 0 else float("nan")
        l_opt = L_inf + A * n_tok ** (-alpha) + B * r_star ** (-beta) + c * r_star / n_tok
        print(f"  N={n_tok/1e6:5.2f}M  r*={r_star:6.1f}  L(r*)={l_opt:.4f}")

    # how fast can the surface improve at all?  the rank terms decay as N^-beta/(1+beta)
    if B > 0 and c > 0:
        gamma = beta / (1.0 + beta)
        # at r = r*(N) ∝ N^(1/(1+beta)) the two rank terms collapse to K * N^-gamma
        K = B * (beta * B / c) ** (-beta / (1.0 + beta))
        print(f"\n  rank-limited decay exponent = beta/(1+beta) = {gamma:.3f} "
              f"(vs data exponent alpha={alpha:.3f})")
        if gamma < alpha:
            print(f"  => at large N the rank terms decay as N^-{gamma:.2f}, SLOWER than data "
                  f"(N^-{alpha:.2f}); capacity becomes the binding constraint.")
        for n_tok in [1.18e7, 4.72e7]:
            r_star = (beta * B * n_tok / c) ** (1.0 / (1.0 + beta))
            l_opt = L_inf + A * n_tok ** (-alpha) + B * r_star ** (-beta) + c * r_star / n_tok
            print(f"  extrapolate N={n_tok/1e6:5.2f}M -> r*={r_star:5.1f}, L={l_opt:.4f}")


if __name__ == "__main__":
    main()
