"""Fit ppl(N) on the measured L0 curve and convert step-speed into loss.

ppl(N) = c + a * N^(-b), fitted by grid search over b (exact for 3 points,
so treat the extrapolation as a guess to be validated by the 1920 run).

Usage:  python scripts/fit_curve.py
"""
import numpy as np

# measured (steps, mean ppl over 3 seeds) - see docs/MFU.md 24.11a / runs/curve*
PTS = [(240, 6.380), (480, 6.158), (960, 5.989)]

N = np.array([p[0] for p in PTS], float)
Y = np.array([p[1] for p in PTS], float)

best = None
for b in np.arange(0.01, 0.60, 0.001):
    x = N ** (-b)
    A = np.c_[x, np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ coef
    sse = float(((pred - Y) ** 2).sum())
    if best is None or sse < best[0]:
        best = (sse, b, coef)

sse, b, (a, c) = best
print(f"fit: ppl = {c:.4f} + {a:.4f} * N^-{b:.3f}   (sse={sse:.2e})")
for n, y in PTS:
    print(f"  N={n:5d}  measured {y:.4f}  fit {c + a * n ** (-b):.4f}")
print(f"  N=1920  predict {c + a * 1920 ** (-b):.4f}")

# wall-clock -> steps -> loss, for three measured step times
print("\nsame wall clock, three configs (bs4, 2048 tok/step):")
print(f"{'T(min)':>7} {'cfg':<26} {'steps':>6} {'ppl_pred':>9} {'tok/s':>7}")
CFG = [
    ("Windows, curve cfg (ckpt)", 1.486),
    ("Linux+compile, curve cfg", 0.691),
    ("Linux+compile, best cfg", 0.525),
]
for tmin in (5, 11, 22, 44):
    T = tmin * 60
    for name, sps in CFG:
        steps = int(T / sps)
        ppl = c + a * steps ** (-b)
        print(f"{tmin:7d} {name:<26} {steps:6d} {ppl:9.4f} {2048 / sps:7.0f}")
    print()
