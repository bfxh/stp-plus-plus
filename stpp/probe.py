"""Is the trajectory a straight line, or a smooth curve?

The STP paper says "locally linear geodesic". The follow-up (arXiv:2604.18464) shows a
3-layer MLP beats linear extrapolation by 3-12x — i.e. the truth is *curved*. This script
turns that into a number for your own runs:

    mse_linear(m)  :  || (z_k + m*(z_k - z_{k-1})) - z_{k+m} ||^2 / ||z_{k+m}||^2
    mse_mlp(m)     :  same, but with a 3-layer MLP trained on the first 80% of anchors

If mlp/linear >> 1 the trajectory is curved and a "straight tube" is the wrong prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def build_pairs(seqs, m: int):
    X, Y, Xlin = [], [], []
    for s in seqs:
        if s.shape[0] <= m + 1:
            continue
        z = torch.from_numpy(s.astype(np.float32))
        for k in range(2, z.shape[0] - m):
            X.append(torch.cat([z[k], z[k] - z[k - 1]], dim=-1))
            Y.append(z[k + m])
            Xlin.append(z[k] + m * (z[k] - z[k - 1]))
    return torch.stack(X), torch.stack(Y), torch.stack(Xlin)


def run(path: str, out: str, m_max: int = 3, epochs: int = 200) -> dict:
    d = np.load(path, allow_pickle=True)
    seqs = list(d["seqs"])

    result = {}
    for m in range(1, m_max + 1):
        X, Y, Xlin = build_pairs(seqs, m)
        n = X.shape[0]
        if n < 32:
            continue
        perm = torch.randperm(n)
        ntr = int(0.8 * n)
        tr, te = perm[:ntr], perm[ntr:]

        # ---- linear ----
        err_lin = ((Xlin[te] - Y[te]) ** 2).sum(-1) / (Y[te] ** 2).sum(-1).clamp_min(1e-8)
        mse_lin = float(err_lin.mean())

        # ---- MLP ----
        din = X.shape[-1]
        mlp = torch.nn.Sequential(
            torch.nn.Linear(din, 512), torch.nn.GELU(),
            torch.nn.Linear(512, 512), torch.nn.GELU(),
            torch.nn.Linear(512, Y.shape[-1]),
        )
        opt = torch.optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-4)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mlp.to(dev)
        Xtr, Ytr = X[tr].to(dev), Y[tr].to(dev)
        for ep in range(epochs):
            pred = mlp(Xtr)
            loss = F.mse_loss(pred, Ytr)
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 50 == 0 and dev == "cuda":
                torch.cuda.empty_cache()
        with torch.no_grad():
            pred = mlp(X[te].to(dev)).cpu()
        err_mlp = ((pred - Y[te]) ** 2).sum(-1) / (Y[te] ** 2).sum(-1).clamp_min(1e-8)
        mse_mlp = float(err_mlp.mean())

        result[f"m{m}"] = {
            "mse_linear": mse_lin,
            "mse_mlp": mse_mlp,
            "curvature_evidence": mse_lin / max(mse_mlp, 1e-9),
            "n": int(n),
        }

    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", default="runs/probe.json")
    ap.add_argument("--m_max", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()
    run(a.traj, a.out, a.m_max, a.epochs)
