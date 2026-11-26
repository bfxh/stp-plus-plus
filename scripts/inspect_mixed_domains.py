"""Per-domain statistics of the mixed corpus scores (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "runs" / "curvity_mixed"
NAMES = ["code", "chat", "zh"]

curv = np.load(D / "train_curv.npy")
nll = np.load(D / "train_nll.npy")


def q(a: np.ndarray) -> dict:
    return {"mean": float(a.mean()), "sd": float(a.std()),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


out = {"overall": {"curv": q(curv), "nll": q(nll),
                   "pearson": float(np.corrcoef(curv, nll)[0, 1])}}
for i, name in enumerate(NAMES):
    out[name] = {"blocks": int(len(curv[i::3])),
                 "curv": q(curv[i::3]), "nll": q(nll[i::3]),
                 "pearson": float(np.corrcoef(curv[i::3], nll[i::3])[0, 1])}

# who occupies the extremes of the curvature ranking?
k = int(len(curv) * 0.25)
order = np.argsort(curv, kind="stable")
hi, lo = order[-k:], order[:k]
out["share_in_top25_curv"] = {n: float((hi % 3 == i).mean()) for i, n in enumerate(NAMES)}
out["share_in_bottom25_curv"] = {n: float((lo % 3 == i).mean()) for i, n in enumerate(NAMES)}
hard = np.argsort(nll)[-k:]
out["share_in_top25_nll"] = {n: float((hard % 3 == i).mean()) for i, n in enumerate(NAMES)}
out["overlap_curv_hi_vs_nll_hard"] = float(len(set(hi.tolist()) & set(hard.tolist())) / k)

(D / "by_domain.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
