"""Build a control pool: same *domain proportions* as nll-hi (88% zh / 5% code / 7% chat),
but sampled **randomly inside each domain**. Isolates "more zh tokens" from "harder zh blocks"."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "runs" / "curvity_mixed"
SHARES = {"zh": 1480, "code": 88, "chat": 114}  # = composition of the nll-hi pool
GROUPS = {"code": 0, "chat": 1, "zh": 2}


def main() -> None:
    nll = np.load(D / "train_nll.npy")
    n = len(nll)
    k = int(n * 0.25)
    hard = np.argsort(nll, kind="stable")[-k:]
    comp = {g: float((hard % 3 == i).mean()) for g, i in GROUPS.items()}

    rng = np.random.default_rng(82)
    picks = []
    for g, i in GROUPS.items():
        pool = np.arange(i, n, 3)
        cnt = SHARES[g]
        picks.append(rng.choice(pool, size=min(cnt, len(pool)), replace=False))
    ids = np.concatenate(picks)

    D.joinpath("ids").mkdir(exist_ok=True)
    np.save(D / "ids" / "mixhi_s82.npy", ids)
    print(json.dumps({"n": int(len(ids)), "nll_hi_composition": comp,
                      "shares": SHARES}, indent=2))


if __name__ == "__main__":
    main()
