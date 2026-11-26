"""Inspect what high/low curvature blocks actually look like, and how curvature
relates to base-model difficulty. CPU only — no model forward pass."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("STPP_CACHE_DIR", "F:/aiwork/real_cache"))
CURV = ROOT / "runs" / "curvity_real" / "train_curv.npy"
NLL = ROOT / "runs" / "curvity_real" / "train_nll.npy"


def rank(a: np.ndarray) -> np.ndarray:
    o = np.argsort(a, kind="stable")
    r = np.empty_like(o, dtype=np.float64)
    r[o] = np.arange(len(a), dtype=np.float64)
    return r


def main() -> None:
    from transformers import AutoTokenizer

    blocks = np.load(CACHE / "train_b512.npy", mmap_mode="r")
    curv = np.load(CURV)
    nll = np.load(NLL)
    tok = AutoTokenizer.from_pretrained(os.environ.get(
        "STPP_MODEL", "unsloth/Llama-3.2-1B-Instruct"))

    n = len(curv)
    k = int(n * 0.25)
    order = np.argsort(curv, kind="stable")
    lo, hi = order[:k], order[-k:]

    pear = float(np.corrcoef(curv, nll)[0, 1])
    spear = float(np.corrcoef(rank(curv), rank(nll))[0, 1])
    out = {
        "n": n,
        "pearson_curv_nll": pear,
        "spearman_curv_nll": spear,
        "curv_mean": float(curv.mean()), "curv_sd": float(curv.std()),
        "nll_mean": float(nll.mean()), "nll_sd": float(nll.std()),
        "nll_hi_curv_pool": float(nll[hi].mean()),
        "nll_lo_curv_pool": float(nll[lo].mean()),
        "nll_all": float(nll.mean()),
        "curv_at_nll_top25": float(curv[np.argsort(nll)[-k:]].mean()),
        "curv_at_nll_bottom25": float(curv[np.argsort(nll)[:k]].mean()),
        "overlap_hi_curv_and_hard": float(len(set(hi.tolist())
                                            & set(np.argsort(nll)[-k:].tolist())) / k),
    }

    def snippet(idx: np.ndarray, m: int = 2) -> list[str]:
        return [tok.decode(np.asarray(blocks[i])[:160], skip_special_tokens=True)
                for i in idx[:m]]

    out["examples_hi_curv"] = snippet(hi[::-1])
    out["examples_lo_curv"] = snippet(lo)
    out["examples_hard"] = snippet(np.argsort(nll)[-k:][::-1])
    out["examples_easy"] = snippet(np.argsort(nll)[:k])

    p = ROOT / "runs" / "curvity_real" / "inspect.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k2: v for k2, v in out.items() if not k2.startswith("examples")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
