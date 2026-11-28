"""Build a weighted training index over the round-robin mixed cache.

The cache is assembled round-robin (`block i` belongs to source `i % n_sources`),
so a "mix" is just a multiset of block indices: take more of a source by
duplicating its indices, fewer by dropping some.

DoReMi-style target: allocate tokens proportionally to each domain's excess loss
over its own floor, w_d ~ (L_d - L_d^inf).  Measured per-domain ppl at r64:

    code 5.32   chat 4.89   zh 2.42        (zh is already near its floor)

hence the 4:4:2 arm: keep all code and chat blocks, halve zh.

Usage:
    python scripts/make_mix_index.py --cache F:/aiwork/mixed_cache_big \
        --weights 4:4:2 --out F:/aiwork/mix_idx/mix442.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--weights", required=True, help="colon list, one per source, e.g. 4:4:2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="", help="source name order; default = meta.json order")
    a = ap.parse_args()

    cache = Path(a.cache)
    train = np.load(cache / "train_b512.npy", mmap_mode="r")
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    names = [s.strip() for s in a.name.split(",") if s.strip()] or \
        list(meta.get("_total", {}).get("sources", [])) or \
        [k for k in meta if not k.startswith("_")]
    w = [float(x) for x in a.weights.split(":")]
    assert len(w) == len(names), f"{len(w)} weights vs {len(names)} sources"

    n_src = len(names)
    counts = {}
    for gi, nm in enumerate(names):
        idx = np.arange(gi, len(train), n_src, dtype=np.int64)
        counts[nm] = len(idx)

    # largest per-source take that every source can satisfy at ratio w_i
    base = min(counts[nm] / w[gi] for gi, nm in enumerate(names))
    take = {nm: int(round(base * w[gi])) for gi, nm in enumerate(names)}

    picks = []
    for gi, nm in enumerate(names):
        idx = np.arange(gi, len(train), n_src, dtype=np.int64)
        k = min(take[nm], len(idx))
        # evenly spaced so the take covers the whole source rather than its head
        sel = idx[np.linspace(0, len(idx) - 1, k).round().astype(np.int64)]
        picks.append(sel)
        print(f"[{nm}] available={len(idx)} take={k} ({k/len(idx)*100:.0f}%)", flush=True)

    out_idx = np.concatenate(picks).astype(np.int64)
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, out_idx)

    total = len(out_idx)
    print(f"[mix] {total} blocks -> {dst}", flush=True)
    for gi, nm in enumerate(names):
        print(f"  share {nm:5s} = {take[nm]/total*100:5.1f}%", flush=True)
    print(f"[mix] tokens = {total*512/1e6:.2f}M; at bs=4, 3840 steps need "
          f"{3840*4} blocks -> {3840*4/total:.2f} epoch", flush=True)


if __name__ == "__main__":
    main()
