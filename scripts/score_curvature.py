"""Score every 512-token block by its mean token curvature at one layer.

    κ_t = 1 − cos(h_t − h_{t−1},  h_{t+1} − h_t)

Forward only — no backprop graph, so it fits an 8GB card comfortably.
The score is a *property of the base model's representation*, not of any adapter,
so it can be computed once and reused for all selection arms.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from stpp.train import HiddenTap, get_layers

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("STPP_CACHE_DIR", str(ROOT / "cache")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=192,
                    help="每块只用前 N 个 token 估曲率（局部量，够用）")
    ap.add_argument("--limit", type=int, default=0, help="0 = all blocks")
    ap.add_argument("--out", default="runs/curvity/train_curv.npy")
    ap.add_argument("--model", default=os.environ.get("STPP_MODEL",
                                                      "unsloth/Llama-3.2-1B-Instruct"))
    a = ap.parse_args()

    blocks = np.load(CACHE / "train_b512.npy", mmap_mode="r")
    n = len(blocks) if a.limit <= 0 else min(a.limit, len(blocks))
    tk = min(a.tokens, blocks.shape[1])
    print(f"[score] {n} blocks × {tk}/{blocks.shape[1]} tokens, layer {a.layer}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                device_map="cuda")
    model.eval()
    tap = HiddenTap(get_layers(model), [a.layer])

    scores = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, a.bs):
            ids = torch.as_tensor(np.asarray(blocks[s:s + a.bs, :tk]), device="cuda")
            attn = torch.ones_like(ids)
            tap.store.clear()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with tap:
                    model(input_ids=ids, attention_mask=attn)
            h = tap.store[a.layer].float()
            v1 = h[:, 1:-1] - h[:, :-2]
            v2 = h[:, 2:] - h[:, 1:-1]
            cos = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)
            scores[s:s + len(ids)] = (1 - cos).mean(dim=1).cpu().numpy()
            if (s // a.bs) % 200 == 0:
                print(f"  {s}/{n}", flush=True)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, scores)
    q = {f"p{p}": float(np.percentile(scores, p)) for p in (1, 10, 25, 50, 75, 90, 99)}
    summary = {"n": int(n), "layer": a.layer, "mean": float(scores.mean()),
               "sd": float(scores.std()), "quantiles": q}
    (out.parent / "curv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done]", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
