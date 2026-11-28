"""Base-model CE on a cache's held-out test blocks. No LoRA, no training.

Use it to put the two corpora on one scale: if the base model is worse on the new
test set, the trained-model ppl gap is (partly) test-set difficulty, not training.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", default="/mnt/f/aiwork/models/Llama-3.2-1B-Instruct")
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()

    arr = np.load(f"{a.cache}/test_b512.npy")[: a.blocks]
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    ce_sum, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(arr), a.bs):
            ids = torch.from_numpy(arr[i:i + a.bs].astype(np.int64)).cuda()
            out = model(input_ids=ids, labels=ids)
            k = ids.shape[1] - 1
            ce_sum += float(out.loss) * k * ids.shape[0]
            ntok += k * ids.shape[0]
    ce = ce_sum / ntok
    print(json.dumps({"cache": a.cache, "blocks": int(len(arr)),
                      "ce": ce, "ppl": math.exp(ce), "tokens": ntok}))


if __name__ == "__main__":
    main()
