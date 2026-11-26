"""Build a real-text 512-token block cache from a HF dataset (streaming).

Output goes straight to `<out>/train_b512.npy` and `<out>/test_b512.npy`, i.e. the same
filenames `stpp.pretrain` expects from `STPP_CACHE_DIR`, so the whole pipeline (scoring,
selection, training) runs unchanged on real text.

Example:
    python scripts/build_real_corpus.py --tokens 3500000 --out F:/aiwork/real_cache
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--config", default="wikitext-103-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokens", type=int, default=3_500_000)
    ap.add_argument("--test-blocks", type=int, default=300)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--out", default="F:/aiwork/real_cache")
    ap.add_argument("--parquet", default="",
                    help="local parquet/jsonl shard; bypasses network streaming")
    ap.add_argument("--model", default=os.environ.get("STPP_MODEL",
                                                      "unsloth/Llama-3.2-1B-Instruct"))
    a = ap.parse_args()

    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.parquet:
        ds = load_dataset("parquet", data_files=a.parquet, split="train", streaming=True)
    else:
        ds = load_dataset(a.dataset, a.config, split=a.split, streaming=True)

    ids: list[int] = []
    docs = 0
    for row in ds:
        t = (row.get("text") or "").strip()
        if not t:
            continue
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(tok.eos_token_id)
        docs += 1
        if len(ids) >= a.tokens:
            break
        if docs % 500 == 0:
            print(f"  docs={docs} tokens={len(ids)/1e6:.2f}M", flush=True)

    arr = np.asarray(ids, dtype=np.int32)
    n = (len(arr) // a.block) * a.block
    blocks = arr[:n].reshape(-1, a.block)
    test = blocks[-a.test_blocks:]
    train = blocks[:-a.test_blocks]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_b512.npy", train)
    np.save(out / "test_b512.npy", test)
    print(f"[done] docs={docs} tokens={len(arr)/1e6:.2f}M "
          f"train_blocks={len(train)} test_blocks={len(test)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
