"""Build a *heterogeneous* 512-token block cache from several parquet sources.

Each source is tokenized separately, cut into 512-token blocks, and the blocks are
then interleaved round-robin, so any random subset keeps the same domain mix. That
is the point of this cache: a selector only has headroom when the corpus actually
contains dissimilar material.

Example:
    python scripts/build_mixed_corpus.py \
        --src code:F:/aiwork/raw/code_val.parquet:whole_func_string \
        --src chat:F:/aiwork/raw/chat_0.parquet:messages \
        --src zh:F:/aiwork/raw/zh_2.parquet:text \
        --tokens-per-source 1200000 --out F:/aiwork/mixed_cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def texts_from(path: str, field: str):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=512, columns=[field]):
        col = batch.column(0).to_pylist()
        for v in col:
            if v is None:
                continue
            if field == "messages":
                parts = []
                for m in v:
                    c = m.get("content") if isinstance(m, dict) else None
                    if c:
                        parts.append(str(c))
                yield "\n\n".join(parts)
            else:
                yield str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True,
                    help="name:parquet_path:text_field")
    ap.add_argument("--tokens-per-source", type=int, default=1_200_000)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--test-blocks", type=int, default=300)
    ap.add_argument("--out", default="F:/aiwork/mixed_cache")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        __import__("os").environ.get("STPP_MODEL", "unsloth/Llama-3.2-1B-Instruct"))

    per_source: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for spec in a.src:
        name, rest = spec.split(":", 1)
        path, field = rest.rsplit(":", 1)
        ids: list[int] = []
        for t in texts_from(path, field):
            t = t.strip()
            if not t:
                continue
            ids.extend(tok(t, add_special_tokens=False)["input_ids"])
            ids.append(tok.eos_token_id)
            if len(ids) >= a.tokens_per_source:
                break
        arr = np.asarray(ids, dtype=np.int32)
        n = (len(arr) // a.block) * a.block
        blocks = arr[:n].reshape(-1, a.block)
        per_source[name] = blocks
        meta[name] = {"blocks": int(len(blocks)), "tokens": int(n)}
        print(f"[{name}] tokens={n/1e6:.2f}M blocks={len(blocks)}", flush=True)

    n = min(len(b) for b in per_source.values())
    names = list(per_source)
    order = [per_source[names[i % len(names)]][i // len(names)] for i in range(n * len(names))]
    merged = np.stack(order)
    test, train = merged[-a.test_blocks:], merged[:-a.test_blocks]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_b512.npy", train)
    np.save(out / "test_b512.npy", test)
    meta["_total"] = {"train_blocks": int(len(train)), "test_blocks": int(len(test)),
                      "sources": names}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] train={len(train)} test={len(test)} sources={names} -> {out}", flush=True)


if __name__ == "__main__":
    main()
