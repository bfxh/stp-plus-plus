"""Evaluation: exact-match accuracy + trajectory geometry diagnostics.

Three numbers matter for the STP story and none of them is plain accuracy:

1. ``exact_match``      — does it still solve the task (the thing data-efficiency buys)
2. ``collision_rate``   — do distinct prompts end up on top of each other
                           (the mode-collapse claim STP only argues theoretically)
3. trajectory dump      — hidden states along the assistant span, for `probe.py`
                           (is the trajectory straight, or is it a smooth curve?)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import read_jsonl
from .train import HiddenTap, get_layers

SYSTEM_FALLBACK = "Convert natural language to regular expression."


def build_prompt(tok, messages) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*assistant\s*[:：]?\s*", "", s)
    s = s.replace("\r", "").strip()
    return re.sub(r"\s+", " ", s)


@torch.no_grad()
def run(args) -> dict:
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval().cuda()
    model.config.use_cache = True

    data = read_jsonl(args.test_file)
    if args.limit > 0:
        data = data[: args.limit]

    prompts = [build_prompt(tok, ex["messages"][:-1]) for ex in data]
    golds = [normalize(ex["messages"][-1]["content"]) for ex in data]

    preds: list[str] = []
    for i in range(0, len(prompts), args.bs):
        chunk = prompts[i: i + args.bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len).to("cuda")
        out = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        for j in range(out.shape[0]):
            new = out[j, enc["input_ids"].shape[1]:]
            preds.append(normalize(tok.decode(new, skip_special_tokens=True)))

    em = float(np.mean([p == g for p, g in zip(preds, golds)])) if preds else 0.0
    contains = float(np.mean([g in p for p, g in zip(preds, golds)])) if preds else 0.0
    unique = len(set(preds)) / max(1, len(preds))

    # ---- trajectory dump (straightness / collision) --------------------------------
    traj = None
    if args.dump_traj and args.limit_traj > 0:
        sub = data[: args.limit_traj]
        pairs = []
        for ex in sub:
            p = build_prompt(tok, ex["messages"][:-1])
            pairs.append((p, p + ex["messages"][-1]["content"]))
        traj = collect_trajectories(model, tok, pairs, args)
        np.savez_compressed(Path(args.out).with_suffix(".traj.npz"), **traj)

    result = {
        "n": len(preds),
        "exact_match": em,
        "contains": contains,
        "unique_ratio": unique,
        "model": args.model,
        "adapter": args.adapter,
    }
    if traj is not None:
        result["collision_rate"] = float(traj["collision_rate"])
        result["mean_pairwise_cos_centered"] = float(traj["mean_cos"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


@torch.no_grad()
def collect_trajectories(model, tok, pairs, args) -> dict:
    """Collect hidden states at anchors inside the *exact* assistant span."""
    from .train import get_body_and_head

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = get_layers(model)
    tap = HiddenTap(layers, [len(layers) - 1])
    body, _ = get_body_and_head(model)

    seqs: list[np.ndarray] = []
    ends: list[np.ndarray] = []

    for i in range(0, len(pairs), 4):
        chunk = pairs[i: i + 4]
        spans = [len(tok(p, add_special_tokens=False)["input_ids"]) for p, _ in chunk]
        enc = tok([f for _, f in chunk], return_tensors="pt", padding=True, truncation=True,
                  max_length=min(args.max_len, 384)).to("cuda")
        attn = enc["attention_mask"]

        tap.store.clear()
        with tap:
            hidden = body(input_ids=enc["input_ids"], attention_mask=attn).last_hidden_state
        h = (body.norm(hidden) if hasattr(body, "norm") else hidden).float()

        for b in range(h.shape[0]):
            n = int(attn[b].sum())
            pad = h.shape[1] - n          # tokenizer pads on the left here
            a0 = pad + max(1, spans[b])
            a1 = pad + n
            if a1 - a0 < 4:
                continue
            idx = np.linspace(a0, a1 - 1, num=min(args.anchors_per_seq, a1 - a0), dtype=int)
            seqs.append(h[b, torch.as_tensor(idx)].cpu().numpy().astype(np.float16))
            ends.append(h[b, a1 - 1].cpu().numpy().astype(np.float16))

    if not ends:
        return {"collision_rate": 0.0, "mean_cos": 0.0}

    E = np.stack(ends).astype(np.float32)
    E = E - E.mean(0, keepdims=True)          # centring kills residual-stream anisotropy
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    S = E @ E.T
    off = S[~np.eye(S.shape[0], dtype=bool)]
    collision = float((off > 0.99).mean())
    mean_cos = float(off.mean())

    return {
        "seqs": np.array(seqs, dtype=object),
        "ends": np.stack(ends),
        "collision_rate": collision,
        "mean_cos": mean_cos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out", default="runs/eval.json")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--dump_traj", action="store_true")
    ap.add_argument("--limit_traj", type=int, default=96)
    ap.add_argument("--anchors_per_seq", type=int, default=16)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
