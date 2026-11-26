"""Teacher-forced accuracy — the sensitive metric when exact-match is still ~0.

Early in training (or at 1/16 data) greedy generation is all-or-nothing, so exact match
sits at 0 for every arm and tells you nothing. Teacher-forcing the gold answer gives a
smooth per-token signal, plus a stricter "whole answer correct" rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import content_start, read_jsonl


def build_prompt(tok, messages) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def run(args) -> dict:
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval().cuda()

    data = read_jsonl(args.test_file)
    if args.limit > 0:
        data = data[: args.limit]

    pairs = []
    for ex in data:
        msgs = ex["messages"]
        # build *exactly* like training: full template + prompt-only template
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        pairs.append((prompt, full))

    tok_correct = 0
    tok_total = 0
    seq_correct = 0

    for i in range(0, len(pairs), args.bs):
        chunk = pairs[i: i + args.bs]
        prompt_lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p, _ in chunk]
        enc = tok([f for _, f in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len).to("cuda")
        out = model(**enc).logits

        specials = set(tok.all_special_ids)
        for b, _s in enumerate(chunk):
            a1 = int(enc["attention_mask"][b].sum())
            a0 = content_start(tok, enc["input_ids"][b, :a1].tolist(), prompt_lens[b])
            if a1 - a0 < 1:
                continue
            gold = enc["input_ids"][b, a0: a1]
            # a single missed <|eot_id|> would zero the whole sample; compare content only
            keep = 0
            for t in gold.tolist():
                if t in specials:
                    break
                keep += 1
            if keep == 0:
                continue
            pred = out[b, a0 - 1: a0 - 1 + keep].argmax(-1)
            ok = (pred == gold[:keep])
            tok_correct += int(ok.sum())
            tok_total += int(ok.numel())
            seq_correct += int(bool(ok.all()))

    res = {
        "n": len(pairs),
        "tf_token_acc": tok_correct / max(1, tok_total),
        "tf_seq_acc": seq_correct / max(1, len(pairs)),
        "model": args.model,
        "adapter": args.adapter,
    }
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out", default="runs/tf.json")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--limit", type=int, default=-1)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
