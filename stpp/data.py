"""Dataset + tokenisation for the NL-RX-SYNTH reproduction and any chat jsonl.

Data format (what the reference repo ships)::

    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user",   "content": "..."},
                  {"role": "assistant", "content": "..."}]}

We tokenise with the model's chat template and mask the loss to the assistant turn,
recording the exact assistant span so the geometric regularisers know where the
"trajectory of interest" lives.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    span: tuple[int, int]          # assistant span in token coordinates
    anchors: list[int]             # optional semantic anchors (step boundaries)
    raw: dict


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _apply_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def content_start(tok, full_ids: list[int], prompt_len: int) -> int:
    """Index of the first *content* token of the assistant turn.

    Tokenising the prompt on its own is not safe here: BPE can merge the template's
    trailing ``\\n\\n`` with whatever follows, so ``len(prompt_ids)`` lands 1-2 tokens
    early. We instead locate the assistant header terminator and skip the whitespace
    that the template emits after it.
    """
    hdr = None
    try:
        hdr = tok.convert_tokens_to_ids("<|end_header_id|>")
    except Exception:
        hdr = None
    if isinstance(hdr, int) and hdr >= 0:
        pos = None
        for i, t in enumerate(full_ids):
            if t == hdr:
                pos = i
        if pos is not None:
            j = pos + 1
            while j < len(full_ids) and j <= pos + 3:
                if tok.decode([full_ids[j]]).strip() == "":
                    j += 1
                else:
                    break
            return j
    return prompt_len


def _detect_anchors(text: str, tokenizer, offset: int, ids: list[int]) -> list[int]:
    """Anchor positions for the geometric constraint.

    Preferred: explicit reasoning-step delimiters (``<|step|>`` / double newline).
    Falls back to evenly spaced token positions (``stride`` handled later).
    """
    delims = ["<|step|>", "\n\n"]
    marks: list[int] = []
    for d in delims:
        if d in text:
            parts = text.split(d)
            running = 0
            for p in parts[:-1]:
                running += len(tokenizer(p, add_special_tokens=False)["input_ids"]) + len(
                    tokenizer(d, add_special_tokens=False)["input_ids"]
                )
                marks.append(offset + running - 1)
            break
    return [m for m in marks if offset <= m]


class ChatSFTDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[dict],
        tokenizer,
        max_length: int = 320,
        fraction: float = 1.0,
        seed: int = 0,
        anchor_mode: str = "random",
        stride: int = 8,
    ):
        self.tok = tokenizer
        self.max_length = max_length
        self.anchor_mode = anchor_mode
        self.stride = stride
        self.items = self._encode(examples, fraction=fraction, seed=seed)

    # ------------------------------------------------------------------
    def _encode(self, examples, fraction: float, seed: int) -> list[Example]:
        if fraction < 1.0:
            rng = random.Random(seed)
            n = max(1, int(len(examples) * fraction))
            examples = rng.sample(list(examples), n)

        out: list[Example] = []
        for ex in examples:
            msgs = ex["messages"]
            if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
                continue
            ctx, answer = msgs[:-1], msgs[-1]

            full_text = _apply_template(self.tok, msgs, add_generation_prompt=False)
            prompt_text = _apply_template(self.tok, ctx, add_generation_prompt=True)

            full_ids = self.tok(full_text, add_special_tokens=False)["input_ids"]
            prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]

            n_prompt = content_start(self.tok, full_ids, len(prompt_ids))
            if n_prompt >= len(full_ids):
                continue
            a0, a1 = n_prompt, len(full_ids)

            full_ids = full_ids[: self.max_length]
            a1 = min(a1, len(full_ids))
            if a1 <= a0:
                continue

            labels = [-100] * a0 + full_ids[a0:]

            anchors: list[int] = []
            if self.anchor_mode == "step":
                anchors = _detect_anchors(answer["content"], self.tok, a0, full_ids)
            if not anchors and self.anchor_mode in ("step", "stride"):
                anchors = list(range(a0, a1, max(1, self.stride)))

            out.append(Example(full_ids, labels, (a0, a1), anchors, ex))
        return out

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        return {
            "input_ids": torch.tensor(it.input_ids, dtype=torch.long),
            "labels": torch.tensor(it.labels, dtype=torch.long),
            "span": torch.tensor(it.span, dtype=torch.long),
            "anchors": torch.tensor(it.anchors or [-1], dtype=torch.long),
            "index": i,
        }


def collate(batch: list[dict], pad_id: int) -> dict:
    T = max(x["input_ids"].numel() for x in batch)
    B = len(batch)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    spans = torch.zeros((B, 2), dtype=torch.long)
    anchors = torch.full((B, T), -1, dtype=torch.long)

    for i, x in enumerate(batch):
        n = x["input_ids"].numel()
        input_ids[i, :n] = x["input_ids"]
        labels[i, :n] = x["labels"]
        attn[i, :n] = 1
        spans[i] = x["span"]
        a = x["anchors"]
        anchors[i, : a.numel()] = a

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "spans": [(int(s[0]), int(s[1])) for s in spans],
        "anchors": anchors,
        "indices": [x["index"] for x in batch],
    }
