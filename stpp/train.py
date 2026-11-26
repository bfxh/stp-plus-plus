"""Single-GPU trainer for the STP reproduction + our curvature-budgeted variants.

Why not reuse the reference `stp.py`? It is a 1.7k-line HF-Trainer/FSDP script written
for 8 GPUs. This is a compact, single-GPU loop that fits an 8 GB card and exposes every
geometric knob as a flag, so ablations are a one-liner.

The NTP loss is computed in position-chunks against `lm_head` instead of letting
`ForCausalLM` materialise a (B, T, 128k) logit tensor — that alone saves ~1.5 GB.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import ChatSFTDataset, collate, read_jsonl
from .geometry import GeomConfig, GeometricRegularizer


# ----------------------------------------------------------------------------------
# hidden-state tap
# ----------------------------------------------------------------------------------
class HiddenTap:
    """Forward hooks on selected decoder layers; keeps the tensors in the autograd graph."""

    def __init__(self, layers: torch.nn.ModuleList, indices: list[int]):
        self.layers = layers
        self.indices = indices
        self.store: dict[int, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        for li in self.indices:
            self._handles.append(self.layers[li].register_forward_hook(self._make(li)))
        return self

    def _make(self, li: int):
        def hook(_m, _inp, out):
            self.store[li] = out[0] if isinstance(out, tuple) else out
        return hook

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def clear(self):
        self.store.clear()


def get_layers(model) -> torch.nn.ModuleList:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for attr in ("model.layers", "model.decoder.layers", "transformer.h"):
        obj = base
        for p in attr.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise RuntimeError("cannot locate decoder layers")


def get_body_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    body = getattr(base, "model", None) or getattr(base, "transformer", None)
    head = base.get_output_embeddings()
    if body is None or head is None:
        raise RuntimeError("cannot locate body / lm_head")
    return body, head


# ----------------------------------------------------------------------------------
# chunked causal-LM loss
# ----------------------------------------------------------------------------------
def chunked_ce(hidden: torch.Tensor, head: torch.nn.Module, labels: torch.Tensor,
               chunk: int = 64) -> torch.Tensor:
    """Causal LM loss, computed in position chunks so the (B,T,vocab) logits never exist.

    logits at position s predict token s+1, hence the shift-by-one on labels.
    """
    B, T, _ = hidden.shape
    total = hidden.new_zeros((), dtype=torch.float32)
    count = 0.0
    shift_labels = labels[:, 1:]
    for s in range(0, T - 1, chunk):
        e = min(s + chunk, T - 1)
        logits = head(hidden[:, s:e]).float()
        tgt = shift_labels[:, s:e]
        total = total + F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100,
            reduction="sum",
        )
        count += float((tgt != -100).sum())
    return total / max(count, 1.0)


def span_ce(hidden: torch.Tensor, head: torch.nn.Module, labels: torch.Tensor,
            spans, chunk: int = 64) -> torch.Tensor:
    """Same loss, but only over the assistant span.

    ~80% of the tokens in a chat sample are prompt tokens that are masked out anyway.
    Gathering just the supervised window cuts the lm_head cost by roughly that factor
    *and* keeps the geometry window and the loss window identical.
    """
    B, T, D = hidden.shape
    lens = [max(0, a1 - a0) for a0, a1 in spans]
    L = max(lens) if lens else 0
    if L == 0:
        return hidden.new_zeros((), dtype=torch.float32)

    hs = hidden.new_zeros((B, L, D))
    tgt = torch.full((B, L), -100, device=hidden.device, dtype=torch.long)
    for b, (a0, a1) in enumerate(spans):
        n = max(0, a1 - a0)
        if n == 0:
            continue
        hs[b, :n] = hidden[b, a0 - 1: a1 - 1]
        tgt[b, :n] = labels[b, a0: a1]

    total = hidden.new_zeros((), dtype=torch.float32)
    count = 0.0
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        logits = head(hs[:, s:e]).float()
        total = total + F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt[:, s:e].reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        count += float((tgt[:, s:e] != -100).sum())
    return total / max(count, 1.0)


# ----------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model: str = "unsloth/Llama-3.2-1B-Instruct"
    train_file: str = ""
    test_file: str = ""
    out_dir: str = "runs/debug"
    epochs: float = 4.0
    bs: int = 8
    grad_accum: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_len: int = 320
    fraction: float = 1.0
    seed: int = 82
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    grad_ckpt: bool = True
    bf16: bool = True
    max_steps: int = -1
    log_every: int = 10
    # geometry
    geom_kind: str = "none"
    lbd: float = 0.02
    geom_layers: str = "-1"
    whiten: bool = False
    anchors: str = "random"
    n_spans: int = 1
    length_adjustment: str = "none"
    tau: float = 0.15
    tau_adaptive: bool = True
    repulsion: float = 0.0
    adaptive_lambda: str = "none"
    target_ratio: float = 0.15


def parse_layers(s: str) -> tuple[int, ...]:
    if s.strip() in ("-1", "last"):
        return (-1,)
    return tuple(int(x) for x in s.split(","))


def build_model(cfg: TrainConfig, dtype):
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kwargs = dict(trust_remote_code=True, torch_dtype=dtype, attn_implementation="sdpa")
    model = AutoModelForCausalLM.from_pretrained(cfg.model, **kwargs)
    model.config.use_cache = False

    if cfg.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    for f, v in asdict(TrainConfig()).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{f}", action=argparse.BooleanOptionalAction, default=v)
        else:
            ap.add_argument(f"--{f}", type=type(v), default=v)
    args = ap.parse_args()

    cfg = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (cfg.bf16 and torch.cuda.is_available()) else torch.float32

    model, tok = build_model(cfg, dtype)
    model.to(device)

    from peft import LoraConfig  # noqa: F401  (already applied)

    train_ex = read_jsonl(cfg.train_file)
    ds = ChatSFTDataset(train_ex, tok, max_length=cfg.max_len,
                        fraction=cfg.fraction, seed=cfg.seed, anchor_mode=cfg.anchors)

    layers = get_layers(model)
    n_layers = len(layers)
    li = [x if x >= 0 else n_layers + x for x in parse_layers(cfg.geom_layers)]
    body, head = get_body_and_head(model)

    d_model = model.config.hidden_size
    gcfg = GeomConfig(
        kind=cfg.geom_kind,
        lbd=cfg.lbd,
        n_spans=cfg.n_spans,
        length_adjustment=None if cfg.length_adjustment == "none" else cfg.length_adjustment,
        whiten=cfg.whiten,
        layers=tuple(li),
        anchors=cfg.anchors,
        tau=cfg.tau,
        tau_adaptive=cfg.tau_adaptive,
        repulsion=cfg.repulsion,
        adaptive_lambda=None if cfg.adaptive_lambda == "none" else cfg.adaptive_lambda,
        target_ratio=cfg.target_ratio,
    )
    reg = GeometricRegularizer(d_model, gcfg).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    # a learned latent predictor (kind="latent_pred") has parameters of its own
    trainable += [p for p in reg.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(ds) / (cfg.bs * cfg.grad_accum))
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    if cfg.max_steps > 0:
        total_steps = min(total_steps, cfg.max_steps)
    warmup = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return cfg.lr * step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * p))

    log = open(out_dir / "train.jsonl", "w", encoding="utf-8")
    tap = HiddenTap(layers, li)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed)

    print(f"[cfg] model={cfg.model} geom={cfg.geom_kind} lbd={cfg.lbd} "
          f"layers={li} whiten={cfg.whiten} anchors={cfg.anchors} "
          f"fraction={cfg.fraction} steps={total_steps} n={len(ds)}")

    model.train()
    step = 0
    t0 = time.time()
    running = {"ntp": 0.0, "aux": 0.0, "n": 0}
    done = False

    for epoch in range(max(1, math.ceil(cfg.epochs))):
        perm = torch.randperm(len(ds), generator=gen).tolist()
        for bstart in range(0, len(perm), cfg.bs):
            if done:
                break
            idxs = perm[bstart:bstart + cfg.bs]
            batch = collate([ds[i] for i in idxs], tok.pad_token_id)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype == torch.bfloat16)):
                tap.store.clear()
                with tap:
                    hidden = body(input_ids=input_ids, attention_mask=attn).last_hidden_state
                # the last decoder output still has to go through the final RMSNorm before
                # lm_head — geometry and CE must both live in *that* space.
                h_last = body.norm(hidden) if hasattr(body, "norm") else hidden
                if li and li[-1] == n_layers - 1:
                    tap.store[li[-1]] = h_last
                ntp = span_ce(h_last, head, labels, batch["spans"], chunk=64)

                pooled = torch.stack([h_last[i, max(0, batch["spans"][i][1] - 1)]
                                      for i in range(h_last.shape[0])])
                aux, stats = reg(tap.store, batch["spans"], generator=gen,
                                 ntp_loss=ntp, pooled_end=pooled)
                lam = reg.lam(ntp, aux)
                loss = ntp + lam * aux
                loss = loss / cfg.grad_accum

            loss.backward()
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            running["ntp"] += float(ntp.detach())
            running["aux"] += float(aux.detach())
            running["n"] += 1

            if step % cfg.log_every == 0:
                rec = {
                    "step": step, "epoch": epoch,
                    "ntp": running["ntp"] / max(1, running["n"]),
                    "aux": running["aux"] / max(1, running["n"]),
                    "lam": float(lam.detach()) if torch.is_tensor(lam) else float(lam),
                    "lr": lr_at(step),
                    "elapsed": round(time.time() - t0, 1),
                    **{k: round(v, 5) for k, v in stats.items() if isinstance(v, float)},
                    "tau": float(reg._tau_ema),
                }
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()
                running = {"ntp": 0.0, "aux": 0.0, "n": 0}
                print(f"[{step}/{total_steps}] ntp={rec['ntp']:.4f} aux={rec['aux']:.4f} "
                      f"lam={rec['lam']:.4f} {int(rec['elapsed'])}s", flush=True)

            step += 1
            if step >= total_steps:
                done = True
                break
        if done:
            break

    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    if reg.predictor is not None:
        torch.save(reg.predictor.state_dict(), out_dir / "predictor.pt")
    meta = {
        "config": asdict(cfg),
        "steps": step,
        "wall_sec": round(time.time() - t0, 1),
        "n_train": len(ds),
        "geom_layers": list(li),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.close()
    print(f"[done] {step} steps in {meta['wall_sec']}s -> {out_dir}")


if __name__ == "__main__":
    main()
