"""Pretraining-regime harness: full next-token supervision, no assistant span, no collapse.

Why this exists
---------------
Every arm in the fine-tuning matrix lives in a regime where 500 examples / 8 epochs
drives training CE to ~0.009 — the task is *memorised*, so any auxiliary objective
can only steal capacity from it.  The STP paper's own framing, though, is a
Chinchilla-style pretraining claim.  This harness puts the same arms in that regime:
all tokens are supervised, the corpus is ~10x larger than a step budget can consume,
and CE never comes close to zero.  Success is measured as held-out perplexity.

Usage:
    python -m stpp.pretrain --geom_kind stp_random_span --lbd 0.02 --steps 300 --seed 82
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .geometry import GeomConfig, GeometricRegularizer
from .optim import build_optimizer
from .train import HiddenTap, chunked_ce, get_body_and_head, get_layers

DS = Path(os.environ.get("STPP_DATA_DIR", "data/llm-jepa/datasets"))
CACHE = Path(os.environ.get("STPP_CACHE_DIR", "cache"))


# --------------------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------------------
def _contents(path: Path, max_rows: int = 200_000) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            try:
                msgs = json.loads(line)["messages"]
            except Exception:
                continue
            for m in msgs:
                c = (m.get("content") or "").strip()
                if c:
                    out.append(c)
    return out


def build_blocks(tok, split: str, block: int = 512) -> np.ndarray:
    """Tokenise every *_<split>.jsonl into one stream, then cut it into fixed blocks."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_f = CACHE / f"{split}_b{block}.npy"
    if cache_f.exists():
        return np.load(cache_f)

    files = sorted(DS.glob(f"*_{split}.jsonl"))
    texts = []
    for f in files:
        if f.stat().st_size > 40 * 1024 * 1024:      # paraphrase_train: cap it
            texts += _contents(f, max_rows=60_000)
        else:
            texts += _contents(f)
    print(f"[corpus] {split}: {len(files)} files, {len(texts)} segments", flush=True)

    eos = tok.eos_token_id
    ids: list[int] = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(eos)
    arr = np.asarray(ids, dtype=np.int32)
    n = (len(arr) // block) * block
    blocks = arr[:n].reshape(-1, block)
    np.save(cache_f, blocks)
    print(f"[corpus] {split}: {len(arr)/1e6:.1f}M tokens -> {len(blocks)} blocks of {block}",
          flush=True)
    return blocks


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
@dataclass
class PreConfig:
    model: str = field(
        default_factory=lambda: os.environ.get("STPP_MODEL", "unsloth/Llama-3.2-1B-Instruct")
    )
    out_dir: str = "runs/pretrain/run"
    steps: int = 300
    bs: int = 4
    block: int = 512
    # 数据选择实验：只在这些 block 上训练。给一个 .npy（int 索引列表）；空 = 全量。
    train_ids: str = ""
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    seed: int = 82
    lora_r: int = 16
    lora_alpha: int = 32
    # rsLoRA：缩放改成 alpha/sqrt(r)，高秩时不会因 alpha/r 变小而削弱增量分支。
    rslora: bool = False
    lora_dropout: float = 0.05
    grad_ckpt: bool = True
    bf16: bool = True
    # fp16 + GradScaler + 低精度累加。Ada 上 fp16 累加的张量吞吐是 fp32 累加的约 1.8x
    # （实测 prec_probe：同一形状 17.4 -> 33.5 TFLOPS）。默认关，bf16 路径完全不变。
    fp16: bool = False
    # 优化器：adamw（基线）/ muon（动量 + Newton-Schulz 正交化，RMS 对齐到 Adam）/
    # ademamix（beta1=0.9 + beta3=0.9999 双 EMA 混合）。见 stpp/optim.py 与 docs/SCALING.md。
    optim: str = "adamw"
    muon_lr_scale: float = 1.0
    ademamix_alpha: float = 5.0
    ademamix_beta3: float = 0.9999
    log_every: int = 20
    geom_kind: str = "none"
    lbd: float = 0.02
    geom_layers: str = "-1"
    whiten: bool = False
    tau: float = 0.15
    tau_adaptive: bool = True
    repulsion: float = 0.0
    adaptive_lambda: str = "none"
    target_ratio: float = 0.15
    eval_blocks: int = 200
    # 8-bit 基座：这台机器 WDDM 的显存预算被主机内存压低到 ~2.6-3.0GB，bf16 权重
    # (2.5GB) 一个人就顶满。8-bit 只给对照实验用（读取的是"测法有没有分辨力"，
    # 不是绝对数值），所有臂在同一子协议下比较。
    load_8bit: bool = False
    # CE 只监督 1/ce_stride 的 token（其余位置交给 latent 目标）—— 用于"替换式"实验
    ce_stride: int = 1
    # 多 token 预测（MTP）：除了 t+1，还让 h_t 直接预测 t+2..t+mtp_k。
    # 共用 lm_head（不新增参数），是"信息型"目标的正例候选：它确实把更长前缀
    # 的信息压进同一个位置。用于校准 ce_stride 缺口测试是否真的能检出信息。
    mtp_k: int = 0
    mtp_weight: float = 1.0
    # 分域评测：混合语料是 round-robin 交错的，测试块 i%domains 属于同一域。
    # 大于 1 时额外报告每域 ppl，用来把"选数据"与"改域配比"解耦。
    domains: int = 1
    domain_names: str = ""
    # 解冻所有 1-D 参数（RMSNorm 的逐通道缩放）。低秩 ΔW=BA 只有 r 个方向，
    # 要表示"逐通道缩放"这个对角算子需要 d 个自由度（d=2048 >> r）；
    # 直接给 norm 权重就是把这部分容量从 rank 预算里摘出来，几乎是免费的。
    train_norms: bool = False


def parse_layers(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


@torch.no_grad()
def evaluate_ppl(model, blocks, body, head, device, dtype, limit: int) -> dict:
    model.eval()
    total, count = 0.0, 0.0
    for i in range(min(limit, len(blocks))):
        ids = torch.as_tensor(blocks[i:i + 1], device=device)
        attn = torch.ones_like(ids)
        with torch.autocast(device_type="cuda", dtype=dtype,
                            enabled=(dtype in (torch.bfloat16, torch.float16))):
            hidden = body(input_ids=ids, attention_mask=attn).last_hidden_state
            h = body.norm(hidden) if hasattr(body, "norm") else hidden
            logits = head(h[:, :-1]).float()
            tgt = ids[:, 1:].long()
            total += float(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum"))
            count += float(tgt.numel())
    model.train()
    ce = total / max(count, 1.0)
    return {"ppl": math.exp(min(ce, 20.0)), "ce": ce, "tokens": int(count), "blocks": min(limit, len(blocks))}


def chunked_offsets(hidden: torch.Tensor, head: torch.nn.Module,
                    terms: list[tuple[torch.Tensor, int]], chunk: int = 64) -> list[torch.Tensor]:
    """One lm_head pass per chunk, several label/offset pairs read off the same logits.

    terms = [(labels, offset), ...]; position s is scored against labels[:, s+offset].
    Numerically identical to calling chunked_ce on each offset separately, but keeps a
    single logits tensor alive instead of one per offset (which is what blew up 8 GB).
    """
    B, T, _ = hidden.shape
    sums = [hidden.new_zeros((), dtype=torch.float32) for _ in terms]
    counts = [0.0 for _ in terms]
    for s in range(0, T - 1, chunk):
        e = min(s + chunk, T - 1)
        # 保持 bf16：fp32 上采样会让每个偏移都留一份 4*511*128256*4B ≈ 1GB 的
        # logits + softmax 在反向图上，三个偏移就 3GB —— 8GB 卡直接 OOM。
        logits = head(hidden[:, s:e])
        for i, (lab, off) in enumerate(terms):
            tgt = lab[:, s + off:e + off]
            k = min(tgt.shape[1], e - s)
            if k <= 0:
                continue
            lg = logits[:, :k]
            sums[i] = sums[i] + F.cross_entropy(
                lg.reshape(-1, lg.size(-1)), tgt[:, :k].reshape(-1),
                ignore_index=-100, reduction="sum",
            )
            counts[i] += float((tgt[:, :k] != -100).sum())
    return [sums[i] / max(counts[i], 1.0) for i in range(len(terms))]


def main() -> None:
    ap = argparse.ArgumentParser()
    inst = PreConfig()                                # 取真实值：default_factory 字段的 .default 是 MISSING
    for f in PreConfig.__dataclass_fields__:          # type: ignore[attr-defined]
        v = getattr(inst, f)
        if isinstance(v, bool):
            ap.add_argument(f"--{f}", action=argparse.BooleanOptionalAction, default=v)
        else:
            ap.add_argument(f"--{f}", type=type(v), default=v)
    cfg = PreConfig(**vars(ap.parse_args()))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    device = "cuda"
    dtype = torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else torch.float32)
    if cfg.fp16 and not cfg.bf16:
        # Ada 的 fp16 张量核在"低精度累加"下才跑满；这开关只影响 fp16 路径。
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if cfg.load_8bit:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0},
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.grad_ckpt)
        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=dtype, device_map=None,
        ).to(device)
        model.config.use_cache = False
        if cfg.grad_ckpt:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    if os.environ.get("STPP_GEMM16") in ("1", "2"):
        # 1 = ctypes 版（Windows 可用，但反传有拷贝）；2 = 零拷贝 C++ 扩展（需 Linux）。
        if os.environ.get("STPP_GEMM16") == "2":
            from .gemm16_ext import patch_linear
            patch_linear()
            print("[gemm16] zero-copy C++ extension enabled for all fp16 linear layers", flush=True)
        else:
            from .gemm16 import linear16
            import torch.nn.functional as _F
            _orig_linear = _F.linear

            def _lin16(x, w, b=None):
                if x.dim() >= 2 and w.dim() == 2 and x.shape[-1] == w.shape[-1] and x.dtype == torch.float16:
                    return linear16(x, w, b)
                return _orig_linear(x, w, b)

            torch.nn.functional.linear = _lin16
            print("[gemm16] ctypes 16F GEMM enabled for all linear layers", flush=True)

    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM", use_rslora=cfg.rslora,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    if cfg.train_norms:
        n_norm = 0
        for _name, p in model.named_parameters():
            if p.ndim == 1:               # LlamaRMSNorm 的 weight（Llama 无 bias）
                p.requires_grad_(True)
                n_norm += p.numel()
        print(f"[norms] unfroze {n_norm/1e6:.3f}M 1-D params (per-channel scales)", flush=True)
    model.print_trainable_parameters()
    if os.environ.get("STPP_FUSE_MLP") == "1":
        from .fuse_mlp import fuse_mlp
        n_fused = fuse_mlp(model)
        print(f"[fuse-mlp] gate+up fused in {n_fused} layers (one 2*d_ff wide GEMM)", flush=True)

    train_blocks = build_blocks(tok, "train", cfg.block)
    eval_blocks = build_blocks(tok, "test", cfg.block)
    if cfg.train_ids:
        sel = np.load(cfg.train_ids)
        print(f"[select] {len(sel)} / {len(train_blocks)} blocks from {cfg.train_ids}", flush=True)
        train_blocks = train_blocks[sel]

    layers = get_layers(model)
    n_layers = len(layers)
    li = [x if x >= 0 else n_layers + x for x in parse_layers(cfg.geom_layers)]
    if os.environ.get("STPP_FP8") == "1":
        # 冻结的 base linear 走 e4m3（我们自己的 Triton 核，(N,K) 天然布局，量化融进 kernel）
        from .fp8 import patch_fp8
        n_fp8, b_fp8 = patch_fp8(model)
        print(f"[fp8] patched {n_fp8} frozen linears ({b_fp8/1e9:.2f} GB fp8, 2 copies)", flush=True)
    body, head = get_body_and_head(model)
    if os.environ.get("STPP_COMPILE") == "1":
        # 把 3800+ 次 elementwise launch 交给 inductor 融合；形状是静态的（bs/block 固定）。
        torch._dynamo.config.cache_size_limit = 64
        body = torch.compile(body, mode=os.environ.get("STPP_COMPILE_MODE", "default"))
    d_model = model.config.hidden_size

    gcfg = GeomConfig(
        kind=cfg.geom_kind, lbd=cfg.lbd, whiten=cfg.whiten, layers=tuple(li),
        tau=cfg.tau, tau_adaptive=cfg.tau_adaptive, repulsion=cfg.repulsion,
        adaptive_lambda=None if cfg.adaptive_lambda == "none" else cfg.adaptive_lambda,
        target_ratio=cfg.target_ratio, anchors="random",  # whole-sequence span
    )
    reg = GeometricRegularizer(d_model, gcfg).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += [p for p in reg.parameters() if p.requires_grad]
    total_steps = cfg.steps
    opt = build_optimizer(trainable, cfg.optim, cfg.lr, cfg.weight_decay,
                          betas=(0.9, 0.95), muon_lr_scale=cfg.muon_lr_scale,
                          ademamix_alpha=cfg.ademamix_alpha,
                          ademamix_beta3=cfg.ademamix_beta3,
                          t_alpha=max(1, total_steps // 2))
    scaler = torch.amp.GradScaler("cuda") if (cfg.fp16 and not cfg.bf16) else None
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
    print(f"[cfg] geom={cfg.geom_kind} lbd={cfg.lbd} layers={li} whiten={cfg.whiten} "
          f"steps={total_steps} bs={cfg.bs} block={cfg.block} "
          f"train_blocks={len(train_blocks)}", flush=True)

    model.train()
    step = 0
    t0 = time.time()
    running = {"ntp": 0.0, "aux": 0.0, "mtp": 0.0, "n": 0}
    done = False
    order = np.random.default_rng(cfg.seed).permutation(len(train_blocks))
    prof_on = os.environ.get("STPP_PROFILE") == "1"
    prof = None
    prof_steps = 0

    for epoch in range(999):
        for bstart in range(0, len(order), cfg.bs):
            if done:
                break
            idx = order[bstart:bstart + cfg.bs]
            if prof_on:
                if step == 2:
                    prof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA])
                    prof.start()
                elif step == 5:
                    prof.stop()
                    ka = prof.key_averages()
                    total = sum(e.self_device_time_total for e in ka) / 3.0
                    print(f"[prof] 3 steps, per-step CUDA time = {total/1000:.1f} ms "
                          f"(sum over 3 steps = {total*3/1000:.1f} ms)", flush=True)
                    print(ka.table(sort_by="self_cuda_time_total", row_limit=30), flush=True)
                    log.close()
                    raise SystemExit(0)
            ids = torch.as_tensor(train_blocks[idx], device=device)
            attn = torch.ones_like(ids)
            labels = ids.clone().long()

            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            with torch.autocast(device_type="cuda", dtype=dtype,
                            enabled=(dtype in (torch.bfloat16, torch.float16))):
                tap.store.clear()
                with tap:
                    hidden = body(input_ids=ids, attention_mask=attn).last_hidden_state
                h_last = body.norm(hidden) if hasattr(body, "norm") else hidden
                if li and li[-1] == n_layers - 1:
                    tap.store[li[-1]] = h_last

                T = h_last.shape[1]
                spans = [(0, T)] * h_last.shape[0]
                # "替换式"：CE 只在每 ce_stride 个 token 上监督，其余位置没有任何
                # token 级目标 —— 这些位置的信息只能由 latent 目标去补。
                labels_ce = labels
                if cfg.ce_stride > 1:
                    keep = (torch.arange(T - 1, device=device) % cfg.ce_stride) == 0
                    labels_ce = labels.clone()
                    labels_ce[:, 1:] = labels[:, 1:].masked_fill(~keep.unsqueeze(0), -100)

                # CE（可能带 stride 掩码）+ MTP 多个偏移，共用同一次 lm_head 前向。
                # 分开调 chunked_ce 会让每份 logits 都留在反向图上 -> 8GB 直接 OOM。
                terms = [(labels_ce, 1)]
                if cfg.mtp_k > 1:
                    terms += [(labels, j) for j in range(2, cfg.mtp_k + 1)]
                # 多偏移时把 chunk 减半：每个 chunk 的 logits 与 CE 中间量都小一倍，
                # 8GB / WDDM 下才稳得住（数值不变，只是切法不同）。
                chunk = int(os.environ.get("STPP_LOGIT_CHUNK",
                                           32 if cfg.mtp_k > 1 else 64))
                ces = chunked_offsets(h_last, head, terms, chunk=chunk)
                ntp = ces[0]
                if cfg.mtp_k > 1:
                    mtp = sum(ces[1:]) / (cfg.mtp_k - 1)
                else:
                    mtp = h_last.new_zeros((), dtype=torch.float32)

                pooled = torch.stack([h_last[i, T - 1] for i in range(h_last.shape[0])])
                aux, stats = reg(tap.store, spans, generator=gen, ntp_loss=ntp,
                                 pooled_end=pooled)
                lam = reg.lam(ntp, aux)
                loss = (ntp + cfg.mtp_weight * mtp + lam * aux) / 1.0

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)

            running["ntp"] += float(ntp.detach())
            running["aux"] += float(aux.detach())
            running["mtp"] += float(mtp.detach())
            running["n"] += 1

            if step % cfg.log_every == 0:
                rec = {"step": step, "epoch": epoch,
                       "ntp": running["ntp"] / max(1, running["n"]),
                       "aux": running["aux"] / max(1, running["n"]),
                       "mtp": running["mtp"] / max(1, running["n"]),
                       "lam": float(lam), "lr": lr_at(step),
                       "elapsed": round(time.time() - t0, 1),
                       **{k: round(v, 5) for k, v in stats.items() if isinstance(v, float)},
                       "tau": float(reg._tau_ema)}
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()
                running = {"ntp": 0.0, "aux": 0.0, "mtp": 0.0, "n": 0}
                print(f"[{step}/{total_steps}] ntp={rec['ntp']:.4f} aux={rec['aux']:.4f} "
                      f"lam={rec['lam']:.4f} {int(rec['elapsed'])}s", flush=True)

            step += 1
            if step >= total_steps:
                done = True
                break
        if done:
            break

    ppl = evaluate_ppl(model, eval_blocks, body, head, device, dtype, cfg.eval_blocks)
    if cfg.domains > 1:
        names = [s.strip() for s in cfg.domain_names.split(",") if s.strip()] or \
                [f"g{g}" for g in range(cfg.domains)]
        ppl["domains"] = {}
        for g, name in enumerate(names[:cfg.domains]):
            sub = eval_blocks[g::cfg.domains]
            r = evaluate_ppl(model, sub, body, head, device, dtype, cfg.eval_blocks)
            ppl["domains"][name] = {"ppl": r["ppl"], "ce": r["ce"], "blocks": r["blocks"]}
    # Adapters are ~50 MB per run and are not needed for the reported metrics
    # (ppl is computed in-process above). Keep them only when explicitly asked,
    # or when the sentinel file exists; otherwise skip to spare the disk.
    keep = os.environ.get("STPP_SAVE_ADAPTER", "") == "1" or \
        (Path(__file__).resolve().parent.parent / "runs" / ".keep_adapter").exists()
    if keep:
        model.save_pretrained(out_dir / "adapter")
        tok.save_pretrained(out_dir / "adapter")
    if reg.predictor is not None:
        torch.save(reg.predictor.state_dict(), out_dir / "predictor.pt")
    meta = {"config": asdict(cfg), "steps": step, "wall_sec": round(time.time() - t0, 1),
            "n_train_blocks": int(len(train_blocks)), "eval": ppl}
    (out_dir / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    log.close()
    print(f"[done] {step} steps in {meta['wall_sec']}s -> ppl={ppl['ppl']:.3f} "
          f"ce={ppl['ce']:.4f}", flush=True)
    if "domains" in ppl:
        print("[domains] " + "  ".join(f"{k}={v['ppl']:.3f}" for k, v in ppl["domains"].items()),
              flush=True)


if __name__ == "__main__":
    main()
