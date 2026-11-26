"""Fuse gate_proj and up_proj into a single GEMM.

Why: after torch.compile the step is 83% GEMM and only runs at ~26 TFLOPS
(55% of the 48.5 achievable in isolation, see docs/MFU.md 24.11c). The MLP
is 75% of the FLOPs and is emitted as two separate 2048x2048x8192 GEMMs per
layer; cuBLAS picks a 128x128 tile for those and leaves bandwidth on the
table. Concatenating the two frozen base weights along the output dim turns
that into one 2048x2048x16384 GEMM with the same FLOPs, half the launches
and a much wider N - which is exactly the knob cuBLAS uses to pick a 256x128
tile.

The base weights are frozen (LoRA training), so the concat costs no extra
memory: the two originals are dropped after the copy. LoRA A/B stay separate
per projection and are added back as two small GEMMs, i.e. numerics are
unchanged up to associativity of the base matmul.

Enable with STPP_FUSE_MLP=1. Default path is untouched.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _add_lora(mod, x, y):
    """y + scaling * (x @ A^T) @ B^T for the (peft) LoRA branch, or y."""
    A = getattr(mod, "lora_A", None)
    if not A:
        return y
    key = next(iter(A.keys()))
    aw = A[key].weight
    bw = mod.lora_B[key].weight
    scale = mod.scaling[key] if hasattr(mod, "scaling") else 1.0
    return y + ((x @ aw.t()) @ bw.t()) * scale


def _fused_mlp_forward(self, x):
    y = F.linear(x, self._fused_W, self._fused_b)
    g, u = y.chunk(2, dim=-1)
    g = _add_lora(self.gate_proj, x, g)
    u = _add_lora(self.up_proj, x, u)
    return self.down_proj(self.act_fn(g) * u)


def _drop_weight(lin):
    try:
        del lin.weight
    except Exception:
        pass
    if getattr(lin, "bias", None) is not None:
        try:
            del lin.bias
        except Exception:
            pass


def fuse_mlp(model) -> int:
    """Patch every LlamaMLP in place. Returns the number of layers fused."""
    patched = 0
    for _, mod in list(model.named_modules()):
        if mod.__class__.__name__ != "LlamaMLP":
            continue
        gate, up = getattr(mod, "gate_proj", None), getattr(mod, "up_proj", None)
        if gate is None or up is None:
            continue
        gb = getattr(gate, "base_layer", gate)
        ub = getattr(up, "base_layer", up)
        if getattr(gb, "weight", None) is None or getattr(ub, "weight", None) is None:
            continue
        w = torch.cat([gb.weight.detach(), ub.weight.detach()], 0).contiguous()
        b = None
        if gb.bias is not None and ub.bias is not None:
            b = torch.cat([gb.bias.detach(), ub.bias.detach()], 0).contiguous()
        mod.register_buffer("_fused_W", w, persistent=False)
        mod.register_buffer("_fused_b", b, persistent=False)
        _drop_weight(gb)
        _drop_weight(ub)
        mod.forward = types.MethodType(_fused_mlp_forward, mod)
        patched += 1
    return patched
