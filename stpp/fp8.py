"""FP8 (e4m3) forward+backward for frozen base linears on sm_89.

Design note (v2)
----------------
`torch._scaled_mm` needs B column-major (stride(0)==1) and only tensor-wise
scaling (one scalar per operand) reaches 88 TFLOPS on sm_89; row-wise variants
fall back to ~35 TFLOPS, slower than bf16.

Forward wants B = (K,N) with stride(0)==1  -> keep w_nk (N,K) contiguous, pass .t()
dGrad   wants B = (N,K) with stride(0)==1  -> keep w_kn (K,N) contiguous, pass .t()
Two e4m3 copies therefore cost the same bytes as one bf16 weight (2 x 1B vs 2B):
memory-neutral, and no per-step re-quantization (v1, which quantized 1.24 B
elements per step, ran 4.2x SLOWER than bf16).
"""
from __future__ import annotations

import torch
import torch.nn as nn

FP8_MAX = 448.0


def _scale(t: torch.Tensor) -> torch.Tensor:
    return (t.abs().amax().clamp_min(1e-6) / FP8_MAX).to(torch.float32)


class _FP8FwdBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_nk, w_kn, s_w):
        ctx.save_for_backward(w_kn, s_w)
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        s_x = _scale(x2)
        x8 = (x2 / s_x).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(x8, w_nk.t(), scale_a=s_x, scale_b=s_w,
                               out_dtype=torch.bfloat16)
        return out.reshape(*shape[:-1], w_nk.shape[0])

    @staticmethod
    def backward(ctx, g):
        w_kn, s_w = ctx.saved_tensors
        shape = g.shape
        g2 = g.reshape(-1, shape[-1])
        s_g = _scale(g2)
        g8 = (g2 / s_g).to(torch.float8_e4m3fn)
        dx = torch._scaled_mm(g8, w_kn.t(), scale_a=s_g, scale_b=s_w,
                              out_dtype=torch.bfloat16)
        return dx.reshape(*shape[:-1], w_kn.shape[0]), None, None, None


class FP8Linear(nn.Module):
    """Frozen linear, e4m3 in both directions. Weight bytes == bf16 original."""

    def __init__(self, src: nn.Linear):
        super().__init__()
        w = src.weight.detach().float()
        s_w = _scale(w)
        q = (w / s_w).to(torch.float8_e4m3fn)
        self.register_buffer("w_nk", q.contiguous(), persistent=False)          # (N,K)
        self.register_buffer("w_kn", q.t().contiguous(), persistent=False)      # (K,N)
        self.register_buffer("s_w", s_w, persistent=False)
        self.in_features = src.in_features
        self.out_features = src.out_features
        self.bias = None
        if src.bias is not None:
            self.bias = nn.Parameter(src.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        out = _FP8FwdBwd.apply(x, self.w_nk, self.w_kn, self.s_w)
        return out if self.bias is None else out + self.bias

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, e4m3 fwd+bwd"


def patch_fp8(module: nn.Module, skip: tuple[str, ...] = ("lm_head",)) -> tuple[int, int]:
    n, nbytes = 0, 0
    for name, child in list(module.named_children()):
        if name in skip:
            continue
        if isinstance(child, nn.Linear) and not child.weight.requires_grad:
            new = FP8Linear(child)
            setattr(module, name, new)
            n += 1
            nbytes += new.w_nk.numel()
            continue
        base = getattr(child, "base_layer", None)
        if isinstance(base, nn.Linear) and not base.weight.requires_grad:
            new = FP8Linear(base)
            child.base_layer = new
            n += 1
            nbytes += new.w_nk.numel()
            continue
        dn, db = patch_fp8(child, skip)
        n, nbytes = n + dn, nbytes + db
    return n, nbytes
