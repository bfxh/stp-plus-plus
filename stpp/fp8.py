"""FP8 base linears using our own Triton kernels (sm_89 mma e4m3).

Why our kernel instead of torch._scaled_mm
------------------------------------------
cuBLASLt only reaches 88 TFLOPS when B is column-major, so the training graph had
to insert a layout copy per call (and again per gradient-checkpoint recompute) —
that is why the first two FP8 integrations ran 3.6-4.2x SLOWER than bf16 even
though the bare GEMM was 1.9x faster.

Our kernel consumes B in nn.Linear's natural (N,K) row-major layout and fuses the
per-row activation quantization, so there is no transpose and no per-step weight
cast.  Measured on real shapes: 86.6 / 63.6 / 83.9 TFLOPS (quant included on the
gate/up case), i.e. on par with cuBLASLt.

Storage: two e4m3 copies (w_nk for fwd, w_kn for dgrad) = 2 bytes/param total,
the same footprint as one bf16 weight.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kernels.fp8_gemm_triton import quant_rows, fp8_gemm, FP8_MAX


def _scale(t: torch.Tensor) -> torch.Tensor:
    return (t.abs().amax().clamp_min(1e-6) / FP8_MAX).to(torch.float32)


class _FP8FwdBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_nk, w_kn, s_w):
        ctx.save_for_backward(w_kn, s_w)
        x8, s_x, shape = quant_rows(x)
        out = fp8_gemm(x8, w_nk, s_x, s_w)
        return out.reshape(*shape[:-1], w_nk.shape[0])

    @staticmethod
    def backward(ctx, g):
        w_kn, s_w = ctx.saved_tensors
        shape = g.shape
        g8, s_g, _ = quant_rows(g)
        dx = fp8_gemm(g8, w_kn, s_g, s_w)
        return dx.reshape(*shape[:-1], w_kn.shape[0]), None, None, None


class FP8Linear(nn.Module):
    def __init__(self, src: nn.Linear):
        super().__init__()
        w = src.weight.detach().float()
        s_w = _scale(w)
        q = (w / s_w).to(torch.float8_e4m3fn)
        self.register_buffer("w_nk", q.contiguous(), persistent=False)      # (N,K)
        self.register_buffer("w_kn", q.t().contiguous(), persistent=False)  # (K,N)
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
        return f"in={self.in_features}, out={self.out_features}, e4m3 triton fwd+bwd"


def patch_fp8(module: nn.Module, skip: tuple[str, ...] = ()) -> tuple[int, int]:
    n, b = 0, 0
    for name, child in list(module.named_children()):
        if name in skip:
            continue
        if isinstance(child, nn.Linear) and not child.weight.requires_grad:
            new = FP8Linear(child)
            setattr(module, name, new)
            n += 1
            b += new.w_nk.numel()
            continue
        base = getattr(child, "base_layer", None)
        if isinstance(base, nn.Linear) and not base.weight.requires_grad:
            new = FP8Linear(base)
            child.base_layer = new
            n += 1
            b += new.w_nk.numel()
            continue
        dn, db = patch_fp8(child, skip)
        n, b = n + dn, b + db
    return n, b
