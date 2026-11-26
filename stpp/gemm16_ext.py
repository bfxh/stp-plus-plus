"""Load and wrap the zero-copy fp16-accumulate GEMM extension.

The ctypes version (stpp/gemm16.py) proved the 1.8x exists but cost more than it
saved inside the trainer: every backward needed `g.t().contiguous()` copies and
the extra buffers pushed an 8GB card into paging.  This version does the three
GEMMs with the transposes folded into the cuBLAS flags, so there is no copy at
all and no Python-side per-call overhead.

Needs: g++ + the CUDA 12.8 headers/libs that torch ships (no nvcc, no MSVC).
"""
from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

_SITE = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))  # .../site-packages
_NV = os.path.join(_SITE, "nvidia")
_CUBLAS_INC = os.path.join(_NV, "cublas", "include")
_CUBLAS_LIB = os.path.join(_NV, "cublas", "lib")
_CUDART_INC = os.path.join(_NV, "cuda_runtime", "include")
_CUDART_LIB = os.path.join(_NV, "cuda_runtime", "lib")

_mod = None

# The pip-shipped cuda_runtime/include is a trimmed copy without crt/host_defines.h,
# so take headers from the real toolkit when it is present and keep the *libraries*
# at 12.8 (the version torch itself links against).
_INCS = [p for p in ("/usr/local/cuda/include", _CUBLAS_INC) if os.path.isdir(p)]


def _load():
    global _mod
    if _mod is not None:
        return _mod
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "csrc", "gemm16.cpp")
    if not os.path.exists(src):
        raise RuntimeError(f"missing {src}")
    _mod = load(
        name="stp_gemm16",
        sources=[src],
        extra_include_paths=_INCS,
        extra_ldflags=[
            os.path.join(_CUBLAS_LIB, "libcublas.so.12"),
            os.path.join(_CUDART_LIB, "libcudart.so.12"),
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return _mod


class _Gemm16Fn(torch.autograd.Function):
    """y = x @ w^T with fp16 accumulate; backward with the same kernel class."""

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return _load().nt(x, w)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        g = g.contiguous()
        return _load().dx(g, w), _load().dw(g, x)


def linear16(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None):
    """Drop-in for F.linear on 2D+ inputs; fp16 only, weights stay fp16."""
    orig = x.shape
    x2 = x.reshape(-1, orig[-1])
    if x2.dtype != torch.float16:
        x2 = x2.to(torch.float16)
    w16 = w if w.dtype == torch.float16 else w.to(torch.float16)
    y = _Gemm16Fn.apply(x2, w16)
    if bias is not None:
        y = y + bias.to(torch.float16)
    return y.reshape(*orig[:-1], w.shape[0]).to(x.dtype)


def patch_linear():
    """Route every 2D fp16 linear through the 16F-accumulate kernel."""
    import torch.nn.functional as F

    orig = F.linear

    def _lin(x, w, b=None):
        if x.dim() >= 2 and w.dim() == 2 and x.shape[-1] == w.shape[-1] and x.dtype == torch.float16:
            return linear16(x, w, b)
        return orig(x, w, b)

    F.linear = _lin
    return orig
