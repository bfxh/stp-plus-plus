"""fp16-accumulate GEMM for Ada, without building a CUDA extension.

Why: on sm_89 the tensor cores run FP16-accumulate at ~1.8x the FP32-accumulate
rate (measured, kernels/prec_probe.cu: 17.4 -> 33.5 TFLOPS on the same shape).
PyTorch's autocast never emits CUBLAS_COMPUTE_16F, so that 1.8x is unreachable
through the normal API.  This module reaches it with ctypes straight into the
cuBLAS DLL that torch already ships (no MSVC / no nvcc needed on Windows).

Numerics (kernels/acc16_check.cu, K=2048/8192, act~N(0,1), w~N(0,0.02)):
    fp32-acc max abs err = 0.06% of rms
    fp16-acc max abs err = 0.59-0.73% of rms, zero non-finite
That is the same order as the bf16 input rounding itself, so it is safe for
LoRA training (and it is what fp16 training did for years).

Usage:
    from stpp.gemm16 import linear16
    y = linear16(x, w, b)      # x: (..., K) fp16/bf16-able, w: (N, K)
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import torch

_LIB = None
_HANDLE = None
_HANDLE_DEV = -1

# cudaDataType / cublasComputeType enums
CUDA_R_16F = 2
CUDA_R_16BF = 14
CUBLAS_COMPUTE_16F = 64
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    candidates = []
    try:
        import torch.utils.cpp_extension as _ce
        candidates.append(Path(torch.__file__).parent / "lib")
    except Exception:
        pass
    for name in ("cublas64_12.dll", "cublas64_11.dll", "libcublas.so.12", "libcublas.so"):
        for d in candidates:
            p = Path(d) / name
            if p.exists():
                candidates.insert(0, p)
                break
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        # let the OS resolve it
        path = "cublas64_12.dll" if sys.platform == "win32" else "libcublas.so.12"
    # torch must already be imported so its CUDA context exists
    _LIB = ctypes.CDLL(str(path))
    return _LIB


def _handle():
    global _HANDLE, _HANDLE_DEV
    dev = torch.cuda.current_device()
    if _HANDLE is None or _HANDLE_DEV != dev:
        lib = _lib()
        h = ctypes.c_void_p()
        st = lib.cublasCreate_v2(ctypes.byref(h))
        if st != 0:
            raise RuntimeError(f"cublasCreate_v2 failed: {st}")
        st = lib.cublasSetStream_v2(h, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if st != 0:
            raise RuntimeError(f"cublasSetStream_v2 failed: {st}")
        _HANDLE, _HANDLE_DEV = h, dev
    return _HANDLE


class _Gemm16(torch.autograd.Function):
    """C = A @ B^T with FP16 accumulate. A:(M,K) B:(N,K) -> (M,N)."""

    @staticmethod
    def forward(ctx, a, b):
        m, k = a.shape
        n = b.shape[0]
        c = torch.empty((m, n), device=a.device, dtype=torch.float16)
        lib = _lib()
        h = _handle()
        lib.cublasSetStream_v2(h, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        # CUBLAS_COMPUTE_16F 要求 alpha/beta 是 __half 本体（2 字节），传 float 会被读成 0
        alpha = ctypes.c_uint16(0x3C00)   # fp16 1.0
        beta = ctypes.c_uint16(0x0000)    # fp16 0.0
        # column-major trick: C^T (n x m) = B (n x k) * A^T
        st = lib.cublasGemmEx(
            h,
            ctypes.c_int(CUBLAS_OP_T), ctypes.c_int(CUBLAS_OP_N),
            ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
            ctypes.byref(alpha),
            ctypes.c_void_p(b.data_ptr()), ctypes.c_int(CUDA_R_16F), ctypes.c_int(k),
            ctypes.c_void_p(a.data_ptr()), ctypes.c_int(CUDA_R_16F), ctypes.c_int(k),
            ctypes.byref(beta),
            ctypes.c_void_p(c.data_ptr()), ctypes.c_int(CUDA_R_16F), ctypes.c_int(n),
            ctypes.c_int(CUBLAS_COMPUTE_16F), ctypes.c_int(0),
        )
        if st != 0:
            raise RuntimeError(f"cublasGemmEx(16F) failed: {st}")
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        # dx = g @ b ; dw = g^T @ a   (both expressed as "A @ W^T")
        ga = _Gemm16.apply(g.contiguous(), b.t().contiguous())
        gb = _Gemm16.apply(g.t().contiguous(), a.t().contiguous())
        return ga, gb


def linear16(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None):
    """y = x @ w.T (+ bias). w is (N, K) like nn.Linear.weight."""
    orig = x.shape
    x2 = x.reshape(-1, orig[-1])
    w16 = w.to(torch.float16)
    if x2.dtype != torch.float16:
        x2 = x2.to(torch.float16)
    y = _Gemm16.apply(x2, w16)
    if bias is not None:
        y = y + bias.to(torch.float16)
    return y.reshape(*orig[:-1], w.shape[0]).to(x.dtype)
