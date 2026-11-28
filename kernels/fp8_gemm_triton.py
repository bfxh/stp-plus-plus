"""Own FP8 GEMM kernels (Triton / sm_89 mma e4m3).

Motivation: cuBLASLt reaches 88 TFLOPS only with B column-major, which forces a
layout copy inside our training graph (that is what made FP8 slower than bf16).
Here we consume B in its *natural* nn.Linear layout (N,K) row-major, and fuse the
per-row activation quantization into the same pass, so the graph needs no extra
transpose and no per-step weight cast.

Two entry points:
  quant_rows(x)  -> x8 (M,K) e4m3 + scale (M,) fp32        (~2 passes over x)
  fp8_gemm(x8, w8, sx, sw) -> (M,N) bf16, w8 is (N,K) e4m3
"""
import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


@triton.jit
def _quant_rows_kernel(X, X8, S, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        mask = mask_m[:, None] & (cols[None, :] < K)
        x = tl.load(X + rows[:, None] * K + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=1))
    s = tl.maximum(amax, 1e-6) / 448.0
    tl.store(S + rows, s, mask=mask_m)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        mask = mask_m[:, None] & (cols[None, :] < K)
        x = tl.load(X + rows[:, None] * K + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        q = (x / s[:, None]).to(tl.float8e4nv)
        tl.store(X8 + rows[:, None] * K + cols[None, :], q, mask=mask)


def quant_rows(x: torch.Tensor):
    assert x.shape[-1] % 16 == 0
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x2.shape
    x8 = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 32),)
    _quant_rows_kernel[grid](x2, x8, s, M, K, BLOCK_M=32, BLOCK_K=512, num_warps=4)
    return x8, s, x.shape


@triton.jit
def _fp8_gemm_nk(A, B, C, SA, SB,
                 M, N, K,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    # B is (N,K) row-major; load (BK,BN) by indexing [k, n] -> strided but Triton vectorises on the fast axis
    b_ptrs = B + offs_n[None, :] * K + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        bt = tl.load(b_ptrs)
        acc = tl.dot(a, bt, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    sa = tl.load(SA + offs_m)
    sw = tl.load(SB)
    acc = acc * sa[:, None] * sw
    c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16))


def fp8_gemm(x8, w8, sx, sw):
    """x8 (M,K) e4m3, w8 (N,K) e4m3 row-major, sx (M,) fp32, sw scalar tensor."""
    M, K = x8.shape
    N = w8.shape[0]
    c = torch.empty((M, N), device=x8.device, dtype=torch.bfloat16)
    BM, BN, BK = 128, 128, 64
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _fp8_gemm_nk[grid](x8, w8, c, sx, sw.reshape(1),
                       M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8,
                       num_warps=8, num_stages=4)
    return c
