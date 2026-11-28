import sys, time, torch
sys.path.insert(0, '/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus')
from kernels.fp8_gemm_triton import quant_rows, fp8_gemm, FP8_MAX

torch.manual_seed(0)
dev = 'cuda'
def bench(fn, warm=10, it=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it

shapes = {
    'gate/up 2048x2048x8192': (2048, 2048, 8192),
    'down    2048x8192x2048': (2048, 8192, 2048),
    'lm_head 2048x2048x128256': (2048, 2048, 128256),
}
for name, (M, K, N) in shapes.items():
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.05
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    sw = (w.abs().amax() / FP8_MAX).float().reshape(())
    w8 = (w / sw).to(torch.float8_e4m3fn)
    tf = 2 * M * K * N / 1e12

    # reference accuracy
    x8, sx, shape = quant_rows(x)
    ours = fp8_gemm(x8, w8, sx, sw).float()
    ref = x @ w.t()
    rel = ((ours - ref.float()).norm() / ref.float().norm()).item()

    bf_ms = bench(lambda: x @ w.t()) * 1e3
    q_ms = bench(lambda: quant_rows(x)) * 1e3
    gemm_ms = bench(lambda: fp8_gemm(x8, w8, sx, sw)) * 1e3
    tot_ms = q_ms + gemm_ms
    print(f'{name:26s} rel={rel:.4f} | bf16 {bf_ms:7.2f}ms({tf/(bf_ms*1e-3):5.1f}TF) | '
          f'quant {q_ms:6.2f} + gemm {gemm_ms:7.2f} = {tot_ms:7.2f}ms({tf/(tot_ms*1e-3):5.1f}TF) | {bf_ms/tot_ms:.2f}x')
