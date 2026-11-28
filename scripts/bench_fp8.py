import torch, time
torch.backends.cuda.matmul.allow_tf32 = True
dev = 'cuda'
def bench(fn, warm=8, it=25):
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
    'logits_c 512x2048x128256': (512, 2048, 128256),
}
print('--- bf16 cuBLAS ---')
for name, (M, K, N) in shapes.items():
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    ms = bench(lambda: a @ b) * 1e3
    tf = 2 * M * K * N / (ms * 1e-3) / 1e12
    print(f'{name:26s} {ms:8.2f} ms  {tf:7.2f} TFLOPS')

print('--- fp8 _scaled_mm (cuBLASLt) ---')
try:
    for name, (M, K, N) in shapes.items():
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = torch.randn(K, N, device=dev, dtype=torch.bfloat16).t().contiguous().t().to(torch.float8_e4m3fn)
        sa = torch.tensor(1.0, device=dev); sb = torch.tensor(1.0, device=dev)
        out_dtype = torch.bfloat16
        try:
            f = lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=out_dtype)
            r = f(); torch.cuda.synchronize()
            ms = bench(f) * 1e3
            tf = 2 * M * K * N / (ms * 1e-3) / 1e12
            print(f'{name:26s} {ms:8.2f} ms  {tf:7.2f} TFLOPS')
        except Exception as e:
            print(f'{name:26s} FAIL: {type(e).__name__}: {str(e)[:110]}')
except Exception as e:
    print('fp8 unavailable:', type(e).__name__, str(e)[:200])
