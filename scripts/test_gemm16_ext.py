"""Validate + time the zero-copy fp16-accumulate extension."""
import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stpp.gemm16_ext import linear16, _load

torch.manual_seed(0)
dev = "cuda"
_mod = _load()
print("extension built:", _mod)

print(f"{'shape (M,N,K)':<22}{'torch ms':>10}{'ext ms':>9}{'speedup':>9}{'err%rms':>9}")
for M, N, K in [(2048, 8192, 2048), (2048, 2048, 8192), (2048, 32000, 2048)]:
    x = (torch.randn(M, K, device=dev) * 1.0).to(torch.float16)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.float16)
    ref = F.linear(x, w)
    got = linear16(x, w)
    rms = ref.float().pow(2).mean().sqrt().item()
    err = (got.float() - ref.float()).abs().max().item() / rms * 100

    def bench(fn, n=30):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(n):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
        return best

    t1 = bench(lambda: F.linear(x, w))
    t2 = bench(lambda: linear16(x, w))
    print(f"({M},{N},{K})".ljust(22) + f"{t1:10.3f}{t2:9.3f}{t1/t2:8.2f}x{err:9.2f}")

# full forward+backward timing (this is what the trainer pays)
M, N, K = 2048, 8192, 2048
x = (torch.randn(M, K, device=dev) * 1.0).to(torch.float16).requires_grad_(True)
w = (torch.randn(N, K, device=dev) * 0.02).to(torch.float16).requires_grad_(True)

def step(ref):
    if ref:
        y = F.linear(x, w)
    else:
        y = linear16(x, w)
    y.float().pow(2).mean().backward()
    x.grad = None; w.grad = None

def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(n):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best

t1 = bench(lambda: step(True))
t2 = bench(lambda: step(False))
print(f"\nfwd+bwd (2048,8192,2048): torch {t1:.3f} ms  ext {t2:.3f} ms  -> {t1/t2:.2f}x")

# gradient correctness
x = torch.randn(256, 256, device=dev, dtype=torch.float16, requires_grad=True)
w = torch.randn(512, 256, device=dev, dtype=torch.float16, requires_grad=True)
y1 = F.linear(x, w)
g1x, g1w = torch.autograd.grad(y1.float().pow(2).sum(), [x, w])
x.grad = None; w.grad = None
y2 = linear16(x, w)
g2x, g2w = torch.autograd.grad(y2.float().pow(2).sum(), [x, w])
for name, a, b in [("dx", g1x, g2x), ("dw", g1w, g2w)]:
    rel = (a.float() - b.float()).abs().max() / (a.float().abs().max() + 1e-6)
    print(f"backward {name}: rel {rel.item():.2e} ({'OK' if rel < 0.05 else 'BAD'})")
