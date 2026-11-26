"""Validate + time fp16-accumulate GEMM against torch's fp32-accumulate path."""
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stpp.gemm16 import linear16

torch.manual_seed(0)
dev = "cuda"

shapes = [(2048, 8192, 2048), (2048, 2048, 8192), (2048, 32000, 2048)]
print(f"{'shape (M,N,K)':<22}{'torch ms':>10}{'gemm16 ms':>11}{'speedup':>9}{'err%rms':>9}")
for M, N, K in shapes:
    x = (torch.randn(M, K, device=dev) * 1.0).to(torch.float16)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.float16)

    ref = F.linear(x, w)                     # fp32 accumulate
    got = linear16(x, w)                     # fp16 accumulate
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
    print(f"({M},{N},{K})".ljust(22) + f"{t1:10.3f}{t2:11.3f}{t1/t2:9.3f}x{err:9.2f}")

# backward sanity: gradcheck-free numeric comparison
M, N, K = 256, 512, 256
x = torch.randn(M, K, device=dev, dtype=torch.float16, requires_grad=True)
w = torch.randn(N, K, device=dev, dtype=torch.float16, requires_grad=True)
y1 = F.linear(x, w); l1 = y1.float().pow(2).sum(); g1 = torch.autograd.grad(l1, [x, w])
y2 = linear16(x, w); l2 = y2.float().pow(2).sum(); g2 = torch.autograd.grad(l2, [x, w])
for name, a, b in [("dx", g1[0], g2[0]), ("dw", g1[1], g2[1])]:
    rel = (a.float() - b.float()).abs().max() / (a.float().abs().max() + 1e-6)
    print(f"backward {name}: max rel diff {rel.item():.2e}  ({'OK' if rel < 0.05 else 'BAD'})")
