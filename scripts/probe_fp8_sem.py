import torch, time
torch.manual_seed(0)
dev = 'cuda'

def q_row(x):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6).float()
    s = (amax / 448.0)
    x8 = (x / s).to(torch.float8_e4m3fn)
    return x8, s.contiguous()

def q_col(w):  # w: (N,K) -> w8 (N,K), scale (1,N)
    amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6).float()   # (N,1)
    s = amax / 448.0
    w8 = (w / s).to(torch.float8_e4m3fn)
    return w8, s.t().contiguous()   # (1,N)

def bench(fn, warm=10, it=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it

cases = {'gate/up': (2048, 2048, 8192), 'down': (2048, 8192, 2048), 'lm_head': (2048, 2048, 128256)}
for name, (M, K, N) in cases.items():
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.05
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    ref = x @ w.t()
    x8, sx = q_row(x)
    w8, sw = q_col(w)
    try:
        out = torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
    except Exception as e:
        print(name, 'FAIL', str(e)[:120]); continue
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    # speed: fwd only, quantization included vs excluded
    bf_ms = bench(lambda: x @ w.t()) * 1e3
    q_ms = bench(lambda: (q_row(x), q_col(w))) * 1e3
    f8_ms = bench(lambda: torch._scaled_mm(*_q(x, w8, sw), out_dtype=torch.bfloat16)) * 1e3 if False else None
    tf_bf = 2*M*K*N/(bf_ms*1e-3)/1e12
    def full():
        x8_, sx_ = q_row(x)
        return torch._scaled_mm(x8_, w8.t(), scale_a=sx_, scale_b=sw, out_dtype=torch.bfloat16)
    fu_ms = bench(full) * 1e3
    tf_fu = 2*M*K*N/(fu_ms*1e-3)/1e12
    print(f'{name:8s} rel={rel:.4f} cos={cos:.4f} | bf16 {bf_ms:6.2f}ms({tf_bf:5.1f}TF) | fp8+q {fu_ms:6.2f}ms({tf_fu:5.1f}TF) | quant overhead {q_ms:5.2f}ms')

print('--- STE autograd (fwd fp8, bwd bf16) ---')
class FP8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        x8, sx = q_row(x); w8, sw = q_col(w)
        return torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        return g @ w, g.t() @ x

M, K, N = 512, 2048, 4096
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
y = FP8Matmul.apply(x, w)
print('fwd', y.shape, 'sum', y.float().abs().mean().item())
gx, gw = torch.autograd.grad(y.float().pow(2).sum(), (x, w))
print('grad ok', gx.shape, gw.shape, 'finite', torch.isfinite(gx).all().item(), torch.isfinite(gw).all().item())
