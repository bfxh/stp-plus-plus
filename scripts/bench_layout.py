import torch, time
torch.manual_seed(0)
dev='cuda'
def bench(fn, warm=10, it=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/it

M,K,N = 2044, 2048, 128256
x  = torch.randn(M,K,device=dev,dtype=torch.bfloat16)*0.05
W  = torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02   # nn.Linear weight layout (N,K) row-major
WT = W.t().contiguous()                                     # (K,N) row-major

ln = torch.nn.Linear(K,N,bias=False,device=dev,dtype=torch.bfloat16)
ln.weight.data.copy_(W)

cases = {
 'TN  x@W.t()      ': lambda: x @ W.t(),
 'NN  x@WT         ': lambda: x @ WT,
 'nn.Linear(TN)    ': lambda: ln(x),
 'F.linear(TN)     ': lambda: torch.nn.functional.linear(x, W),
 'fp8 TN (row/col) ': None,
}
for name, f in cases.items():
    if f is None: continue
    ms = bench(f)*1e3
    tf = 2*M*K*N/(ms*1e-3)/1e12
    print(f'{name} {ms:7.2f} ms  {tf:6.2f} TFLOPS  ({2*M*K*N/1e12:.3f} TFLOP)')

# fp8 with pre-quantized weight
def q_row(x):
    s=(x.abs().amax(-1,keepdim=True).clamp_min(1e-6).float()/448.0)
    return (x/s).to(torch.float8_e4m3fn), s
def q_col(w):
    s=(w.abs().amax(-1,keepdim=True).clamp_min(1e-6).float()/448.0)
    return (w/s).to(torch.float8_e4m3fn), s.t().contiguous()
W8, sw = q_col(W)          # once
def f8():
    x8,sx = q_row(x)
    return torch._scaled_mm(x8, W8.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
ms = bench(f8)*1e3
print(f'fp8 preQ+act    {ms:7.2f} ms  {2*M*K*N/(ms*1e-3)/1e12:6.2f} TFLOPS')
ref = x @ WT
out = f8()
print('fp8 rel', ((out.float()-ref.float()).norm()/ref.float().norm()).item())
