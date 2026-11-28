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
x = torch.randn(M,K,device=dev,dtype=torch.bfloat16)*0.05
W = torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02     # (N,K) row-major = nn.Linear
ref = x @ W.t()
tf = 2*M*K*N/1e12

def q_row(t):
    s=(t.abs().amax(-1,keepdim=True).clamp_min(1e-6).float()/448.0)
    return (t/s).to(torch.float8_e4m3fn), s
def q_col(w):
    s=(w.abs().amax(-1,keepdim=True).clamp_min(1e-6).float()/448.0)
    return (w/s).to(torch.float8_e4m3fn), s

W8, sw_col = q_col(W)      # W8 (N,K)
x8, sx = q_row(x)          # cached activation quant (aim: measure GEMM only)
bf = bench(lambda: x @ W.t())*1e3
print(f'bf16                      {bf:7.2f} ms  {tf/(bf*1e-3):6.2f} TFLOPS')

variants = {
 'A tensorwise scalar': (torch.tensor(1.0,device=dev), torch.tensor(1.0,device=dev)),
 'B row-a + scalar-b ': (sx, torch.tensor(1.0,device=dev)),
 'C row-a + row-b    ': (sx, sw_col.t().contiguous()),
}
for name,(sa,sb) in variants.items():
    try:
        f = lambda sa=sa, sb=sb: torch._scaled_mm(x8, W8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        f(); torch.cuda.synchronize()
        ms = bench(f)*1e3
        print(f'{name}      {ms:7.2f} ms  {tf/(ms*1e-3):6.2f} TFLOPS  ({bf/ms:.2f}x)')
    except Exception as e:
        print(f'{name}      FAIL {type(e).__name__}: {str(e)[:90]}')

# dequant accuracy of variant C
out = torch._scaled_mm(x8, W8.t(), scale_a=sx, scale_b=sw_col.t().contiguous(), out_dtype=torch.bfloat16)
print('rel', ((out.float()-ref.float()).norm()/ref.float().norm()).item())

# cost of quantizing activations per call (what training would pay)
def quant_only(): q_row(x)
print(f'act-quant only            {bench(quant_only)*1e3:7.2f} ms')
