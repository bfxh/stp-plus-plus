import torch, time
torch.manual_seed(0)
dev='cuda'
def bench(fn, warm=8, it=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/it

B,T,K,N = 4,511,2048,128256
x = torch.randn(B,T,K,device=dev,dtype=torch.bfloat16)*0.05
W = torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
tgt = torch.randint(0,N,(B,T),device=dev)
print(f'vocab tensor per step: {B*T*N*2/1e9:.3f} GB logits')

# 1) fwd only
ms = bench(lambda: torch.nn.functional.linear(x, W))*1e3
print(f'fwd linear only            {ms:7.2f} ms')

# 2) fwd + CE + backward (no weight grad, like frozen head)
def full():
    xg = x.detach().requires_grad_(True)
    lg = torch.nn.functional.linear(xg, W)
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1,N), tgt.reshape(-1), reduction="sum")
    loss.backward()
full(); torch.cuda.synchronize()
ms = bench(full)*1e3
print(f'fwd+CE.sum+bwd (1 chunk)   {ms:7.2f} ms')

# 3) same but chunked 64 (what the trainer does when STPP_LOGIT_CHUNK is small)
def chunked(chunk=64):
    xg = x.detach().requires_grad_(True)
    tot = 0.0
    for s in range(0,T-1,chunk):
        e = min(s+chunk, T-1)
        lg = torch.nn.functional.linear(xg[:, s:e], W)
        tot = tot + torch.nn.functional.cross_entropy(lg.reshape(-1,N), tgt[:, s:e].reshape(-1), reduction="sum")
    tot.backward()
chunked(); torch.cuda.synchronize()
ms = bench(chunked)*1e3
print(f'fwd+CE.sum+bwd chunk=64    {ms:7.2f} ms')

# 4) fp8 with CORRECT layout: W8 stored (K,N) row-major
def q_row(t):
    s=(t.abs().amax(-1,keepdim=True).clamp_min(1e-6).float()/448.0)
    return (t/s).to(torch.float8_e4m3fn), s
def q_col(w):
    s=(w.abs().amax(-1,keepdim=True).clamp_min(1e-6).float()/448.0)
    return (w/s).to(torch.float8_e4m3fn), s.t().contiguous()
W8, sw = q_col(W)
W8t = W8.t().contiguous()          # (K,N) row-major
x2 = x.reshape(B*T, K)
def f8():
    x8,sx = q_row(x2)
    return torch._scaled_mm(x8, W8t, scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
f8(); torch.cuda.synchronize()
ms = bench(f8)*1e3
print(f'fp8 preQ (K,N row-major)   {ms:7.2f} ms  {2*B*T*K*N/(ms*1e-3)/1e12:6.2f} TFLOPS')
ms = bench(lambda: torch.nn.functional.linear(x2, W))*1e3
print(f'bf16 same flat             {ms:7.2f} ms  {2*B*T*K*N/(ms*1e-3)/1e12:6.2f} TFLOPS')
