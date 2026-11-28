"""Sanity test for stpp/optim.py: each optimizer must drive a small least-squares loss down."""
import torch

from stpp.optim import AdEMAMix, Muon, zeropower_via_newtonschulz5


def ns_check():
    torch.manual_seed(0)
    g = torch.randn(16, 64, dtype=torch.float32)
    u = zeropower_via_newtonschulz5(g).float()
    s = torch.linalg.svdvals(u)
    print(f"[ns] 16x64 singular values min={s.min():.4f} max={s.max():.4f} (want ~1)")


def lsq(opt_name, steps=200):
    torch.manual_seed(1)
    w_true = torch.randn(32, 8)
    x = torch.randn(256, 32)
    y = x @ w_true
    w = torch.nn.Parameter(torch.zeros(32, 8))
    if opt_name == "muon":
        opt = Muon([w], lr=0.05)
    elif opt_name == "ademamix":
        opt = AdEMAMix([w], lr=0.05, alpha=5.0, t_alpha=100)
    else:
        opt = torch.optim.AdamW([w], lr=0.05, betas=(0.9, 0.95))
    first = last = None
    for i in range(steps):
        opt.zero_grad()
        loss = ((x @ w - y) ** 2).mean()
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    print(f"[{opt_name}] loss {first:.4f} -> {last:.4f}  (ratio {last/first:.4f})")
    assert last < first * 0.2, f"{opt_name} did not converge"


if __name__ == "__main__":
    ns_check()
    for name in ("adamw", "muon", "ademamix"):
        lsq(name)
    print("ok")
