"""Optimizers that are supposed to beat AdamW at equal tokens.

Muon     - Keller Jordan et al.: momentum + Newton-Schulz orthogonalization of the
           update, then RMS-matched to the Adam update so `--lr` keeps its meaning.
           Moonshot reports ~52% of AdamW's FLOPs at compute-optimal (2502.16982).
AdEMAMix - two EMAs (beta1=0.9 and beta3=0.9999) mixed by alpha; 1.3B/101B tokens
           reported on par with AdamW at 197B (2409.03137).

Both leave 1-D params (norms/biases) on AdamW, as in the papers.
"""
from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration for the matrix sign / polar factor."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.to(torch.bfloat16)
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0, rms_match=True, betas=(0.9, 0.95), eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, rms_match=rms_match, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                st["step"] = st.get("step", 0) + 1
                if p.ndim == 2 and min(p.shape) > 1:
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(g)
                    buf = st["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(g)
                    d = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    u = zeropower_via_newtonschulz5(d, steps=group["ns_steps"])
                    u = u.to(d.dtype)
                    if group["rms_match"]:
                        u = u * (u.numel() ** 0.5) / (u.norm() + 1e-12)
                    else:
                        u = u * (max(1.0, p.shape[0] / p.shape[1]) ** 0.5)
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(u, alpha=-lr)
                else:
                    if "exp_avg" not in st:
                        st["exp_avg"] = torch.zeros_like(g)
                        st["exp_avg_sq"] = torch.zeros_like(g)
                    m, v = st["exp_avg"], st["exp_avg_sq"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bc1 = 1 - b1 ** st["step"]
                    bc2 = 1 - b2 ** st["step"]
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(m / bc1, (v / bc2).sqrt().add_(group["eps"]), value=-lr)
        return loss


class AdEMAMix(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.95), beta3=0.9999, alpha=5.0,
                 t_alpha=1000, weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, beta3=beta3, alpha=alpha, t_alpha=t_alpha,
                        weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            b1, b2 = group["betas"]
            b3 = group["beta3"]
            eps = group["eps"]
            t_alpha = max(1, int(group["t_alpha"]))
            alpha_max = group["alpha"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "m1" not in st:
                    st["m1"] = torch.zeros_like(g)
                    st["m2"] = torch.zeros_like(g)
                    st["v"] = torch.zeros_like(g)
                    st["step"] = 0
                st["step"] += 1
                t = st["step"]
                m1, m2, v = st["m1"], st["m2"], st["v"]
                m1.mul_(b1).add_(g, alpha=1 - b1)
                m2.mul_(b3).add_(g, alpha=1 - b3)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                alpha_t = alpha_max * min(1.0, t / t_alpha)
                denom = (v / (1 - b2 ** t)).sqrt().add_(eps)
                num = m1 / (1 - b1 ** t) + alpha_t * m2
                if wd:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(num, denom, value=-lr)
        return loss


def build_optimizer(params, name: str, lr: float, weight_decay: float,
                    betas=(0.9, 0.95), muon_lr_scale: float = 1.0,
                    ademamix_alpha: float = 5.0, ademamix_beta3: float = 0.9999,
                    t_alpha: int = 1000):
    name = (name or "adamw").lower()
    if name == "muon":
        return Muon(params, lr=lr * muon_lr_scale, momentum=0.95, nesterov=True,
                    ns_steps=5, weight_decay=weight_decay, betas=betas)
    if name == "ademamix":
        return AdEMAMix(params, lr=lr, betas=betas, beta3=ademamix_beta3,
                        alpha=ademamix_alpha, t_alpha=t_alpha, weight_decay=weight_decay)
    if name in ("adamw", "adam"):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    raise SystemExit(f"unknown --optim {name}")
