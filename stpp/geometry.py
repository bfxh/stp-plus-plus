"""Geometric regularizers for LLM hidden-state trajectories.

Everything here operates on a *packed batch* of hidden states from one layer:

    h      : (B, T, D) float32   hidden states
    spans  : list[(start, end)]  assistant span per sample, start inclusive / end exclusive

The conventions follow Huang, LeCun & Balestriero 2026 (arXiv:2602.22617) so that
`stp_random_span` reproduces the official `linear="random_span"` objective in
galilai-group/llm-jepa *exactly*:

    v_patch = h[end_of_patch] - h[start_of_patch]
    v_out   = (h[patch_start] - h[seq_start]) + (h[seq_end] - h[patch_end])
    L       = 1 - cos(v_out, v_patch)

Everything else (`curvature_budget`, `whiten`, `repulsion`) is what we add on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1, eps=eps)


def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return a / b.clamp_min(eps)


# --------------------------------------------------------------------------------------
# 1. official STP: random span, "rest displacement vs patch displacement"
# --------------------------------------------------------------------------------------
def stp_random_span(
    h: torch.Tensor,
    spans: Sequence[tuple[int, int]],
    *,
    generator: torch.Generator | None = None,
    n_spans: int = 1,
    length_adjustment: str | None = None,
    seq_start: int = 1,
) -> tuple[torch.Tensor, dict]:
    """Original Semantic Tube term (repo: ``linear="random_span"``).

    A random contiguous patch is carved out of the conversation. The model is pushed
    to make the displacement *across the patch* collinear with the displacement
    accumulated *outside* the patch. Straight trajectory -> loss 0.
    """
    B, T, D = h.shape
    losses: list[torch.Tensor] = []
    weights: list[float] = []

    for b in range(B):
        a0, a1 = spans[b]
        lo, hi = max(seq_start, 1), min(a1, T)
        length = hi - lo
        if length < 3:
            continue
        for _ in range(n_spans):
            p0 = int(torch.randint(lo, hi, (1,), generator=generator).item())
            p1 = int(torch.randint(p0 + 1, hi + 1, (1,), generator=generator).item())
            if p1 <= p0:
                continue
            v_patch = h[b, p1 - 1] - h[b, p0 - 1]
            v_out = (h[b, p0 - 1] - h[b, lo - 1]) + (h[b, hi - 1] - h[b, p1 - 1])
            if v_out.norm() < 1e-6 or v_patch.norm() < 1e-6:
                continue
            losses.append(1.0 - _cos(v_out, v_patch))
            weights.append(_length_weight(p1 - p0, length - (p1 - p0), length_adjustment))

    if not losses:
        return h.new_zeros(()), {"stp": 0.0, "n": 0}

    loss_vec = torch.stack(losses)
    w = torch.tensor(weights, device=h.device, dtype=loss_vec.dtype)
    loss = (loss_vec * w).sum() / w.sum().clamp_min(1e-8)
    return loss, {"stp": float(loss.detach()), "n": len(losses)}


def _length_weight(patch_len: int, rest_len: int, mode: str | None) -> float:
    """Same re-weighting as the reference implementation."""
    if mode is None:
        return 1.0
    if patch_len <= 0 or rest_len <= 0:
        return 0.0
    if mode == "cosine_like":
        return 2.0 * patch_len * rest_len / (patch_len**2 + rest_len**2)
    if mode == "jaccard_like":
        return 1.0 - abs(patch_len - rest_len) / (patch_len + rest_len)
    raise ValueError(f"unknown length_adjustment: {mode}")


# --------------------------------------------------------------------------------------
# 2. literal three-point triplet (the formula as written in the paper)
# --------------------------------------------------------------------------------------
def stp_triplet(
    h: torch.Tensor,
    spans: Sequence[tuple[int, int]],
    *,
    generator: torch.Generator | None = None,
    n_triplets: int = 4,
) -> tuple[torch.Tensor, dict]:
    """L = 1 - cos(h_t - h_r, h_r - h_s), sampled s < r < t inside the assistant span."""
    B, T, D = h.shape
    losses: list[torch.Tensor] = []
    for b in range(B):
        a0, a1 = spans[b]
        length = min(a1, T) - a0
        if length < 4:
            continue
        for _ in range(n_triplets):
            idx = torch.randint(a0, a0 + length, (3,), generator=generator).to(h.device)
            idx, _ = torch.sort(idx)
            if idx[0] == idx[1] or idx[1] == idx[2]:
                continue
            v1 = h[b, idx[1]] - h[b, idx[0]]
            v2 = h[b, idx[2]] - h[b, idx[1]]
            losses.append(1.0 - _cos(v1, v2))
    if not losses:
        return h.new_zeros(()), {"triplet": 0.0, "n": 0}
    loss = torch.stack(losses).mean()
    return loss, {"triplet": float(loss.detach()), "n": len(losses)}


# --------------------------------------------------------------------------------------
# 3. curvature baselines
# --------------------------------------------------------------------------------------
def curvature_angle(
    h: torch.Tensor,
    spans: Sequence[tuple[int, int]],
) -> tuple[torch.Tensor, dict]:
    """Reference repo's `linear="curvature"`: mean |turning angle| between consecutive steps.

    This *forces* the trajectory to be straight and is known to over-constrain legitimate
    curvature, which is exactly the failure mode we exploit.
    """
    per_sample: list[torch.Tensor] = []
    total = 0
    for b in range(h.shape[0]):
        a0, a1 = spans[b]
        a1 = min(a1, h.shape[1])
        if a1 - a0 < 3:
            continue
        seg = h[b, a0:a1]
        v = seg[1:] - seg[:-1]
        cosang = F.cosine_similarity(v[:-1], v[1:], dim=-1, eps=1e-6).clamp(-1.0, 1.0)
        ang = torch.acos(cosang)
        per_sample.append(ang.mean())
        total += 1
    if not per_sample:
        return h.new_zeros(()), {"curv": 0.0, "n": 0}
    loss = torch.stack(per_sample).mean()
    return loss, {"curv": float(loss.detach()), "n": total}


def curvature_budget(
    h: torch.Tensor,
    spans: Sequence[tuple[int, int]],
    *,
    tau: torch.Tensor | float = 0.15,
    hard: bool = True,
) -> tuple[torch.Tensor, dict]:
    """**Our curvature-budgeted tube.**

    Why: a real trajectory is a *smooth curve*, not a straight line — the follow-up work
    (arXiv:2604.18464) shows a 3-layer MLP predicts the next states 3-12x better than
    linear extrapolation, i.e. the geodesic has genuine curvature. Penalising all
    curvature (repo baseline) fights the data; penalising none (STP) lets noise in.

    We instead measure the *scale-invariant* discrete curvature of consecutive triples

        kappa_i = || (h_i - h_{i-1})_perp(chord) || / || chord ||,
        chord   = h_{i+1} - h_{i-1}

    (dimensionless, i.e. curvature x arc-length) and only penalise the part that
    exceeds a budget ``tau``:

        L = mean_i  relu(kappa_i - tau)^2

    With ``tau`` set per-layer to the observed median curvature, the model is free to
    keep its natural, *predictable* bending and is only pushed to remove the noise that
    blows the tube up.
    """
    if isinstance(tau, torch.Tensor):
        tau_val = float(tau.detach().mean())
    else:
        tau_val = float(tau)

    kappas: list[torch.Tensor] = []
    for b in range(h.shape[0]):
        a0, a1 = spans[b]
        a1 = min(a1, h.shape[1])
        if a1 - a0 < 3:
            continue
        seg = h[b, a0:a1]
        # three-point window: p_{i-1}, p_i, p_{i+1}
        chord = seg[2:] - seg[:-2]
        mid_disp = seg[1:-1] - seg[:-2]
        chord_dir = F.normalize(chord, dim=-1, eps=1e-6)
        proj = (mid_disp * chord_dir).sum(-1, keepdim=True) * chord_dir
        perp = mid_disp - proj
        kappa = perp.norm(dim=-1) / chord.norm(dim=-1).clamp_min(1e-6)  # dimensionless
        kappas.append(kappa)

    if not kappas:
        return h.new_zeros(()), {"kmean": 0.0, "n": 0}

    k = torch.cat(kappas)
    excess = (k - tau_val)
    if hard:
        excess = excess.clamp_min(0.0)
    loss = (excess**2).mean()
    q90 = float(k.quantile(0.90).detach()) if k.numel() > 2 else float(k.mean())
    return loss, {
        "kmean": float(k.mean().detach()),
        "k90": q90,
        "k95": float(k.quantile(0.95).detach()) if k.numel() > 2 else float(k.mean()),
        "n": int(k.numel()),
        "tau": tau_val,
    }


# --------------------------------------------------------------------------------------
# 3b. our main alternative: a *learned nonlinear* latent dynamics model (JEPA-style)
# --------------------------------------------------------------------------------------
class LatentPredictor(torch.nn.Module):
    """Residual 2-layer MLP mapping h_t -> h_{t+k} in latent space.

    Why this and not a cosine? Our probe measures how much better a 3-layer MLP
    forecasts the next hidden state than plain linear extrapolation, at horizons of
    1/2/3 reasoning steps. The ratio is 7.6x / 15.0x / 19.5x for the plain baseline —
    i.e. the trajectory is nowhere near straight. STP's loss still *asks* for straight.
    Instead of hard-coding the (wrong) shape of the path, let a small head learn it:

        L = 1 - cos( g(h_t), sg(h_{t+k}) ),   g = identity + MLP   (zero-init last layer)

    ``sg`` = stop-gradient, so the target is not dragged along — this is the
    BYOL/JEPA asymmetry that prevents representational collapse.
    """

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, dim)
        # near-identity start, but NOT exactly zero: a hard zero on fc2 kills the
        # gradient into fc1 on step 0 and the head never leaves its initialisation.
        torch.nn.init.normal_(self.fc2.weight, std=1e-3)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(x)))


def latent_prediction(
    h: torch.Tensor,
    spans: Sequence[tuple[int, int]],
    predictor: torch.nn.Module,
    *,
    k_max: int = 3,
    n_pairs: int = 8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """Sample (t, t+k) pairs inside the assistant span and score the forecast."""
    B, T, D = h.shape
    losses: list[torch.Tensor] = []
    by_k: dict[int, list[float]] = {}

    for b in range(B):
        a0, a1 = spans[b]
        a1 = min(a1, T)
        if a1 - a0 < 3:
            continue
        for _ in range(n_pairs):
            k = int(torch.randint(1, k_max + 1, (1,), generator=generator).item())
            hi = a1 - k
            if hi <= a0:
                continue
            t0 = int(torch.randint(a0, hi, (1,), generator=generator).item())
            pred = predictor(h[b, t0])
            tgt = h[b, t0 + k].detach()
            if tgt.norm() < 1e-6:
                continue
            l = 1.0 - _cos(pred, tgt)
            losses.append(l)
            by_k.setdefault(k, []).append(float(l.detach()))

    if not losses:
        return h.new_zeros(()), {"pred": 0.0, "n": 0}
    loss = torch.stack(losses).mean()
    stats = {"pred": float(loss.detach()), "n": len(losses)}
    for k, v in by_k.items():
        stats[f"k{k}"] = sum(v) / len(v)
    return loss, stats


def goal_prediction(
    h: torch.Tensor,
    spans: Sequence[tuple[int, int]],
    predictor: torch.nn.Module,
    *,
    n_pairs: int = 8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """Predict the *destination* of the answer from any interior position.

    STP asks a one-step question: "is the next displacement collinear with the last
    one?"  For a language model that is nearly free — adjacent hidden states already
    have high cosine, so the answer carries almost no information, and the loss
    saturates at 1.65 (cos = -0.65) because the model has no reason to reshape its
    whole trajectory to satisfy a local constraint.

    This asks the *planning* question instead: from position t, predict the
    representation at the end of the answer (stop-gradient).  To minimise it the
    model has to encode "where this answer is going" at every token — which is
    exactly the information CE cannot supply, because CE only ever asks for the
    next token.

    This is the "replacement" form: the latent objective is no longer a geometric
    regulariser bolted onto CE, it is a second, information-bearing target.
    """

    B, T, D = h.shape
    losses: list[torch.Tensor] = []
    pos: list[float] = []

    for b in range(B):
        a0, a1 = spans[b]
        a1 = min(a1, T)
        if a1 - a0 < 3:
            continue
        goal = h[b, a1 - 1].detach()
        if goal.norm() < 1e-6:
            continue
        L = a1 - 1 - a0
        for _ in range(n_pairs):
            off = int(torch.randint(0, L, (1,), generator=generator).item())
            t0 = a0 + off
            pred = predictor(h[b, t0])
            losses.append(1.0 - _cos(pred, goal))
            pos.append(off / max(1, L))

    if not losses:
        return h.new_zeros(()), {"goal": 0.0, "n": 0}

    loss = torch.stack(losses).mean()
    # report the easy/hard split so we can see whether the head is actually learning
    early = [float(l.detach()) for l, p in zip(losses, pos) if p < 0.34]
    mid = [float(l.detach()) for l, p in zip(losses, pos) if 0.34 <= p < 0.67]
    late = [float(l.detach()) for l, p in zip(losses, pos) if p >= 0.67]
    stats = {"goal": float(loss.detach()), "n": len(losses)}
    if early:
        stats["p_early"] = sum(early) / len(early)
    if mid:
        stats["p_mid"] = sum(mid) / len(mid)
    if late:
        stats["p_late"] = sum(late) / len(late)
    return loss, stats


# --------------------------------------------------------------------------------------
# 4. geometry hygiene: whitening
# --------------------------------------------------------------------------------------
class RunningWhitener(torch.nn.Module):
    """Per-dimension z-scoring of hidden states with running statistics.

    Residual streams carry a handful of giant outlier dimensions (attention sinks /
    massive activations). A raw cosine is then dominated by ~10 coordinates and the
    "angle" we regularise is not the semantic angle at all. Whitening restores it.
    """

    def __init__(self, dim: int, momentum: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("n", torch.zeros(()))
        self.momentum = momentum
        self.eps = eps

    @torch.no_grad()
    def update(self, h: torch.Tensor) -> None:
        x = h.detach().reshape(-1, h.shape[-1]).float()
        if x.shape[0] == 0:
            return
        bmean = x.mean(0)
        bvar = x.var(0, unbiased=False)
        if float(self.n) == 0.0:
            self.mean.copy_(bmean)
            self.var.copy_(bvar)
        else:
            m = self.momentum
            self.mean.mul_(m).add_(bmean, alpha=1 - m)
            self.var.mul_(m).add_(bvar, alpha=1 - m)
        self.n += 1

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (h.float() - self.mean) / (self.var.clamp_min(1e-8).sqrt() + self.eps)


# --------------------------------------------------------------------------------------
# 5. cross-sample separation (the thing STP only argues about theoretically)
# --------------------------------------------------------------------------------------
def repulsion(z: torch.Tensor, margin: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Hinge repulsion on pooled trajectory endpoints.

    STP claims ODE uniqueness rules out trajectory collisions, but never enforces
    separation *between different prompts*. This is the explicit counter-force:
    distinct inputs must not end up on top of each other.
    """
    if z.shape[0] < 2:
        return z.new_zeros(()), {"rep": 0.0}
    zn = F.normalize(z.float(), dim=-1, eps=1e-6)
    sim = zn @ zn.t()
    off = sim - torch.eye(z.shape[0], device=z.device, dtype=sim.dtype) * 2.0
    loss = F.relu(off - margin).mean()
    return loss, {"rep": float(loss.detach())}


# --------------------------------------------------------------------------------------
# 6. composition + adaptive lambda
# --------------------------------------------------------------------------------------
@dataclass
class GeomConfig:
    kind: str = "stp_random_span"          # none | stp_random_span | stp_triplet | curvature_angle | curvature_budget
    lbd: float = 0.02                       # lambda
    n_spans: int = 1
    n_triplets: int = 4
    length_adjustment: str | None = None   # None | cosine_like | jaccard_like
    whiten: bool = False
    layers: tuple[int, ...] = (-1,)         # which decoder layers to regularise
    anchors: str = "random"                 # random | stride | step
    tau: float = 0.15                       # curvature budget (only for curvature_budget)
    tau_adaptive: bool = True               # set tau to running median curvature
    repulsion: float = 0.0                  # weight of the cross-sample term (0 = off)
    adaptive_lambda: str | None = None      # None | loss_ratio
    target_ratio: float = 0.15              # STP share of total loss when adaptive


class GeometricRegularizer(torch.nn.Module):
    """Computes the auxiliary geometric loss from tapped hidden states."""

    def __init__(self, dim: int, cfg: GeomConfig):
        super().__init__()
        self.cfg = cfg
        self.whiteners = torch.nn.ModuleDict(
            {str(li): RunningWhitener(dim) for li in cfg.layers} if cfg.whiten else {}
        )
        self.predictor = LatentPredictor(dim) if cfg.kind in ("latent_pred", "goal_pred") else None
        self.register_buffer("_tau_ema", torch.tensor(float(cfg.tau)))
        self.register_buffer("_ntp_ema", torch.tensor(1.0))
        self.register_buffer("_aux_ema", torch.tensor(1.0))

    # ---- tau -------------------------------------------------------------------
    @torch.no_grad()
    def _refresh_tau(self, kmedian: float) -> None:
        if not self.cfg.tau_adaptive:
            return
        m = 0.9
        self._tau_ema.mul_(m).add_(float(kmedian), alpha=1 - m)

    # ---- lambda ----------------------------------------------------------------
    def lam(self, ntp_loss: torch.Tensor | None = None, aux: torch.Tensor | None = None) -> torch.Tensor:
        """Return the effective lambda.

        ``adaptive_lambda="loss_ratio"`` replaces the brittle fixed lambda (the paper
        needs 0.0025-0.16 depending on model/dataset, and overshooting it collapses
        accuracy) with a controller that keeps the geometric term at a fixed *share*
        of the total loss. Zero hyper-parameter search across model sizes.
        """
        if self.cfg.adaptive_lambda is None or ntp_loss is None or aux is None:
            return torch.tensor(self.cfg.lbd)
        with torch.no_grad():
            m = 0.95
            self._ntp_ema.mul_(m).add_(float(ntp_loss.detach()), alpha=1 - m)
            self._aux_ema.mul_(m).add_(float(aux.detach()), alpha=1 - m)
            rho = self.cfg.target_ratio
            target = (rho / (1.0 - rho)) * (self._ntp_ema / self._aux_ema.clamp_min(1e-6))
            # never let the controller wander more than 4x away from lbd
            lo, hi = self.cfg.lbd / 4.0, self.cfg.lbd * 4.0
            return torch.clamp(target, lo, hi)

    # ---- forward ---------------------------------------------------------------
    def forward(
        self,
        hiddens: dict[int, torch.Tensor],
        spans: Sequence[tuple[int, int]],
        *,
        generator: torch.Generator | None = None,
        ntp_loss: torch.Tensor | None = None,
        pooled_end: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        if cfg.kind == "none":
            return hiddens[next(iter(hiddens))].new_zeros(()), {"aux": 0.0}

        device = next(iter(hiddens.values())).device
        total = torch.zeros((), device=device, dtype=torch.float32)
        stats: dict = {}

        for li, h in hiddens.items():
            h32 = h.float()
            if cfg.whiten and str(li) in self.whiteners:
                w = self.whiteners[str(li)]
                if self.training:
                    w.update(h32)
                h32 = w(h32)

            if cfg.kind == "stp_random_span":
                loss, s = stp_random_span(
                    h32, spans, generator=generator, n_spans=cfg.n_spans,
                    length_adjustment=cfg.length_adjustment,
                )
            elif cfg.kind == "stp_triplet":
                loss, s = stp_triplet(h32, spans, generator=generator, n_triplets=cfg.n_triplets)
            elif cfg.kind == "curvature_angle":
                loss, s = curvature_angle(h32, spans)
            elif cfg.kind == "curvature_budget":
                loss, s = curvature_budget(h32, spans, tau=float(self._tau_ema))
                if "k90" in s:
                    self._refresh_tau(s["k90"])
            elif cfg.kind == "latent_pred":
                loss, s = latent_prediction(
                    h32, spans, self.predictor, generator=generator
                )
            elif cfg.kind == "goal_pred":
                loss, s = goal_prediction(
                    h32, spans, self.predictor, generator=generator
                )
            else:
                raise ValueError(f"unknown geometric kind: {cfg.kind}")

            total = total + loss.float()
            for k, v in s.items():
                stats[f"L{li}/{k}"] = v

        aux = total / max(1, len(hiddens))
        stats["aux"] = float(aux.detach())

        if cfg.repulsion > 0 and pooled_end is not None:
            rep, rs = repulsion(pooled_end)
            aux = aux + cfg.repulsion * rep
            stats.update({f"rep/{k}": v for k, v in rs.items()})
            stats["aux"] = float(aux.detach())

        return aux, stats
