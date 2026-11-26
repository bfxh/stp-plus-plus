"""Sanity checks for the geometry module — run: python tests/test_geometry.py"""

import torch
import torch.nn.functional as F

from stpp.geometry import (
    curvature_angle, curvature_budget, repulsion, stp_random_span, stp_triplet,
)

torch.manual_seed(0)
B, T, D = 2, 12, 64
spans = [(1, T)] * B
g = torch.Generator().manual_seed(7)


def line(b=0, d=None, curve=0.0):
    """A straight line, optionally bent into a circular arc of `curve` radians per step."""
    d = d if d is not None else torch.randn(D)
    d = d / d.norm()
    perp = torch.randn(D); perp = perp - perp.dot(d) * d; perp = perp / perp.norm()
    h = torch.zeros(T, D)
    v = d.clone()
    for i in range(1, T):
        ang = curve * (i - 1)
        v_i = torch.cos(torch.tensor(ang)) * d + torch.sin(torch.tensor(ang)) * perp
        h[i] = h[i - 1] + v_i
    return h


print("== straight line ==")
h = torch.stack([line(0), line(1)])
print(" stp_random_span  ", float(stp_random_span(h, spans, generator=g)[0]))
print(" stp_triplet       ", float(stp_triplet(h, spans, generator=g)[0]))
print(" curvature_angle   ", float(curvature_angle(h, spans)[0]))
print(" curvature_budget  ", curvature_budget(h, spans, tau=0.05)[0])

print("== curved trajectory (0.12 rad/step) ==")
h = torch.stack([line(0, curve=0.12), line(1, curve=0.12)])
print(" stp_random_span  ", float(stp_random_span(h, spans, generator=g)[0]))
print(" curvature_angle   ", float(curvature_angle(h, spans)[0]))
print(" curvature_budget  ", curvature_budget(h, spans, tau=0.05)[0])

print("== zigzag ==")
h = torch.zeros(B, T, D)
for i in range(1, T):
    sign = -1.0 if i % 2 else 1.0
    h[:, i] = h[:, i - 1] + sign * F.normalize(torch.randn(B, D), dim=-1)
print(" stp_random_span  ", float(stp_random_span(h, spans, generator=g)[0]))
print(" curvature_angle   ", float(curvature_angle(h, spans)[0]))

print("== repulsion ==")
z = torch.eye(4, 16)
print(" orthogonal ->", float(repulsion(z)[0]))
print(" duplicated ->", float(repulsion(torch.stack([z[0]] * 4))[0]))

print("== whiten kills the outlier dimension ==")
h = torch.randn(2, T, D) * 0.02
h[:, :, 0] = 500.0 + 80.0 * torch.randn(2, T)     # dim 0: huge magnitude AND huge variance
h[:, :, 1] = -300.0 + 40.0 * torch.randn(2, T)    # dim 1: same trick
cos_raw = F.cosine_similarity(h[:, 3] - h[:, 2], h[:, 2] - h[:, 1], dim=-1)
print(" raw cos  ", [round(x, 4) for x in cos_raw.tolist()], "<- pinned by dims 0/1 (std ~1.0)")
from stpp.geometry import RunningWhitener
w = RunningWhitener(D)
for _ in range(3):
    w.update(h)
hw = w(h)
cos_white = F.cosine_similarity(hw[:, 3] - hw[:, 2], hw[:, 2] - hw[:, 1], dim=-1)
print(" whitened cos", [round(x, 4) for x in cos_white.tolist()], "<- reflects the real semantic angle (std ~0.06)")
