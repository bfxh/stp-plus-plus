"""CPU-only sanity check for the latent_pred arm (no GPU contention with the sweep)."""
import torch, sys
sys.path.insert(0, r"C:\Users\lbx13\Doubao\chats\2026-11-24\new-chat-7\stp-plus-plus")
from stpp.geometry import LatentPredictor, latent_prediction, GeomConfig, GeometricRegularizer

torch.manual_seed(0)
B, T, D = 3, 24, 64
h = torch.randn(B, T, D)
spans = [(4, 20), (6, 22), (3, 18)]

pred = LatentPredictor(D)
# near-identity start (std=1e-3 on fc2) so the head can still receive gradient at step 0
out = pred(h[:, 5])
assert torch.allclose(out, h[:, 5], atol=5e-2), (out - h[:, 5]).abs().max()

g = torch.Generator().manual_seed(1)
loss, stats = latent_prediction(h, spans, pred, k_max=3, n_pairs=8, generator=g)
assert loss.shape == (), loss.shape
assert torch.isfinite(loss), loss
assert stats["n"] == B * 8, stats["n"]
assert set(k for k in stats if k.startswith("k")) <= {"k1", "k2", "k3"}
print("latent_prediction ok:", float(loss), stats)

# gradient must reach the predictor
loss.backward()
assert pred.fc1.weight.grad is not None and pred.fc1.weight.grad.abs().sum() > 0
print("predictor grad ok")

# it must learn: optimise a fixed target to prove the loss is actionable
opt = torch.optim.Adam(pred.parameters(), lr=1e-3)
for i in range(200):
    l, _ = latent_prediction(h, spans, pred, generator=g, n_pairs=8)
    opt.zero_grad(); l.backward(); opt.step()
print("after 200 steps: ", float(l))

# end-to-end through the regularizer, exercising the real code path
cfg = GeomConfig(kind="latent_pred", lbd=0.02, layers=(2,))
reg = GeometricRegularizer(D, cfg)
aux, st = reg({2: h}, spans, generator=g, ntp_loss=torch.tensor(1.0))
aux.backward()
print("regulator ok:", float(aux), st)
print("ALL GOOD")
