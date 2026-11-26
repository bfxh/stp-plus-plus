"""CPU-only sanity check for the goal_pred arm."""
import sys
import torch

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.parent.as_posix())
from stpp.geometry import LatentPredictor, goal_prediction, GeomConfig, GeometricRegularizer

torch.manual_seed(0)
B, T, D = 4, 30, 64
h = torch.randn(B, T, D)
spans = [(3, 20), (5, 26), (2, 14), (8, 29)]

pred = LatentPredictor(D)
g = torch.Generator().manual_seed(2)
loss, st = goal_prediction(h, spans, pred, n_pairs=8, generator=g)
assert loss.shape == (), loss.shape
assert torch.isfinite(loss)
assert st["n"] == B * 8, st["n"]
assert "p_early" in st and "p_late" in st
print("goal_prediction ok:", round(float(loss), 4), st)

loss.backward()
assert pred.fc1.weight.grad is not None and pred.fc1.weight.grad.abs().sum() > 0
print("predictor grad ok")

# the head must be able to learn the target: fixed tensor, 300 Adam steps
opt = torch.optim.Adam(pred.parameters(), lr=1e-3)
first = None
for i in range(300):
    l, _ = goal_prediction(h, spans, pred, generator=g)
    if i == 0:
        first = float(l)
    opt.zero_grad(); l.backward(); opt.step()
print("loss %.3f -> %.3f after 300 steps" % (first, float(l)))
assert float(l) < first, "goal predictor is not learning"

# end-to-end through the regularizer
reg = GeometricRegularizer(D, GeomConfig(kind="goal_pred", lbd=0.02, layers=(2,)))
aux, s = reg({2: h}, spans, generator=g, ntp_loss=torch.tensor(1.0))
aux.backward()
print("regulator ok:", round(float(aux), 4), s)
print("ALL GOOD")
