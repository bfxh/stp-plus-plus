"""chunked_offsets must reproduce chunked_ce exactly (it's the same math, one head pass)."""
import torch
import torch.nn as nn

from stpp.pretrain import chunked_offsets
from stpp.train import chunked_ce

torch.manual_seed(0)
B, T, D, V = 2, 130, 64, 500
hidden = torch.randn(B, T, D, requires_grad=True)
head = nn.Linear(D, V, bias=False)
labels = torch.randint(0, V, (B, T))
labels[0, 10:20] = -100          # exercise the mask path

off1 = chunked_ce(hidden, head, labels, chunk=32)
got = chunked_offsets(hidden, head, [(labels, 1)], chunk=32)[0]
print("chunked_ce     =", float(off1))
print("chunked_offsets=", float(got))
assert torch.allclose(off1, got, atol=1e-5), "mismatch"

# multi-offset: each entry must equal the standalone one-head-call computation
terms = [(labels, 1), (labels, 2), (labels, 3)]
multi = chunked_offsets(hidden, head, terms, chunk=32)
for i, (lab, off) in enumerate(terms):
    ref = chunked_ce(hidden[:, : T - off + 1], head, lab[:, off - 1:], chunk=32)
    print(f"offset {off}: multi={float(multi[i]):.6f} ref={float(ref):.6f}")
    assert torch.allclose(multi[i], ref, atol=1e-5), f"offset {off} mismatch"
print("OK")
