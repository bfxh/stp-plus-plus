import json, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOTS = ["runs\\curve", "runs\\regime", "runs\\seeds5", "runs\\pilot"]

cells = defaultdict(dict)      # (arm, frac) -> {seed: acc}
for r in ROOTS:
    for tf in (ROOT / r).glob("*_f*_s*/tf.json"):
        tag = tf.parent.name
        head, seed = tag.rsplit("_s", 1)
        arm, frac = head.rsplit("_f", 1)
        try:
            acc = json.loads(tf.read_text(encoding="utf-8"))["tf_seq_acc"]
        except Exception:
            continue
        cells[(arm, float(frac))][int(seed)] = acc

CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def paired(a, b, frac):
    A, B = cells.get((a, frac), {}), cells.get((b, frac), {})
    common = sorted(set(A) & set(B))
    if len(common) < 2:
        return None
    d = [A[s] - B[s] for s in common]
    m = sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    t = m / (sd / math.sqrt(len(d))) if sd > 0 else float("inf")
    return m, sd, t, len(common)


for frac in [0.0625, 0.25, 1.0]:
    print(f"\n=== fraction {frac:g} ===")
    for a, b, label in [("stp", "reg", "stp - reg"), ("goal_pred", "reg", "goal - reg")]:
        r = paired(a, b, frac)
        if not r:
            print(f"  {label:<12} insufficient seeds")
            continue
        m, sd, t, n = r
        crit = CRIT.get(n - 1, 2.571)
        common = sorted(set(cells[(a, frac)]) & set(cells[(b, frac)]))
        diffs = [round(cells[(a, frac)][s] - cells[(b, frac)][s], 3) for s in common]
        verdict = "显著" if abs(t) >= crit else "不显著"
        print(f"  {label:<12} {m:+.4f}  sd={sd:.4f}  t={t:+.2f}  n={n}  crit={crit:.2f}  "
              f"{verdict}   per-seed={diffs}")

print("\n=== per-seed acc ===")
for frac in [0.0625, 0.25, 1.0]:
    print(f"frac {frac:g}")
    for a in ["reg", "stp", "goal_pred"]:
        if (a, frac) in cells:
            print("  %-10s %s" % (a, {s: round(v, 3) for s, v in sorted(cells[(a, frac)].items())}))
