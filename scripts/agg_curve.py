"""Data-efficiency curve: does STP actually buy data?

Pulls teacher-forced seq acc for every completed run across the three sweep roots,
prints arm x fraction means, then fits acc ~ a + b*log2(fraction) to estimate how
much data an arm needs to match the *full-data* baseline.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\lbx13\Doubao\chats\2026-11-24\new-chat-7\stp-plus-plus")
ROOTS = ["runs\\curve", "runs\\regime", "runs\\seeds5", "runs\\pilot"]
ARMS = ["reg", "stp", "goal_pred", "curv_budget", "stpp", "stpp_pred"]


def load() -> dict:
    table = defaultdict(lambda: defaultdict(list))
    for r in ROOTS:
        base = ROOT / r
        if not base.exists():
            continue
        for tf in base.glob("*_f*_s*/tf.json"):
            tag = tf.parent.name
            head, seed = tag.rsplit("_s", 1)
            arm, frac = head.rsplit("_f", 1)
            try:
                acc = json.loads(tf.read_text(encoding="utf-8"))["tf_seq_acc"]
            except Exception:
                continue
            table[arm][float(frac)].append(acc)
    return table


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def fit_needed(table_arm: dict, target: float) -> float:
    """fraction needed to hit `target`, via least squares on log2(fraction)."""
    xs = sorted(table_arm)
    if len(xs) < 2:
        return float("nan")
    lx = [math.log2(x) for x in xs]
    ys = [mean(table_arm[x]) for x in xs]
    n = len(xs)
    mx, my = mean(lx), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(lx, ys))
    den = sum((x - mx) ** 2 for x in lx)
    if den == 0:
        return float("nan")
    b = num / den
    a = my - b * mx
    if b == 0:
        return float("nan")
    return 2 ** ((target - a) / b)


def main() -> None:
    table = load()
    fr = sorted({f for a in table.values() for f in a})
    print("fractions:", fr)
    print()
    hdr = "%-12s" % "arm" + "".join("%12s" % f"f={f:g}" for f in fr)
    print(hdr)
    print("-" * len(hdr))
    for a in ARMS:
        if a not in table:
            continue
        line = "%-12s" % a
        for f in fr:
            xs = table[a].get(f)
            if xs:
                line += "%8.3f±%4.3f" % (mean(xs), sd(xs))
            else:
                line += "%12s" % "-"
        print(line)

    print()
    print("  n per cell:")
    for a in ARMS:
        if a not in table:
            continue
        print("   %-12s" % a + "".join("%8d" % len(table[a].get(f, [])) for f in fr))

    if 1.0 not in table.get("reg", {}):
        print("\n[no reg@1.0 - can't compute data efficiency]")
        return
    target = mean(table["reg"][1.0])
    print(f"\nbaseline = reg@1.0 = {target:.3f}\n")
    print("%-12s%10s%10s%14s" % ("arm", "best_frac", "best_acc", "frac_needed"))
    for a in ARMS:
        if a not in table:
            continue
        needed = fit_needed(table[a], target)
        best_f = max(table[a], key=lambda f: mean(table[a][f]))
        best = mean(table[a][best_f])
        note = ""
        if needed == needed:      # not nan
            note = f"{needed:.3f}"
            if needed > 1.0:
                note += "  (>1 => 要更多数据)"
            else:
                note += f"  (={1/needed:.1f}x 数据效率)"
        print("%-12s%10.3f%10.3f%14s" % (a, best_f, best, note))


if __name__ == "__main__":
    main()
