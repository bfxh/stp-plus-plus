"""Aggregate a multi-seed sweep: mean ± std per arm, plus a paired comparison vs baseline.

Paired by seed (same subset, same order), which is the correct comparison here:
seed 82 for every arm sees the *same* 500 examples in the *same* order, so the
difference is attributable to the regulariser, not to data luck.

Critical values are hard-coded t(0.975, df) so we don't need scipy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def mean(xs): return sum(xs) / len(xs)
def std(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def metric(row: dict, key: str):
    if key == "tf_seq_acc":
        return (row.get("tf") or {}).get("tf_seq_acc")
    if key == "tf_token_acc":
        return (row.get("tf") or {}).get("tf_token_acc")
    if key == "exact_match":
        return row.get("exact_match")
    return None


def paired(a: list, b: list, label: str) -> str:
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return f"{label}: n={n}, 样本太少"
    m, s = mean(diffs), std(diffs)
    se = s / math.sqrt(n)
    t = m / se if se > 0 else float("inf")
    crit = T975.get(n - 1, 2.776)
    sig = "**显著**" if abs(t) >= crit else "不显著"
    return (f"{label}: 平均差 {m:+.4f}  (sd {s:.4f}, se {se:.4f}, "
            f"t={t:+.2f}, 临界值 {crit}) -> {sig}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/seeds5/results.json")
    ap.add_argument("--metric", default="tf_seq_acc")
    a = ap.parse_args()

    rows = json.loads(Path(a.results).read_text(encoding="utf-8"))
    by_arm: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        v = metric(r, a.metric)
        if v is None:
            continue
        by_arm[r["arm"]][r["seed"]] = v

    print(f"[metric] {a.metric}\n")
    print(f"{'arm':<14}{'n':>3}{'mean':>9}{'sd':>9}{'min':>8}{'max':>8}")
    print("-" * 51)
    for arm, d in sorted(by_arm.items()):
        xs = list(d.values())
        print(f"{arm:<14}{len(xs):>3}{mean(xs):>9.4f}{std(xs):>9.4f}{min(xs):>8.4f}{max(xs):>8.4f}")

    base = "reg"
    if base in by_arm:
        seeds = sorted(set(by_arm[base]) & set().union(*[set(v) for v in by_arm.values()]))
        print("\n[配对比较 vs reg]")
        for arm, d in sorted(by_arm.items()):
            if arm == base:
                continue
            common = [s for s in seeds if s in d]
            print(" " + paired([d[s] for s in common], [by_arm[base][s] for s in common], arm))


if __name__ == "__main__":
    main()
