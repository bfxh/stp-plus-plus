"""Aggregate selection arms for any corpus: pass one or more `label:results.json` roots.

    python scripts/agg_real.py --root curv:runs/select_real/results.json \
                               --root nll:runs/select_real_nll/results.json \
                               --base curv:all --out runs/select_real/results.txt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LABEL = {
    "all": "full (all blocks)",
    "rand": "random 25%",
    "hi": "top 25% by score",
    "lo": "bottom 25% by score",
    "mid": "mid band (50-75%)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="label:results.json")
    ap.add_argument("--base", default="", help="label:arm used as the reference baseline")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    keys: dict[tuple[str, str], list[float]] = {}
    for spec in a.root:
        lab, path = spec.split(":", 1)
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            print(f"[warn] missing {p}")
            continue
        for r in json.loads(p.read_text(encoding="utf-8")):
            keys.setdefault((lab, r["arm"]), []).append(r["ppl"])

    base_lab, base_arm = (a.base.split(":", 1) if a.base else (None, None))
    if base_lab and (base_lab, base_arm) in keys:
        xs = keys[(base_lab, base_arm)]
        mu0 = sum(xs) / len(xs)
    else:  # first arm of the first root
        first = next(iter(keys))
        xs = keys[first]
        mu0 = sum(xs) / len(xs)

    lines = [a.title] if a.title else []
    lines.append(f"baseline ({base_lab}/{base_arm}) ppl = {mu0:.4f}")
    lines.append("")
    lines.append(f"{'arm':<26}{'n':>3}{'mean_ppl':>10}{'sd':>8}{'delta':>9}{'t':>8}")
    for (lab, arm), xs in sorted(keys.items()):
        name = f"{lab}/{DEFAULT_LABEL.get(arm, arm)}"
        mu = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / max(len(xs) - 1, 1))
        d = mu - mu0
        t = d / (sd / math.sqrt(len(xs))) if sd > 0 else float("nan")
        lines.append(f"{name:<26}{len(xs):>3}{mu:>10.4f}{sd:>8.4f}{d:>+9.4f}{t:>8.2f}")
    text = "\n".join(lines)
    print(text)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
