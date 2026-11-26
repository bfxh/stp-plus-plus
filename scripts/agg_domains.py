"""Per-domain ppl table for the mixed-corpus selection arms (single seed).

Reads `runs/<root>/<tag>/run.json` for the given roots and prints
arm | overall | code | chat | zh, plus the delta of each domain against `all`.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOTS = ["dom_curv", "dom_nll", "dom_strat_curv", "dom_strat_nll", "dom_mixhi"]
NAME = {"selall": "full", "selrand": "random 25%", "selhi": "hi 25%", "sello": "lo 25%",
        "selshi": "strat hi 25%", "selsrand": "strat rand",
        "mixhi": "88/5/7 mix, random"}


def main() -> None:
    rows = []
    for r in ROOTS:
        for f in sorted((ROOT / "runs" / r).glob("*/run.json")):
            tag = f.parent.name
            if not tag.endswith("_s82"):
                continue
            m = json.loads(f.read_text(encoding="utf-8"))
            d = m.get("eval", {}).get("domains")
            if not d:
                continue
            rows.append({"root": r, "tag": tag, "overall": m["eval"]["ppl"],
                         **{k: v["ppl"] for k, v in d.items()}})

    base = next(x for x in rows if x["tag"] == "selall_s82")
    lines = [f"{'arm':<18}{'overall':>9}{'code':>8}{'chat':>8}{'zh':>8}"
             f"{'  d_code':>9}{'d_chat':>8}{'d_zh':>8}"]
    for x in rows:
        name = f"{x['root'].replace('dom_', '')}/{NAME.get(x['tag'].rsplit('_s', 1)[0], x['tag'])}"
        dc = x.get("code", float("nan")) - base.get("code", 0)
        dch = x.get("chat", float("nan")) - base.get("chat", 0)
        dz = x.get("zh", float("nan")) - base.get("zh", 0)
        lines.append(f"{name:<18}{x['overall']:>9.3f}{x.get('code', 0):>8.3f}"
                     f"{x.get('chat', 0):>8.3f}{x.get('zh', 0):>8.3f}"
                     f"{dc:>+9.3f}{dch:>+8.3f}{dz:>+8.3f}")
    text = "\n".join(lines)
    print(text)
    (ROOT / "runs" / "mixed_curv" / "by_domain.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
