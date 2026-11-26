"""Collect every finished run under a directory into one results.json.

Scans ``<root>/<arm>_f<frac>_s<seed>/`` for eval.json / probe.json / tf.json / run.json,
so partial or resumed sweeps still produce a complete table.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PAT = re.compile(r"^(?P<arm>.+)_f(?P<frac>[0-9.]+)_s(?P<seed>\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/pilot")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    root = Path(a.root)
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = PAT.match(d.name)
        if not m:
            continue
        row = {"tag": d.name, "arm": m["arm"], "fraction": float(m["frac"]),
               "seed": int(m["seed"])}
        for key, fn in (("run", "run.json"), ("eval", "eval.json"),
                        ("probe", "probe.json"), ("tf", "tf.json")):
            p = d / fn
            if p.exists():
                row[key] = json.loads(p.read_text(encoding="utf-8"))
        if "eval" in row:
            row["exact_match"] = row["eval"].get("exact_match")
            row["unique_ratio"] = row["eval"].get("unique_ratio")
            row["collision_rate"] = row["eval"].get("collision_rate")
        if "tf" in row:
            row["tf_token_acc"] = row["tf"].get("tf_token_acc")
            row["tf_seq_acc"] = row["tf"].get("tf_seq_acc")
        rows.append(row)

    out = Path(a.out) if a.out else root / "results.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    hdr = f"{'arm':<14}{'frac':>7}{'EM':>8}{'TF-tok':>9}{'TF-seq':>8}{'uniq':>7}{'coll':>7}{'curv m1':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        probe = r.get("probe") or {}
        ce = probe.get("m1", {}).get("curvature_evidence")
        print(f"{r['arm']:<14}{r['fraction']:>7.4f}{r.get('exact_match', float('nan')):>8.3f}"
              f"{r.get('tf_token_acc', float('nan')):>9.3f}{r.get('tf_seq_acc', float('nan')):>8.3f}"
              f"{r.get('unique_ratio', float('nan')):>7.2f}{r.get('collision_rate', float('nan')):>7.2f}"
              f"{(ce if ce is not None else float('nan')):>9.2f}")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
