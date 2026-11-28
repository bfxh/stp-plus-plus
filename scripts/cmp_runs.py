"""Print a one-line summary per run dir: steps, wall, ppl, final train ntp, epochs seen."""
import json, sys
from pathlib import Path

dirs = sys.argv[1:]
print(f"{'dir':<34}{'steps':>7}{'wall_s':>9}{'ppl':>9}{'ce':>8}{'ntp':>8}{'ep':>6}")
for d in dirs:
    p = Path(d)
    rj = p / "run.json"
    if not rj.exists():
        print(f"{d:<34}{'--':>7}")
        continue
    r = json.loads(rj.read_text())
    last = None
    tj = p / "train.jsonl"
    if tj.exists():
        for line in tj.read_text().splitlines():
            if line.strip():
                last = json.loads(line)
    ntp = f"{last['ntp']:.4f}" if last else "  --"
    ep = f"{last['epoch']:.2f}" if last else "  --"
    e = r.get("eval", {})
    print(f"{d:<34}{r['steps']:>7}{r['wall_sec']:>9.0f}{e.get('ppl', float('nan')):>9.3f}"
          f"{e.get('ce', 0):>8.3f}{ntp:>8}{ep:>6}")
