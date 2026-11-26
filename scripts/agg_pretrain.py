import json, math, pathlib

R = pathlib.Path(r"C:\Users\lbx13\Doubao\chats\2026-11-24\new-chat-7\stp-plus-plus\runs\pretrain")
rows = {}
for p in R.glob("*_s*/run.json"):
    tag = p.parent.name
    if "_s" not in tag:
        continue
    arm, s = tag.rsplit("_s", 1)
    if not s.isdigit():
        continue
    rows.setdefault(arm, {})[int(s)] = json.loads(p.read_text(encoding="utf-8"))["eval"]["ppl"]

ORDER = ["reg", "stp", "goal_pred", "stp_l0.5", "goal_l0.5", "ce25", "ce25_goal", "ce25_stp"]
print("%-12s%3s%10s%8s%9s" % ("arm", "n", "mean_ppl", "sd", "min"))
for a in ORDER:
    if a not in rows:
        continue
    xs = list(rows[a].values())
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else float("nan")
    print("%-12s%3d%10.4f%8.4f%9.4f" % (a, len(xs), m, sd, min(xs)))


def paired(a, b, label):
    common = sorted(set(rows.get(a, {})) & set(rows.get(b, {})))
    if not common:
        return
    d = [rows[a][s] - rows[b][s] for s in common]
    m = sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else float("nan")
    t = m / (sd / math.sqrt(len(d))) if sd and sd > 0 else float("inf")
    crit = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(d) - 1, 2.776)
    base = sum(rows[b][s] for s in common) / len(common)
    print("%-26s seeds=%s  dPPL=%+.4f (%+.1f%%)  t=%+.2f  crit=%.2f  %s"
          % (label, common, m, 100 * m / base, t, crit, "显著" if abs(t) >= crit else "不显著"))


print()
paired("ce25", "reg", "ce25  vs reg (CE 降到 1/4)")
paired("ce25_goal", "ce25", "ce25_goal vs ce25")
paired("ce25_stp", "ce25", "ce25_stp  vs ce25")
paired("ce25_goal", "reg", "ce25_goal vs reg")
paired("ce25_stp", "reg", "ce25_stp  vs reg")
print()
paired("stp", "reg", "stp(0.02) vs reg")
paired("goal_pred", "reg", "goal(0.02) vs reg")
paired("stp_l0.5", "reg", "stp(0.5) vs reg")
paired("goal_l0.5", "reg", "goal(0.5) vs reg")
