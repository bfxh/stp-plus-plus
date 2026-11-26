import json, collections, pathlib

ROOT = pathlib.Path(r"C:\Users\lbx13\Doubao\chats\2026-11-24\new-chat-7\stp-plus-plus")
rows = json.loads((ROOT / "runs" / "seeds5" / "results.json").read_text(encoding="utf-8"))

d = collections.defaultdict(dict)
for r in rows:
    tf = r.get("tf") or {}
    d[r["arm"]][r["seed"]] = (
        round(tf.get("tf_seq_acc", 0), 3),
        round(tf.get("tf_token_acc", 0), 3),
    )

seeds = sorted({s for v in d.values() for s in v})
order = ["reg", "stp", "curv_budget", "stpp", "stpp_pred"]

print("=== tf_seq_acc per seed ===")
print("arm".ljust(14) + "".join(str(s).rjust(8) for s in seeds))
for arm in order:
    print(arm.ljust(14) + "".join(str(d[arm].get(s, ("-",))[0]).rjust(8) for s in seeds))

print("\n=== final train ntp / aux / lambda-dot-aux share (seed 82) ===")
for arm in order:
    p = ROOT / "runs" / "seeds5" / f"{arm}_f0.0625_s82" / "train.jsonl"
    last = json.loads(p.read_text(encoding="utf-8").strip().splitlines()[-1])
    ntp, aux, lam = last["ntp"], last["aux"], last["lam"]
    contrib = lam * aux
    share = 100.0 * contrib / (ntp + contrib + 1e-12)
    print("  %-14s ntp=%.4f  aux=%.4f  lam=%.4f  lam*aux=%.4f  (=%5.1f%% of total)"
          % (arm, ntp, aux, lam, contrib, share))
