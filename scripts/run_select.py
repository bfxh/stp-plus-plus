"""Geometry-as-selector: pick which blocks to train on, by curvature score.

Arms (each gets 25% of the corpus as its sampling pool, i.e. the same token budget):
    rand  — random 25%                (control)
    hi    — top 25% by curvature      (bending trajectories = "informative"?)
    lo    — bottom 25%                (straight trajectories = "redundant"?)
    mid   — the 50–75% quartile       (middle band, sanity check)

Each run trains 120 steps × bs 4 × 512 tokens = 245,760 tokens, exactly the same
budget as the `reg` arms, so any difference is caused only by *which* tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCORES = ROOT / "runs" / "curvity" / "train_curv.npy"

# "all" = no subset at all (the ordinary training baseline, same budget)
ARMS = ("all", "rand", "hi", "lo", "mid")


def pools(scores: np.ndarray, seed: int, frac: float = 0.25,
          strata: int = 1) -> dict[str, np.ndarray]:
    """Subsets of the corpus. With strata>1 the quota is filled *within* each
    interleaved group (i % strata), so the domain mix is preserved exactly."""
    n = len(scores)
    k = int(n * frac)
    out: dict[str, np.ndarray] = {}
    if strata > 1:
        srand, shi, slo, smid = [], [], [], []
        for g in range(strata):
            gi = np.arange(g, n, strata)
            kk = max(1, int(len(gi) * frac))
            order = gi[np.argsort(scores[gi], kind="stable")]
            lo_g, hi_g = order[:kk], order[-kk:]
            mid_g = order[int(len(gi) * 0.5):int(len(gi) * 0.75)]
            rng = np.random.default_rng(seed + 1000 * g)
            srand.append(rng.choice(gi, size=kk, replace=False))
            shi.append(hi_g)
            slo.append(lo_g)
            smid.append(mid_g)
        out.update(srand=np.concatenate(srand), shi=np.concatenate(shi),
                   slo=np.concatenate(slo), smid=np.concatenate(smid))
        return out

    order = np.argsort(scores, kind="stable")
    lo, hi = order[:k], order[-k:]
    q1, q3 = int(n * 0.5), int(n * 0.75)
    mid = order[q1:q3]
    rng = np.random.default_rng(seed)
    rand = rng.choice(n, size=k, replace=False)
    return {"lo": lo, "hi": hi, "mid": mid, "rand": rand}


def main() -> None:
    import argparse as _a
    ap = _a.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[82, 23, 37])
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out-root", default="runs/select")
    ap.add_argument("--scores", default=str(DEFAULT_SCORES))
    ap.add_argument("--strata", type=int, default=1,
                    help="域内分层：对 i%%strata 的每组分别取子集（保持域配比）")
    ap.add_argument("--domains", type=int, default=1,
                    help="在 test 块上按 i%%domains 分域报告 ppl")
    ap.add_argument("--domain-names", default="")
    ap.add_argument("--model", default=os.environ.get("STPP_MODEL",
                                                      "unsloth/Llama-3.2-1B-Instruct"))
    a = ap.parse_args()

    scores = np.load(a.scores) if "all" not in a.arms else (
        np.load(a.scores) if Path(a.scores).exists() else None)
    ids_dir = ROOT / a.out_root / "ids"
    ids_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in a.seeds:
        pmap = pools(scores, seed, strata=a.strata) if scores is not None else {}
        for arm in a.arms:
            tag = f"sel{arm}_s{seed}"
            out = ROOT / a.out_root / tag
            run_json = out / "run.json"
            if run_json.exists():
                print(f"[skip] {tag}", flush=True)
            else:
                cmd = [sys.executable, "-m", "stpp.pretrain",
                       "--model", a.model, "--out_dir", str(out),
                       "--steps", str(a.steps), "--bs", str(a.bs),
                       "--block", "512", "--seed", str(seed)]
                if a.domains > 1:
                    cmd += ["--domains", str(a.domains)]
                    if a.domain_names:
                        cmd += ["--domain_names", a.domain_names]
                if arm != "all":
                    ids_f = ids_dir / f"{arm}_s{seed}.npy"
                    np.save(ids_f, pmap[arm])
                    cmd += ["--train_ids", str(ids_f)]
                pool_n = len(pmap[arm]) if arm != "all" else -1
                print(f"\n=== {tag}  pool={pool_n} ===", flush=True)
                out.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, PYTHONPATH=str(ROOT), TOKENIZERS_PARALLELISM="false")
                with open(out / "train.log", "w", encoding="utf-8") as f:
                    rc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                        stdout=f, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    print(f"[fail] {tag} rc={rc}", flush=True)
                    continue
            meta = json.loads(run_json.read_text(encoding="utf-8"))
            rows.append({"arm": arm, "seed": seed, "ppl": meta["eval"]["ppl"]})
            print(f"  {tag}: ppl={meta['eval']['ppl']:.3f}", flush=True)

    (ROOT / a.out_root / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[table]")
    for arm in a.arms:
        xs = [r["ppl"] for r in rows if r["arm"] == arm]
        if xs:
            print(f"  sel{arm:<4} n={len(xs)} mean_ppl={sum(xs)/len(xs):.3f} "
                  f"min={min(xs):.3f} max={max(xs):.3f}")


if __name__ == "__main__":
    main()
