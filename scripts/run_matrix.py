"""Sweep driver: arms x data-fractions x seeds, then eval + probe, then a markdown table.

Usage (from the project root, inside the venv):

    python scripts/run_matrix.py --smoke
    python scripts/run_matrix.py --fractions 1.0 0.25 --seeds 82 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each arm is an independent scientific claim, not a grid point.
ARMS: dict[str, dict] = {
    "reg": dict(geom_kind="none"),
    "stp": dict(geom_kind="stp_random_span", lbd=0.02),
    "stp_whiten": dict(geom_kind="stp_random_span", lbd=0.02, whiten=True),
    "curv": dict(geom_kind="curvature_angle", lbd=0.02),
    "curv_budget": dict(geom_kind="curvature_budget", lbd=0.02, tau_adaptive=True),
    "stpp": dict(geom_kind="curvature_budget", lbd=0.02, tau_adaptive=True,
                 whiten=True, repulsion=0.05, adaptive_lambda="loss_ratio"),
    # our answer to "what would actually be stronger than STP": drop the straight-line
    # assumption, learn the curve.  residual 2-layer MLP head + stop-gradient target.
    "stpp_pred": dict(geom_kind="latent_pred", lbd=0.02),
    # the diagnosis arm: *identical* STP geometry, but lambda chosen to hold the
    # auxiliary term at ~3% of total loss instead of the paper's fixed 0.02, which
    # lands at 77% once CE has collapsed.  Same prior, sane controller.
    "stp_share3": dict(geom_kind="stp_random_span", lbd=0.0002,
                       adaptive_lambda="loss_ratio", target_ratio=0.03),
    # dose-response on the *bounded* force: the P90 budget was inert (aux 0.0027).
    # tau=0.5 sits below the observed median kappa (~0.77), so the hinge is active
    # everywhere yet still zero-gradient below tau.  Is there a sweet spot?
    "curv_tau50": dict(geom_kind="curvature_budget", lbd=0.02, tau=0.5, tau_adaptive=False),
    # the *replacement* form: not a geometric regulariser on the path, but a second
    # information-bearing target — from any interior position, predict where the
    # answer ends.  lambda is set to make it a co-primary objective, not a nudge.
    "goal_pred": dict(geom_kind="goal_pred", lbd=0.02),
}

SMOKE_ARMS = ["reg", "stp", "curv_budget"]


def run_cmd(cmd: list[str], log_path: Path) -> int:
    env = dict(os.environ, PYTHONPATH=str(ROOT), TOKENIZERS_PARALLELISM="false",
               HF_HOME=os.environ.get("HF_HOME", r"F:\aiwork\hf"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--train", default=r"F:\aiwork\llm-jepa-ref\datasets\synth_train.jsonl")
    ap.add_argument("--test", default=r"F:\aiwork\llm-jepa-ref\datasets\synth_test.jsonl")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--fractions", nargs="*", type=float, default=[1.0, 0.25])
    ap.add_argument("--seeds", nargs="*", type=int, default=[82])
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-limit", type=int, default=300)
    ap.add_argument("--traj-limit", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--eval-mode", choices=["tf", "full"], default="tf",
                    help="tf = teacher-forced only (fast, sweep-friendly); full = generation + trajectory")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-trained", action="store_true", help="only evaluate existing dirs")
    args = ap.parse_args()

    arms = args.arms or (SMOKE_ARMS if args.smoke else list(ARMS))
    if args.smoke:
        args.max_steps = args.max_steps if args.max_steps > 0 else 60

    results = []
    for frac in args.fractions:
        for arm in arms:
            for seed in args.seeds:
                tag = f"{arm}_f{frac:g}_s{seed}"
                out = ROOT / args.out_root / tag
                cfg = dict(ARMS[arm], fraction=frac, seed=seed)

                if not args.skip_trained:
                    cmd = [sys.executable, "-m", "stpp.train",
                           "--model", args.model,
                           "--train_file", args.train,
                           "--out_dir", str(out),
                           "--epochs", str(args.epochs),
                           "--bs", str(args.bs),
                           "--lr", str(args.lr),
                           "--seed", str(seed),
                           "--fraction", str(frac),
                           "--max_steps", str(args.max_steps)]
                    for k, v in cfg.items():
                        if k in ("fraction", "seed"):
                            continue
                        if isinstance(v, bool):
                            cmd.append(f"--{k}" if v else f"--no-{k}")
                        else:
                            cmd += [f"--{k}", str(v)]
                    print(f"\n=== {tag} ===", flush=True)
                    rc = run_cmd(cmd, out / "train.log")
                    if rc != 0:
                        print(f"[fail] {tag} rc={rc}", flush=True)
                        results.append({"tag": tag, "error": rc})
                        continue

                ev = out / "eval.json"
                tf = out / "tf.json"
                if args.eval_mode == "full":
                    if not ev.exists():
                        cmd = [sys.executable, "-m", "stpp.evaluate",
                               "--model", args.model,
                               "--adapter", str(out / "adapter"),
                               "--test_file", args.test,
                               "--out", str(ev),
                               "--limit", str(args.eval_limit),
                               "--dump_traj",
                               "--limit_traj", str(args.traj_limit)]
                        run_cmd(cmd, out / "eval.log")
                if not tf.exists():
                    cmd = [sys.executable, "-m", "stpp.eval_tf",
                           "--model", args.model,
                           "--adapter", str(out / "adapter"),
                           "--test_file", args.test,
                           "--out", str(tf),
                           "--limit", str(args.eval_limit)]
                    run_cmd(cmd, out / "tf.log")

                rec = {"tag": tag, "arm": arm, "fraction": frac, "seed": seed}
                if ev.exists():
                    rec.update(json.loads(ev.read_text(encoding="utf-8")))
                    pr = out / "probe.json"
                    if not pr.exists():
                        run_cmd([sys.executable, "-m", "stpp.probe",
                                 "--traj", str(ev).replace("eval.json", "eval.traj.npz"),
                                 "--out", str(pr)], out / "probe.log")
                    if pr.exists():
                        rec["probe"] = json.loads(pr.read_text(encoding="utf-8"))
                if tf.exists():
                    rec["tf"] = json.loads(tf.read_text(encoding="utf-8"))
                results.append(rec)

    res_path = ROOT / args.out_root / "results.json"
    res_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[results] {res_path}")
    print_table(results)


def print_table(results: list[dict]) -> None:
    hdr = f"{'arm':<14}{'frac':>7}{'EM':>8}{'TF-seq':>8}{'uniq':>8}{'coll':>8}{'curv/lin':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r or ("exact_match" not in r and "tf" not in r):
            print(f"{r.get('tag','?'):<14}  FAILED")
            continue
        probe = r.get("probe") or {}
        ratio = probe.get("m1", {}).get("curvature_evidence", float("nan"))
        tf = r.get("tf") or {}
        print(f"{r['arm']:<14}{r['fraction']:>7.3f}{r.get('exact_match', float('nan')):>9.3f}"
              f"{tf.get('tf_seq_acc', float('nan')):>8.3f}"
              f"{r.get('unique_ratio', float('nan')):>8.2f}{r.get('collision_rate', float('nan')):>8.2f}"
              f"{ratio:>10.2f}")


if __name__ == "__main__":
    main()
