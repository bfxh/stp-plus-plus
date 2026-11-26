"""Sweep the pretraining-regime arms: reg / stp / goal_pred across seeds."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARMS = {
    "reg": dict(geom_kind="none"),
    "stp": dict(geom_kind="stp_random_span", lbd=0.02),
    "goal_pred": dict(geom_kind="goal_pred", lbd=0.02),
    # the same two objectives, but with lambda scaled so they actually carry weight
    # once CE sits at ~1.2 instead of ~0.01 (share ~1.5% -> ~40%)
    "stp_l0.5": dict(geom_kind="stp_random_span", lbd=0.5),
    "goal_l0.5": dict(geom_kind="goal_pred", lbd=0.5),
    # ---- 替换式：CE 只监督 1/4 的 token，剩下的 3/4 没有任何 token 级目标 ----
    # 若 latent 目标真的"携带 CE 之外的信息"，它就该把 ce25 的坑填回来。
    # 若它只是个正则项，ce25_goal 只能等于 ce25 或更差。
    "ce25": dict(geom_kind="none", ce_stride=4),
    "ce25_goal": dict(geom_kind="goal_pred", lbd=0.5, ce_stride=4),
    "ce25_stp": dict(geom_kind="stp_random_span", lbd=0.5, ce_stride=4),
    # ---- MTP：信息型目标的正例对照（校准 §11 的缺口测试）----
    # h_t 直接预测 t+2..t+k。它确实把更长前缀的信息压进同一个位置，所以若
    # §11 的测法有效，ce25_mtp3 必须把缺口填回来一块。
    "mtp3": dict(geom_kind="none", mtp_k=3, mtp_weight=0.15),
    "ce25_mtp3": dict(geom_kind="none", ce_stride=4, mtp_k=3, mtp_weight=0.15),
    # 轻剂量版：MTP 初始损失 ~12.4，0.05 的权重让它的首批贡献 ~0.6 ≈ ntp 的一半。
    # 若"填不回缺口"是剂量问题，这两条臂应该能填回来一块。
    "mtp3_l": dict(geom_kind="none", mtp_k=3, mtp_weight=0.05),
    "ce25_mtp3_l": dict(geom_kind="none", ce_stride=4, mtp_k=3, mtp_weight=0.05),
    # ---- 8GB 版正例对照：两个偏移正好覆盖 stride 3 抽掉的 2/3 位置 ----
    # (三个偏移会留三份 logits，5.1GB + 64MB 就爆；两个偏移刚好在预算内)
    "ce33": dict(geom_kind="none", ce_stride=3),
    "ce33_mtp2": dict(geom_kind="none", ce_stride=3, mtp_k=2, mtp_weight=0.05),
    "mtp2": dict(geom_kind="none", mtp_k=2, mtp_weight=0.05),
}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[82, 23, 37])
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--out-root", default="runs/pretrain")
    ap.add_argument("--model", default=r"F:\aiwork\models\Llama-3.2-1B-Instruct")
    a = ap.parse_args()

    rows = []
    for arm in a.arms:
        for seed in a.seeds:
            tag = f"{arm}_s{seed}"
            out = ROOT / a.out_root / tag
            run_json = out / "run.json"
            if run_json.exists():
                print(f"[skip] {tag}", flush=True)
            else:
                cmd = [sys.executable, "-m", "stpp.pretrain",
                       "--model", a.model, "--out_dir", str(out),
                       "--steps", str(a.steps), "--bs", str(a.bs),
                       "--block", str(a.block), "--seed", str(seed)]
                for k, v in ARMS[arm].items():
                    cmd += [f"--{k}", str(v)]
                print(f"\n=== {tag} ===", flush=True)
                out.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, PYTHONPATH=str(ROOT), TOKENIZERS_PARALLELISM="false")
                with open(out / "train.log", "w", encoding="utf-8") as f:
                    rc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                        stdout=f, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    print(f"[fail] {tag} rc={rc}", flush=True)
                    continue
            meta = json.loads(run_json.read_text(encoding="utf-8"))
            rows.append({"arm": arm, "seed": seed, "ppl": meta["eval"]["ppl"],
                         "ce": meta["eval"]["ce"], "wall": meta["wall_sec"]})
            print(f"  {tag}: ppl={meta['eval']['ppl']:.3f} ce={meta['eval']['ce']:.4f}",
                  flush=True)

    (ROOT / a.out_root / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[table]")
    for arm in a.arms:
        xs = [r["ppl"] for r in rows if r["arm"] == arm]
        if xs:
            print(f"  {arm:<12} n={len(xs)} mean_ppl={sum(xs)/len(xs):.3f} "
                  f"min={min(xs):.3f} max={max(xs):.3f}")


if __name__ == "__main__":
    main()
