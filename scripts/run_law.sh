#!/usr/bin/env bash
# Rank-law verification at 2x data + the "norm-scale capacity" arm.
#
# Predictions being tested (docs/SCALING.md §26):
#   r*(N) = (0.720*N/1013)^0.566  ->  r* = 89 @1.92k steps, 132 @3840 steps
#   => at 3840 steps r128 should BEAT r64 (opposite of the 1920-step result).
#   If r128 <= r64 at 3840, the cr/N overfit term is wrong.
#
# Arm 3 tests: 1-D RMSNorm scales give d per-channel DOF that a rank-r dW must
# otherwise spend rank budget to imitate.  Predicted -0.02..-0.05 ppl.
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
mkdir -p runs/big
exec > runs/big/driver8.log 2>&1
BIG=/mnt/f/aiwork/mixed_cache_big
M=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct
COMMON="--bs 4 --seed 82 --log_every 160 --lora_dropout 0.05 --domains 3 --domain_names code,chat,zh"
export STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1

echo "[law] r64@3840 start $(date +%H:%M:%S)  (baseline for r128)"
/home/lbx13/stpenv/bin/python -m stpp.pretrain --out_dir runs/big/r64_s3840b_s82 \
  --steps 3840 --lora_r 64 --lora_alpha 128 $COMMON || echo "[warn] r64@3840 failed"
echo "[law] r64@3840 done $(date +%H:%M:%S)"

echo "[law] r128@3840 start $(date +%H:%M:%S)  (r*(7.86M)=132 says this should win)"
/home/lbx13/stpenv/bin/python -m stpp.pretrain --out_dir runs/big/r128_s3840_s82 \
  --steps 3840 --lora_r 128 --lora_alpha 256 $COMMON || echo "[warn] r128@3840 failed"
echo "[law] r128@3840 done $(date +%H:%M:%S)"

echo "[law] r64+norm@3840 start $(date +%H:%M:%S)  (free d-DOF per layer)"
/home/lbx13/stpenv/bin/python -m stpp.pretrain --out_dir runs/big/r64n_s3840_s82 \
  --steps 3840 --lora_r 64 --lora_alpha 128 --train_norms $COMMON || echo "[warn] norm arm failed"
echo "[law] r64+norm@3840 done $(date +%H:%M:%S)"
echo "[law] ALL DONE $(date +%H:%M:%S)"
