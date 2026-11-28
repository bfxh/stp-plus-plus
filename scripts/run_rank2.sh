#!/usr/bin/env bash
# Rank, second batch: does r128 keep helping (saturation?), and does r64 want more steps?
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
mkdir -p runs/big
exec > runs/big/driver6.log 2>&1
BIG=/mnt/f/aiwork/mixed_cache_big
M=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct

echo "[train] r128 steps=1920 start $(date +%H:%M:%S)"
STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
/home/lbx13/stpenv/bin/python -m stpp.pretrain \
  --out_dir runs/big/r128_s82 --steps 1920 --bs 4 --seed 82 \
  --log_every 160 --lora_dropout 0.05 --lora_r 128 --lora_alpha 256 \
  || echo "[warn] r128 arm failed (likely 8GB OOM)"
echo "[train] r128 done $(date +%H:%M:%S)"

echo "[train] r64 steps=3840 start $(date +%H:%M:%S)"
STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
/home/lbx13/stpenv/bin/python -m stpp.pretrain \
  --out_dir runs/big/r64_s3840_s82 --steps 3840 --bs 4 --seed 82 \
  --log_every 160 --lora_dropout 0.05 --lora_r 64 --lora_alpha 128
echo "[train] r64 3840 done $(date +%H:%M:%S)"
echo "[rank2] all done $(date +%H:%M:%S)"
