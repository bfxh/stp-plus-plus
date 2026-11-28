#!/usr/bin/env bash
# Combine the two winners: r64 (capacity, -0.073 at flat wall clock) + fp16/16F (1.20x speed).
# Goal: r64's ppl inside the r16-bf16 wall clock (~1262 s for 1920 steps).
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
mkdir -p runs/big
exec > runs/big/driver7.log 2>&1
BIG=/mnt/f/aiwork/mixed_cache_big
M=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct

echo "[train] r64+fp16 start $(date +%H:%M:%S)"
STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
/home/lbx13/stpenv/bin/python -m stpp.pretrain \
  --out_dir runs/big/r64_fp16_s82 --steps 1920 --bs 4 --seed 82 \
  --log_every 160 --lora_dropout 0.05 --lora_r 64 --lora_alpha 128 \
  --fp16 --no-bf16
echo "[train] r64+fp16 done $(date +%H:%M:%S)"
echo "[combo] all done $(date +%H:%M:%S)"
