#!/usr/bin/env bash
# Capacity arm: does the 3.80 floor come from LoRA rank rather than data/optimizer?
# r=32 and r=64, lora_alpha = 2r so the delta scaling stays at 2x (same as the baseline).
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
mkdir -p runs/big
exec > runs/big/driver4.log 2>&1
BIG=/mnt/f/aiwork/mixed_cache_big
M=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct

for R in 32 64; do
  A=$((R * 2))
  echo "[train] rank=$R alpha=$A start $(date +%H:%M:%S)"
  STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
  /home/lbx13/stpenv/bin/python -m stpp.pretrain \
    --out_dir runs/big/r${R}_s82 --steps 1920 --bs 4 --seed 82 \
    --log_every 160 --lora_dropout 0.05 --lora_r $R --lora_alpha $A
  echo "[train] rank=$R done $(date +%H:%M:%S)"
done
echo "[rank] all done $(date +%H:%M:%S)"
