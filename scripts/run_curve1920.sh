#!/usr/bin/env bash
# L0 curve: seeds 23 and 37 at 1920 steps, same config as runs/curve960.
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
cd /mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
for S in 23 37; do
  echo "[curve] seed=$S start $(date +%H:%M:%S)"
  STPP_MODEL=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct \
  STPP_CACHE_DIR=/mnt/f/aiwork/mixed_cache \
  STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
  /home/lbx13/stpenv/bin/python -m stpp.pretrain \
    --out_dir runs/curve1920/selall_s$S --steps 1920 --bs 4 --seed $S \
    --log_every 40 --lora_dropout 0.05
  echo "[curve] seed=$S done $(date +%H:%M:%S)"
done
echo "[curve] all done"
