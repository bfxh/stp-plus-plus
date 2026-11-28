#!/usr/bin/env bash
# Big-corpus curve, second batch: 960 / 5760 / 7680 steps (cache must already exist).
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
mkdir -p runs/big
exec > runs/big/driver2.log 2>&1
BIG=/mnt/f/aiwork/mixed_cache_big
for S in 960 5760 7680; do
  echo "[train] steps=$S start $(date +%H:%M:%S)"
  STPP_MODEL=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct \
  STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
  /home/lbx13/stpenv/bin/python -m stpp.pretrain \
    --out_dir runs/big/steps${S}_s82 --steps $S --bs 4 --seed 82 \
    --log_every 160 --lora_dropout 0.05
  echo "[train] steps=$S done $(date +%H:%M:%S)"
done
echo "[big2] all done $(date +%H:%M:%S)"
