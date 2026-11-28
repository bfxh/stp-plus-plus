#!/usr/bin/env bash
# Objective arm: multi-token prediction (Meta reports 1.3-1.8x compute efficiency).
# Shares lm_head, adds no parameters. Compare against runs/big/steps1920_s82 (ppl 4.004).
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
mkdir -p runs/big
exec > runs/big/driver5.log 2>&1
BIG=/mnt/f/aiwork/mixed_cache_big
M=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct

for W in 0.2 0.5; do
  echo "[train] mtp_k=1 w=$W start $(date +%H:%M:%S)"
  STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
  /home/lbx13/stpenv/bin/python -m stpp.pretrain \
    --out_dir runs/big/mtp_w${W}_s82 --steps 1920 --bs 4 --seed 82 \
    --log_every 160 --lora_dropout 0.05 --mtp_k 1 --mtp_weight $W
  echo "[train] mtp_k=1 w=$W done $(date +%H:%M:%S)"
done
echo "[mtp] all done $(date +%H:%M:%S)"
