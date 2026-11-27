#!/usr/bin/env bash
# L0 curve: extend to 3840 steps (seed 82), same config as runs/curve1920.
# Launched via a Windows scheduled task so it survives the agent tool call.
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
mkdir -p runs/curve3840
exec > runs/curve3840/driver.log 2>&1
echo "[curve] seed=82 start $(date +%H:%M:%S)"
STPP_MODEL=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct \
STPP_CACHE_DIR=/mnt/f/aiwork/mixed_cache \
STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
/home/lbx13/stpenv/bin/python -m stpp.pretrain \
  --out_dir runs/curve3840/selall_s82 --steps 3840 --bs 4 --seed 82 \
  --log_every 80 --lora_dropout 0.05
echo "[curve] seed=82 done $(date +%H:%M:%S)"
