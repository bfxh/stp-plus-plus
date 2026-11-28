#!/usr/bin/env bash
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
BIG=/mnt/f/aiwork/mixed_cache_big
M=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct
echo "[fp8probe] start $(date +%H:%M:%S)"
STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 STPP_FP8=1 \
/home/lbx13/stpenv/bin/python -m stpp.pretrain \
  --out_dir runs/big/fp8_probe --steps 60 --bs 4 --seed 82 --log_every 20 \
  --lora_dropout 0.05 --lora_r 64 --lora_alpha 128 \
  > runs/big/fp8_probe.log 2>&1 || echo "[fp8probe] FAILED"
echo "[fp8probe] done $(date +%H:%M:%S)"
grep -E '^\[|out of memory|OutOfMemory|Error|error' runs/big/fp8_probe.log | tail -12