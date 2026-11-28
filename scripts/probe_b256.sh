#!/usr/bin/env bash
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
export STPP_CACHE_DIR=/mnt/f/aiwork/mixed_cache_big
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
/home/lbx13/stpenv/bin/python scripts/make_b256.py || exit 1
M=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct
echo "[b256] start $(date +%H:%M:%S)"
STPP_MODEL=$M STPP_CACHE_DIR=/mnt/f/aiwork/mixed_cache_big STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=256 STPP_COMPILE=1 \
/home/lbx13/stpenv/bin/python -m stpp.pretrain \
  --out_dir runs/big/b256s8_probe --steps 60 --bs 8 --block 256 --seed 82 --log_every 20 \
  --lora_dropout 0.05 --lora_r 64 --lora_alpha 128 \
  > runs/big/b256s8_probe.log 2>&1 || echo "[b256] FAILED"
echo "[b256] done $(date +%H:%M:%S)"
grep -E '^\[' runs/big/b256s8_probe.log | tail -8