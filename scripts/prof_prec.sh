#!/usr/bin/env bash
# Profile only: 5 steps, dump the CUDA kernel table for bf16 vs fp16 at r64.
# Purpose: prove/disprove that the residual fp32-accumulate GEMMs sit in the backward.
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
mkdir -p runs/big
BIG=/mnt/f/aiwork/mixed_cache_big
M=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct
COMMON="--steps 8 --bs 4 --lora_r 64 --lora_alpha 128 --lora_dropout 0.05"

echo "[prof] bf16 start $(date +%H:%M:%S)"
STPP_PROFILE=1 STPP_COMPILE=1 STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 \
STPP_LOGIT_CHUNK=512 /home/lbx13/stpenv/bin/python -m stpp.pretrain \
  --out_dir runs/big/prof_bf16_r64 $COMMON > runs/big/prof_bf16_r64.log 2>&1 || true
echo "[prof] bf16 done $(date +%H:%M:%S)"

echo "[prof] fp16 start $(date +%H:%M:%S)"
STPP_PROFILE=1 STPP_COMPILE=1 STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 \
STPP_LOGIT_CHUNK=512 /home/lbx13/stpenv/bin/python -m stpp.pretrain \
  --out_dir runs/big/prof_fp16_r64 $COMMON --fp16 --no-bf16 > runs/big/prof_fp16_r64.log 2>&1 || true
echo "[prof] fp16 done $(date +%H:%M:%S)"
echo "[prof] ALL DONE"
