#!/usr/bin/env bash
# Optimizer A/B on the big corpus: Muon, AdEMAMix, then the fp16/16F speed arm.
# Baseline: runs/big/steps1920_s82 (AdamW, ppl 4.004).
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
mkdir -p runs/big
exec > runs/big/driver3.log 2>&1
BIG=/mnt/f/aiwork/mixed_cache_big
M=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct

for OPT in muon ademamix; do
  echo "[train] optim=$OPT start $(date +%H:%M:%S)"
  STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
  /home/lbx13/stpenv/bin/python -m stpp.pretrain \
    --out_dir runs/big/opt_${OPT}_s82 --steps 1920 --bs 4 --seed 82 \
    --log_every 160 --lora_dropout 0.05 --optim $OPT
  echo "[train] optim=$OPT done $(date +%H:%M:%S)"
done

echo "[train] fp16/16F arm start $(date +%H:%M:%S)"
STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
/home/lbx13/stpenv/bin/python -m stpp.pretrain \
  --out_dir runs/big/fp16_s82 --steps 1920 --bs 4 --seed 82 \
  --log_every 160 --lora_dropout 0.05 --fp16 --no-bf16 \
  || echo "[warn] fp16 arm failed"
echo "[train] fp16/16F arm done $(date +%H:%M:%S)"
echo "[optab] all done $(date +%H:%M:%S)"
