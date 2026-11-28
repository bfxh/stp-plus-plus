#!/usr/bin/env bash
# Big-corpus curve: build a 3x cache (3M tokens/source) then train 1920 and 3840 steps.
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
mkdir -p runs/big
exec > runs/big/driver.log 2>&1

BIG=/mnt/f/aiwork/mixed_cache_big
if [ ! -f "$BIG/train_b512.npy" ]; then
  echo "[build] start $(date +%H:%M:%S)"
  STPP_MODEL=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct \
  /home/lbx13/stpenv/bin/python scripts/build_mixed_corpus.py \
    --src code:/mnt/f/aiwork/raw/code_val.parquet:whole_func_string \
    --src chat:/mnt/f/aiwork/raw/chat_0.parquet:messages \
    --src zh:/mnt/f/aiwork/raw/zh_2.parquet:text \
    --tokens-per-source 3000000 --out $BIG
  echo "[build] done $(date +%H:%M:%S)"
else
  echo "[build] cache already present, skip"
fi

for S in 1920 3840; do
  echo "[train] steps=$S start $(date +%H:%M:%S)"
  STPP_MODEL=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct \
  STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
  /home/lbx13/stpenv/bin/python -m stpp.pretrain \
    --out_dir runs/big/steps${S}_s82 --steps $S --bs 4 --seed 82 \
    --log_every 80 --lora_dropout 0.05
  echo "[train] steps=$S done $(date +%H:%M:%S)"
done
echo "[big] all done $(date +%H:%M:%S)"
