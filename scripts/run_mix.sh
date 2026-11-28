#!/usr/bin/env bash
# Mix arm: 4:4:2 over (code, chat, zh) at the current best capacity point (r64, 3840 steps).
#
# Why 4:4:2: per-domain ppl at r64@3840 is code 5.321 / chat 4.885 / zh 2.419.
# DoReMi allocates w_d ~ (L_d - L_d^inf); zh is already near its floor, code/chat are not.
# Baseline for comparison: r64@3840 = 3.866 (1:1:1).
set -e
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
REPO=/mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
cd "$REPO"
mkdir -p runs/big
exec > runs/big/driver9.log 2>&1
BIG=/mnt/f/aiwork/mixed_cache_big
M=/mnt/f/aiwork/models/Llama-3.2-1B-Instruct
IDX=/mnt/f/aiwork/mix_idx/mix442.npy

if [ ! -f "$IDX" ]; then
  echo "[mix] building index $(date +%H:%M:%S)"
  STPP_MODEL=$M /home/lbx13/stpenv/bin/python scripts/make_mix_index.py \
    --cache "$BIG" --weights 4:4:2 --out "$IDX"
fi

echo "[mix] r64@3840 mix=4:4:2 start $(date +%H:%M:%S)"
STPP_MODEL=$M STPP_CACHE_DIR=$BIG STPP_SKIP_SAVE=1 STPP_LOGIT_CHUNK=512 STPP_COMPILE=1 \
/home/lbx13/stpenv/bin/python -m stpp.pretrain \
  --out_dir runs/big/mix442_s3840_s82 --steps 3840 --bs 4 --seed 82 \
  --log_every 160 --lora_dropout 0.05 --lora_r 64 --lora_alpha 128 \
  --train_ids "$IDX" --domains 3 --domain_names code,chat,zh \
  || echo "[warn] mix arm failed"
echo "[mix] done $(date +%H:%M:%S)"
