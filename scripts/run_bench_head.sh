#!/usr/bin/env bash
export PATH=/home/lbx13/stpenv/bin:/usr/local/bin:/usr/bin:/bin
cd /mnt/c/Users/lbx13/Doubao/chats/2026-11-24/new-chat-7/stp-plus-plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/lbx13/stpenv/bin/python scripts/bench_head.py 2>&1 | grep -v Warning