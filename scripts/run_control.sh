#!/usr/bin/env bash
# Run the missing control arms (gap test + MTP positive control) on Linux.
# A run that fails leaves no run.json, so the next pass retries it.
#
#   bash scripts/run_control.sh                 # 全部 7 臂（约 18 个 run）
#   STPP_CORE_ONLY=1 bash scripts/run_control.sh  # 只跑决定性三臂（9 个 run）
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
py="${STPP_PYTHON:-python3}"

core=(ce33 ce33_mtp2 mtp2)
extra=(mtp3 ce25_mtp3 mtp3_l ce25_mtp3_l)

if [ "${STPP_CORE_ONLY:-0}" = "1" ]; then
    arms=("${core[@]}")
else
    arms=("${core[@]}" "${extra[@]}")
fi

extra_args=()
if [ "${STPP_LOAD_8BIT:-0}" = "1" ]; then
    extra_args+=(--load_8bit)
fi

for pass in 1 2 3; do
    echo "=== pass $pass  $(date +%H:%M) ==="
    "$py" "$here/run_pretrain.py" \
        --arms "${arms[@]}" \
        --seeds 82 23 37 \
        --steps 120 --bs 4 \
        --out-root runs/pretrain "${extra_args[@]}" 2>&1 | grep -E "ppl=|fail" || true
    sleep 20
done

echo "--- done $(date +%H:%M)"
"$py" "$here/agg_pretrain.py"
