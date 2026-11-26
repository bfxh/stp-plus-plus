#!/usr/bin/env bash
# Run the three control arms (gap test + MTP positive control) on Linux.
# A run that fails leaves no run.json, so the next pass retries it.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
py="${STPP_PYTHON:-python3}"

arms=(ce33 ce33_mtp2 mtp2 mtp3 ce25_mtp3 mtp3_l ce25_mtp3_l)

for pass in 1 2 3; do
    echo "=== pass $pass  $(date +%H:%M) ==="
    "$py" "$here/run_pretrain.py" \
        --arms "${arms[@]}" \
        --seeds 82 23 37 \
        --steps 120 --bs 4 \
        --out-root runs/pretrain 2>&1 | grep -E "ppl=|fail" || true
    sleep 20
done

echo "--- done $(date +%H:%M)"
"$py" "$here/agg_pretrain.py"
