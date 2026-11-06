#!/usr/bin/env bash
# ADR-034: screen -> refine -> confirm -> select, resumable. The selection is not applied.
#
# Every stage is guarded by its merged output, so re-running the script after an
# interruption continues where it stopped. Shards are independent processes; the
# GPU is shared, so SHARDS trades wall clock for contention.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/data/Island/ML/ml-env/bin/python}
# the cap is memory: about 3.6 GB per process, 30 GB box without swap
SHARDS=${SHARDS:-5}
export PYTHONPATH=. MPLBACKEND=Agg OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 TORCH_THREADS=3
LOGS=artifacts/adr/logs
mkdir -p "$LOGS"

common=(--fold 1 --negatives graded --readout softmax)

stage () {
    local name=$1 out=$2; shift 2
    if [ -s "$out" ]; then echo "$(date +%T) $name already merged, skipping"; return 0; fi
    local pids=()
    for i in $(seq 0 $((SHARDS - 1))); do
        "$PY" hipo.py --stage "$name" --shard "$i/$SHARDS" "$@" \
            --out "artifacts/hipo_${name}__${i}.json" \
            >> "$LOGS/hipo_${name}__${i}.log" 2>&1 &
        pids+=($!)
    done
    local failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [ "$failed" -eq 0 ] || { echo "$(date +%T) $name shard failed; see $LOGS"; return 1; }
    "$PY" hipo.py --merge "hipo_${name}__*.json" --out "$out" \
        >> "$LOGS/hipo_${name}__merge.log" 2>&1
    echo "$(date +%T) $name merged into $out"
}

stage screen artifacts/hipo_screen.json "${common[@]}"
stage refine artifacts/hipo_refine.json --from artifacts/hipo_screen.json "${common[@]}"
stage confirm artifacts/hipo_confirm.json --from artifacts/hipo_refine.json "${common[@]}"

if [ ! -s artifacts/hipo_selection.json ]; then
    "$PY" hipo.py --select \
        --select-screen artifacts/hipo_screen.json \
        --select-refine artifacts/hipo_refine.json \
        --select-confirm artifacts/hipo_confirm.json \
        --selection-out artifacts/hipo_selection.json \
        >> "$LOGS/hipo_select.log" 2>&1
fi

echo "$(date +%T) chain finished"
