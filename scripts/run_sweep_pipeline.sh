#!/usr/bin/env bash
# ADR-004: screen -> refine -> confirm -> select, each stage in 6 parallel shards.
# Run from the repo root. Skip a stage that is already merged by leaving its
# merged json in artifacts/; set SKIP_SCREEN=1 when the screen shards are
# already running or done.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/data/Island/ML/ml-env/bin/python}
N=${N:-6}
LOGS=artifacts/adr/logs
mkdir -p "$LOGS"
export PYTHONPATH=. MPLBACKEND=Agg

stage() {  # stage <name> [--from file]
    local name=$1; shift
    local pids=()
    for i in $(seq 0 $((N - 1))); do
        "$PY" hipo.py --stage "$name" --shard "$i/$N" --out "hipo_${name}__$i.json" "$@" \
            > "$LOGS/hipo_${name}__$i.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

merge() {
    "$PY" hipo.py --merge "hipo_$1__*.json" --out "hipo_$1.json" > "$LOGS/hipo_$1__merge.log" 2>&1
    echo "$(date +%T) merged $1: $(grep -c . "$LOGS/hipo_$1__merge.log") report lines"
}

if [ "${SKIP_SCREEN:-0}" != "1" ]; then
    stage screen
else
    while pgrep -f "hipo.py --stage screen" > /dev/null; do sleep 20; done
fi
merge screen

stage refine --from hipo_screen.json
merge refine

stage confirm --from hipo_refine.json
merge confirm

"$PY" hipo.py --select > "$LOGS/hipo_select.log" 2>&1
echo "$(date +%T) selection written"
cat "$LOGS/hipo_select.log"
