#!/usr/bin/env bash
# ADR-005: all 32 cells of the 2^5 factorial x SEEDS, split over N processes by
# arm, one log per process, concatenated into artifacts/ablation_runs__cost.jsonl.
# Reads the coefficients from src/config.py, so update that first (ADR-004).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/data/Island/ML/ml-env/bin/python}
N=${N:-6}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
LOGS=artifacts/adr/logs
mkdir -p "$LOGS"
export PYTHONPATH=. MPLBACKEND=Agg

mapfile -t ARMS < <("$PY" -c "import ablation_study as a; print('\n'.join(s.label for s in a.ARMS))")
echo "$(date +%T) ${#ARMS[@]} arms x seeds [$SEEDS] over $N processes"

pids=()
for i in $(seq 0 $((N - 1))); do
    slice=()
    for j in "${!ARMS[@]}"; do
        if [ $((j % N)) -eq "$i" ]; then slice+=("${ARMS[$j]}"); fi
    done
    # shellcheck disable=SC2086
    "$PY" ablation_study.py --arms "${slice[@]}" --seeds $SEEDS --quiet \
        --log "artifacts/ablation_runs__cost__$i.jsonl" > "$LOGS/ablation__$i.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

cat artifacts/ablation_runs__cost__*.jsonl > artifacts/ablation_runs__cost.jsonl
echo "$(date +%T) $(wc -l < artifacts/ablation_runs__cost.jsonl) rows"
"$PY" ablation_study.py --table --figures --log artifacts/ablation_runs__cost.jsonl \
    > "$LOGS/ablation_report__test.txt" 2>&1
"$PY" ablation_study.py --table --split val --log artifacts/ablation_runs__cost.jsonl \
    > "$LOGS/ablation_report__val.txt" 2>&1
echo "$(date +%T) report written"
