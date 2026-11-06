#!/usr/bin/env bash
# ADR-024: the 64 cells of the 2^6 factorial x FOLDS x SEEDS, one worker per fold (seeds in sequence),
# one log per worker, merged into artifacts/ablation_runs.jsonl at the end.
# Rerun the same command to resume: each worker appends and skips logged runs.
#
# THREADS caps torch per worker so N workers own the machine instead of
# oversubscribing it; FOLDS sets how many workers run at once.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/data/Island/ML/ml-env/bin/python}
THREADS=${THREADS:-3}
SEEDS=${SEEDS:-"0 1"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
LOGS=artifacts/adr/logs
mkdir -p "$LOGS"
export PYTHONPATH=. MPLBACKEND=Agg OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

readout_arg=()
if [ -n "${READOUT:-}" ]; then readout_arg=(--readout "$READOUT"); fi
# Fold 0's legacy arms resolve to existing checkpoint names, and the study
# refuses to overwrite them; NO_SAVE=1 runs the block without writing checkpoints.
save_arg=()
if [ "${NO_SAVE:-0}" = "1" ]; then save_arg=(--no-save); fi

mapfile -t ARMS < <("$PY" -c "import ablation_study as a; print('\n'.join(s.label for s in a.ARMS))")
echo "$(date +%T) ${#ARMS[@]} arms x folds [$FOLDS] x seeds [$SEEDS], one worker per fold, $THREADS threads each"

pids=()
# the cap is memory: about 3.6 GB per process, 30 GB box without swap, so one worker per fold
for f in $FOLDS; do
    "$PY" ablation_study.py --arms "${ARMS[@]}" --seeds $SEEDS --fold "$f" \
        --quiet --resume "${readout_arg[@]}" "${save_arg[@]}" \
        --log "artifacts/ablation_runs__f${f}.jsonl" \
        >> "$LOGS/ablation__f${f}.log" 2>&1 &
    pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
echo "$(date +%T) workers finished, failed=$failed"

merged=artifacts/ablation_runs.jsonl
if [ -e "$merged" ]; then
    echo "refusing to overwrite $merged; move it aside to merge the per-fold logs again" >&2
    exit 1
fi
cat artifacts/ablation_runs__f?.jsonl > "$merged"
echo "$(date +%T) $(wc -l < "$merged") rows in $merged"
