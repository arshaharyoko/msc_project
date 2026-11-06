#!/usr/bin/env bash
# ADR-045 part B: baseline and the six single-factor arms, seeds 0 and 1, folds 0 to 4, with
# checkpoints saved. Same arguments as abl_spec in run_rerun.sh minus --no-save.
#
#   WORKERS=3 bash scripts/run_ablation_checkpoints.sh
#
# Cells are dealt round-robin to WORKERS processes, each with its own log
# artifacts/ablation_runs_single_factor__p<k>.jsonl, then merged into artifacts/ablation_runs_single_factor.jsonl.
# A cell that fails twice stops that worker; running the script again continues (--resume).
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=. MPLBACKEND=Agg OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 TORCH_THREADS=3
PY=${PY:-/data/Island/ML/ml-env/bin/python}
WORKERS=${WORKERS:-3}
MEM_MIN_MB=${MEM_MIN_MB:-4000}
MEM_WAIT_S=${MEM_WAIT_S:-600}
ARMS=(baseline no_svd no_mmd no_rgcn no_gate no_cost baseline+graded)
LOGS=artifacts/adr/logs
CHAIN=$LOGS/ablation_single_factor_chain.log
mkdir -p "$LOGS"

say () { echo "$(date '+%F %T') $*" | tee -a "$CHAIN"; }

mem_ok () {
    local waited=0
    while [ "$(awk '/MemAvailable/ { printf "%d", $2 / 1024 }' /proc/meminfo)" -lt "$MEM_MIN_MB" ]; do
        [ "$waited" -ge "$MEM_WAIT_S" ] && return 1
        sleep 5; waited=$((waited + 5))
    done
}

cells=()
for fold in 0 1 2 3 4; do for seed in 0 1; do for arm in "${ARMS[@]}"; do
    cells+=("$arm $fold $seed")
done; done; done

worker () {
    local k=$1 i=0 arm fold seed try ok
    local out=artifacts/ablation_runs_single_factor__p${k}.jsonl log=$LOGS/ablation_single_factor__p${k}.log
    for cell in "${cells[@]}"; do
        if [ $((i % WORKERS)) -eq "$k" ]; then
            read -r arm fold seed <<< "$cell"
            ok=0
            for try in 1 2; do
                say "p$k start $arm fold $fold seed $seed attempt $try"
                if mem_ok >> "$log" 2>&1 && "$PY" ablation_study.py --arms "$arm" --seeds "$seed" --fold "$fold" \
                        --quiet --resume --log "$out" >> "$log" 2>&1; then ok=1; break; fi
                say "p$k FAILED $arm fold $fold seed $seed attempt $try (see $log)"
            done
            if [ "$ok" = 0 ]; then say "p$k stopped"; return 1; fi
            say "p$k done $arm fold $fold seed $seed"
            sleep 10
        fi
        i=$((i + 1))
    done
}

say "start: ${#cells[@]} cells, $WORKERS workers"
pids=()
for k in $(seq 0 $((WORKERS - 1))); do worker "$k" & pids+=($!); sleep 20; done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
if [ "$status" = 0 ]; then
    cat $(for k in $(seq 0 $((WORKERS - 1))); do echo artifacts/ablation_runs_single_factor__p${k}.jsonl; done) > artifacts/ablation_runs_single_factor.jsonl
    say "merged into artifacts/ablation_runs_single_factor.jsonl"
fi
say "finished status=$status"
exit $status
