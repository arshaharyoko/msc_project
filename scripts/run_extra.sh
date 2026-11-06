#!/usr/bin/env bash
# ADR-043: graded anchor strength (5 alphas x 2 arms x 10 seeds), protein-floor scale
# (4 scales x 10 seeds) and the alpha = 1 sanity sweeps (2), 142 sweeps in all, through
# scripts/predict_extra.py. The sweeps share the GPU, MAXPROC at a time.
#
# An output that loads with the 7 registered cost weights is never recomputed, so running the
# script again continues from the first incomplete sweep. A sweep that fails twice stops the batch.
#
#   bash scripts/run_extra.sh --dry-run   print commands and counts, write nothing
#   bash scripts/run_extra.sh --check     the 2 sanity sweeps (alpha = 1, seed 0)
#   $PY scripts/check_extra.py alpha1     compare them with artifacts/optimizer/*__s0.json
#   bash scripts/run_extra.sh             the 140 sweeps; refuses to start without check_alpha1.ok
#
# Environment: PY, MAXPROC, MEM_MIN_MB, MEM_WAIT_S, SETTLE, EXPECTED_FP.
set -uo pipefail
cd "$(dirname "$0")/.."

DRY=0 CHECK=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY=1 ;;
        --check) CHECK=1 ;;
        *) echo "usage: $0 [--dry-run] [--check]" >&2; exit 2 ;;
    esac
done

export PYTHONPATH=. MPLBACKEND=Agg OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 TORCH_THREADS=3
export PY=${PY:-/data/Island/ML/ml-env/bin/python}
export MAXPROC=${MAXPROC:-4}
export MEM_MIN_MB=${MEM_MIN_MB:-4000}
export MEM_WAIT_S=${MEM_WAIT_S:-600}
export SETTLE=${SETTLE:-10}
export EXPECTED_FP=${EXPECTED_FP:-ef513dfd5a24111c}
export CKPT_ID=728320fe947b
export SEEDS="0 1 2 3 4 5 6 7 8 9"
export ALPHAS="0.1 0.25 0.5 0.75 0.9"
export SCALES="1 2 5 base"

export OUTDIR=artifacts/optimizer_sweeps LOGS=artifacts/adr/logs STATE=/tmp/opencode/extra_state
export CHAIN_LOG=$LOGS/predict_extra_chain.log
export DRY
CHECKER="$PY scripts/check_extra.py"
export CHECKER

say () {
    local line="$(date '+%F %T') $*"
    echo "$line"
    if [ "$DRY" = 0 ]; then echo "$line" >> "$CHAIN_LOG"; fi
}
export -f say

trap 'say "interrupted"; pkill -f scripts/predict_extra.py; exit 143' INT TERM HUP

if [ "$DRY" = 0 ]; then
    mkdir -p "$LOGS" "$STATE" "$OUTDIR"
    jobs_file=$STATE/jobs
else
    jobs_file=$(mktemp)
    trap 'rm -f "$jobs_file"' EXIT
fi

# One line per sweep: name, env variable, its value, protected (0/1), seed
if [ "$CHECK" = 1 ]; then
    for arm in free prot; do
        p=0; [ "$arm" = prot ] && p=1
        echo "check_alpha1_${arm}__s0 ANCHOR_ALPHA 1 $p 0"
    done > "$jobs_file"
else
    {
        for a in $ALPHAS; do for arm in free prot; do for s in $SEEDS; do
            p=0; [ "$arm" = prot ] && p=1
            echo "latent_alpha${a}_${arm}__s${s} ANCHOR_ALPHA $a $p $s"
        done; done; done
        for f in $SCALES; do for s in $SEEDS; do
            echo "latent_floor${f}__s${s} FLOOR_SCALE $f 1 $s"
        done; done
    } > "$jobs_file"
fi

job_cmd () {
    local name=$1 var=$2 val=$3 prot=$4 seed=$5
    CMD=(env "$var=$val" "$PY" scripts/predict_extra.py
         --checkpoint "artifacts/ablation__selected__${CKPT_ID}__s${seed}__graded.pth"
         --sweep --decode anchored)
    if [ "$prot" = 1 ]; then CMD+=(--protect-proteins); fi
    CMD+=(--out "optimizer_sweeps/${name}.json")
    OUT=$OUTDIR/${name}.json
    LOG=$LOGS/predict_extra_${name}.log
}
export -f job_cmd

if [ "$DRY" = 1 ]; then
    n=$(wc -l < "$jobs_file")
    say "dry run: $n sweeps, $MAXPROC at a time, outputs under $OUTDIR, logs $LOGS/predict_extra_<name>.log"
    say "  by kind: alpha $(grep -c ' ANCHOR_ALPHA ' "$jobs_file"), floor $(grep -c ' FLOOR_SCALE ' "$jobs_file"); optimizations $((n * 7))"
    say "  fingerprint check ($EXPECTED_FP) and CUDA check skipped in a dry run; checkpoints present: $(ls artifacts/ablation__selected__${CKPT_ID}__s{0..9}__graded.pth 2>/dev/null | wc -l)/10"
    first=$(head -1 "$jobs_file"); last=$(tail -1 "$jobs_file")
    for line in "$first" "$last"; do
        read -r name var val prot seed <<< "$line"
        job_cmd "$name" "$var" "$val" "$prot" "$seed"
        echo "  ${CMD[*]} >> $LOG"
    done
    if [ "$CHECK" = 0 ] && [ ! -e "$OUTDIR/check_alpha1.ok" ]; then
        say "  note: a real run stops at once, $OUTDIR/check_alpha1.ok is missing (run --check, then check_extra.py alpha1)"
    fi
    exit 0
fi

if [ "$CHECK" = 0 ] && [ ! -e "$OUTDIR/check_alpha1.ok" ]; then
    echo "$OUTDIR/check_alpha1.ok is missing: run --check, then $CHECKER alpha1" >&2
    exit 1
fi

exec 8>"$STATE/chain.lock"
if ! flock -n 8; then echo "another extra batch holds $STATE/chain.lock" >&2; exit 1; fi
rm -f "$STATE/stop" "$STATE/failed"

if [ ! -f data/database_graded.pt ] || [ data/database_graded.json -nt data/database_graded.pt ]; then
    echo "data/database_graded.pt is missing or older than the JSON; each process would parse the JSON" >&2
    exit 1
fi

read -r device fp nrec < <($PY scripts/rerun_tools.py fingerprint)
say "device=$device dataset_fingerprint=$fp records=$nrec python=$PY threads=3 check=$CHECK"
case $device in
    cuda:*) ;;
    *) say "abort: CUDA is not available (device=$device)"; exit 1 ;;
esac
if [ "$fp" != "$EXPECTED_FP" ]; then
    say "abort: dataset fingerprint $fp, expected $EXPECTED_FP"
    exit 1
fi

mem_available_mb () { awk '/MemAvailable/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
export -f mem_available_mb

wait_mem () {
    local waited=0
    while [ "$(mem_available_mb)" -lt "$MEM_MIN_MB" ]; do
        if [ "$waited" -ge "$MEM_WAIT_S" ]; then
            say "memory guard: MemAvailable $(mem_available_mb) MB < $MEM_MIN_MB MB after ${MEM_WAIT_S}s"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
}
export -f wait_mem

# One launch at a time, then SETTLE seconds before the next guard reads MemAvailable, so the
# new process has been counted. A guard timeout is a failed attempt.
attempt () {
    local log=$1; shift
    exec 9>"$STATE/launch.lock"
    flock 9
    if ! wait_mem >> "$log" 2>&1; then
        flock -u 9; exec 9>&-
        return 1
    fi
    "$@" >> "$log" 2>&1 9>&- &
    local pid=$!
    sleep "$SETTLE"
    flock -u 9; exec 9>&-
    wait "$pid"
}
export -f attempt

run_job () {
    local name var val prot seed OUT LOG CMD try
    read -r name var val prot seed <<< "$1"
    [ -e "$STATE/stop" ] && return 0
    job_cmd "$name" "$var" "$val" "$prot" "$seed"
    if $CHECKER validate "$OUT" 2>/dev/null; then say "skip $OUT (valid)"; return 0; fi
    rm -f "$OUT"
    for try in 1 2; do
        [ -e "$STATE/stop" ] && return 1
        say "start $name attempt $try"
        if attempt "$LOG" "${CMD[@]}" && $CHECKER validate "$OUT" 2>>"$LOG"; then
            say "done $name"
            return 0
        fi
        say "FAILED $name attempt $try (see $LOG)"
        rm -f "$OUT"
    done
    echo "$name" >> "$STATE/failed"
    touch "$STATE/stop"
    return 1
}
export -f run_job

n=$(wc -l < "$jobs_file")
say "batch: $n sweeps, $MAXPROC at a time"
start=$SECONDS
xargs -a "$jobs_file" -d '\n' -P "$MAXPROC" -I{} bash -c 'run_job "$1"' _ {}
wall=$((SECONDS - start))

if [ -e "$STATE/stop" ]; then
    say "batch stopped after ${wall}s: a sweep failed twice. Failed:"
    sort -u "$STATE/failed" 2>/dev/null | while read -r f; do say "  $f"; done
    exit 1
fi
missing=0
while read -r name _; do
    $CHECKER validate "$OUTDIR/$name.json" 2>/dev/null || { say "missing or invalid: $name"; missing=$((missing + 1)); }
done < "$jobs_file"
if [ "$missing" != 0 ]; then say "batch incomplete: $missing of $n sweeps without a valid output"; exit 1; fi
say "extra batch finished: $n sweeps valid, wall ${wall}s"
