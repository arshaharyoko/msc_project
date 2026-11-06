#!/usr/bin/env bash
# ADR-041: the corrected sweep and ablation on the GPU, one restartable chain:
#   screen -> refine -> confirm -> select -> 640-cell ablation -> latent figures.
#
# hipo.py writes a stage's rows only when a shard ends, so each stage is cut into shards of
# at most MAX_ROWS rows and run from a job pool of at most MAXPROC processes. An output that
# loads and has the expected rows is never recomputed, so running the script again continues
# from the first incomplete step. A shard that fails twice stops the chain.
#
#   bash scripts/run_rerun.sh --dry-run   print commands, shard counts, partition
#   bash scripts/run_rerun.sh             run the chain
#   bash scripts/run_rerun.sh --smoke     one screen row (smallest tsvd_k) under /tmp/opencode/smoke
#
# Environment: EXPECTED_FP, MEM_MIN_MB, MEM_WAIT_S, SMOKE_EPOCHS (smoke only), MAXPROC, MAX_ROWS.
set -uo pipefail
cd "$(dirname "$0")/.."

DRY=0 SMOKE=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY=1 ;;
        --smoke) SMOKE=1 ;;
        *) echo "usage: $0 [--dry-run] [--smoke]" >&2; exit 2 ;;
    esac
done

export PYTHONPATH=. MPLBACKEND=Agg OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 TORCH_THREADS=3
export PY=${PY:-/data/Island/ML/ml-env/bin/python}
export MAXPROC=${MAXPROC:-4}
export MAX_ROWS=${MAX_ROWS:-11}
export MEM_MIN_MB=${MEM_MIN_MB:-4000}
export MEM_WAIT_S=${MEM_WAIT_S:-600}
export SETTLE=${SETTLE:-10}
export EXPECTED_FP=${EXPECTED_FP:-ef513dfd5a24111c}
export COMMON_ARGS="--fold 1 --negatives graded --readout softmax"
export ABL_FOLDS="0 1 2 3 4" ABL_SEEDS="0 1"
ABL_PROCS=$MAXPROC

export OUTDIR=artifacts LOGS=artifacts/adr/logs STATE=/tmp/opencode/rerun_state
EPOCHS_ARG=""
if [ "$SMOKE" = 1 ]; then
    OUTDIR=/tmp/opencode/smoke LOGS=/tmp/opencode/smoke/logs STATE=/tmp/opencode/smoke/state
    if [ -n "${SMOKE_EPOCHS:-}" ]; then EPOCHS_ARG="--epochs $SMOKE_EPOCHS"; fi
fi
export EPOCHS_ARG
export CHAIN_LOG=$LOGS/rerun_chain.log
RSS_LOG=$LOGS/rerun_rss.log
TOOLS_STR="$PY scripts/rerun_tools.py"
export TOOLS_STR
TOOLS=($TOOLS_STR)
export DRY

say () {
    local line="$(date '+%F %T') $*"
    echo "$line"
    if [ "$DRY" = 0 ]; then echo "$line" >> "$CHAIN_LOG"; fi
}
export -f say

descendants () {
    ps -eo pid=,ppid=,rss=,comm= | awk -v root="$1" '
        { pp[$1] = $2; rss[$1] = $3; comm[$1] = $4 }
        END { for (p in pp) { q = p
                while (q != "" && q != 0) { if (q == root) { print p, rss[p], comm[p]; break }; q = pp[q] } } }'
}
export -f descendants

ROOTPID=$$
RSS_PID=""
rss_loop () {
    while :; do
        local sample gpu
        sample=$(descendants "$ROOTPID" | awk '$3 ~ /^python/ { n++; s += $2 }
            END { printf "procs=%d rss_mb=%.0f", n, s / 1024 }')
        gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>&1 | head -1)
        echo "$(date '+%F %T') $sample mem_available_mb=$(awk '/MemAvailable/ { printf "%d", $2 / 1024 }' /proc/meminfo) gpu_util_pct,mem_used_mib,mem_total_mib=$gpu" >> "$RSS_LOG"
        sleep 600
    done
}

kill_children () {
    descendants "$ROOTPID" | awk '$3 ~ /^python/ { print $1 }' | xargs -r kill 2>/dev/null
}
cleanup () {
    if [ -n "$RSS_PID" ]; then pkill -P "$RSS_PID" 2>/dev/null; kill "$RSS_PID" 2>/dev/null; fi
}
trap cleanup EXIT
trap 'say "interrupted"; kill_children; exit 143' INT TERM HUP

mkdir -p "$LOGS" "$STATE" "$OUTDIR"

# two chains would race for the same shard files
exec 8>"$STATE/chain.lock"
if ! flock -n 8; then echo "another chain holds $STATE/chain.lock" >&2; exit 1; fi
rm -f "$STATE/stop" "$STATE/failed"

# the loader reads the float32 sidecar only when it is not older than the JSON
if [ ! -f data/database_graded.pt ] || [ data/database_graded.json -nt data/database_graded.pt ]; then
    echo "data/database_graded.pt is missing or older than the JSON; each process would parse the JSON" >&2
    exit 1
fi

read -r device fp nrec < <("${TOOLS[@]}" fingerprint)
say "device=$device dataset_fingerprint=$fp records=$nrec python=$PY threads=3 dry_run=$DRY smoke=$SMOKE"
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

# Sets OUT, LOG and the arrays CMD, CHECK for one hipo shard.
hipo_spec () {
    local stage=$1 i=$2 total=$3 n=$4 src=()
    case $stage in
        refine) src=(--from "$OUTDIR/hipo_screen.json") ;;
        confirm) src=(--from "$OUTDIR/hipo_refine.json") ;;
    esac
    OUT=$OUTDIR/hipo_${stage}__${i}.json
    LOG=$LOGS/hipo_${stage}__${i}.log
    CMD=("$PY" hipo.py --stage "$stage" --shard "$i/$n" "${src[@]}" $COMMON_ARGS $EPOCHS_ARG --out "$OUT")
    CHECK=($TOOLS_STR validate "$OUT" "$stage" "$total" "$n" "$i")
}
export -f hipo_spec

abl_spec () {
    local k=$1 fold=$2 seed=$3
    OUT=$OUTDIR/ablation_runs__p${k}.jsonl
    LOG=$LOGS/ablation__p${k}.log
    # --no-save: 640 checkpoints of about 50 MB would be 32 GB, and fold 0 names collide with the deployed ones
    CMD=("$PY" ablation_study.py --arms $(cat "$STATE/arms_p${k}.txt") --seeds "$seed" --fold "$fold"
         --quiet --resume --no-save --log "$OUT")
    CHECK=($TOOLS_STR validate-ablation "$OUT" "$STATE/arms_p${k}.txt" --folds $ABL_FOLDS --seeds $ABL_SEEDS)
}
export -f abl_spec

# retry once; a second failure raises the stop flag
run_with_retry () {
    local label=$1 log=$2; shift 2
    local try
    for try in 1 2; do
        [ -e "$STATE/stop" ] && return 1
        say "start $label attempt $try"
        if attempt "$log" "$@"; then say "done $label"; return 0; fi
        say "FAILED $label attempt $try (see $log)"
    done
    echo "$label" >> "$STATE/failed"
    touch "$STATE/stop"
    return 1
}
export -f run_with_retry

run_job () {
    local OUT LOG CMD CHECK kind a b c d
    read -r kind a b c d <<< "$1"
    [ -e "$STATE/stop" ] && return 0
    case $kind in
        hipo)
            hipo_spec "$a" "$b" "$c" "$d"
            if "${CHECK[@]}" 2>/dev/null; then say "skip $OUT (valid)"; return 0; fi
            rm -f "$OUT"
            run_with_retry "$OUT" "$LOG" "${CMD[@]}"
            [ -s "$OUT" ] && "${CHECK[@]}" 2>>"$LOG" || { say "FAILED $OUT did not validate"; echo "$OUT" >> "$STATE/failed"; touch "$STATE/stop"; return 1; }
            ;;
        abl)
            local k=$a f s
            abl_spec "$k" 0 0
            if "${CHECK[@]}" 2>/dev/null; then say "skip $OUT (all cells logged)"; return 0; fi
            for f in $ABL_FOLDS; do for s in $ABL_SEEDS; do
                abl_spec "$k" "$f" "$s"
                run_with_retry "ablation p$k fold $f seed $s" "$LOG" "${CMD[@]}" || return 1
            done; done
            abl_spec "$k" 0 0
            "${CHECK[@]}" 2>>"$LOG" || { say "FAILED $OUT incomplete after all folds"; echo "$OUT" >> "$STATE/failed"; touch "$STATE/stop"; return 1; }
            ;;
    esac
}
export -f run_job

pool () {
    xargs -a "$1" -d '\n' -P "$MAXPROC" -I{} bash -c 'run_job "$1"' _ {}
}

stop_if_failed () {
    if [ -e "$STATE/stop" ]; then
        say "chain stopped: a job failed twice. Failed:"
        sort -u "$STATE/failed" 2>/dev/null | while read -r f; do say "  $f"; done
        if [ -n "${1:-}" ]; then say "  shards without a valid output: $1"; fi
        exit 1
    fi
}


backup_step () {
    local bk=artifacts/adr/backup_pre_rerun/his_lsc_oct29
    local have; have=$(find paper/his_lsc -type f | wc -l)
    if [ "$DRY" = 1 ]; then
        if [ -d "$bk" ]; then say "backup $bk present"; else say "backup: cp -r paper/his_lsc $bk ($have files)"; fi
        return 0
    fi
    if [ ! -d "$bk" ]; then mkdir -p "$(dirname "$bk")" && cp -r paper/his_lsc "$bk"; fi
    local got; got=$(find "$bk" -type f | wc -l)
    if [ "$got" != "$have" ]; then say "abort: backup holds $got files, paper/his_lsc $have"; exit 1; fi
    say "backup $bk verified, $got files"
}

hipo_stage () {
    local stage=$1 only=${2:-}
    local merged=$OUTDIR/hipo_${stage}.json src="" total n
    case $stage in
        refine) src=$OUTDIR/hipo_screen.json ;;
        confirm) src=$OUTDIR/hipo_refine.json ;;
    esac

    if [ -n "$src" ] && [ ! -s "$src" ]; then
        local assumed=192; [ "$stage" = confirm ] && assumed=30
        n=$(( (assumed + MAX_ROWS - 1) / MAX_ROWS ))
        say "stage $stage: row count computed after $(basename "$src") is merged; assuming $assumed rows -> $n shards of at most $MAX_ROWS rows"
        return 0
    fi
    total=$("${TOOLS[@]}" rows "$stage" ${src:+"$src"}) || { say "abort: cannot count $stage rows"; exit 1; }
    n=$("${TOOLS[@]}" shards "$total" "$MAX_ROWS")
    if [ -n "$only" ]; then n=$total; fi

    if [ "$DRY" = 1 ]; then
        say "stage $stage: $total rows, ceil($total/$MAX_ROWS) = $n shards, $((total / n)) to $(((total + n - 1) / n)) rows per shard, merged $merged"
        local i OUT LOG CMD CHECK
        for i in 0 $((n - 1)); do hipo_spec "$stage" "$i" "$total" "$n"; echo "  ${CMD[*]} >> $LOG"; done
        echo "  merge: $PY hipo.py --merge \"hipo_${stage}__*.json\" --out $merged"
        return 0
    fi

    if "${TOOLS[@]}" validate-merged "$merged" "$stage" "$total" 2>/dev/null; then
        say "stage $stage already merged, skipping"
        return 0
    fi
    "${TOOLS[@]}" plan "$OUTDIR" "$stage" "$total" "$n" "$MAX_ROWS" 2> >(tee -a "$CHAIN_LOG" >&2) \
        || { say "abort: shard plan for $stage conflicts with files on disk"; exit 1; }
    say "stage $stage: $total rows, $n shards of $((total / n)) to $(((total + n - 1) / n)) rows"
    local jobs=$STATE/jobs_$stage i
    if [ -n "$only" ]; then echo "hipo $stage $only $total $n" > "$jobs"
    else for i in $(seq 0 $((n - 1))); do echo "hipo $stage $i $total $n"; done > "$jobs"; fi
    pool "$jobs"
    stop_if_failed "$("${TOOLS[@]}" missing "$OUTDIR" "$stage" "$total" "$n")"
    if [ -n "$only" ]; then return 0; fi
    "${TOOLS[@]}" checkset "$OUTDIR" "$stage" "$total" "$n" 2> >(tee -a "$CHAIN_LOG" >&2) \
        || { say "chain stopped: $stage shard set is incomplete or mixed"; exit 1; }
    "$PY" hipo.py --merge "hipo_${stage}__*.json" --out "$merged" >> "$LOGS/hipo_${stage}__merge.log" 2>&1 \
        && "${TOOLS[@]}" validate-merged "$merged" "$stage" "$total" \
        || { say "chain stopped: merge of $stage failed (see $LOGS/hipo_${stage}__merge.log)"; exit 1; }
    say "stage $stage merged into $merged"
}

select_step () {
    local out=$OUTDIR/hipo_selection.json
    local cmd=("$PY" hipo.py --select
        --select-screen "$OUTDIR/hipo_screen.json"
        --select-refine "$OUTDIR/hipo_refine.json"
        --select-confirm "$OUTDIR/hipo_confirm.json"
        --selection-out "$out")
    if [ "$DRY" = 1 ]; then say "select: ${cmd[*]} >> $LOGS/hipo_select.log"; return 0; fi
    if "$PY" -c "import json,sys; json.load(open(sys.argv[1]))['chosen']" "$out" 2>/dev/null; then
        say "selection $out exists, skipping"; return 0
    fi
    "${cmd[@]}" >> "$LOGS/hipo_select.log" 2>&1 \
        || { say "chain stopped: --select failed (see $LOGS/hipo_select.log)"; exit 1; }
    say "selection written to $out (not applied to src/config.py)"
}

ablation_step () {
    local merged=$OUTDIR/ablation_runs.jsonl cells=640 k
    "${TOOLS[@]}" partition "$STATE" "$ABL_PROCS" > "$STATE/partition.txt" || { say "abort: ablation partition failed"; exit 1; }
    local narms=0 sizes=""
    while read -r k n; do sizes="$sizes p$k=$n"; narms=$((narms + n)); done < "$STATE/partition.txt"

    if [ "$DRY" = 1 ]; then
        local per=$(( narms / ABL_PROCS * 10 ))
        say "ablation: $narms arms x 5 folds x 2 seeds = $((narms * 10)) cells; $ABL_PROCS processes by arm (round-robin), arms per process:$sizes; $per cells each"
        say "  estimate at 204 s per cell with $ABL_PROCS processes: $per x 204 s = $((per * 204)) s = $(awk "BEGIN { printf \"%.1f\", $per * 204 / 3600 }") h"
        for k in $(seq 0 $((ABL_PROCS - 1))); do
            local OUT LOG CMD CHECK; abl_spec "$k" "<fold>" "<seed>"
            echo "  p$k (folds $ABL_FOLDS x seeds $ABL_SEEDS, log $OUT): ${CMD[*]} >> $LOG"
        done
        say "  merge: cat $OUTDIR/ablation_runs__p?.jsonl > $merged ($cells cells)"
        return 0
    fi

    if [ "$narms" != 64 ]; then say "abort: ablation_study has $narms arms, expected 64"; exit 1; fi
    if "${TOOLS[@]}" validate-ablation-merged "$merged" "$cells" 2>/dev/null; then
        say "ablation already merged ($cells cells), skipping"; return 0
    fi
    local jobs=$STATE/jobs_ablation
    for k in $(seq 0 $((ABL_PROCS - 1))); do echo "abl $k"; done > "$jobs"
    say "ablation: $cells cells over $ABL_PROCS processes"
    pool "$jobs"
    stop_if_failed
    local parts=()
    for k in $(seq 0 $((ABL_PROCS - 1))); do parts+=("$OUTDIR/ablation_runs__p${k}.jsonl"); done
    cat "${parts[@]}" > "$merged.tmp"
    "${TOOLS[@]}" validate-ablation-merged "$merged.tmp" "$cells" \
        || { say "chain stopped: merged ablation log has the wrong cell count"; rm -f "$merged.tmp"; exit 1; }
    mv "$merged.tmp" "$merged"
    say "ablation merged into $merged"
}

latent_step () {
    local out=$OUTDIR/latent_characterization.json log=$LOGS/latent.log
    if [ "$DRY" = 1 ]; then
        say "latent: $PY scripts/latent_figures.py >> $log (figures in $OUTDIR/figures/latent/, thresholds via hipo_sweep.json if present, else src/config.py)"
        return 0
    fi
    if [ -s "$out" ] && [ -s "$OUTDIR/latent_cluster_recipes.json" ]; then
        say "latent outputs exist, skipping"; return 0
    fi
    run_with_retry "latent figures" "$log" "$PY" scripts/latent_figures.py
    if [ ! -s "$out" ]; then say "chain stopped: latent figures failed twice (see $log)"; exit 1; fi
    say "latent figures written to $OUTDIR/figures/latent/"
}

if [ "$DRY" = 0 ]; then
    rss_loop > /dev/null 2>&1 &
    RSS_PID=$!
fi

if [ "$SMOKE" = 1 ]; then
    idx=$("${TOOLS[@]}" smallest-k-index | awk '{ print $2 }')
    say "smoke: screen shard $idx of the one-row-per-shard split"
    hipo_stage screen "$idx"
    say "smoke finished"
    exit 0
fi

backup_step
hipo_stage screen
hipo_stage refine
hipo_stage confirm
select_step
ablation_step
latent_step
say "chain finished"
