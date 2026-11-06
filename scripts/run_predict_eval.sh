#!/usr/bin/env bash
# ADR-006: predict-time cost evaluation on the baseline and no_cost checkpoints of
# the current config (ids read from the spec), 4 checkpoints at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/data/Island/ML/ml-env/bin/python}
OUT=artifacts/adr/predict_eval
mkdir -p "$OUT" artifacts/adr/logs
export PYTHONPATH=. MPLBACKEND=Agg
read -r BASE NOCOST < <("$PY" -c "
import ablation_study as a
ids = {s.label: s.config_id for s in a.ARMS}
print(ids['baseline'], ids['no_cost'])")
want=()
for s in 0 1 2 3 4 5 6 7 8 9; do
    want+=("artifacts/ablation__baseline__${BASE}__s$s.pth" "artifacts/ablation__no_cost__${NOCOST}__s$s.pth")
done
for f in "${want[@]}"; do until [ -f "$f" ]; do sleep 30; done; done
echo "$(date +%T) all ${#want[@]} checkpoints present"
running=0
for f in "${want[@]}"; do
    name=$(basename "$f" .pth)
    "$PY" benchmark_predict.py "$f" "$OUT/$name.json" > "artifacts/adr/logs/predict_eval__$name.log" 2>&1 &
    running=$((running + 1))
    if [ "$running" -ge 4 ]; then wait -n; running=$((running - 1)); fi
done
wait
"$PY" benchmark_predict.py --summarize "$OUT/*.json" > artifacts/adr/logs/predict_eval_summary.txt 2>&1
echo "$(date +%T) done"
