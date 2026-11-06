#!/usr/bin/env bash
# Rebuild the deployed checkpoints and the numbers derived from them, one stage per command.
#
#   bash scripts/reproduce.sh data       check raw inputs, rebuild the dataset into a temp dir, compare
#   bash scripts/reproduce.sh train      retrain seeds 0 to 9 into $RETRAINED (about 19 min on the RTX 5060 Ti)
#   bash scripts/reproduce.sh ablation   factorial analysis from the logs; the 9 h ablation rerun only prints its plan
#   bash scripts/reproduce.sh latent     latent curves of the ten deployed checkpoints (under a minute)
#   bash scripts/reproduce.sh optimize   prints the optimizer commands; set FULL=1 to run them (hours)
#   bash scripts/reproduce.sh figures    prints the figure commands; set FULL=1 to run them
#   bash scripts/reproduce.sh verify     checksums, factorial.json byte check, retrained vs deployed checkpoints
#
# Stages that take hours are dry runs unless FULL=1. Environment: PY, RETRAINED, FULL.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=. MPLBACKEND=Agg
PY=${PY:-/data/Island/ML/ml-env/bin/python}
RETRAINED=${RETRAINED:-artifacts/retrained}
FULL=${FULL:-0}
TMP=${TMPDIR:-/tmp}/reproduce_$$

run() {
    echo "+ $*"
    if [ "$FULL" = 1 ] || [ "${DRY:-0}" = 0 ]; then "$@"; fi
}

stage_data() {
    sha256sum -c artifacts/inputs.sha256
    mkdir -p "$TMP"
    trap 'rm -rf "$TMP"' EXIT
    "$PY" preprocess_database.py --merged data/database_with_cost.json \
        --out "$TMP/pre.json" --scaling-stats "$TMP/scaling_stats.pt" \
        --cost-stats "$TMP/cost_stats.pt" --prices "$TMP/component_prices.json"
    cmp "$TMP/component_prices.json" artifacts/component_prices.json
    "$PY" -I - "$TMP" <<'PYEOF'
import json, sys, torch
tmp = sys.argv[1]
a, b = json.load(open(f"{tmp}/pre.json")), json.load(open("data/database_preprocessed_cost.json"))
assert len(a) == len(b)
for p, q in zip(a, b):
    for c, d in zip(p["components"], q["components"]):
        assert c["component"] == d["component"] and c["component_vec"] == d["component_vec"]
for name in ("scaling_stats", "cost_stats"):
    x = torch.load(f"{tmp}/{name}.pt", weights_only=False)
    y = torch.load(f"artifacts/{name}.pt", weights_only=False)
    assert all(torch.equal(torch.as_tensor(x[k]), torch.as_tensor(y[k])) for k in x), name
print("embeddings, scaling stats, cost stats and prices match the stored files")
PYEOF
    "$PY" -m src.modules.negatives --out "$TMP/graded.json"
    cmp "$TMP/graded.json" data/database_graded.json
    "$PY" scripts/compact_dataset.py --out "$TMP/graded.pt" "$TMP/graded.json"
    "$PY" -I - "$TMP" <<'PYEOF'
import sys, torch
a = torch.load(f"{sys.argv[1]}/graded.pt", weights_only=False)
b = torch.load("data/database_graded.pt", weights_only=False)
def same(x, y):
    if torch.is_tensor(x): return torch.equal(x, y)
    if isinstance(x, dict): return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)): return len(x) == len(y) and all(same(p, q) for p, q in zip(x, y))
    return x == y
assert same(a, b), "database_graded.pt contents differ"
print("database_graded.json is byte identical and database_graded.pt has equal contents")
PYEOF
    sha256sum -c artifacts/datasets.sha256
    echo "not run here: scripts/data_pipeline/01_names_to_smiles and the live price calls in 03_cost need network access and API keys (.env.example)"
}

stage_train() {
    run "$PY" scripts/train_selected.py --all --out-dir "$RETRAINED"
}

stage_ablation() {
    bash scripts/run_rerun.sh --dry-run
    if [ "$FULL" = 1 ]; then bash scripts/run_rerun.sh; fi
    "$PY" scripts/ablation_factorial.py
}

stage_latent() {
    run "$PY" scripts/latent_curves.py
}

stage_optimize() {
    DRY=$((1 - FULL))
    export DRY
    for k in 0 1 2 3 4 5 6 7 8 9; do
        ck=artifacts/ablation__selected__728320fe947b__s${k}__graded.pth
        run "$PY" predict.py --checkpoint "$ck" --sweep --out optimizer/latent_raw__s${k}.json
        run "$PY" predict.py --checkpoint "$ck" --sweep --decode anchored --out optimizer/latent_anch__s${k}.json
        run "$PY" predict.py --checkpoint "$ck" --sweep --decode anchored --protect-proteins --out optimizer/latent_prot__s${k}.json
        run "$PY" predict.py --mode input --checkpoint "$ck" --out optimizer/input__s${k}.json
    done
    bash scripts/run_extra.sh --dry-run
}

stage_figures() {
    DRY=$((1 - FULL))
    export DRY
    run "$PY" scripts/ablation_factorial_figures.py
    run "$PY" scripts/figures_optimization_existing.py
    run "$PY" scripts/figures_optimization_main.py
    run "$PY" scripts/optimization_table.py
    run "$PY" scripts/figures_sweep.py
    run "$PY" scripts/supp_refs.py --write
}

stage_verify() {
    (cd artifacts && sha256sum -c checkpoints.sha256)
    sha256sum -c artifacts/inputs.sha256
    mkdir -p "$TMP"
    trap 'rm -rf "$TMP"' EXIT
    "$PY" scripts/ablation_factorial.py --out-json "$TMP/factorial.json" --out-md "$TMP/factorial.md"
    cmp "$TMP/factorial.json" artifacts/ablation_analysis/factorial.json
    "$PY" scripts/supp_refs.py --check
    if [ ! -d "$RETRAINED" ]; then
        echo "no retrained checkpoints in $RETRAINED, run 'bash scripts/reproduce.sh train' first" >&2
        exit 1
    fi
    "$PY" scripts/verify_retrained.py --retrained-dir "$RETRAINED"
}

case "${1:-}" in
    data|train|ablation|latent|optimize|figures|verify) "stage_$1" ;;
    *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
