"""
Does the cost term hold a formulation's cost down at predict time? (ADR-006)

For each saved ablation checkpoint matching the pattern, every usable val+test
recipe is optimized with predict.py's own machinery under the four objectives in
``src/benchmark.py:PREDICT_OBJECTIVES``, and scored on the readout reached and the
realized USD/L of the decoded, named recipe.

    PYTHONPATH=. python benchmark_predict.py "artifacts/ablation__baseline__*__s0.pth" out.json
    PYTHONPATH=. python benchmark_predict.py --summarize "artifacts/adr/predict_eval/*.json"
"""

import glob
import json
import os
import sys

import torch

import predict
from src import benchmark
from src.ablation import AblationSpec, _build, split_dataset
from src.vector_store import VectorStore

STEPS = int(os.environ.get("EVAL_STEPS", 30))


def main(pattern, out):
    prices = predict.load_prices()
    records = []
    for path in sorted(glob.glob(pattern)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        spec = AblationSpec(**payload["spec"])
        run = _build(spec)
        run.model.load_state_dict(payload["state_dict"])
        run.model.eval()
        store = VectorStore.from_run(run)
        store.load()
        store.toxic_compounds = set()
        if not getattr(store, "oracle", None):
            store.fit_price_oracle(run.dataset, prices, run.cost_bounds[0])

        splits = split_dataset(run.dataset, spec)
        held = list(splits["val"]) + list(splits["test"])
        for idx, item in enumerate(held):
            ctx = benchmark.predict_context(run, store, item)
            if ctx is None:
                continue
            true_cost, _ = predict.realized_cost(
                [{"component": c["component"], "concentration": c["concentration"]}
                 for c in item["components"]])
            for name, weights in benchmark.PREDICT_OBJECTIVES.items():
                torch.manual_seed(0)
                rec = predict.optimize_weighted(run, store, ctx, dict(weights), steps=STEPS,
                                                verbose=False)
                records.append({
                    "checkpoint": path, "arm": spec.label, "seed": spec.seed, "recipe": idx,
                    "product": item["product"], "objective": name, "true_cost": true_cost,
                    "readout": rec["readout"], "pathway": rec["terms"]["pathway"],
                    "predicted_cost_start": rec["predicted_cost_start"],
                    "predicted_cost": rec["predicted_cost"],
                    "realized_cost": rec["realized_cost"], "n_unpriced": rec["n_unpriced"],
                    "n_components": rec["n_pubchem"] + rec["n_uniprot"],
                    "step_norm": rec["step_norm"],
                })
        print(f"{path}: {len(held)} held-out recipes", flush=True)
        with open(out, "w") as f:
            json.dump(records, f, indent=1)

    benchmark.predict_objective_summary(records)


if __name__ == "__main__":
    if sys.argv[1] == "--summarize":
        benchmark.predict_objective_summary(
            [r for path in sorted(glob.glob(sys.argv[2])) for r in json.load(open(path))])
    else:
        main(sys.argv[1], sys.argv[2])
