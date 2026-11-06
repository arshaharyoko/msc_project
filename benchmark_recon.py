"""
Reconstruction usefulness of saved checkpoints.

Slot-level identity and priced decoded-recipe cost, split by dose band and by
database-vs-poison (``src/benchmark.py:recon_usefulness``).

    PYTHONPATH=. python benchmark_recon.py out.json ckpt.pth [ckpt.pth ...]
"""

import json
import sys

import torch

import predict
from src import benchmark
from src.ablation import AblationSpec, _build, split_dataset
from src.vector_store import VectorStore


def run_one(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec = AblationSpec(**payload["spec"])
    run = _build(spec)
    run.model.load_state_dict(payload["state_dict"])
    run.model.eval()
    store = VectorStore.from_run(run)
    prices = predict.load_prices()
    table = benchmark.recon_tables(run, store)
    splits = split_dataset(run.dataset, spec)
    held = list(splits["val"]) + list(splits["test"])
    base = next(i for i in run.dataset if i["source"] == predict.BASE_SOURCE)
    return {"checkpoint": path, "arm": spec.label, "seed": spec.seed,
            "held": benchmark.recon_usefulness(run, store, held, table, prices),
            "base": benchmark.recon_usefulness(run, store, [base], table, prices)}


def main(out, paths):
    results = [run_one(p) for p in paths]
    for r in results:
        h = r["held"]
        print(json.dumps({"arm": r["arm"], "seed": r["seed"], "all": h["all"],
                          "by_dose": h["by_dose"], "by_class": h["by_class"],
                          "cost_err_median": h["cost_err_median"]}), flush=True)
        print("   base:", json.dumps(r["base"]["all"]), flush=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
