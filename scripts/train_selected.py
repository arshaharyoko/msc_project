"""Retrain the ten deployed checkpoints, one seed per run.

    PYTHONPATH=. python scripts/train_selected.py --seed 0 --out-dir artifacts/retrained
    PYTHONPATH=. python scripts/train_selected.py --all --out-dir artifacts/retrained

Each deployed model is the arm `selected` of AblationSpec: graded negatives, softmax readout,
fold 0 as the test fold (groups 2 to 4 train, group 1 validates), every other field at its
src/config.py default. EXPECTED below is that recipe written out. The script builds the spec
from the defaults and stops if any field other than the seed differs, so a change to
src/config.py cannot silently change what this command trains.

Checkpoints go to --out-dir under the deployed names, never to artifacts/ itself. No torch
determinism flag is set, as in the original runs (ablation.seed_everything only), so two runs of
one seed agree up to GPU nondeterminism. scripts/verify_retrained.py measures that.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src import ablation, config
from src.ablation import AblationSpec, run_spec

EXPECTED = {
    "label": "selected", "use_svd": True, "use_mmd": True, "use_rgcn": True, "use_gate": True,
    "split_seed": 67, "fold": 0, "encoding": "dose_split", "epochs": 16, "batch_size": 1,
    "latent_dim": 32, "hidden_dim": 128, "tsvd_k": 32, "pathway_alpha": 1.0, "div_beta": 1e-05,
    "gate_gamma": 0.01, "recon_w1": 1, "recon_w2": 1.0, "use_cost": True,
    "cost_delta": 0.5208333333333334, "cost_input": False, "cost_pregate": True,
    "cost_bounded": True, "cost_consist": 5.0, "cost_consist_mode": "delta", "cost_recipe": 0.015625,
    "readout_width": None, "blacklist": False, "augment": False, "dep_w": 1.0, "dep_margin": 3.0,
    "negatives": "graded", "readout": "softmax", "ident_w2": 0.0,
}
DEPLOYED = config.ARTIFACT_DIR / "ablation__selected__728320fe947b__s{seed}__graded.pth"


def selected_spec(seed):
    return AblationSpec(label="selected", negatives="graded", readout="softmax", fold=0, seed=seed)


def check_recipe(spec):
    got = {k: v for k, v in asdict(spec).items() if k != "seed"}
    drift = {k: (got.get(k), EXPECTED.get(k)) for k in set(got) | set(EXPECTED)
             if got.get(k) != EXPECTED.get(k)}
    if drift:
        raise SystemExit(f"config drifted from the deployed recipe (current, expected): {drift}")
    stored = Path(str(DEPLOYED).format(seed=0))
    if stored.exists():
        old = torch.load(stored, map_location="cpu", weights_only=False)["spec"]
        old = {k: v for k, v in old.items() if k != "seed"}
        if old != got:
            raise SystemExit(f"stored seed 0 spec differs from the recipe: {old} vs {got}")


name_of = ablation.checkpoint_path


def train(seed, out_dir):
    spec = selected_spec(seed)
    check_recipe(spec)
    target = out_dir / DEPLOYED.name.format(seed=seed)
    if name_of(spec).name != target.name:
        raise SystemExit(f"checkpoint name {name_of(spec).name} != {target.name}")
    ablation.checkpoint_path = lambda s: out_dir / name_of(s).name
    started = time.perf_counter()
    _, res, _ = run_spec(spec, save=True, path=out_dir / "train_runs.jsonl")
    wall = time.perf_counter() - started
    print(json.dumps({"seed": seed, "checkpoint": str(target), "wall_seconds": round(wall, 1),
                      "test": {k: res["test"][k] for k in ("recon", "nmse_x2", "cost_r2")}}),
          flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, help="one seed from 0 to 9")
    p.add_argument("--all", action="store_true", help="seeds 0 to 9 in sequence")
    p.add_argument("--out-dir", default="artifacts/retrained")
    args = p.parse_args()
    if args.all == (args.seed is not None):
        p.error("give exactly one of --seed and --all")
    out_dir = Path(args.out_dir).resolve()
    if out_dir == config.ARTIFACT_DIR.resolve():
        p.error("--out-dir must not be artifacts/ itself")
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in (range(10) if args.all else [args.seed]):
        train(seed, out_dir)


if __name__ == "__main__":
    main()
