"""Readout sensitivity of one checkpoint on the cleaned Wu base.

    PYTHONPATH=. python scripts/readout_drift.py CKPT --out out.json

Scenarios run through scripts/eval_readout.py's own forward path (_logits), so
the numbers are comparable with the AC-M2 dilution and AC-M3 deletions.
(a) every UniProt dose of the base scaled by a factor, plus a glucose control;
(b) each UniProt row deleted one at a time, and 30 random small-molecule rows.

Two forward paths. "eval" is eval_readout._logits, which keeps the posterior
and gate noise running (the model is not called with deterministic=True), so it
is re-seeded before every call to make base and scenario share their draws, and
its noise floor is measured. "judge" is constraints.encode_recipe, the noise-free
path that decides feasibility in predict.round_trip.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch

from predict import BASE_SOURCE
from src import constraints
from src.modules import negatives

spec_ = importlib.util.spec_from_file_location("eval_readout", Path(__file__).with_name("eval_readout.py"))
ev = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(ev)

NAMES = ("proliferation", "survival", "differentiation")
PROTEIN_SCALES = (1e-3, 1e-2, 0.1, 10.0, 100.0, 1000.0)
GLUCOSE_KEY = "WQZGKKKJIJFFOK"
N_RANDOM = 30
RANDOM_SEED = 20261102
NOISE_SEED = 0
NOISE_DRAWS = 30


def _is_glucose(smiles):
    from rdkit import Chem
    from rdkit.Chem import inchi
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and inchi.MolToInchiKey(mol).startswith(GLUCOSE_KEY)


def _scale(components, pick, factor):
    out = []
    for comp in components:
        conc = comp.get("concentration")
        if pick(comp) and isinstance(conc, (int, float)):
            out.append(negatives._with_concentration(comp, conc * factor))
        else:
            out.append(comp)
    return out


def _eval_logits(run, store, components):
    torch.manual_seed(NOISE_SEED)
    return ev._logits(run, store, components)


def _judge_logits(run, store, components):
    comps = ev._components(components)
    x1 = [store.make_x1_vec(c, run.device) for c in comps if c["component_source"] == "pubchem"]
    x2 = [store.make_x2_vec(c, run.device) for c in comps if c["component_source"] == "uniprot"]
    if not x1 or not x2:
        return None
    with torch.no_grad():
        logits, _, _ = constraints.encode_recipe(run, torch.stack(x1).unsqueeze(0),
                                                 torch.stack(x2).unsqueeze(0))
    return logits.cpu().numpy()


PATHS = {"eval": _eval_logits, "judge": _judge_logits}


def _shifts(run, store, bases, components):
    out = {}
    for path, fn in PATHS.items():
        logits = fn(run, store, components)
        out[path] = None if logits is None else {
            n: float(logits[node] - bases[path][node]) for n, node in zip(NAMES, ev.NODES)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    spec, run, store = ev._load(args.checkpoint)
    item = next(i for i in run.dataset if i["source"] == BASE_SOURCE)
    comps, _ = constraints.clean_base(item["components"])
    bases = {path: fn(run, store, comps) for path, fn in PATHS.items()}
    unseeded = np.stack([ev._logits(run, store, comps) for _ in range(NOISE_DRAWS)])[:, list(ev.NODES)]
    noise = {n: {"sd": float(unseeded[:, j].std(ddof=1)),
                 "range": float(np.ptp(unseeded[:, j]))} for j, n in enumerate(NAMES)}
    judge_repeat = float(np.abs(bases["judge"] - _judge_logits(run, store, comps)).max())

    is_protein = lambda c: c["component_source"] == "uniprot"
    glucose_rows = [c for c in comps if c["component_source"] == "pubchem" and _is_glucose(c["component"])]
    scaling = {"proteins": {}, "glucose": {}, "all_components_1e-3": None}
    for factor in PROTEIN_SCALES:
        scaling["proteins"][f"{factor:g}"] = _shifts(run, store, bases, _scale(comps, is_protein, factor))
    glucose_ids = {id(c) for c in glucose_rows}
    for factor in (1e-3, 1e3):
        scaling["glucose"][f"{factor:g}"] = _shifts(
            run, store, bases, _scale(comps, lambda c: id(c) in glucose_ids, factor))
    scaling["all_components_1e-3"] = _shifts(run, store, bases, negatives._scaled(comps, 1e-3))

    proteins = [i for i, c in enumerate(comps) if is_protein(c)]
    small = [i for i, c in enumerate(comps) if c["component_source"] == "pubchem"]
    sample = sorted(random.Random(RANDOM_SEED).sample(small, N_RANDOM))
    deletion = {"proteins": [], "random_small": []}
    for key, rows in (("proteins", proteins), ("random_small", sample)):
        for i in rows:
            shift = _shifts(run, store, bases, comps[:i] + comps[i + 1:])
            deletion[key].append({"row": i, "component": comps[i]["component"],
                                  "concentration": comps[i]["concentration"], "shift": shift})

    payload = {
        "checkpoint": args.checkpoint,
        "seed": spec.seed,
        "device": str(run.device),
        "n_base_rows": len(comps),
        "base_logits": {path: {n: float(b[node]) for n, node in zip(NAMES, ev.NODES)}
                        for path, b in bases.items()},
        "eval_path_noise_unseeded": noise,
        "judge_path_repeat_max_abs_diff": judge_repeat,
        "scaling": scaling,
        "deletion": deletion,
        "random_seed": RANDOM_SEED,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
