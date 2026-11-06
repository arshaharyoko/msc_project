"""Score one readout checkpoint against AC-M1 to AC-M6.

    PYTHONPATH=. python scripts/eval_readout.py CKPT --out out.json \
        [--data data/database_graded.json] [--split test] [--children N]

AC-M1 needs graded held-out recipes and held-out real recipes are all positives, so
evaluation-only children of held-out parents are generated here through
src/modules/negatives.py. Nothing is written back to any data file.

Legacy view: "test" is the fold the checkpoint's own spec selects from its own
dataset; every metric names the population it was computed over.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from src import ablation, benchmark, config
from src.modules import mvrrae_gv, negatives, priors
from src.vector_store import VectorStore

NODES = (22, 23, 24)                    # proliferation, survival/qui, differentiation
PROLIF = ablation.PROLIF_NODE           # node 22 of the 25-node pathway readout
TAU = ablation.TAU_FLOOR
RANDOM_DRAWS = 10
AC_M4_RECIPES = 8
AC_M4_SLOTS = 3


def _components(components):
    """The JSON stores component_vec as a list; every vectorizer wants a tensor."""
    return [dict(c, component_vec=torch.as_tensor(c["component_vec"], dtype=torch.float32))
            for c in components]


def _logits(run, store, components, shut=None):
    comps = _components(components)
    x1 = [store.make_x1_vec(c, run.device) for c in comps if c["component_source"] == "pubchem"]
    x2 = [store.make_x2_vec(c, run.device) for c in comps if c["component_source"] == "uniprot"]
    if not x1 or not x2:
        return None
    x1 = torch.stack(x1).unsqueeze(0)
    x2 = torch.stack(x2).unsqueeze(0)
    with torch.no_grad():
        if shut is None:
            out = run.model(x1, x2, run.adjacency_matrices, tau=TAU)
        else:
            with _gate_shut(shut):
                out = run.model(x1, x2, run.adjacency_matrices, tau=TAU)
    return out[3][0, :, 0].cpu().numpy()


@contextmanager
def _gate_shut(index):
    """Force one gate to exactly 0 inside the forward pass, in the forward's own order."""
    original = mvrrae_gv.gumbel_sigmoid

    def shut(logits, tau=1.0, sample=True, hard=False, threshold=0.5):
        gate = original(logits, tau, sample=sample, hard=hard, threshold=threshold).clone()
        gate[:, index] = 0.0
        return gate

    mvrrae_gv.gumbel_sigmoid = shut
    try:
        yield
    finally:
        mvrrae_gv.gumbel_sigmoid = original


def _ordered(components):
    return ([c for c in components if c["component_source"] == "pubchem"]
            + [c for c in components if c["component_source"] == "uniprot"])


def _p(record):
    return float(record.get("p", record.get("proliferation", 1.0)))


def _item(record):
    """A dataset-sample lookalike, which the benchmark helpers accept."""
    return {
        "product": record.get("product"),
        "flag": record.get("flag"),
        "proliferation": record.get("proliferation", 1),
        "components": _components(record["components"]),
    }


def _median_dose(record):
    return benchmark._median_dose(_item(record))


def _rho(xs, ys):
    if len(xs) < 3 or len(set(np.round(ys, 9))) < 2:
        return None
    value = spearmanr(xs, ys).statistic
    return None if np.isnan(value) else float(value)


def _stratified(children, n, rng):
    buckets = {}
    for child in children:
        buckets.setdefault(child["generator"], []).append(child)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    picked, names = [], sorted(buckets)
    while len(picked) < min(n, len(children)):
        for name in names:
            if buckets[name] and len(picked) < n:
                picked.append(buckets[name].pop())
    return picked


def _load(ckpt):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "spec" not in payload:
        raise SystemExit(f"{ckpt} carries no spec; it predates the ablation harness")
    spec = ablation.AblationSpec(**payload["spec"])
    ablation.seed_everything(spec.seed)
    run = ablation._build(spec)
    run.model.load_state_dict(payload["state_dict"])
    run.model.eval()
    # The price oracle is only used by predict-time objectives and needs faiss indexes.
    store = VectorStore._from_run(run)
    return spec, run, store


def _heldout(dataset, split, eval_records):
    """Held-out real parents, matched by publication order against the evaluation file."""
    data = dataset.data
    real = [i for i, r in enumerate(data) if r.get("parent") is None]
    parents = [r for r in eval_records if r.get("parent") is None]
    if len(real) != len(parents):
        raise SystemExit(f"checkpoint has {len(real)} real recipes, {len(parents)} in the "
                         f"evaluation file; the split cannot be matched")
    for k, i in enumerate(real):
        if data[i]["product"] != parents[k]["product"]:
            raise SystemExit(f"recipe {k} differs between checkpoint data and evaluation file")
    position = {i: k for k, i in enumerate(real)}
    indices = list(split.indices) if hasattr(split, "indices") else list(range(len(dataset)))
    heldout = [parents[position[i]] for i in indices if i in position]
    return heldout, [data[i] for i in indices]


def _ac_m1(run, store, eval_all, heldout, eval_parent_positions, spec, n_children):
    spec_yaml = negatives.load_spec()
    rng = random.Random(spec.seed)
    children = negatives.generate(eval_all, spec_yaml, eval_parent_positions)
    by_parent = {}
    for child in children:
        by_parent.setdefault(child["parent"], []).append(child)

    grouped = [c for parent, pool in sorted(by_parent.items())
               for c in _stratified(pool, n_children, rng)]

    real_rows, child_rows, skipped = [], [], 0
    for record in heldout:
        logits = _logits(run, store, record["components"])
        if logits is None:
            skipped += 1
            continue
        real_rows.append((_p(record), float(logits[PROLIF])))
    for child in grouped:
        logits = _logits(run, store, child["components"])
        if logits is None:
            skipped += 1
            continue
        child_rows.append((float(child["p"]), float(logits[PROLIF])))

    rows = real_rows + child_rows
    rho = _rho([y for _, y in rows], [p for p, _ in rows])
    rho_real = _rho([y for _, y in real_rows], [p for p, _ in real_rows])
    notes = []
    if rho is None:
        notes.append("rho undefined: fewer than 3 forwardable recipes or a constant p")
    if rho_real is None:
        notes.append("real-only rho undefined: held-out real recipes carry a single p")
    return {
        "verdict": "PASS" if rho is not None and rho >= 0.6 else ("FAIL" if rho is not None else "REPORT"),
        "value": rho,
        "n": len(rows),
        "split": "test",
        "bar": 0.6,
        "detail": {
            "rho_children_and_real": rho,
            "rho_real_only": rho_real,
            "n_children": len(child_rows),
            "n_real": len(real_rows),
            "n_real_positives": sum(1 for p, _ in real_rows if p == 1.0),
            "children_per_parent": n_children,
            "n_heldout_parents_generated": len(by_parent),
            "n_skipped_one_sided": skipped,
            "p_range": [min(p for p, _ in rows), max(p for p, _ in rows)] if rows else None,
        },
        "notes": notes,
    }


def _ac_m2(run, store, heldout, eval_all, spec):
    positives = [r for r in heldout if _p(r) == 1.0]
    drops, skipped = [], 0
    for record in positives:
        full = _logits(run, store, record["components"])
        if full is None:
            skipped += 1
            continue
        diluted = _logits(run, store, negatives._scaled(record["components"], 1e-3))
        drops.append(float(full[PROLIF] - diluted[PROLIF]))

    templates = {}
    for record in eval_all:
        for comp in record.get("components", []):
            templates.setdefault(comp["component"], comp)
    spec_yaml = negatives.load_spec()
    toxicants = list(spec_yaml["toxicants"])
    spikes = {0.01: [], 0.1: []}
    missing = 0
    for k, record in enumerate(positives):
        toxicant_id = toxicants[k % len(toxicants)]
        template = templates.get(toxicant_id)
        if template is None:
            missing += 1
            continue
        toxicant = spec_yaml["toxicants"][toxicant_id]
        for multiple in spikes:
            components = list(record["components"]) + [
                negatives._with_concentration(template, toxicant["ic50"] * multiple)]
            logits = _logits(run, store, components)
            if logits is not None:
                spikes[multiple].append(float(logits[PROLIF] > 0))

    share = float(np.mean([d >= 3.0 for d in drops])) if drops else None
    per_dose = {f"{m:g}x": {"n": len(v), "frac_proliferative": float(np.mean(v)) if v else None}
                for m, v in spikes.items()}
    pooled = spikes[0.01] + spikes[0.1]
    return {
        "verdict": "PASS" if share is not None and share >= 0.8 else ("FAIL" if share is not None else "REPORT"),
        "value": share,
        "n": len(drops),
        "split": "test",
        "bar": 0.8,
        "detail": {
            "dilution_1000x": {"n": len(drops), "frac_drop_ge_3": share,
                               "median_drop": float(np.median(drops)) if drops else None,
                               "min_drop": float(min(drops)) if drops else None,
                               "max_drop": float(max(drops)) if drops else None},
            "spike_below_ic50": {**per_dose,
                                 "pooled": {"n": len(pooled),
                                            "frac_proliferative": float(np.mean(pooled)) if pooled else None}},
            "n_skipped_one_sided": skipped,
            "n_toxicants_without_template": missing,
        },
        "notes": [],
    }


def _ac_m3(run, store, heldout, spec):
    rng = random.Random(spec.seed + 1)
    growth = set(priors.GROWTH_FACTORS)
    rows, skipped = [], 0
    for record in heldout:
        if _p(record) != 1.0:
            continue
        components = record["components"]
        gfs = [c for c in components if c["component"] in growth]
        if not gfs:
            continue
        full = _logits(run, store, components)
        if full is None:
            skipped += 1
            continue
        kept = [c for c in components if c["component"] not in growth]
        deleted = _logits(run, store, kept)
        if deleted is None:
            skipped += 1
            continue
        pool = [i for i, c in enumerate(components)
                if c["component_source"] == "pubchem" and c["component"] not in growth]
        if len(pool) < len(gfs):
            skipped += 1
            continue
        draws = []
        for _ in range(RANDOM_DRAWS):
            drop = set(rng.sample(pool, len(gfs)))
            control = _logits(run, store, [c for i, c in enumerate(components) if i not in drop])
            if control is not None:
                draws.append(float(full[PROLIF] - control[PROLIF]))
        if not draws:
            skipped += 1
            continue
        rows.append({"product": record.get("product"),
                     "n_growth_factors": len(gfs),
                     "gf_drop": float(full[PROLIF] - deleted[PROLIF]),
                     "random_drop": float(np.mean(draws)),
                     "random_draws": draws})

    gf = [r["gf_drop"] for r in rows]
    random_all = [d for r in rows for d in r["random_draws"]]
    random_med = [float(np.median(r["random_draws"])) for r in rows]
    paired = [r["gf_drop"] - r["random_drop"] for r in rows]
    detail = {
        "n_recipes": len(rows),
        "median_gf_drop": float(np.median(gf)) if gf else None,
        "median_random_drop": float(np.median(random_all)) if random_all else None,
        "median_of_recipe_random_medians": float(np.median(random_med)) if random_med else None,
        "median_paired_difference": float(np.median(paired)) if paired else None,
        "frac_gf_moves_right_way": float(np.mean([d > 0 for d in paired])) if paired else None,
        "frac_gf_drop_ge_3": float(np.mean([d >= 3 for d in gf])) if gf else None,
        "random_draws_per_recipe": RANDOM_DRAWS,
        "per_recipe": rows,
        "n_skipped": skipped,
    }
    ok = (gf and np.median(gf) >= 3.0 and random_all and np.median(random_all) < 1.0)
    return {
        "verdict": "PASS" if ok else ("FAIL" if gf and random_all else "REPORT"),
        "value": detail["median_gf_drop"],
        "n": len(rows),
        "split": "test",
        "bar": {"gf_median": 3.0, "random_median": 1.0},
        "detail": detail,
        "notes": [] if rows else ["no forwardable held-out positive contains a growth factor"],
    }


def _ac_m4(run, store, heldout, spec):
    rng = random.Random(spec.seed + 2)
    recipes = [r for r in heldout if _p(r) == 1.0
               and any(c["component_source"] == "pubchem" for c in r["components"])
               and any(c["component_source"] == "uniprot" for c in r["components"])]
    rng.shuffle(recipes)
    diffs, skipped = [], 0
    for record in recipes[:AC_M4_RECIPES]:
        ordered = _ordered(record["components"])
        counts = {"pubchem": sum(c["component_source"] == "pubchem" for c in ordered),
                  "uniprot": sum(c["component_source"] == "uniprot" for c in ordered)}
        candidates = [i for i, c in enumerate(ordered) if counts[c["component_source"]] > 1]
        if not candidates:
            continue
        for index in rng.sample(candidates, min(AC_M4_SLOTS, len(candidates))):
            shut = _logits(run, store, ordered, shut=index)
            deleted = _logits(run, store, ordered[:index] + ordered[index + 1:])
            if shut is None or deleted is None:
                skipped += 1
                continue
            diffs.append(float(np.abs(shut[list(NODES)] - deleted[list(NODES)]).max()))
    worst = max(diffs) if diffs else None
    return {
        "verdict": "PASS" if worst is not None and worst < 0.1 else ("FAIL" if worst is not None else "REPORT"),
        "value": worst,
        "n": len(diffs),
        "split": "test",
        "bar": 0.1,
        "detail": {
            "max_abs_logit_difference": worst,
            "mean_abs_logit_difference": float(np.mean(diffs)) if diffs else None,
            "frac_under_0.1": float(np.mean([d < 0.1 for d in diffs])) if diffs else None,
            "n_comparisons": len(diffs),
            "n_recipes": len(recipes[:AC_M4_RECIPES]),
            "n_skipped": skipped,
        },
        "notes": [],
    }


def _ac_m5(run, store, heldout_records):
    values, skipped = [], 0
    for record in heldout_records:
        logits = _logits(run, store, record["components"])
        if logits is None:
            skipped += 1
            continue
        values.append(logits)
    if not values:
        return {"verdict": "REPORT", "value": None, "n": 0, "split": "test",
                "bar": 0.05, "detail": {"n_skipped": skipped},
                "notes": ["no forwardable held-out record"]}
    V = np.stack(values)
    per_node = {}
    for node in NODES:
        col = V[:, node]
        per_node[f"node_{node}"] = {
            "n": int(col.size),
            "frac_abs_gt_12": float(np.mean(np.abs(col) > 12)),
            "p10": float(np.percentile(col, 10)),
            "p50": float(np.percentile(col, 50)),
            "p90": float(np.percentile(col, 90)),
        }
    flat = V[:, list(NODES)].reshape(-1)
    frac = float(np.mean(np.abs(flat) > 12))
    return {
        "verdict": "PASS" if frac < 0.05 else "FAIL",
        "value": frac,
        "n": int(flat.size),
        "split": "test",
        "bar": 0.05,
        "detail": {
            "frac_abs_gt_12_all_nodes": frac,
            "p10": float(np.percentile(flat, 10)),
            "p50": float(np.percentile(flat, 50)),
            "p90": float(np.percentile(flat, 90)),
            "per_node": per_node,
            "n_recipes": len(values),
            "n_skipped_one_sided": skipped,
        },
        "notes": [],
    }


def _ac_m6(run, store, heldout):
    items = [_item(r) for r in heldout if r.get("proliferation", 1) == 1]
    if not items:
        return {"verdict": "REPORT", "value": None, "n": 0, "split": "test", "bar": 0.6,
                "detail": {}, "notes": ["no held-out real positive"]}
    # recon_usefulness already owns the recipe-level low/high dose split (DOSE_BANDS).
    pooled = benchmark.recon_usefulness(run, store, items)
    low_key = next((k for k in pooled["by_dose"] if k.startswith("0-0.1")), None)
    low = pooled["by_dose"].get(low_key) if low_key else None

    table = benchmark.recon_tables(run, store)
    named = {"P04085": 0, "P05019": 0, "n_slots": 0}
    for item in items:
        x1v = [store.make_x1_vec(c, run.device) for c in item["components"]
               if c["component_source"] == "pubchem"]
        x2v = [store.make_x2_vec(c, run.device) for c in item["components"]
               if c["component_source"] == "uniprot"]
        if not x1v or not x2v:
            continue
        x1, x2 = torch.stack(x1v).unsqueeze(0), torch.stack(x2v).unsqueeze(0)
        with torch.no_grad():
            code = benchmark.mean_code(run, x1, x2, TAU)
            x1_hat, x2_hat = run.model.decoder(code, x1.size(1), x2.size(1))
        for source, x_hat in (("pubchem", x1_hat), ("uniprot", x2_hat)):
            names, _ = benchmark.name_slots(store, x_hat, store.n_scalars, *table[source])
            named["n_slots"] += len(names)
            for name in names:
                if name in named:
                    named[name] += 1
    value = low["id_acc"] if low else None
    return {
        "verdict": "PASS" if value is not None and value >= 0.6 else ("FAIL" if value is not None else "REPORT"),
        "value": value,
        "n": low["n"] if low else 0,
        "split": "test",
        "bar": 0.6,
        "detail": {
            "low_dose_band": low_key,
            "band_definition": "benchmark.DOSE_BANDS: median UniProt dose < 0.1 mg/L",
            "low_dose_slot_id_accuracy": value,
            "low_dose_slots": low["n"] if low else 0,
            "all_dose_slot_id_accuracy": pooled["all"]["id_acc"],
            "all_dose_slots": pooled["all"]["n"],
            "n_recipes_in_band": sum(
                1 for i in items
                if benchmark._median_dose(i) < 0.1
                and any(c["component_source"] == "pubchem" for c in i["components"])
                and any(c["component_source"] == "uniprot" for c in i["components"])),
            "naming_counts": named,
            "n_recipes": pooled["n_recipes"],
        },
        "notes": [] if low else ["no held-out recipe falls in the low-dose band"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="data/database_graded.json")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--children", type=int, default=6)
    args = parser.parse_args()

    started = time.time()
    spec, run, store = _load(args.checkpoint)
    split = ablation.split_dataset(run.dataset, spec)[args.split]
    eval_all = json.loads(Path(args.data).read_text())
    heldout, heldout_records = _heldout(run.dataset, split, eval_all)

    parent_position = {id(r): i for i, r in enumerate(eval_all) if r.get("parent") is None}
    heldout_positions = [parent_position[id(r)] for r in heldout]

    ac = {
        "AC-M1": _ac_m1(run, store, eval_all, heldout, heldout_positions, spec, args.children),
        "AC-M2": _ac_m2(run, store, heldout, eval_all, spec),
        "AC-M3": _ac_m3(run, store, heldout, spec),
        "AC-M4": _ac_m4(run, store, heldout, spec),
        "AC-M5": _ac_m5(run, store, heldout_records),
        "AC-M6": _ac_m6(run, store, heldout),
        "AC-M7": {"verdict": "REPORT", "value": None, "n": 0,
                  "detail": {"note": "decoder-side criterion; measured by the decoder check, "
                                     "not by this readout harness"}},
    }
    if args.split != "test":
        for row in ac.values():
            row["split"] = args.split

    payload = {
        "checkpoint": str(args.checkpoint),
        "spec": {
            "config_id": spec.config_id,
            "label": spec.label,
            "seed": spec.seed,
            "split_seed": spec.split_seed,
            "readout": spec.readout,
            "negatives": spec.negatives,
            "ident_w2": spec.ident_w2,
        },
        "data": str(args.data),
        "split": args.split,
        "split_sizes": {name: len(sub) for name, sub in
                        ablation.split_dataset(run.dataset, spec).items()},
        "n_heldout_real": len(heldout),
        "n_heldout_real_positives": sum(1 for r in heldout if _p(r) == 1.0),
        "ac": ac,
        "wall_seconds": time.time() - started,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
