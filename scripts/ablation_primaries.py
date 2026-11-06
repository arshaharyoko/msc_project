"""ADR-045 part B: determinism check, registered primaries from saved checkpoints, single-factor contrasts.

    PYTHONPATH=. python scripts/ablation_primaries.py determinism
    PYTHONPATH=. python scripts/ablation_primaries.py primaries
    PYTHONPATH=. python scripts/ablation_primaries.py contrasts

ac_m1_rho and ac_m3_gf_drop are scripts/eval_readout.py `_ac_m1` and `_ac_m3` unchanged. ks_prior is
defined in ADR-045: mean over the 32 latent coordinates of the one-sample KS statistic of the pooled
noise-free ungated test-split slot codes against N(0, 1).
"""

import glob
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import eval_readout as er                       # noqa: E402
from src import ablation, benchmark             # noqa: E402

NEW = ROOT / "artifacts/ablation_runs_single_factor.jsonl"
OLD = ROOT / "artifacts/ablation_runs.jsonl"
OUT = ROOT / "artifacts/ablation_analysis"
ARMS = ["baseline", "no_svd", "no_mmd", "no_rgcn", "no_gate", "no_cost", "baseline+graded"]
REGISTERED = {"no_svd": "active", "no_mmd": "ks_prior", "no_rgcn": "ac_m1_rho", "no_gate": "ac_m1_rho",
              "no_cost": "cost_probe_r2", "baseline+graded": "ac_m3_gf_drop"}
METRICS = ["active", "cost_probe_r2", "ks_prior", "ac_m1_rho", "ac_m3_gf_drop"]
RESAMPLES, SEED = 2000, 20261030


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def determinism():
    new = {(r["label"], r["seed"], r["fold"], r["split"], r["metric"]): r["value"] for r in rows(NEW)}
    old = {(r["label"], r["seed"], r["fold"], r["split"], r["metric"]): r["value"] for r in rows(OLD)}
    report = {}
    for metric in ("active", "cost_probe_r2"):
        for split in ("test", "val"):
            keys = [k for k in new if k[0] == "baseline" and k[3] == split and k[4] == metric]
            diffs = [abs(new[k] - old[k]) for k in keys if k in old]
            report[f"{metric}/{split}"] = {
                "cells": len(keys), "matched": len(diffs), "exactly_equal": int(sum(d == 0 for d in diffs)),
                "max_abs_diff": float(max(diffs)), "median_abs_diff": float(np.median(diffs)),
                "new_median": float(np.median([new[k] for k in keys])),
                "old_median": float(np.median([old[k] for k in keys if k in old]))}
    (OUT / "determinism.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def ks_prior(run, store, items):
    codes = []
    for item in items:
        comps = er._components(item["components"])
        x1 = [store.make_x1_vec(c, run.device) for c in comps if c["component_source"] == "pubchem"]
        x2 = [store.make_x2_vec(c, run.device) for c in comps if c["component_source"] == "uniprot"]
        if not x1 or not x2:
            continue
        with torch.no_grad():
            z = benchmark.mean_code(run, torch.stack(x1).unsqueeze(0), torch.stack(x2).unsqueeze(0),
                                    tau=ablation.TAU_FLOOR, gated=False)
        codes.append(z.reshape(-1, z.size(-1)).cpu().numpy().astype(np.float64))
    Z = np.concatenate(codes)
    return float(np.mean([kstest(Z[:, d], "norm").statistic for d in range(Z.shape[1])])), int(len(Z))


def one(path):
    spec, run, store = er._load(path)
    split = ablation.split_dataset(run.dataset, spec)["test"]
    eval_all = json.loads((ROOT / "data/database_graded.json").read_text())
    out = {"label": spec.label, "seed": spec.seed, "fold": spec.fold, "checkpoint": Path(path).name}
    out["ks_prior"], out["ks_n_slots"] = ks_prior(run, store, list(split))
    try:
        heldout, _ = er._heldout(run.dataset, split, eval_all)
    except SystemExit as err:
        out["error"] = f"heldout match: {err}"
        return out
    position = {id(r): i for i, r in enumerate(eval_all) if r.get("parent") is None}
    m1 = er._ac_m1(run, store, eval_all, heldout, [position[id(r)] for r in heldout], spec, 6)
    m3 = er._ac_m3(run, store, heldout, spec)
    out.update(ac_m1_rho=m1["value"], ac_m1_detail={k: m1["detail"][k] for k in
                                                      ("rho_real_only", "n_children", "n_real", "n_skipped_one_sided")},
               ac_m3_gf_drop=m3["detail"]["median_gf_drop"], ac_m3_n=m3["n"],
               ac_m3_median_random_drop=m3["detail"]["median_random_drop"])
    return out


def primaries():
    results = []
    started = time.time()
    for label in ARMS:
        for path in sorted(glob.glob(str(ROOT / f"artifacts/ablation__{label}__*.pth"))):
            try:
                results.append(one(path))
            except Exception as err:
                payload = torch.load(path, map_location="cpu", weights_only=False)["spec"]
                results.append({"label": label, "seed": payload["seed"], "fold": payload["fold"],
                                "checkpoint": Path(path).name, "error": f"{type(err).__name__}: {err}"})
            print(results[-1]["label"], results[-1]["seed"], results[-1]["fold"],
                  {k: results[-1].get(k) for k in ("ks_prior", "ac_m1_rho", "ac_m3_gf_drop", "error")}, flush=True)
    (OUT / "ablation_primaries.json").write_text(json.dumps(results, indent=1))
    print(f"{len(results)} checkpoints, {time.time() - started:.0f} s")


def cell_values():
    values = {}
    for r in rows(NEW):
        if r["split"] == "test" and r["metric"] in ("active", "cost_probe_r2"):
            values.setdefault((r["label"], r["fold"], r["seed"]), {})[r["metric"]] = r["value"]
    for r in json.loads((OUT / "ablation_primaries.json").read_text()):
        cell = values.setdefault((r["label"], r["fold"], r["seed"]), {})
        for metric in ("ks_prior", "ac_m1_rho", "ac_m3_gf_drop"):
            if r.get(metric) is not None:
                cell[metric] = r[metric]
    return values


def pct(ordered, q):
    pos = (len(ordered) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def contrasts():
    values = cell_values()
    rng = random.Random(SEED)
    table = []
    for arm in ARMS[1:]:
        for metric in METRICS:
            diffs, undefined = [], 0
            for fold in range(5):
                for seed in range(2):
                    a, b = values.get((arm, fold, seed), {}), values.get(("baseline", fold, seed), {})
                    if metric in a and metric in b and a[metric] == a[metric] and b[metric] == b[metric]:
                        diffs.append(a[metric] - b[metric])
                    else:
                        undefined += 1
            entry = {"arm": arm, "metric": metric, "registered": REGISTERED[arm] == metric,
                     "n_cells": len(diffs), "n_undefined": undefined}
            if len(diffs) >= 2:
                boot = sorted(float(np.mean([diffs[rng.randrange(len(diffs))] for _ in diffs]))
                              for _ in range(RESAMPLES))
                entry.update(mean_diff=float(np.mean(diffs)), sd=float(np.std(diffs, ddof=1)),
                             ci_low=pct(boot, 2.5), ci_high=pct(boot, 97.5))
            table.append(entry)
    arm_means = {arm: {m: [values[(arm, f, s)][m] for f in range(5) for s in range(2)
                           if m in values.get((arm, f, s), {})] for m in METRICS} for arm in ARMS}
    (OUT / "ablation_contrasts.json").write_text(json.dumps({"contrasts": table, "arm_values": arm_means}, indent=1))
    lines = ["| arm | metric | registered | cells | undefined | mean diff vs baseline | sd | 95% CI | CI contains 0 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for e in table:
        if "mean_diff" in e:
            contains = e["ci_low"] <= 0 <= e["ci_high"]
            lines.append(f"| {e['arm']} | {e['metric']} | {'yes' if e['registered'] else ''} | {e['n_cells']} | {e['n_undefined']} | "
                         f"{e['mean_diff']:.4f} | {e['sd']:.4f} | [{e['ci_low']:.4f}, {e['ci_high']:.4f}] | {'yes' if contains else 'no'} |")
        else:
            lines.append(f"| {e['arm']} | {e['metric']} | {'yes' if e['registered'] else ''} | {e['n_cells']} | {e['n_undefined']} | undefined | | | |")
    lines += ["", "Baseline values (mean over defined cells): " + ", ".join(
        f"{m} {np.mean(v):.4f} (n = {len(v)})" for m, v in arm_means["baseline"].items() if v)]
    (OUT / "ablation_contrasts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("determinism", "primaries", "contrasts"):
        raise SystemExit(__doc__)
    {"determinism": determinism, "primaries": primaries, "contrasts": contrasts}[sys.argv[1]]()
