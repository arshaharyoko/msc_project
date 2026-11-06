#!/usr/bin/env python3
"""Paired bootstrap of the ablation main effects (ADR-024 decisions 3 and 4)."""

from __future__ import annotations

import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
FACTORS = ("svd", "mmd", "rgcn", "gate", "cost", "negatives")
COLUMNS = dict(zip(FACTORS, ("use_svd", "use_mmd", "use_rgcn", "use_gate", "use_cost", "negatives")))
REGISTERED = {"svd": ("active", "surviving latent width"), "mmd": ("ks_prior", "KS distance to the prior"),
              "rgcn": ("ac_m1_rho", "AC-M1 readout validity rho"), "gate": ("ac_m1_rho", "AC-M1 readout validity rho"),
              "cost": ("cost_probe_r2", "cost probe R2"), "negatives": ("ac_m3_gf_drop", "AC-M3 growth-factor drop")}
RESAMPLES, SEED, SPLIT = 2000, 20261030, "test"

def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path)
    parser.add_argument("--metrics", help="comma-separated diagnostic set, not the ADR primaries")
    parser.add_argument("--out-json", type=Path, default=ARTIFACTS / "ablation_bootstrap.json")
    parser.add_argument("--out-md", type=Path, default=ARTIFACTS / "ablation_bootstrap.md")
    return parser.parse_args()

def _load(paths):
    values, factors = defaultdict(dict), {}
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip(): continue
            row = json.loads(line)
            if row["split"] != SPLIT: continue
            label, metric = row["label"], row["metric"]
            vector = tuple(row[COLUMNS[f]] for f in FACTORS)
            if label in factors and factors[label] != vector:
                raise SystemExit(f"{path}:{number}: arm {label!r} changes its factor columns")
            factors[label] = vector
            cell, value = (int(row["fold"]), int(row["seed"])), float(row["value"])
            if metric in values[(label, cell)] and values[(label, cell)][metric] != value:
                raise SystemExit(f"{path}:{number}: duplicate {label} {cell} {metric} with a new value")
            values[(label, cell)][metric] = value
    if not values: raise SystemExit(f"no {SPLIT} rows in {[str(p) for p in paths]}")
    return values, factors

def _design(factors):
    if len(factors) != 64: raise SystemExit(f"found {len(factors)} arms, the 2^6 design needs 64")
    for index, factor in enumerate(FACTORS):
        levels = {vector[index] for vector in factors.values()}
        if len(levels) < 2:
            raise SystemExit(f"factor {factor!r} is constant ({levels.pop()!r}); no main effect exists")
    if len(set(factors.values())) != 64:
        raise SystemExit("factor columns repeat across arms; the rows are not 64 distinct cells")

def _pairs(args, values):
    if args.metrics:
        names = [name.strip() for name in args.metrics.split(",") if name.strip()]
        return [(factor, name, name) for factor in FACTORS for name in names], False
    available = {metric for cell in values.values() for metric in cell}
    missing = [(f, m, t) for f, (m, t) in REGISTERED.items() if m not in available]
    if missing:
        detail = "; ".join(f"{f} needs {m} ({t})" for f, m, t in missing)
        raise SystemExit("ADR-024 registers a primary metric absent from the run rows: "
                         f"{detail}. Available: {', '.join(sorted(available))}")
    return [(factor, *REGISTERED[factor]) for factor in FACTORS], True

def _on(vector, index):
    return vector[index] == "graded" if FACTORS[index] == "negatives" else bool(vector[index])

def _pct(ordered, quantile):
    position = (len(ordered) - 1) * quantile / 100
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

def _bh(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 1.0
    for rank in range(len(p_values), 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = running
    return adjusted

def main():
    args = _args()
    paths = args.runs or ([ARTIFACTS / "ablation_runs.jsonl"] if (ARTIFACTS / "ablation_runs.jsonl").exists() else sorted(ARTIFACTS.glob("ablation_runs__f*.jsonl")))
    values, factors = _load(paths)
    _design(factors)
    pairs, registered = _pairs(args, values)
    per_arm = defaultdict(dict)
    for (label, cell), cell_values in values.items(): per_arm[label][cell] = cell_values
    all_cells = set().union(*(set(cells) for cells in per_arm.values()))
    needed = {metric for _, metric, _ in pairs}
    for label, cells in per_arm.items():
        for cell, got in cells.items():
            if needed - set(got):
                raise SystemExit(f"metric(s) {sorted(needed - set(got))} missing for arm {label} at cell {cell}")
    complete = sum(1 for cells in per_arm.values() if set(cells) >= all_cells)
    cells = sorted(set.intersection(*(set(c) for c in per_arm.values())))
    if not cells: raise SystemExit("no cell is present for all 64 arms")
    dropped = sorted(all_cells - set(cells))

    rng, results = random.Random(SEED), []
    for factor, metric, text in pairs:
        index = FACTORS.index(factor)
        effects = []
        for cell in cells:
            on = [per_arm[arm][cell][metric] for arm in factors if _on(factors[arm], index)]
            off = [per_arm[arm][cell][metric] for arm in factors if not _on(factors[arm], index)]
            if len(on) != 32 or len(off) != 32:
                raise SystemExit(f"{factor} at cell {cell}: {len(on)} on and {len(off)} off, expected 32/32")
            effects.append(sum(on) / 32 - sum(off) / 32)
        point = sum(effects) / len(effects)
        boot = [sum(effects[rng.randrange(len(effects))] for _ in effects) / len(effects)
                for _ in range(RESAMPLES)]
        ordered = sorted(boot)
        wrong = sum(v <= 0 for v in boot) if point >= 0 else sum(v >= 0 for v in boot)
        results.append({"factor": factor, "metric": metric, "description": text, "effect": point,
                        "ci_low": _pct(ordered, 2.5), "ci_high": _pct(ordered, 97.5),
                        "p_raw": min(1.0, 2 * wrong / RESAMPLES)})
    for result, adjusted in zip(results, _bh([r["p_raw"] for r in results])): result["p_bh"] = adjusted

    note = ("Diagnostic metric set, not the ADR-024 primaries. " if not registered else "") + (
            "The 95% CI is the registered statistic. The p-value is auxiliary: two-sided bootstrap "
            "tail probability, doubled and capped at 1, then Benjamini-Hochberg across all pairs.")
    payload = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "source_files": [str(path) for path in paths], "split": SPLIT,
               "registered_primaries": registered, "resamples": RESAMPLES, "seed": SEED,
               "arms": len(factors), "arms_dropped": 0, "pairs": len(results), "complete_arms": complete,
               "cells_used": [[f, s] for f, s in cells], "cells_dropped": [[f, s] for f, s in dropped],
               "note": note, "effects": results}
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")

    used = ", ".join(f"s{s}f{f}" for f, s in cells)
    gone = ", ".join(f"s{s}f{f}" for f, s in dropped) or "none"
    lines = ["# Ablation paired bootstrap", "",
             f"Source: {', '.join(path.name for path in paths)}. Split: {SPLIT}. Arms: {len(factors)} "
             f"(dropped: 0). "
             f"Pairs: {len(results)}. Cells used ({len(cells)}): {used}. Cells dropped ({len(dropped)}): {gone}. "
             f"Complete arms: {complete}/{len(factors)}. Resamples: {RESAMPLES}.", "",
             note, "", "| factor | metric | effect | 95% CI | p raw | p BH |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['factor']} | {r['metric']} | {r['effect']:.4f} | "
                     f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['p_raw']:.4f} | {r['p_bh']:.4f} |")
    table = "\n".join(lines)
    args.out_md.write_text(table + "\n")
    print(table)

if __name__ == "__main__":
    sys.exit(main())
