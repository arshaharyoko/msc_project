#!/usr/bin/env python3
"""Full-factorial ablation analysis: paired main effects and two-way interactions with unit-level bootstrap."""

from __future__ import annotations

import argparse, itertools, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTORS5 = ("svd", "mmd", "rgcn", "gate", "cost")
COL = {"svd": "use_svd", "mmd": "use_mmd", "rgcn": "use_rgcn", "gate": "use_gate", "cost": "use_cost"}
DRAWS, SEED = 10000, 20261104
ORIG = ROOT / "artifacts/original/ablation_runs.jsonl"
FOCUS = ("active", "recon", "gate_auc")
DIRECTION = {"active": "higher", "dims95": "higher", "pr": "higher", "eff_rank": "higher", "collapsed": "lower",
             "recon": "lower", "recon_x1": "lower", "recon_x2": "lower", "recon_recipe_mean": "lower",
             "nmse_x1": "lower", "nmse_x2": "lower", "nmse2_recipe_mean": "lower",
             "cost_probe_r2": "higher", "cost_r2": "higher", "cost_mae_dec": "lower", "recipe_cost_rho": "higher",
             "gate_auc": "none", "balance": "none", "gate_open": "none", "div_loss": "none", "d_z": "none",
             "comp_per_recipe": "none", "n_recipes": "none",
             "ks_prior": "lower", "ac_m1_rho": "higher", "ac_m3_gf_drop": "higher"}
SINGLE_FACTOR_RUNS = ROOT / "artifacts/ablation_runs_single_factor.jsonl"
PRIMARIES = ROOT / "artifacts/ablation_analysis/ablation_primaries.json"


def load(path, split="test"):
    d = pd.read_json(path, lines=True)
    d = d[d.split == split]
    key = [c for c in ("label", "negatives", "seed", "fold") if c in d.columns]
    wide = d.pivot_table(index=key, columns="metric", values="value", aggfunc="first")
    meta = d.drop_duplicates(key).set_index(key)[[c for c in list(COL.values()) + ["params_live"] if c in d.columns]]
    return wide.join(meta).reset_index()


def tensor(df, factors, unit_cols, metrics, flip=None):
    """Array (units, 2, ..., 2); axis i is factor i with index 1 meaning removed (graded for 'negatives')."""
    df = df.copy()
    for f in factors:
        df[f"_{f}"] = (df["negatives"] == "graded") if f == "negatives" else ~df[COL[f]].astype(bool)
        df[f"_{f}"] = df[f"_{f}"].astype(int)
    order = unit_cols + [f"_{f}" for f in factors]
    df = df.sort_values(order)
    units = df[unit_cols].drop_duplicates().to_records(index=False).tolist()
    shape = (len(units),) + (2,) * len(factors)
    out = {}
    for m in metrics:
        if m not in df or df[m].isna().all():
            continue
        if len(df) != int(np.prod(shape)):
            raise SystemExit(f"incomplete design: {len(df)} rows for shape {shape}")
        out[m] = df[m].to_numpy(float).reshape(shape)
    return units, out


warnings.filterwarnings("ignore", message="Mean of empty slice")


def per_unit(Y, axes_keep=None):
    k = Y.ndim - 1
    main, inter = {}, {}
    for i in range(k):
        hi, lo = np.take(Y, 1, axis=i + 1), np.take(Y, 0, axis=i + 1)
        main[i] = np.nanmean((hi - lo).reshape(Y.shape[0], -1), axis=1)
    for i, j in itertools.combinations(range(k), 2):
        t = lambda p, q: np.take(np.take(Y, q, axis=j + 1), p, axis=i + 1)
        inter[(i, j)] = 0.5 * np.nanmean((t(1, 1) - t(0, 1) - t(1, 0) + t(0, 0)).reshape(Y.shape[0], -1), axis=1)
    return main, inter


def summarise(u, bootstrap):
    u = u[~np.isnan(u)]
    if not len(u):
        return {"est": None, "lo": None, "hi": None, "excludes_zero": None, "n_units": 0}
    est = float(u.mean())
    if not bootstrap:
        return {"est": est, "lo": None, "hi": None, "excludes_zero": None, "n_units": len(u)}
    idx = np.random.default_rng(SEED).integers(0, len(u), size=(DRAWS, len(u)))
    lo, hi = np.percentile(u[idx].mean(axis=1), [2.5, 97.5])
    return {"est": est, "lo": float(lo), "hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0), "n_units": len(u)}


def effects(Y, factors, bootstrap):
    main, inter = per_unit(Y, None)
    return ({factors[i]: summarise(v, bootstrap) for i, v in main.items()},
            {f"{factors[i]}:{factors[j]}": summarise(v, bootstrap) for (i, j), v in inter.items()})


RESTRICT = {"gate_auc": "gate", "cost_r2": "cost", "cost_mae_dec": "cost", "recipe_cost_rho": "cost"}


def restrict(Y, factors, metric):
    fs = list(factors)
    f = RESTRICT.get(metric)
    if f in fs:
        ax = fs.index(f)
        Y = np.take(Y, 0, axis=ax + 1)
        fs.pop(ax)
    keep = ~np.all(np.isnan(Y.reshape(Y.shape[0], -1)), axis=1)
    return Y[keep], fs


def design_results(tens, factors, bootstrap):
    res = {}
    for m, Y in tens.items():
        Y, fs = restrict(Y, factors, m)
        if np.nanstd(Y) == 0 or not len(fs):
            continue
        mains, inters = effects(Y, fs, bootstrap)
        res[m] = {"factors": fs, "main": mains, "interaction": inters,
                  "pairs_per_unit": int(np.prod(Y.shape[1:]) // 2), "n_units": int(Y.shape[0])}
    return res


def gate_deltas(tens, factors, bootstrap):
    Y = tens["gate_auc"]
    Y = Y[~np.all(np.isnan(Y.reshape(Y.shape[0], -1)), axis=1)]
    k = len(factors)
    base = Y[(slice(None),) + (0,) * k]
    out = {"baseline_mean_auc": float(np.nanmean(base)), "baseline_below_chance": int(np.sum(base < 0.5)),
           "baseline_n": int(np.sum(~np.isnan(base))),
           "baseline_mean_abs_dev": float(np.nanmean(np.abs(base - 0.5)))}
    on = np.take(Y, 0, axis=factors.index("gate") + 1)
    out["gate_on_cells_below_chance"] = int(np.sum(on < 0.5))
    out["gate_on_cells_n"] = int(np.sum(~np.isnan(on)))
    out["single"], out["single_abs_dev"] = {}, {}
    for i, f in enumerate(factors):
        sel = [0] * k
        sel[i] = 1
        arm = Y[(slice(None),) + tuple(sel)]
        out["single"][f] = summarise(arm - base, bootstrap)
        out["single_abs_dev"][f] = summarise(np.abs(arm - 0.5) - np.abs(base - 0.5), bootstrap)
    return out


def flip(e):
    if e["est"] is None or e["lo"] is None:
        return dict(e, est=None if e["est"] is None else 0.0 - e["est"])
    return dict(e, est=0.0 - e["est"], lo=0.0 - e["hi"], hi=0.0 - e["lo"])


def verdict(m, pooled):
    if pooled["est"] is None or pooled["lo"] is None:
        return "n/a"
    d = DIRECTION.get(m, "none")
    if pooled["lo"] == pooled["hi"] == 0:
        return "constant across arms"
    if not pooled["excludes_zero"]:
        return "indistinguishable"
    if d == "none":
        return "differs, no preferred direction"
    better = (pooled["est"] > 0) == (d == "higher")
    return "favours graph network" if better else "favours MLP readout"


def rgcn_report(tg, tl, A, B):
    i = list(FACTORS5).index("rgcn")
    out = {}
    for m in tg:
        e = {"direction_assumed": DIRECTION.get(m, "none")}
        e["pooled_graded"] = flip(A[m]["main"]["rgcn"]) if m in A else None
        e["pooled_legacy"] = flip(B[m]["main"]["rgcn"]) if m in B else None
        for name, Y, boot in (("single_graded", tg[m], True), ("single_legacy", tl[m], False)):
            sel = [0] * len(FACTORS5)
            sel[i] = 1
            base = Y[(slice(None),) + (0,) * len(FACTORS5)]
            e[name] = flip(summarise(Y[(slice(None),) + tuple(sel)] - base, boot))
        e["verdict_pooled_graded"] = verdict(m, e["pooled_graded"]) if e["pooled_graded"] else "constant across arms"
        e["verdict_single_graded"] = verdict(m, e["single_graded"])
        out[m] = e
    return out


def checkpoint_descriptive():
    import pandas as pd
    prim = pd.DataFrame(json.loads(PRIMARIES.read_text()))
    prim = prim[prim.label.isin(["baseline", "no_rgcn"])]
    rows = {}
    for (label, seed), g in prim.groupby(["label", "seed"]):
        rows[f"{label}|{seed}"] = {k: float(g[k].mean()) for k in ("ks_prior", "ac_m1_rho", "ac_m3_gf_drop")}
        rows[f"{label}|{seed}"]["ac_m1_rho_real_only"] = float(np.mean([d["rho_real_only"] for d in g.ac_m1_detail]))
        rows[f"{label}|{seed}"]["fold_spread"] = {k: float(g[k].max() - g[k].min()) for k in ("ks_prior", "ac_m1_rho", "ac_m3_gf_drop")}
    d = pd.read_json(SINGLE_FACTOR_RUNS, lines=True)
    d = d[(d.split == "test") & d.label.isin(["baseline", "no_rgcn"])]
    w = d.pivot_table(index=["label", "seed"], columns="metric", values="value", aggfunc="mean")
    for (label, seed), r in w.iterrows():
        rows[f"{label}|{seed}"].update({k: (None if np.isnan(v) else float(v)) for k, v in r.items()})
    return rows


def run_new(path):
    df = load(path)
    metrics = [c for c in df.columns if c not in ("label", "negatives", "seed", "fold", "params_live") and not c.startswith("use_")]
    out = {"n_rows_test": len(df)}
    graded = df[df.negatives == "graded"]
    units, tg = tensor(graded, FACTORS5, ["seed", "fold"], metrics)
    out["A_graded"] = {"units": units, "metrics": design_results(tg, FACTORS5, True),
                       "gate_auc_single": gate_deltas(tg, list(FACTORS5), True) if "gate_auc" in tg else None}
    legacy = df[df.negatives == "legacy"]
    units_f, tl_f = tensor(legacy, FACTORS5, ["seed", "fold"], metrics)
    seeds = sorted({u[0] for u in units_f})
    tl, spread = {}, {}
    for m, Y in tl_f.items():
        Yr = Y.reshape((len(seeds), -1) + Y.shape[1:])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spread[m] = float(np.nanmax(np.nanmax(Yr, axis=1) - np.nanmin(Yr, axis=1)))
            tl[m] = np.nanmean(Yr, axis=1)
    units_l = [(s,) for s in seeds]
    out["B_legacy"] = {"units": units_l, "note": "n=2 independent seeds; no interval", "fold_spread_max": {k: v for k, v in spread.items() if v and v > 0},
                       "metrics": design_results(tl, FACTORS5, False),
                       "gate_auc_single": gate_deltas(tl, list(FACTORS5), False)}
    out["rgcn"] = {"sign": "on minus off (graph network minus mean-pool MLP readout)",
                   "metrics": rgcn_report(tg, tl, out["A_graded"]["metrics"], out["B_legacy"]["metrics"]),
                   "per_seed": checkpoint_descriptive()}
    factors6 = FACTORS5 + ("negatives",)
    units_c, tc = tensor(df, factors6, ["seed", "fold"], metrics)
    out["C_negatives"] = {"units": units_c, "sign": "graded minus legacy for negatives; removed minus retained for the others",
                          "caveat": "confounded with the test split; legacy cells repeat across folds so only 2 legacy values are independent, and the 10-unit interval understates their uncertainty",
                          "metrics": design_results(tc, factors6, True)}
    return out, df


def run_orig(path):
    df = load(path)
    f4 = ("svd", "mmd", "rgcn", "gate")
    units, t = tensor(df, f4, ["seed"], ["active", "recon", "gate_auc"])
    return {"units": len(units), "metrics": design_results(t, f4, True)}


def fmt(e):
    if e["est"] is None:
        return "n/a"
    if e["lo"] is None:
        return f"{e['est']:+.4g}"
    return f"{e['est']:+.4g} [{e['lo']:+.4g}, {e['hi']:+.4g}]{'' if e['excludes_zero'] else ' (incl 0)'}"


def write_rgcn_md(L, r):
    L += ["## RGCN versus mean-pool MLP readout, every logged test metric", "",
          "Sign here is on minus off: graph network minus MLP readout (the opposite of the tables above). "
          "`pooled` is the main effect over the other four factors (graded: 16 pairs per unit, interval over 10 units; legacy: 2 seeds, no interval). "
          "`single` is the no_rgcn arm against the baseline arm with every other factor on. "
          "Direction column: the better direction assumed before looking, `none` where no direction is asserted. "
          "Verdict rule, fixed before the numbers: pooled graded interval includes 0 gives indistinguishable; otherwise the sign of the effect against the assumed direction.", "",
          "| metric | better | pooled graded | pooled legacy | single graded | single legacy | verdict (pooled graded) | verdict (single graded) |", "|---|---|---|---|---|---|---|---|"]
    for m, e in r["metrics"].items():
        f = lambda x: fmt(x) if x else "n/a"
        L.append(f"| {m} | {e['direction_assumed']} | {f(e['pooled_graded'])} | {f(e['pooled_legacy'])} | {f(e['single_graded'])} | {f(e['single_legacy'])} | {e['verdict_pooled_graded']} | {e['verdict_single_graded']} |")
    groups = {}
    for m, e in r["metrics"].items():
        groups.setdefault(e["verdict_pooled_graded"], []).append(m)
    L += [""] + [f"- {k}: {', '.join(v)}" for k, v in groups.items()]
    groups = {}
    for m, e in r["metrics"].items():
        groups.setdefault(e["verdict_single_graded"], []).append(m)
    L += ["", "Single-factor (baseline vs no_rgcn, graded):"] + [f"- {k}: {', '.join(v)}" for k, v in groups.items()]
    L += ["", "### Checkpoint-based values, legacy negatives, per seed (folds identical, so one value per seed)", "",
          "ks_prior, ac_m1_rho, ac_m1_rho_real_only and ac_m3_gf_drop come from `artifacts/ablation_analysis/ablation_primaries.json` (mean over folds, fold spread in the json); the other metrics from `artifacts/ablation_runs_single_factor.jsonl`, test split. Descriptive only, n=1 per seed per arm.", ""]
    seeds = sorted({int(k.split("|")[1]) for k in r["per_seed"]})
    keys = [k for k in r["per_seed"]["baseline|0"] if k != "fold_spread"]
    L += ["| metric | better | " + " | ".join(f"baseline s{s} | no_rgcn s{s} | on-off s{s}" for s in seeds) + " |", "|---|---|" + "---|" * (3 * len(seeds))]
    for k in keys:
        cells = []
        for s in seeds:
            b, n = r["per_seed"][f"baseline|{s}"].get(k), r["per_seed"][f"no_rgcn|{s}"].get(k)
            cells += [f"{b:.4g}" if b is not None else "n/a", f"{n:.4g}" if n is not None else "n/a",
                      f"{b - n:+.4g}" if b is not None and n is not None else "n/a"]
        L.append(f"| {k} | {DIRECTION.get(k, 'none')} | " + " | ".join(cells) + " |")
    L.append("")


def write_md(path, res, orig):
    L = ["# Factorial ablation (test split)", "",
         "Sign: removed minus retained for svd, mmd, rgcn, gate, cost; graded minus legacy for negatives. "
         f"Intervals: percentile bootstrap, {DRAWS} draws, seed {SEED}, resampling units.",
         "Interaction = half difference of differences, averaged over the remaining factors. `(incl 0)` = interval includes 0.", ""]
    for key, title, n in (("A_graded", "A. Graded arms, 5 factors, 10 units (seed, fold), 16 pairs per factor per unit", 10),
                          ("B_legacy", "B. Legacy arms, 5 factors, n=2 units (seeds), NO interval", 2),
                          ("C_negatives", "C. Negatives as sixth factor pooled over the five others (confounded with the test split; legacy cells repeat over folds, 2 independent)", 10)):
        r = res[key]
        L += [f"## {title}", ""]
        if key == "B_legacy" and r["fold_spread_max"]:
            L += [f"Max within-seed spread across folds for legacy arms (nonzero metrics only): {r['fold_spread_max']}", ""]
        mets = list(r["metrics"])
        L += ["### Main effects", "", "| metric | units | " + " | ".join(r["metrics"]["active"]["main"]) + " |", "|---|---|" + "---|" * len(r["metrics"]["active"]["main"])]
        for m in mets:
            facs = list(r["metrics"]["active"]["main"])
            cells = [fmt(r["metrics"][m]["main"][f]) if f in r["metrics"][m]["main"] else "n/a" for f in facs]
            L.append(f"| {m} | {r['metrics'][m]['n_units']} | " + " | ".join(cells) + " |")
        L += ["", "In section C, effects of negatives use only units where both levels are defined (gate_auc: 8). gate_auc is estimated on gate-on cells and cost_r2, cost_mae_dec, recipe_cost_rho on cost-on cells (undefined otherwise), so the restricted factor is n/a. Units with no defined value are dropped (graded gate_auc is undefined in fold 4, leaving 8 units).", ""]
        L += ["### Two-way interactions", ""]
        for m in mets:
            inter = r["metrics"][m]["interaction"]
            show = inter if m in FOCUS else {k: v for k, v in inter.items() if v["excludes_zero"]}
            if not show:
                L.append(f"- {m}: none excludes 0" if key != "B_legacy" else f"- {m}: not shown")
                continue
            L += ["", f"**{m}**", "", "| pair | estimate |", "|---|---|"] + [f"| {k} | {fmt(v)} |" for k, v in show.items()]
        L.append("")
        if key != "C_negatives":
            g = r["gate_auc_single"]
            L += ["### Gate AUC against baseline", "",
                  f"Baseline mean AUC {g['baseline_mean_auc']:.4f}, below 0.5 in {g['baseline_below_chance']}/{g['baseline_n']} units; "
                  f"mean |AUC-0.5| {g['baseline_mean_abs_dev']:.4f}. All gate-on cells below 0.5: {g['gate_on_cells_below_chance']}/{g['gate_on_cells_n']}.", "",
                  "| removed | delta AUC | delta abs(AUC-0.5) |", "|---|---|---|"]
            L += [f"| {f} | {fmt(g['single'][f])} | {fmt(g['single_abs_dev'][f])} |" for f in g["single"]]
            L.append("")
    write_rgcn_md(L, res["rgcn"])
    L += ["## Method check on the original data (growthmedium, 4 factors, 10 seeds, test split)", "",
          "| metric | svd | mmd | rgcn | gate |", "|---|---|---|---|---|"]
    for m, v in orig["metrics"].items():
        L.append(f"| {m} | " + " | ".join(fmt(v["main"][f]) if f in v["main"] else "n/a" for f in ("svd", "mmd", "rgcn", "gate")) + " |")
    L += ["", "Interactions excluding 0 in the original data:"]
    for m, v in orig["metrics"].items():
        ex = [k for k, e in v["interaction"].items() if e["excludes_zero"]]
        L.append(f"- {m}: {ex if ex else 'none'}")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=ROOT / "artifacts/ablation_runs.jsonl")
    ap.add_argument("--orig", type=Path, default=ORIG)
    ap.add_argument("--out-json", type=Path, default=ROOT / "artifacts/ablation_analysis/factorial.json")
    ap.add_argument("--out-md", type=Path, default=ROOT / "artifacts/adr/logs/factorial.md")
    a = ap.parse_args()
    res, _ = run_new(a.runs)
    orig = run_orig(a.orig) if a.orig.exists() else None
    res["original_check"] = orig
    for p in (a.out_json, a.out_md):
        p.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(res, indent=1, default=str))
    if orig:
        write_md(a.out_md, res, orig)


if __name__ == "__main__":
    main()
