"""ADR-045 part A: diagnostics re-measured on the ten deployed checkpoints.

    PYTHONPATH=. python scripts/diagnostics.py run [stage ...]
    PYTHONPATH=. python scripts/diagnostics.py aggregate

Stages: predict (A1), slots (A2 B1 and B2), norms (A2 B3), geometry (A3), recovery (A4).
Per-model results are cached under artifacts/ablation_analysis/diag_parts/ and a finished (stage, seed) is
not recomputed. `aggregate` writes artifacts/ablation_analysis/diagnostics.json and
artifacts/adr/logs/diagnostics.md. Definitions and decision rules are in ADR-045.
"""

import itertools
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "artifacts/adr/scripts"), str(ROOT / "artifacts/adr/priors")]

import benchmark_predict as bp                  # noqa: E402
import latent_geometry as lg                    # noqa: E402
import latent_geometry_controls as lgc          # noqa: E402
import predict                                  # noqa: E402
import recon_slot_metrics as rsm                # noqa: E402
import recovery_knowns as rk                    # noqa: E402
from src.ablation import AblationSpec, _build, seed_everything, split_dataset  # noqa: E402
from src.vector_store import VectorStore        # noqa: E402

CKPT = str(ROOT / "artifacts/ablation__selected__728320fe947b__s{}__graded.pth")
SEEDS = range(10)
OUT = ROOT / "artifacts/ablation_analysis"
PARTS = OUT / "diag_parts"
LOG_MD = ROOT / "artifacts/adr/logs/diagnostics.md"
DOSES = {"0.01": 0.01, "1": 1.0, "10": 10.0, "100": 100.0, "own": "own"}
LOW, HIGH = 0.1, 1.0


def load(seed, untrained=False):
    payload = torch.load(CKPT.format(seed), map_location="cpu", weights_only=False)
    spec = AblationSpec(**payload["spec"])
    seed_everything(spec.seed + (1000 if untrained else 0))
    run = _build(spec)
    if not untrained:
        run.model.load_state_dict(payload["state_dict"])
    run.model.eval()
    return spec, run, VectorStore.from_run(run)


def part(stage, seed):
    return PARTS / f"{stage}__s{seed}.json"


def cached(stage, seed, fn):
    path = part(stage, seed)
    if path.exists():
        return json.loads(path.read_text())
    started = time.time()
    result = fn()
    result["wall_s"] = time.time() - started
    path.write_text(json.dumps(result))
    print(f"{stage} s{seed}: {result['wall_s']:.0f} s", flush=True)
    return result


def stage_predict(seed):
    def fn():
        out = OUT / "predict_eval" / f"s{seed}.json"
        out.parent.mkdir(exist_ok=True)
        bp.main(CKPT.format(seed), str(out))
        return {"records": str(out.relative_to(ROOT))}
    return cached("predict", seed, fn)


def stage_slots(seed):
    def fn():
        spec, run, store = load(seed)
        table = rsm.tables(run, store)
        splits = split_dataset(run.dataset, spec)
        rows = []
        for name in ("val", "test"):
            for item in splits[name]:
                rows += rsm.slots(run, store, item, table)
        uni = [r for r in rows if r["source"] == "uniprot"]
        groups = {"low": lambda r: r["band"] == "low", "mid": lambda r: r["band"] == "mid",
                  "high": lambda r: r["band"] == "high", "database": lambda r: r["prolif"] == 1,
                  "poison": lambda r: r["prolif"] == 0, "all": lambda r: True}
        bands = {}
        for key, pred in groups.items():
            hit = [r["hit"] for r in uni if pred(r)]
            bands[key] = {"acc": float(np.mean(hit)) if hit else None, "n": len(hit)}
        base = predict.encode_base(run, store)["base"]
        base_rows = [r for r in rsm.slots(run, store, base, table) if r["source"] == "uniprot"]
        return {"bands": bands,
                "base": [{k: r[k] for k in ("truth", "decoded", "hit", "true_conc")} for r in base_rows]}
    return cached("slots", seed, fn)


def stage_norms(seed):
    def fn():
        _, run, store = load(seed)
        k = store.n_scalars
        ident, dose, band = [], [], []
        for item in run.dataset:
            for c in item["components"]:
                if c["component_source"] != "uniprot" or not c["concentration"]:
                    continue
                v = store.make_x2_vec(c, run.device)
                ident.append(float(v[:-k].norm()))
                dose.append(float(v[-k]))
                lc = float(np.log1p(c["concentration"]))
                band.append("low" if lc < LOW else ("high" if lc > HIGH else "mid"))
        ident, dose, band = np.array(ident), np.array(dose), np.array(band)
        share = dose ** 2 / (ident ** 2 + dose ** 2)
        res = {"n_scalars": k, "dim": int(v.numel())}
        for b in ("low", "mid", "high"):
            m = band == b
            res[b] = {"n": int(m.sum()), "median_ident_norm": float(np.median(ident[m])),
                      "median_dose_share": float(np.median(share[m]))}
        res["ratio_high_over_low"] = res["high"]["median_ident_norm"] / res["low"]["median_ident_norm"]
        return res
    return cached("norms", seed, fn)


def stage_geometry(seed):
    def fn():
        names = codes = None
        out = {}
        for untrained in (False, True):
            spec, run, store = load(seed, untrained)
            rows = lg.protein_rows(run)
            names = list(rows)
            for tag, conc in DOSES.items():
                lg.CONC = conc
                code = lg.codes(run, store, names, rows, run.device)[0]
                out[f"{'untrained' if untrained else 'trained'}__{tag}"] = code.tolist()
            if not untrained:
                raw = np.array([rows[n]["vec"] for n in names])
                cooc = lgc.cooccurrence(run.dataset, names)
        return {"names": names, "raw": raw.tolist(), "cooc": cooc.tolist(), "codes": out}
    return cached("geometry", seed, fn)


def stage_recovery(seed):
    def fn():
        _, run, store = load(seed)
        res = {"trained": rk.evaluate_run(run, store, np.random.default_rng(0), placebo=True)}
        if seed == 0:
            _, urun, ustore = load(0, untrained=True)
            res["untrained"] = rk.evaluate_run(urun, ustore, np.random.default_rng(0), placebo=True)
        return res
    return cached("recovery", seed, fn)


STAGES = {"predict": stage_predict, "slots": stage_slots, "norms": stage_norms,
          "geometry": stage_geometry, "recovery": stage_recovery}


def run(stages):
    PARTS.mkdir(parents=True, exist_ok=True)
    for stage in stages or STAGES:
        for seed in SEEDS:
            STAGES[stage](seed)


def rng_summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": float(np.median(values)), "min": float(min(values)), "max": float(max(values)),
            "mean": float(np.mean(values)), "n": len(values)}


def sign_p(k, n=10):
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def agg_predict():
    per_model, pooled = {}, {}
    labels = {"pathway + proxy": "proxy", "pathway + latent": "latent", "pathway + decoded": "decoded"}
    excluded = 0
    diff = {}
    for seed in SEEDS:
        recs = json.loads((OUT / "predict_eval" / f"s{seed}.json").read_text())
        by = {}
        for r in recs:
            by.setdefault(r["recipe"], {})[r["objective"]] = r
        res = {}
        for obj, short in labels.items():
            d, bce = [], []
            for recipe, objs in by.items():
                a, b = objs.get("pathway"), objs.get(obj)
                if not a or not b:
                    continue
                if a["realized_cost"] <= 0 or b["realized_cost"] <= 0:
                    excluded += 1
                    continue
                d.append(float(np.log(b["realized_cost"]) - np.log(a["realized_cost"])))
                bce.append(b["pathway"])
            res[short] = {"n": len(d), "median_delta": float(np.median(d)), "mean_delta": float(np.mean(d)),
                          "share_cheaper": float(np.mean(np.array(d) < 0)), "median_bce": float(np.median(bce)),
                          "deltas": d}
            pooled.setdefault(short, []).extend(d)
        by_recipe = [(float(np.log(o["pathway + decoded"]["realized_cost"]) - np.log(o["pathway + latent"]["realized_cost"])))
                     for o in by.values() if all(k in o and o[k]["realized_cost"] > 0
                                                 for k in ("pathway + decoded", "pathway + latent"))]
        res["decoded_minus_latent_median"] = float(np.median(by_recipe))
        diff[seed] = res["decoded_minus_latent_median"]
        per_model[seed] = res
    k = sum(v < 0 for v in diff.values())
    pooled_stats = {s: {"median_delta": float(np.median(v)), "mean_delta": float(np.mean(v)),
                        "share_cheaper": float(np.mean(np.array(v) < 0)), "n": len(v)} for s, v in pooled.items()}
    across = {s: {"median_delta": rng_summary([per_model[m][s]["median_delta"] for m in SEEDS]),
                  "share_cheaper": rng_summary([per_model[m][s]["share_cheaper"] for m in SEEDS]),
                  "median_bce": rng_summary([per_model[m][s]["median_bce"] for m in SEEDS])}
              for s in labels.values()}
    for m in per_model.values():
        for s in labels.values():
            m[s].pop("deltas")
    return {"per_model": per_model, "pooled_over_recipes": pooled_stats, "across_models": across,
            "decoded_cheaper_than_latent_models": k, "sign_test_p": sign_p(k), "excluded_nonpositive_cost": excluded}


def agg_slots():
    bands = {key: {"per_model": [], "n": []} for key in ("low", "mid", "high", "database", "poison", "all")}
    for seed in SEEDS:
        res = json.loads(part("slots", seed).read_text())
        for key in bands:
            bands[key]["per_model"].append(res["bands"][key]["acc"])
            bands[key]["n"].append(res["bands"][key]["n"])
    out = {key: {**(rng_summary(v["per_model"]) or {"median": None, "min": None, "max": None, "mean": None, "n": 0}),
                 "per_model": v["per_model"], "n_slots_per_model": v["n"]}
           for key, v in bands.items()}
    low = [a for a in bands["low"]["per_model"] if a is not None]
    out["low_band_at_least_0.40"] = int(sum(a >= 0.40 for a in low))
    out["low_band_models_with_slots"] = len(low)

    base = {}
    for seed in SEEDS:
        for r in json.loads(part("slots", seed).read_text())["base"]:
            e = base.setdefault(r["truth"], {"correct": 0, "n": 0, "true_conc": r["true_conc"], "decoded_as": {}})
            e["correct"] += r["hit"]
            e["n"] += 1
            e["decoded_as"][r["decoded"]] = e["decoded_as"].get(r["decoded"], 0) + 1
    return out, base


def agg_norms():
    res = [json.loads(part("norms", s).read_text()) for s in SEEDS]
    keys = ("ratio_high_over_low",)
    out = {k: rng_summary([r[k] for r in res]) for k in keys}
    for b in ("low", "mid", "high"):
        out[b] = {"n": res[0][b]["n"], "median_ident_norm": rng_summary([r[b]["median_ident_norm"] for r in res]),
                  "median_dose_share": rng_summary([r[b]["median_dose_share"] for r in res])}
    out["n_scalars"], out["dim"] = res[0]["n_scalars"], res[0]["dim"]
    return out


def agg_geometry():
    parts = [json.loads(part("geometry", s).read_text()) for s in SEEDS]
    names = parts[0]["names"]
    raw = np.array(parts[0]["raw"])
    v_raw, v_cooc = lgc.off(lgc.simmat(raw)), lgc.off(np.array(parts[0]["cooc"]))
    res = {"n_proteins": len(names), "names": names}

    def mats(kind, tag):
        return [lgc.simmat(np.array(p["codes"][f"{kind}__{tag}"], dtype=np.float64)) for p in parts]

    def cosmed(kind, tag):
        out = []
        for p in parts:
            X = np.array(p["codes"][f"{kind}__{tag}"], dtype=np.float64)
            out.append(float(np.median(lgc.off(lgc.simmat(X, center=False)))))
        return out

    def per_model(M, nuis):
        out = []
        for i in range(len(M)):
            vals = []
            for j in range(len(M)):
                if i == j:
                    continue
                va, vb = lgc.off(M[i]), lgc.off(M[j])
                if nuis:
                    va, vb = lgc.resid(va, nuis), lgc.resid(vb, nuis)
                vals.append(spearmanr(va, vb).statistic)
            out.append(float(np.mean(vals)))
        return out

    for kind in ("trained", "untrained"):
        res[kind] = {}
        for tag in DOSES:
            M = mats(kind, tag)
            entry = {"pair": list(lgc.pair_rsa(M)), "per_model_pair_rsa": per_model(M, ()),
                     "median_uncentered_cosine": rng_summary(cosmed(kind, tag)),
                     "agree_with_10": rng_summary([float(spearmanr(lgc.off(a), lgc.off(b)).statistic)
                                                   for a, b in zip(M, mats(kind, "10"))])}
            if tag == "10":
                entry["partial_protT5"] = list(lgc.pair_rsa(M, [v_raw]))
                entry["partial_both"] = list(lgc.pair_rsa(M, [v_raw, v_cooc]))
                entry["per_model_partial_protT5"] = per_model(M, [v_raw])
                entry["vs_protT5"] = rng_summary([float(spearmanr(lgc.off(m), v_raw).statistic) for m in M])
            res[kind][tag] = entry
    res["protT5_vs_cooc"] = float(spearmanr(v_raw, v_cooc).statistic)
    return res


def agg_recovery():
    per = {s: json.loads(part("recovery", s).read_text()) for s in SEEDS}
    untrained = {r["accession"]: r for r in per[0]["untrained"]}
    out = []
    for r0 in per[0]["trained"]:
        acc = r0["accession"]
        rows = [r for s in SEEDS for r in per[s]["trained"] if r["accession"] == acc]
        ratio = float(np.mean([r["ratio"] for r in rows]))
        placebo = [r["placebo_ratio"] for r in rows if r.get("placebo_ratio")]
        placebo_r = float(np.mean(placebo)) if placebo else None
        un = untrained[acc]["ratio"] if acc in untrained else None
        out.append({"accession": acc, "name": r0["name"], "n_recipes": r0["n_recipes"],
                    "ratio_mean": ratio, "ratio_median": float(np.median([r["ratio"] for r in rows])),
                    "ratio_range": [float(min(r["ratio"] for r in rows)), float(max(r["ratio"] for r in rows))],
                    "placebo_ratio": placebo_r, "untrained_ratio": un,
                    "models_ratio_gt1_p_lt05": int(sum(r["ratio"] > 1 and r["null_p"] < 0.05 for r in rows)),
                    "specific": bool(placebo_r is not None and un is not None and ratio > placebo_r and ratio > un)})
    return out


def f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def rs(d, n=3):
    if d["median"] is None:
        return "undefined (no slots)"
    return f"{f(d['median'], n)} ({f(d['min'], n)} to {f(d['max'], n)})"


def write(payload):
    p, s, g, r = payload["predict"], payload["slots"], payload["geometry"], payload["recovery"]
    L = ["# diagnostics: new analyses on the current encoding (ADR-045)", "",
         f"Checkpoints: `artifacts/ablation__selected__728320fe947b__s{{0..9}}__graded.pth`. Script `scripts/diagnostics.py`, "
         f"data `artifacts/ablation_analysis/diagnostics.json`. Median (min to max) over the ten models unless stated. "
         f"Every figure below comes from that JSON.", "",
         "## A1 cost control (ADR-006 protocol, 30 L-BFGS steps, delta = ln realized USD/L minus pathway-only)", "",
         "| term | median delta across models | share cheaper across models | pathway BCE | pooled over recipes: median, mean, cheaper (n) |",
         "|---|---|---|---|---|"]
    for t in ("proxy", "latent", "decoded"):
        a, q = p["across_models"][t], p["pooled_over_recipes"][t]
        L.append(f"| {t} | {rs(a['median_delta'])} | {rs(a['share_cheaper'], 2)} | {rs(a['median_bce'], 3)} | "
                 f"{f(q['median_delta'])}, {f(q['mean_delta'])}, {f(q['share_cheaper'], 2)} ({q['n']}) |")
    L += ["", f"Decoded term cheaper than the latent term (per-model median of the per-recipe difference below 0) in "
          f"{p['decoded_cheaper_than_latent_models']} of 10 models, one-sided sign test p = {p['sign_test_p']:.4f}. "
          f"Recipes dropped for a non-positive realized cost: {p['excluded_nonpositive_cost']}.", "",
          "Per model, median delta (share cheaper): latent | decoded", ""]
    for m in SEEDS:
        pm = p["per_model"][m]
        L.append(f"- s{m}: {pm['latent']['median_delta']:.3f} ({pm['latent']['share_cheaper']:.2f}) | "
                 f"{pm['decoded']['median_delta']:.3f} ({pm['decoded']['share_cheaper']:.2f}), n = {pm['decoded']['n']}")
    L += ["", "## A2 dose-split encoding", "", "B1 UniProt slot identity accuracy, val + test, per model:", "",
          "| band | median (min to max) | slots per model |", "|---|---|---|"]
    for k in ("low", "mid", "high", "database", "poison", "all"):
        L.append(f"| {k} | {rs(s['bands'][k], 3)} | {s['bands'][k]['n_slots_per_model'][0]} to "
                 f"{max(s['bands'][k]['n_slots_per_model'])} |")
    L += ["", f"Low-band accuracy at least 0.40 in {s['bands']['low_band_at_least_0.40']} of "
          f"{s['bands']['low_band_models_with_slots']} models (bar scaled from 2 of 3: 7 of 10).", "",
          "B2 base recipe, noise-free decode, UniProt slots correct out of 10 models:", "",
          "| protein | dose mg/L | correct | most frequent decoded name |", "|---|---|---|---|"]
    for name, e in sorted(s["base"].items(), key=lambda kv: kv[0]):
        top = max(e["decoded_as"].items(), key=lambda kv: kv[1])
        L.append(f"| {name} | {e['true_conc']} | {e['correct']}/{e['n']} | {top[0]} ({top[1]}) |")
    n = payload["norms"]
    L += ["", f"B3 input row (dim {n['dim']}, {n['n_scalars']} scalar channel(s)): median identity-block norm "
          f"low {rs(n['low']['median_ident_norm'], 2)} (n = {n['low']['n']} slots), high "
          f"{rs(n['high']['median_ident_norm'], 2)} (n = {n['high']['n']}); ratio high over low "
          f"{rs(n['ratio_high_over_low'], 3)}; median dose-channel share of squared row norm low "
          f"{rs(n['low']['median_dose_share'], 5)}, high {rs(n['high']['median_dose_share'], 5)}. Input-side only.", "",
          f"## A3 latent geometry ({g['n_proteins']} UniProt components, one-slot probes, mean-centered cosine)", "",
          "| probe dose mg/L | trained pair RSA | untrained pair RSA | trained median cosine (uncentered) | untrained median cosine | trained same-model agreement with 10 mg/L |",
          "|---|---|---|---|---|---|"]
    for tag in DOSES:
        t, u = g["trained"][tag], g["untrained"][tag]
        L.append(f"| {tag} | {t['pair'][0]:.3f} +- {t['pair'][1]:.3f} | {u['pair'][0]:.3f} +- {u['pair'][1]:.3f} | "
                 f"{rs(t['median_uncentered_cosine'], 5)} | {rs(u['median_uncentered_cosine'], 5)} | {rs(t['agree_with_10'], 3)} |")
    t, u = g["trained"]["10"], g["untrained"]["10"]
    L += ["", f"At 10 mg/L, partialling out raw ProtT5: trained {t['partial_protT5'][0]:.3f} +- {t['partial_protT5'][1]:.3f}, "
          f"untrained {u['partial_protT5'][0]:.3f} +- {u['partial_protT5'][1]:.3f}; ProtT5 and co-occurrence: trained "
          f"{t['partial_both'][0]:.3f} +- {t['partial_both'][1]:.3f}, untrained {u['partial_both'][0]:.3f} +- {u['partial_both'][1]:.3f}. "
          f"Mean +- sd over the 45 model pairs. Per-model mean pair RSA, trained: "
          f"{', '.join(f'{v:.3f}' for v in t['per_model_pair_rsa'])}.", "",
          "## A4 recovery of knowns (ADR-013 probe, 10 models, one untrained model, placebo)", "",
          "| ligand | recipes | ratio mean | ratio median (range) | placebo | untrained | models with ratio > 1 and p < 0.05 | specific |",
          "|---|---|---|---|---|---|---|---|"]
    for e in r:
        L.append(f"| {e['name']} ({e['accession']}) | {e['n_recipes']} | {f(e['ratio_mean'], 2)} | {f(e['ratio_median'], 2)} "
                 f"({f(e['ratio_range'][0], 2)} to {f(e['ratio_range'][1], 2)}) | {f(e['placebo_ratio'], 2)} | "
                 f"{f(e['untrained_ratio'], 2)} | {e['models_ratio_gt1_p_lt05']}/10 | {'yes' if e['specific'] else 'no'} |")
    LOG_MD.write_text("\n".join(L) + "\n")


def aggregate():
    slots, base = agg_slots()
    slots_payload = {"bands": slots, "base": base}
    payload = {"label": "new analysis, current encoding (ADR-045)", "checkpoints": CKPT, "predict": agg_predict(),
               "slots": slots_payload, "norms": agg_norms(), "geometry": agg_geometry(), "recovery": agg_recovery()}
    (OUT / "diagnostics.json").write_text(json.dumps(payload, indent=1))
    write(payload)
    print("wrote", OUT / "diagnostics.json", LOG_MD)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "aggregate"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "run":
        run(sys.argv[2:])
    else:
        aggregate()
