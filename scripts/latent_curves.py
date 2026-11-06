"""ADR-046: hole, marginal, cluster and spectrum curves of the ten deployed checkpoints.

    PYTHONPATH=. python scripts/latent_curves.py

Writes artifacts/ablation_analysis/latent_curves.json, artifacts/adr/logs/latent_curves.md and
artifacts/figures/ablation/latent_{hole_curves,marginals,clusters}.{pdf,png}.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
from scipy.stats import gaussian_kde, kstest                      # noqa: E402
from torch.utils.data import DataLoader                           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import latent_space_characterization as lsc                       # noqa: E402
from scripts import okabe_ito as oi                                # noqa: E402
from src import benchmark                                         # noqa: E402
from src.ablation import AblationSpec, _build, seed_everything, split_dataset  # noqa: E402
from src.modules.dataset import collate_fn                        # noqa: E402
from src.vector_store import VectorStore                          # noqa: E402

CKPT = str(ROOT / "artifacts/ablation__selected__728320fe947b__s{}__graded.pth")
SEEDS = range(10)
PERCENTILES = (1, 2, 3, 5, 8, 10, 15, 20, 25)
BW_MULTIPLES = (0.5, 0.75, 1, 1.5, 2)
STORED = ROOT / "artifacts/latent_characterization.json"
OUT_JSON = ROOT / "artifacts/ablation_analysis/latent_curves.json"
OUT_MD = ROOT / "artifacts/adr/logs/latent_curves.md"
FIG = ROOT / "artifacts/figures/ablation"


def load(seed):
    payload = torch.load(CKPT.format(seed), map_location="cpu", weights_only=False)
    spec = AblationSpec(**payload["spec"])
    seed_everything(spec.seed)
    run = _build(spec)
    run.model.load_state_dict(payload["state_dict"])
    run.model.eval()
    return spec, run, VectorStore.from_run(run)


def holes_at(projection, percentile=None, bw=None):
    kwargs = {}
    if percentile is not None:
        kwargs["percentile"] = float(percentile)
    if bw is not None:
        kwargs["bw_method"] = float(bw)
    return benchmark.find_holes(projection, **kwargs)["hole_fraction"]


def cluster_dims(item, n_top=3):
    codes, labels = item["codes"], item["cluster_labels"]
    mean, sd = codes.mean(axis=0), codes.std(axis=0)
    out = []
    for cid in range(item["k"]):
        m = labels == cid
        shift = (codes[m].mean(axis=0) - mean) / np.where(sd > 0, sd, 1.0)
        top = np.argsort(-np.abs(shift))[:n_top]
        out.append({"id": cid, "size": int(m.sum()),
                    "top_dims": [{"dim": int(d), "z": float(shift[d])} for d in top]})
    return out


def one_model(seed):
    spec, run, store = load(seed)
    splits = split_dataset(run.dataset, spec)
    loader = DataLoader(splits["test"], run.batch_size, shuffle=False, collate_fn=collate_fn)
    cfg = {"label": f"s{seed}", "ablate_mmd": "1" if run.ablate_mmd else "0",
           "ablate_svd": "1" if run.ablate_svd else "0"}
    item = lsc.characterize(cfg, run, store, loader)

    proj, raw = item["projection"], item["raw_projection"]
    Z = torch.as_tensor(item["codes"], dtype=torch.float32)
    lam = benchmark.latent_spectrum(Z).numpy().astype(float)
    cum = np.cumsum(lam) / lam.sum()
    marg = [float(kstest(proj[:, d], "norm").statistic) for d in range(2)]

    res = {
        "seed": seed,
        "n_codes": int(len(Z)),
        "n_test_recipes": len(splits["test"]),
        "divergence": item["divergence"],
        "svd": item["svd"],
        "ks_dir1": marg[0], "ks_dir2": marg[1], "ks_mean": item["ks"],
        "mean_dir1": float(proj[:, 0].mean()), "mean_dir2": float(proj[:, 1].mean()),
        "hole_fraction": item["hole_fraction"],
        "raw_hole_fraction": item["raw_hole_fraction"],
        "var_top2_pca": item["var_top2_pca"], "var_top2_raw": item["var_top2_raw"],
        "bandwidth": item["bandwidth"],
        "top_dims": [int(d) for d in item["top_dims"]],
        "hole_vs_percentile": {"pca": [holes_at(proj, p) for p in PERCENTILES],
                               "raw": [holes_at(raw, p) for p in PERCENTILES]},
        "hole_vs_bandwidth": {},
        "k": item["k"], "silhouette": item["silhouette"],
        "clusters": cluster_dims(item),
        "active": item["active"], "pr": item["pr"], "d_z": item["d_z"],
        "scale_ratio": item["scale_ratio"],
        "cum_variance": cum.tolist(), "eigenvalues": lam.tolist(),
    }
    for name, p in (("pca", proj), ("raw", raw)):
        factor = gaussian_kde(p.T).factor
        res["hole_vs_bandwidth"][name] = [holes_at(p, bw=m * factor) for m in BW_MULTIPLES]
    keep = {"proj": proj, "labels": item["cluster_labels"], "codes": item["codes"]}
    return res, keep


def pool(values):
    v = np.asarray(values, dtype=float)
    return {"median": np.median(v, axis=0).tolist(), "min": v.min(axis=0).tolist(),
            "max": v.max(axis=0).tolist()}


SCALARS = ("ks_dir1", "ks_dir2", "ks_mean", "mean_dir1", "mean_dir2", "hole_fraction",
           "raw_hole_fraction", "var_top2_pca", "var_top2_raw", "bandwidth", "k",
           "silhouette", "active", "pr", "scale_ratio")


def agreement(pooled):
    stored = json.load(open(STORED))
    keys = (("hole_fraction", "hole_fraction"), ("raw_hole_fraction", "raw_hole_fraction"),
            ("var_top2_pca", "var_top2_pca"), ("var_top2_raw", "var_top2_raw"),
            ("bandwidth", "bandwidth"), ("ks_mean", "ks"), ("pr", "pr"),
            ("active", "active"), ("scale_ratio", "scale_ratio"))
    rows = []
    for mine, theirs in keys:
        vals = [s[theirs] for s in stored]
        lo, hi = pooled[mine]["min"], pooled[mine]["max"]
        rows.append({"scalar": mine, "deployed_median": pooled[mine]["median"],
                     "deployed_min": lo, "deployed_max": hi,
                     "stored_min": min(vals), "stored_max": max(vals),
                     "stored_by_variant": {s["label"]: s[theirs] for s in stored},
                     "stored_inside_deployed_range": [bool(lo <= v <= hi) for v in vals]})
    return rows


def figures(models, keeps):
    FIG.mkdir(parents=True, exist_ok=True)
    blue, grey = oi.BLUE, oi.ORANGE

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, xs, xlabel in (
            (axes[0], "hole_vs_percentile", PERCENTILES, "density percentile cut (percent)"),
            (axes[1], "hole_vs_bandwidth", BW_MULTIPLES, "bandwidth (times Scott)")):
        for plane, color, name in (("pca", blue, "PCA plane"), ("raw", grey, "raw-dimension plane")):
            arr = np.array([m[key][plane] for m in models])
            ax.fill_between(xs, arr.min(0), arr.max(0), color=color, alpha=0.2, linewidth=0)
            ax.plot(xs, np.median(arr, 0), color=color, marker="o", label=name + ", median")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("hole fraction")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    benchmark.panel_labels(axes)
    axes[0].text(0.98, 0.04, "band: range over ten models", transform=axes[0].transAxes,
                 ha="right", fontsize=8.5)
    save(fig, "latent_hole_curves")

    grid = np.linspace(-4, 4, 300)
    normal = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for d, ax in enumerate(axes):
        dens = np.array([gaussian_kde(k["proj"][:, d])(grid) for k in keeps])
        ks = np.array([m[f"ks_dir{d + 1}"] for m in models])
        ax.fill_between(grid, dens.min(0), dens.max(0), color=blue, alpha=0.2, linewidth=0)
        ax.plot(grid, np.median(dens, 0), color=blue,
                label=f"deployed, median KS {np.median(ks):.3f} (range {ks.min():.3f} to {ks.max():.3f})")
        ax.plot(grid, normal, "--", color=oi.DARK_GRAY, label="N(0,1)")
        ax.set_xlabel(f"direction {d + 1} (standardized)")
        ax.set_ylim(0, dens.max() * 1.45)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("density")
    benchmark.panel_labels(axes)
    save(fig, "latent_marginals")

    order = np.argsort([m["silhouette"] for m in models])
    mid = int(order[len(order) // 2])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    ax.scatter(range(10), [m["silhouette"] for m in models], color=blue, zorder=3)
    for i, m in enumerate(models):
        ax.annotate(f"k={m['k']}", (i, m["silhouette"]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(range(10))
    ax.set_xlabel("checkpoint (seed)")
    ax.set_ylabel("silhouette at selected k")
    ax.set_ylim(0, max(m["silhouette"] for m in models) * 1.2)
    k = keeps[mid]
    ax = axes[1]
    for c, (color, marker) in benchmark.cluster_styles(k["labels"]).items():
        sel = k["labels"] == c
        ax.scatter(k["proj"][sel, 0], k["proj"][sel, 1], color=color, marker=marker, s=9,
                   label=f"cluster {c + 1}")
    ax.set_xlabel("direction 1 (standardized)")
    ax.set_ylabel("direction 2 (standardized)")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="best", markerscale=1.2)
    benchmark.panel_labels(axes)
    save(fig, "latent_clusters")


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fmt(p, digits=3):
    return f"{p['median']:.{digits}f} ({p['min']:.{digits}f} to {p['max']:.{digits}f})"


def write_md(models, pooled, curves, agree):
    L = ["# latent curves from the ten deployed checkpoints (ADR-046)", "",
         "Analysis on the deployed ten models, held-out test split, tau = TAU_FLOOR. "
         "Pooled: median (min to max) over the ten checkpoints.", "",
         "## scalars at the reported setting (5 percent, Scott)", "",
         "| scalar | pooled |", "|---|---|"]
    L += [f"| {k} | {fmt(pooled[k])} |" for k in SCALARS]
    L += ["", "## per model", "",
          "| seed | codes | KS d1 | KS d2 | mean d1 | mean d2 | holes PCA | holes raw | var PCA | var raw | k | sil | live | PR | scale |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        L.append(f"| {m['seed']} | {m['n_codes']} | {m['ks_dir1']:.3f} | {m['ks_dir2']:.3f} | "
                 f"{m['mean_dir1']:.3f} | {m['mean_dir2']:.3f} | {m['hole_fraction']:.3f} | "
                 f"{m['raw_hole_fraction']:.3f} | {m['var_top2_pca']:.3f} | {m['var_top2_raw']:.3f} | "
                 f"{m['k']} | {m['silhouette']:.3f} | {m['active']}/{m['d_z']} | {m['pr']:.2f} | "
                 f"{m['scale_ratio']:.2f} |")
    L += ["", "## hole fraction against density percentile", "",
          "| plane | " + " | ".join(f"{p}%" for p in PERCENTILES) + " |",
          "|---|" + "---|" * len(PERCENTILES)]
    for plane in ("pca", "raw"):
        c = curves["hole_vs_percentile"][plane]
        L.append(f"| {plane} | " + " | ".join(
            f"{a:.3f} ({lo:.3f} to {hi:.3f})" for a, lo, hi in zip(c["median"], c["min"], c["max"])) + " |")
    L += ["", "## hole fraction against bandwidth (times Scott)", "",
          "| plane | " + " | ".join(str(b) for b in BW_MULTIPLES) + " |",
          "|---|" + "---|" * len(BW_MULTIPLES)]
    for plane in ("pca", "raw"):
        c = curves["hole_vs_bandwidth"][plane]
        L.append(f"| {plane} | " + " | ".join(
            f"{a:.3f} ({lo:.3f} to {hi:.3f})" for a, lo, hi in zip(c["median"], c["min"], c["max"])) + " |")
    cv = curves["cum_variance"]
    L += ["", "## cumulative covariance variance (median, min, max over models)", "",
          "| directions | median | min | max |", "|---|---|---|---|"]
    for n in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32):
        L.append(f"| {n} | {cv['median'][n - 1]:.4f} | {cv['min'][n - 1]:.4f} | {cv['max'][n - 1]:.4f} |")
    L += ["", "## clusters (top dimensions by standardized centroid shift)", ""]
    for m in models:
        L.append(f"seed {m['seed']}: k={m['k']}, silhouette={m['silhouette']:.3f}")
        for c in m["clusters"]:
            dims = ", ".join(f"z{d['dim']} ({d['z']:+.2f})" for d in c["top_dims"])
            L.append(f"  - cluster {c['id']} (n={c['size']}): {dims}")
    L += ["", "## agreement with artifacts/latent_characterization.json", "",
          "The stored file holds four retrained seed-67 variants, not the deployed checkpoints.", "",
          "| scalar | deployed median (range) | stored range over four variants | stored values inside deployed range |",
          "|---|---|---|---|"]
    for r in agree:
        inside = ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in
                           zip(r["stored_by_variant"], r["stored_inside_deployed_range"]))
        L.append(f"| {r['scalar']} | {r['deployed_median']:.4f} ({r['deployed_min']:.4f} to "
                 f"{r['deployed_max']:.4f}) | {r['stored_min']:.4f} to {r['stored_max']:.4f} | {inside} |")
    OUT_MD.write_text("\n".join(L) + "\n")


def main():
    models, keeps = [], []
    for seed in SEEDS:
        res, keep = one_model(seed)
        models.append(res)
        keeps.append(keep)
        print(f"s{seed}: codes {res['n_codes']} holes {res['hole_fraction']:.3f} "
              f"raw {res['raw_hole_fraction']:.3f} k {res['k']} sil {res['silhouette']:.3f} "
              f"live {res['active']}", flush=True)

    pooled = {k: {"median": float(np.median([m[k] for m in models])),
                  "min": float(min(m[k] for m in models)),
                  "max": float(max(m[k] for m in models))} for k in SCALARS}
    curves = {"percentiles": list(PERCENTILES), "bw_multiples": list(BW_MULTIPLES),
              "hole_vs_percentile": {p: pool([m["hole_vs_percentile"][p] for m in models])
                                     for p in ("pca", "raw")},
              "hole_vs_bandwidth": {p: pool([m["hole_vs_bandwidth"][p] for m in models])
                                    for p in ("pca", "raw")},
              "cum_variance": pool([m["cum_variance"] for m in models])}
    agree = agreement(pooled)
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({"models": models, "pooled": pooled, "curves": curves,
                                    "agreement_with_stored": agree}, indent=1))
    write_md(models, pooled, curves, agree)
    figures(models, keeps)


if __name__ == "__main__":
    main()
