"""
Hyperparameter optimization for MVRRAE_GV.

Samples the loss weights and the rank cap from geomspace grids, retraining from
scratch per configuration, and measures how much of the latent survives:

    PATHWAY_ALPHA   graph readout weight
    KL_BETA         divergence weight on the closed form arms
    MMD_BETA        divergence weight on the sample based arms
    GATE_GAMMA      Gumbel gate weight
    TSVD_K          reduced rank cap
    RECON_W2        UniProt reconstruction weight
    COST_DELTA      latent cost readout weight (ADR-002)

The full product is a few thousand runs, so configurations are drawn at random from
it, stratified across the four MMD/KL x SVD/no-SVD variants. KL_BETA and MMD_BETA span
the same decades because at their config.py defaults the arms sit 100x apart, and a
comparison between them would measure the weight rather than the divergence.

Collapse is measured spectrally on the noise free code: per axis variance cannot see a
code confined to a tilted subspace, and a spectrum of the sampled code is filled back
in by the sampling noise (a collapsed dimension has mu -> 0 and std -> 1).

No checkpoints are written, so the sweep cannot overwrite the artifacts of an ablation.

Selection runs in three stages (ADR-004), because one random draw over six axes leaves
about five runs per variant at any one (RECON_W2, COST_DELTA) pair:

    screen    every axis sampled; fixes the two weights shared by every arm,
              RECON_W2 on modality balance and COST_DELTA as the smallest weight
              whose cost fit is within COST_R2_TOL of the best
    refine    those two held, the remaining axes resampled per variant
    confirm   the refine leaders retrained over CONFIRM_SEEDS, ranked on the mean

Run:
    python hipo.py                                        # screen + report
    python hipo.py --stage screen --shard 0/3 --out hipo_screen__0.json
    python hipo.py --merge hipo_screen__*.json --out hipo_screen.json
    python hipo.py --stage refine --from hipo_screen.json --shard 0/3 --out hipo_refine__0.json
    python hipo.py --stage confirm --from hipo_refine.json --out hipo_confirm.json
    python hipo.py --select                               # writes hipo_selection.json
    python hipo.py --apply                                # writes them into src/config.py
"""

import argparse
import glob
import json
import re
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src import benchmark, config
from src.ablation import (
    N_FOLDS,
    AblationSpec,
    evaluate as evaluate_spec,
    parent_folds,
    train_spec,
)

__all__ = [
    "evaluate",
    "main",
    "plot_hyperparameter_space",
    "report",
    "save_results",
    "run_sweep",
    "sample_configs",
    "select_final",
    "select_shared",
    "train_config",
]

SEED = 67
TAU_FLOOR = 0.1                     # tau the training anneal bottoms out at

# Grouped folds (ADR-023 decision 6). Fold 0 is the test fold and enters no stage
# (AC-H2), so every stage validates on fold 1.
FOLD = 1
NEGATIVES = "graded"

LATENT_DIM = 32                     # d_z_max, the colour scale runs 0..this

# loss_reg defaults sigma=1, but the codes carry norms around 15, so exp(-d^2/2)
# underflows and the MMD is a constant with an exactly zero gradient. Held at
# the pairwise distance scale so the term trains something.
SIGMA_REF = 16

# Geomspace axes, each config.py default sits inside its span
PATHWAY_ALPHA = np.geomspace(1e-3, 1e0, 4)      # default 1e-1
KL_BETA = np.geomspace(1e-5, 1e-1, 5)           # default 1e-4
MMD_BETA = np.geomspace(1e-5, 1e-1, 5)          # default 1e-2
GATE_GAMMA = np.geomspace(1e-4, 1e-1, 4)        # default 1e-2
TSVD_K = np.unique(np.geomspace(2, LATENT_DIM, 5).round().astype(int))
RECON_W2 = np.geomspace(1e0, 1e3, 4)
COST_DELTA = np.geomspace(1e-3, 1e1, 5)         # default 1e-1

SAMPLES_PER_VARIANT = 64            # screen
REFINE_PER_VARIANT = 48             # refine, RECON_W2 and COST_DELTA held
CONFIRM_TOP = 5                     # refine leaders retrained per confirmed variant
CONFIRM_SEEDS = (67, 1, 2)
CONFIRM_VARIANTS = ("MMD + tSVD", "KL + tSVD")   # the ablation baseline and its no_mmd arm

# COST_DELTA is the smallest weight whose median held-out cost R^2 is within this
# of the best weight's median. Larger weights fit cost no better and only take
# capacity from the rest of the objective.
COST_R2_TOL = 0.05

# (label, ABLATE_MMD, ABLATE_SVD)
VARIANTS = (
    ("MMD + tSVD", "0", "0"),
    ("MMD", "0", "1"),
    ("KL + tSVD", "1", "0"),
    ("KL", "1", "1"),
)

def sample_configs(rng=None, fixed=None, samples=SAMPLES_PER_VARIANT):
    """
    Draw configurations from the geomspace product evenly across variants.

    ``fixed`` pins ``recon_w2`` and/or ``cost_delta`` to one value, the refine stage.
    """
    rng = np.random.default_rng(SEED) if rng is None else rng
    fixed = fixed or {}
    w2_axis = np.array([fixed["recon_w2"]]) if "recon_w2" in fixed else RECON_W2
    delta_axis = np.array([fixed["cost_delta"]]) if "cost_delta" in fixed else COST_DELTA
    configs = []

    for label, ablate_mmd, ablate_svd in VARIANTS:
        beta_axis = KL_BETA if ablate_mmd == "1" else MMD_BETA
        # TSVD_K is inert without the SVD branch, so it is not sampled there
        k_axis = np.array([LATENT_DIM]) if ablate_svd == "1" else TSVD_K

        product = [(a, b, g, k, w, c)
                   for a in PATHWAY_ALPHA for b in beta_axis
                   for g in GATE_GAMMA for k in k_axis for w in w2_axis
                   for c in delta_axis]
        take = min(samples, len(product))

        for i in rng.choice(len(product), size=take, replace=False):
            alpha, beta, gamma, tsvd_k, recon_w2, cost_delta = product[i]
            configs.append({
                "label": label,
                "divergence": "KL" if ablate_mmd == "1" else "MMD",
                "svd": ablate_svd == "0",
                "ablate_mmd": ablate_mmd,
                "ablate_svd": ablate_svd,
                "pathway_alpha": float(alpha),
                "div_beta": float(beta),
                "gate_gamma": float(gamma),
                "tsvd_k": int(tsvd_k),
                "recon_w2": float(recon_w2),
                "cost_delta": float(cost_delta),
                "seed": SEED,
            })

    return configs





def _spec_fold(fold):
    """Map hipo's validation fold onto the harness's test-fold argument.

    The harness names a fold by its *test* group, while a stage here scores its
    *validation* fold and never iterates test. Fold 1 is the harness's fold-0 split (test
    group 0, validation 1, train 2 to 4); folds 2-4 validate at ``fold - 1``.
    """
    return 0 if fold == 1 else fold - 1


# ADR-020: softmax unless src/config.py names the additive readout the pilot adopted.
def _default_readout():
    return getattr(config, "READOUT", "softmax")


def _spec(cfg, epochs=None, fold=None, readout=None, negatives=None):
    return AblationSpec(
        label=cfg["label"],
        use_svd=bool(cfg["svd"]),
        use_mmd=cfg["divergence"] == "MMD",
        seed=int(cfg.get("seed", SEED)),
        fold=_spec_fold(FOLD if fold is None else fold),
        epochs=epochs or config.EPOCHS,
        latent_dim=LATENT_DIM,
        tsvd_k=int(cfg["tsvd_k"]),
        pathway_alpha=float(cfg["pathway_alpha"]),
        div_beta=float(cfg["div_beta"]),
        gate_gamma=float(cfg["gate_gamma"]),
        recon_w2=float(cfg["recon_w2"]),
        cost_delta=float(cfg.get("cost_delta", config.COST_DELTA)),
        negatives=NEGATIVES if negatives is None else negatives,
        readout=_default_readout() if readout is None else readout,
    )


def _parent(dataset, index):
    data = getattr(dataset, "data", None)
    return (data[index] if data is not None else dataset[index]).get("parent")


def _fold_index(dataset, spec, indices):
    """The grouped fold a set of record indices belongs to, by the harness's own split."""
    indices = sorted(indices)
    for k in range(N_FOLDS):
        if sorted(parent_folds(dataset, replace(spec, fold=k))["test"]) == indices:
            return k
    return None


def _check_fold(dataset, spec, loaders, fold):
    """Refuse a split that reaches fold 0, or validates off the requested fold (AC-H2)."""
    if all(_parent(dataset, i) is None for i in range(len(dataset))):
        return

    used = set(loaders["train"].dataset.indices) | set(loaders["val"].dataset.indices)
    held = set(parent_folds(dataset, replace(spec, fold=0))["test"])
    leaked = [i for i in used if i in held or _parent(dataset, i) in held]
    if leaked:
        raise SystemExit(f"fold 0 reaches the train or validation loader "
                         f"({len(leaked)} of {len(used)} records); AC-H2 forbids it")

    val = [i for i in loaders["val"].dataset.indices if _parent(dataset, i) is None]
    if not val or _fold_index(dataset, spec, val) != fold:
        raise SystemExit(f"the validation loader is not fold {fold} of the grouped split")


def train_config(cfg, epochs=None, dataset_path=None):
    """Retrain from scratch under one point of the sweep; returns its validation loader."""
    fold = FOLD if cfg.get("fold") is None else int(cfg["fold"])
    spec = _spec(cfg, epochs, fold, cfg.get("readout"), cfg.get("negatives"))
    run, store, loaders, _ = train_spec(spec, dataset_path, verbose=False)
    _check_fold(run.dataset, spec, loaders, fold)
    return run, store, loaders["val"]


def evaluate(run, store, loader, tau=TAU_FLOOR):
    """Held-out metrics of one trained model, computed by the shared ablation harness."""
    return evaluate_spec(run, store, loader, tau=tau)


def run_sweep(configs=None, epochs=None, fold=FOLD, readout=None, negatives=NEGATIVES,
              dataset_path=None):
    configs = sample_configs() if configs is None else configs
    readout = _default_readout() if readout is None else readout
    results = []

    header = (f"{'variant':<14} {'alpha':>8} {'beta':>8} {'gamma':>8} {'k':>4} {'w2':>8} "
              f"{'delta':>8} | {'active':>7} {'PR':>6} {'eff_rank':>9} {'recon':>9} {'bal':>7} "
              f"{'div':>9} {'gate':>6} {'costR2':>7}")
    bar = tqdm(configs, desc="sweep", unit="run", dynamic_ncols=True)
    bar.write(f"\n{header}\n" + "-" * 125)

    for cfg in bar:
        # Every row carries its fold, readout and negative set: merging or selecting
        # rows scored under different settings is refused, not averaged over.
        cfg = {**cfg, "fold": int(fold), "readout": readout, "negatives": negatives}
        bar.set_postfix_str(f"{cfg['label']}  k={cfg['tsvd_k']}")
        run, store, val_loader = train_config(cfg, epochs, dataset_path)
        metrics = evaluate(run, store, val_loader)
        metrics.update(cfg)
        run = store = val_loader = None
        results.append(metrics)

        flag = "  <- collapsed" if metrics["collapsed"] else ""
        cost_r2 = metrics["cost_r2"] if metrics["cost_r2"] is not None else float("nan")
        bar.write(f"{cfg['label']:<14} {cfg['pathway_alpha']:8.1e} {cfg['div_beta']:8.1e} "
                  f"{cfg['gate_gamma']:8.1e} {cfg['tsvd_k']:4d} {cfg['recon_w2']:8.1e} "
                  f"{cfg['cost_delta']:8.1e} | "
                  f"{metrics['active']:7d} "
                  f"{metrics['pr']:6.2f} {metrics['eff_rank']:9.3f} {metrics['recon']:9.3f} "
                  f"{metrics['balance']:7.2f} "
                  f"{metrics['div_loss']:9.4f} {metrics['gate_open']:6.2f} {cost_r2:7.3f}{flag}")

    return results







def plot_hyperparameter_space(results, filename="hipo_hyperparameter_space.png"):
    """The sweep's three loss weights, drawn with the shared collapse language."""
    return benchmark.plot_latent_scatter3d(
        results,
        axes=(("div_beta", "div_beta"),
              ("gate_gamma", "gate_gamma"),
              ("pathway_alpha", "pathway_alpha")),
        d_z=LATENT_DIM,
        size_key="tsvd_k",
        size_range=(TSVD_K.min(), TSVD_K.max()),
        filename=filename,
    )


def save_results(results, filename="hipo_sweep.json", stage="screen", fixed=None):
    """Persist every explored point so the sweep does not have to be rerun."""
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACT_DIR / filename
    with open(path, "w") as f:
        json.dump({
            "seed": SEED,
            "latent_dim": LATENT_DIM,
            "sigma": SIGMA_REF,
            "samples_per_variant": SAMPLES_PER_VARIANT,
            "stage": stage,
            "fixed": fixed or {},
            "axes": {
                "PATHWAY_ALPHA": PATHWAY_ALPHA.tolist(),
                "KL_BETA": KL_BETA.tolist(),
                "MMD_BETA": MMD_BETA.tolist(),
                "GATE_GAMMA": GATE_GAMMA.tolist(),
                "TSVD_K": TSVD_K.tolist(),
                "RECON_W2": RECON_W2.tolist(),
                "COST_DELTA": COST_DELTA.tolist(),
            },
            "results": results,
        }, f, indent=2)

    print(f"\nsaved {len(results)} explored points to {path}")
    return path


def _coefficients(row):
    return (f"PATHWAY_ALPHA={row['pathway_alpha']:.3e}  "
            f"{'KL_BETA' if row['divergence'] == 'KL' else 'MMD_BETA'}={row['div_beta']:.3e}  "
            f"GATE_GAMMA={row['gate_gamma']:.3e}  TSVD_K={row['tsvd_k']}  "
            f"RECON_W2={row['recon_w2']:.0e}  COST_DELTA={row.get('cost_delta', 0):.0e}")


def report(results):
    print("\n============================")
    for label, _, _ in VARIANTS:
        rows = [r for r in results if r["label"] == label]
        if not rows:
            continue
        best = max(rows, key=lambda r: (r["active"], -r["recon"]))
        n_dead = sum(r["collapsed"] for r in rows)
        print(f"{label:<14} best {best['active']:2d}/{best['d_z']} live at "
              f"alpha={best['pathway_alpha']:.1e} beta={best['div_beta']:.1e} "
              f"gamma={best['gate_gamma']:.1e} k={best['tsvd_k']:<2d} "
              f"recon {best['recon']:6.3f}  |  {n_dead}/{len(rows)} runs collapsed")

    overall = max(results, key=lambda r: (r["active"], -r["recon"]))
    print(f"\nmost directions kept alive: {overall['label']} with "
          f"{overall['active']}/{overall['d_z']}")

    # Best coefficients out of the explored geomspace, per arm
    # Ranked within a divergence only, an MMD value and a KL value are not the same quantity.
    print("\nbest coefficients by smallest divergence loss")
    for divergence in ("MMD", "KL"):
        arm = [r for r in results if r["divergence"] == divergence]
        if not arm:
            continue
        best = min(arm, key=lambda r: r["div_loss"])
        print(f"  {divergence:<3} {best['label']:<14} div={best['div_loss']:.4f}  "
              f"active={best['active']:2d}/{best['d_z']}  recon={best['recon']:.3f}")
        print(f"      {_coefficients(best)}")

    print("\nbest coefficients by surviving width")
    for divergence in ("MMD", "KL"):
        arm = [r for r in results if r["divergence"] == divergence]
        if not arm:
            continue
        best = max(arm, key=lambda r: (r["active"], -r["recon"]))
        print(f"  {divergence:<3} {best['label']:<14} active={best['active']:2d}/{best['d_z']}  "
              f"recon={best['recon']:.3f}  div={best['div_loss']:.4f}")
        print(f"      {_coefficients(best)}")
    print("============================")


def _rank(row):
    """The ordering hipo.report and latent_space_characterization share."""
    return (row["active"], -row["recon"])


def select_shared(results):
    """
    The two weights held common to every arm, read off the screen.

    RECON_W2 on modality balance alone, the rule latent_space_characterization
    already uses: the grid value whose held-out nmse_x1 and nmse_x2 sit closest.
    COST_DELTA as the smallest grid value whose median held-out cost R^2 is
    within COST_R2_TOL of the best one's. Both are medians over every variant
    and every other axis, which is the only slice the screen has enough runs in.
    """
    def median_at(key, value, metric):
        vals = [metric(r) for r in results if np.isclose(r[key], value)]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    weights = sorted({r["recon_w2"] for r in results})
    imbalance = {w: median_at("recon_w2", w, lambda r: abs(np.log(max(r["balance"], 1e-12))))
                 for w in weights}
    recon_w2 = min(weights, key=lambda w: imbalance[w])

    deltas = sorted({r["cost_delta"] for r in results})
    fit = {d: median_at("cost_delta", d, lambda r: r["cost_r2"]) for d in deltas}
    best = max(v for v in fit.values() if np.isfinite(v))
    cost_delta = min(d for d in deltas if np.isfinite(fit[d]) and fit[d] >= best - COST_R2_TOL)

    print("\nshared weights from the screen")
    for w in weights:
        print(f"  RECON_W2   {w:9.3e}  median |log nmse_x2/nmse_x1| {imbalance[w]:.4f}"
              f"{'  <- chosen' if w == recon_w2 else ''}")
    for d in deltas:
        print(f"  COST_DELTA {d:9.3e}  median cost R^2 {fit[d]:.4f}"
              f"{'  <- chosen' if d == cost_delta else ''}")

    return {"recon_w2": float(recon_w2), "cost_delta": float(cost_delta),
            "imbalance": {str(k): v for k, v in imbalance.items()},
            "cost_r2_by_delta": {str(k): v for k, v in fit.items()}}


def confirm_configs(refine):
    """The refine leaders of each confirmed variant, once per seed."""
    configs = []
    for label in CONFIRM_VARIANTS:
        arm = sorted([r for r in refine if r["label"] == label], key=_rank, reverse=True)
        for rank, row in enumerate(arm[:CONFIRM_TOP]):
            cfg = {k: row[k] for k in ("label", "divergence", "svd", "ablate_mmd", "ablate_svd",
                                       "pathway_alpha", "div_beta", "gate_gamma", "tsvd_k",
                                       "recon_w2", "cost_delta")}
            for seed in CONFIRM_SEEDS:
                configs.append({**cfg, "seed": seed, "refine_rank": rank})
    return configs


def select_final(confirm):
    """Per confirmed variant, the candidate with the best mean over seeds.

    A candidate that did not cover every entry of CONFIRM_SEEDS is dropped rather
    than ranked: its mean would rest on a different seed subset than its rivals'.
    """
    keys = ("pathway_alpha", "div_beta", "gate_gamma", "tsvd_k", "recon_w2", "cost_delta")
    chosen = {}
    for label in CONFIRM_VARIANTS:
        groups = {}
        for row in (r for r in confirm if r["label"] == label):
            groups.setdefault(tuple(row[k] for k in keys), []).append(row)
        if not groups:
            continue

        def summary(rows):
            out = {m: float(np.mean([r[m] for r in rows]))
                   for m in ("active", "recon", "cost_r2", "recipe_cost_rho", "nmse_x1",
                             "nmse_x2", "div_loss", "gate_open")
                   if all(r.get(m) is not None for r in rows)}
            out["active_sd"] = float(np.std([r["active"] for r in rows]))
            out["n_seeds"] = len(rows)
            return out

        complete, dropped = {}, {}
        for values, rows in groups.items():
            seeds = {r.get("seed") for r in rows}
            absent = [s for s in CONFIRM_SEEDS if s not in seeds]
            if absent:
                dropped[values] = absent
            else:
                complete[values] = rows

        print(f"\n{label}: confirm over {len(CONFIRM_SEEDS)} seeds")
        for values, absent in dropped.items():
            print(f"  dropped {dict(zip(keys, values))}: no seed {absent}")
        if not complete:
            print(f"  no candidate ran every seed; {label} not ranked")
            continue

        ranked = sorted(complete.items(), key=lambda kv: (summary(kv[1])["active"],
                                                          -summary(kv[1])["recon"]), reverse=True)
        for values, rows in ranked:
            m = summary(rows)
            print(f"  {dict(zip(keys, values))}\n      active {m['active']:.2f} +/- {m['active_sd']:.2f}"
                  f"  recon {m['recon']:.3f}  cost R2 {m.get('cost_r2', float('nan')):.3f}"
                  f"  recipe rho {m.get('recipe_cost_rho', float('nan')):.3f}")
        values, rows = ranked[0]
        chosen[label] = {**dict(zip(keys, values)), **summary(rows)}
    return chosen


def _stage_settings(rows, name, strict=True):
    """
    The (fold, readout, negatives) a stage artifact was produced under.

    Rows from different settings would compare a fold against itself, so a mixed merge is
    refused. Shards that carry none of the three only merge as a set. ``epochs`` and
    ``dataset_path`` join the tuple once any row records them.
    """
    if not rows:
        raise SystemExit(f"{name} matched no rows")
    keys = ["fold", "readout", "negatives"]
    keys += [k for k in ("epochs", "dataset_path")
             if any(r.get(k) is not None for r in rows)]
    seen = {tuple(r.get(k) for k in keys) for r in rows}
    missing = [k for k in seen if k[0] is None or k[1] is None]
    if missing and len(missing) != len(seen):
        raise SystemExit(f"{name} mixes rows with and without fold/readout")
    if missing:
        if strict:
            raise SystemExit(f"{name} rows carry no fold/readout")
        return None
    if len(seen) != 1:
        raise SystemExit(f"{name} rows mix {'/'.join(keys)} "
                         f"{sorted(seen, key=str)}; re-run the stage under one setting")
    return dict(zip(keys, seen.pop()))


def _check_stage_agreement(settings):
    """Refuse a selection whose stages were produced under different settings."""
    keys = sorted({k for found in settings.values() for k in found})
    unrecorded = object()
    groups = {}
    for name, found in settings.items():
        groups.setdefault(tuple(found.get(k, unrecorded) for k in keys), []).append(name)
    if len(groups) == 1:
        return
    reference = max(groups.values(), key=len)

    def describe(names):
        found = settings[names[0]]
        parts = ", ".join(f"{k}={found[k]!r}" if k in found else f"{k} unrecorded"
                          for k in keys)
        return "{" + parts + "}"

    for names in groups.values():
        if names is reference:
            continue
        raise SystemExit(f"{' and '.join(names)} was produced under {describe(names)}, "
                         f"{' and '.join(reference)} under {describe(reference)}; "
                         f"--select reads one setting across screen/refine/confirm")


def _load(pattern):
    rows = []
    for path in sorted(glob.glob(str(config.ARTIFACT_DIR / pattern))):
        rows += json.load(open(path))["results"]
    return rows


def _shard(configs, shard):
    if not shard:
        return configs
    i, n = (int(v) for v in shard.split("/"))
    return configs[i::n]


# The 2026-10-06 artifacts the defaults still name. ADR-023 replaces what they
# hold, so reaching one through an explicit path takes an opt-in.
LEGACY_ARTIFACTS = ("hipo_screen.json", "hipo_refine.json", "hipo_confirm.json",
                    "hipo_selection.json")


def _artifact_path(value, flag, allow_legacy):
    """A bare filename sits under config.ARTIFACT_DIR; a path with a directory is taken as given."""
    path = Path(value)
    if path.parent == Path("."):
        path = config.ARTIFACT_DIR / path
    path = path.resolve()
    legacy = {p.resolve() for p in (config.ARTIFACT_DIR / n for n in LEGACY_ARTIFACTS)}
    if path in legacy and not allow_legacy:
        raise SystemExit(f"{flag} names the 2026-10-06 artifact {path}; "
                         f"pass --allow-legacy to use it")
    return path


SELECTION_KEYS = {
    "PATHWAY_ALPHA": ("MMD + tSVD", "pathway_alpha"),
    "MMD_BETA": ("MMD + tSVD", "div_beta"),
    "GATE_GAMMA": ("MMD + tSVD", "gate_gamma"),
    "TSVD_K": ("MMD + tSVD", "tsvd_k"),
    "RECON_W2": ("MMD + tSVD", "recon_w2"),
    "COST_DELTA": ("MMD + tSVD", "cost_delta"),
    "KL_BETA": ("KL + tSVD", "div_beta"),
}


def apply_selection(chosen, path=Path(config.__file__)):
    """Write ``chosen`` (from --select) into src/config.py, printing old -> new."""
    text = path.read_text()
    for name, (variant, key) in SELECTION_KEYS.items():
        value = chosen[variant][key]
        new = str(int(value)) if name == "TSVD_K" else f"{float(value):.0e}"
        pattern = re.compile(rf"^({name} = )(\S+)(.*)$", re.M)
        match = pattern.search(text)
        if match is None:
            raise SystemExit(f"{name} not found in {path}")
        print(f"{name:<14} {match.group(2):>10} -> {new}")
        text = pattern.sub(lambda m: f"{m.group(1)}{new}{m.group(3)}", text, count=1)
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--stage", choices=("screen", "refine", "confirm"), default=None)
    parser.add_argument("--shard", default=None, help="i/n, every n-th config from i")
    parser.add_argument("--from", dest="source", default=None,
                        help="artifact the stage builds on (screen for refine, refine for confirm)")
    parser.add_argument("--out", default=None,
                        help="stage output, a bare filename under artifacts/ or a path")
    parser.add_argument("--merge", nargs="*", default=None,
                        help="concatenate shard files under artifacts/ into --out")
    parser.add_argument("--select", action="store_true",
                        help="read the screen/refine/confirm artifacts, write the selection artifact")
    parser.add_argument("--figures", action="store_true", help="draw the sweep figures")
    parser.add_argument("--screen-out", default=None,
                        help="screen output (default hipo_screen.json)")
    parser.add_argument("--refine-out", default=None,
                        help="refine output (default hipo_refine.json)")
    parser.add_argument("--confirm-out", default=None,
                        help="confirm output (default hipo_confirm.json)")
    parser.add_argument("--select-screen", default=None,
                        help="screen artifact --select reads (default hipo_screen.json)")
    parser.add_argument("--select-refine", default=None,
                        help="refine artifact --select reads (default hipo_refine.json)")
    parser.add_argument("--select-confirm", default=None,
                        help="confirm artifact --select reads (default hipo_confirm.json)")
    parser.add_argument("--selection-out", default=None,
                        help="--select output (default hipo_selection.json)")
    parser.add_argument("--selection-in", default=None,
                        help="--apply input (default hipo_selection.json)")
    parser.add_argument("--allow-legacy", action="store_true",
                        help="permit an explicit path naming a 2026-10-06 artifact")
    parser.add_argument("--data", default=None,
                        help="database to train on (default data/database_graded.json, or "
                             "the legacy database under --negatives legacy)")
    parser.add_argument("--fold", type=int, default=FOLD,
                        help=f"grouped validation fold (default {FOLD})")
    parser.add_argument("--readout", choices=("softmax", "additive"), default=None,
                        help="pathway readout (default src/config.py's READOUT)")
    parser.add_argument("--negatives", choices=("legacy", "graded"), default=NEGATIVES,
                        help=f"negative set, recorded in every row (default {NEGATIVES})")
    parser.add_argument("--epochs", type=int, default=None,
                        help="epochs per run (default config.EPOCHS)")
    parser.add_argument("--apply", action="store_true",
                        help="write the selected coefficients into src/config.py")
    args = parser.parse_args()

    if args.fold == 0:
        parser.error("--fold 0 is the test fold and enters no stage (AC-H2)")
    if not 1 <= args.fold < N_FOLDS:
        parser.error(f"--fold must be in 1..{N_FOLDS - 1}")
    readout = _default_readout() if args.readout is None else args.readout
    data = args.data
    if data is None and args.negatives == "graded":
        data = config.DATA_DIR / "database_graded.json"
    sweep = dict(epochs=args.epochs, fold=args.fold, readout=readout,
                 negatives=args.negatives, dataset_path=data)

    def check_source(rows, name):
        found = _stage_settings(rows, name)
        wanted = {"fold": args.fold, "readout": readout, "negatives": args.negatives}
        if any(found[k] != v for k, v in wanted.items()):
            raise SystemExit(f"{name} was produced under {found}, this stage runs under "
                             f"{wanted}; pass the matching --fold/--readout/--negatives")

    def path_or_default(value, flag, default):
        if value is None:
            return (config.ARTIFACT_DIR / default).resolve()
        return _artifact_path(value, flag, args.allow_legacy)

    stage_out = {s: path_or_default(getattr(args, f"{s}_out"), f"--{s}-out",
                                    f"hipo_{s}.json")
                 for s in ("screen", "refine", "confirm")}
    select_in = {s: path_or_default(getattr(args, f"select_{s}"), f"--select-{s}",
                                    f"hipo_{s}.json")
                 for s in ("screen", "refine", "confirm")}
    selection_out = path_or_default(args.selection_out, "--selection-out",
                                    "hipo_selection.json")
    selection_in = path_or_default(args.selection_in, "--selection-in",
                                   "hipo_selection.json")
    if args.out is not None:
        args.out = str(_artifact_path(args.out, "--out", args.allow_legacy))

    if args.merge is not None:
        paths = [p for pattern in args.merge
                 for p in sorted(glob.glob(str(config.ARTIFACT_DIR / pattern)))]
        if not paths:
            raise SystemExit(f"--merge matched no shard files under {config.ARTIFACT_DIR}")
        payloads = [json.load(open(p)) for p in paths]
        first = payloads[0]
        # The merged payload keeps one stage tag and one fixed dict, so shards that
        # disagree about either cannot be concatenated.
        for path, payload in zip(paths[1:], payloads[1:]):
            for key in ("stage", "fixed"):
                if payload.get(key) != first.get(key):
                    raise SystemExit(
                        f"--merge refuses {Path(path).name} ({key}={payload.get(key)!r}) "
                        f"with {Path(paths[0]).name} ({key}={first.get(key)!r})")
        results = [r for payload in payloads for r in payload["results"]]
        _stage_settings(results, "--merge", strict=False)
        save_results(results, args.out, first.get("stage", "screen"),
                     first.get("fixed"))
        report(results)
        return results

    if args.select:
        screen = json.load(open(select_in["screen"]))["results"]
        refine = json.load(open(select_in["refine"]))["results"]
        confirm = json.load(open(select_in["confirm"]))["results"]
        settings = {name: _stage_settings(rows, name)
                    for name, rows in (("screen", screen), ("refine", refine),
                                       ("confirm", confirm))}
        _check_stage_agreement(settings)
        shared = select_shared(screen)
        chosen = select_final(confirm)
        path = selection_out
        with open(path, "w") as f:
            json.dump({"shared": shared, "chosen": chosen, "cost_r2_tol": COST_R2_TOL,
                       "confirm_seeds": list(CONFIRM_SEEDS), "confirm_top": CONFIRM_TOP,
                       "settings": settings}, f, indent=2)
        print(f"\nwrote {path}")
        return chosen

    if args.apply:
        chosen = json.load(open(selection_in))["chosen"]
        apply_selection(chosen)
        return chosen

    if args.stage is None:
        # Single pass: screen everything, report, draw.
        results = run_sweep(**sweep)
        report(results)
        save_results(results)
        args.figures = True
    else:
        fixed = None
        if args.stage == "screen":
            configs = sample_configs()
        elif args.stage == "refine":
            source = stage_out["screen"] if args.source is None else _artifact_path(
                args.source, "--from", args.allow_legacy)
            screen = json.load(open(source))
            check_source(screen["results"], source.name)
            fixed = {k: v for k, v in select_shared(screen["results"]).items()
                     if k in ("recon_w2", "cost_delta")}
            configs = sample_configs(np.random.default_rng(SEED + 1), fixed, REFINE_PER_VARIANT)
        else:
            source = stage_out["refine"] if args.source is None else _artifact_path(
                args.source, "--from", args.allow_legacy)
            refine = json.load(open(source))
            check_source(refine["results"], source.name)
            fixed = refine.get("fixed")
            configs = confirm_configs(refine["results"])

        configs = _shard(configs, args.shard)
        results = run_sweep(configs, **sweep)
        report(results)
        save_results(results, args.out or str(stage_out[args.stage]), args.stage, fixed)

    if args.figures:
        plot_hyperparameter_space(results)
        benchmark.plot_tradeoff(results, d_z=LATENT_DIM,
                                size_range=(TSVD_K.min(), TSVD_K.max()),
                                filename="hipo_tradeoff.png")
        benchmark.plot_weight_marginals(
            results,
            params=(("pathway_alpha", "PATHWAY_ALPHA"), ("div_beta", "div beta"),
                    ("gate_gamma", "GATE_GAMMA"), ("tsvd_k", "TSVD_K"),
                    ("cost_delta", "COST_DELTA")),
            d_z=LATENT_DIM,
            filename="hipo_marginals.png")
        plt.show()

    return results


if __name__ == "__main__":
    main()
