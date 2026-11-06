"""Benchmarks and plots for the learned latent space: per-dimension activity, collapse counts, spectra, holes, clustering."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from scripts import okabe_ito as oi
from src import config
from src.modules.dataset import safe_stack
from src.modules.layers import gumbel_sigmoid
from src.modules.mvrrae_gv import mmd, kl

__all__ = [
    "COLLAPSE_FRACTION",
    "EIGEN_TOL",
    "KL_THRESHOLD",
    "MMD_THRESHOLD",
    "OKABE_ITO",
    "RING_KL",
    "RING_LIVE",
    "RING_MMD",
    "VAR_THRESHOLD",
    "active_directions",
    "best_of",
    "by_divergence",
    "collapse_rings",
    "drop_lines_2d",
    "drop_lines_3d",
    "collect_latent",
    "cost_fit",
    "cost_probe",
    "recipe_cost_rank",
    "count_dead",
    "dim_activity_kl",
    "dim_activity_mmd",
    "dim_activity_var",
    "effective_rank",
    "latent_activity",
    "latent_spectrum",
    "log3d_axis",
    "marker_sizes",
    "mean_code",
    "panel_labels",
    "plot_dim_activity",
    "plot_latent_scatter3d",
    "plot_tradeoff",
    "plot_weight_marginals",
    "ring_color",
    "ringed",
    "jitter",
    "ring_legend",
    "GRID_PER_AXIS",
    "HOLE_BW_SCALES",
    "HOLE_PERCENTILE",
    "SUMMARY_METRICS",
    "find_holes",
    "hole_bandwidth_curve",
    "hole_curve",
    "plot_hole_bandwidth",
    "plot_hole_curve",
    "plot_hole_robustness",
    "plot_latent_manifold",
    "plot_prior_marginals",
    "plot_projection_comparison",
    "plot_reconstruction",
    "plot_spectrum_comparison",
    "plot_summary",
    "prior_quantile_axis",
    "project_active",
    "project_raw",
    "threshold_label",
    "variance_share",
    "CLUSTER_CMAP",
    "plot_latent_clusters_grid",
    "cluster_styles",
    "plot_latent_clusters",
    "top_variance_dims",
    "PRIOR_RING_RADIUS",
    "style",
    "style_3d",
    "DOSE_BANDS",
    "recon_tables",
    "name_slots",
    "recon_score",
    "recon_usefulness",
    "PREDICT_OBJECTIVES",
    "predict_context",
    "predict_objective_summary",
]

OKABE_ITO = list(oi.OKABE_ITO)

# Posterior collapse := dead dimension iff its activity -> 0
VAR_THRESHOLD = 1e-2    # active units, Var[z_d] carries no signal
KL_THRESHOLD = 1e-2     # posterior is the prior, D_KL[d] -> 0
MMD_THRESHOLD = 1e-3    # marginal is indistinguishable from a prior sample


def dim_activity_var(z):
    # Active units, divergence agnostic so every ablation is measured the same way.
    # Per axis, so it cannot see a code that collapsed onto a subspace lying at an
    # angle to the coordinates, and the prior is isotropic so nothing holds the
    # code to the axes. Use latent_spectrum for that.
    z = z.reshape(-1, z.size(-1))                               # (n_samples, d_z)
    return z.var(dim=0)                                         # (d_z,)


def dim_activity_kl(mu, logvar):
    # Closed form D_KL held back from its sum(dim=1).mean() reduction.
    # logvar is (..., 1) under reduced-rank SVD, it broadcasts over d_z.
    per_dim = 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar)     # (n_samples, d_z)
    return per_dim.reshape(-1, per_dim.size(-1)).mean(dim=0)    # (d_z,)


def dim_activity_mmd(z, zr, sigma=1):
    # The Gaussian kernel mixes every coordinate into one distance
    # Each dimension is scored on its own 1D marginal against the prior sample
    activity = [
        mmd(z[..., d:d + 1], zr[..., d:d + 1], sigma).mean()
        for d in range(z.size(-1))
    ]
    return torch.stack(activity)                                # (d_z,)


def count_dead(activity, threshold):
    return int((activity < threshold).sum())


def collect_latent(run, store, loader, tau=1.0):
    """Gated latent code and divergence args of a trained model over a loader."""
    model, device = run.model, run.device
    codes, div_a, div_b = [], [], []

    # z_c is fed the gated Yk in every ablation branch, so its input is the one
    # code all four variants share. forward() only hands back (mu, logvar)
    # under ABLATE_MMD, and those live in the SVD coefficient space instead.
    def capture(module, args):
        Yk = args[0]                                            # (batch, n_comp, d_z)
        codes.append(Yk.detach().reshape(-1, Yk.size(-1)).cpu())

    handle = model.z_c.register_forward_pre_hook(capture)
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                x1_batch, x2_batch = store.vectorize_batch(batch, device)

                if not any(x1_batch) or not any(x2_batch):
                    continue

                x1_in = safe_stack(x1_batch, dim=run.input_dim1)
                x2_in = safe_stack(x2_batch, dim=run.input_dim2)
                _, _, divergence_args, _, _, _ = model(
                    x1_in, x2_in, run.adjacency_matrices, tau=tau
                )

                a, b = divergence_args
                div_a.append(a.reshape(-1, a.size(-1)).cpu())
                div_b.append(b.reshape(-1, b.size(-1)).cpu())
    finally:
        handle.remove()

    return torch.cat(codes), torch.cat(div_a), torch.cat(div_b)


def latent_activity(run, store, loader, sigma=1, tau=1.0):
    """Per dimension activity of a trained model, keyed by metric."""
    code, div_a, div_b = collect_latent(run, store, loader, tau=tau)

    if run.ablate_mmd:
        divergence = dim_activity_kl(div_a, div_b)              # (mu, logvar)
    else:
        divergence = dim_activity_mmd(div_a, div_b, sigma)      # (Yk, Yk_pz)

    return {"variance": dim_activity_var(code), "divergence": divergence}


# Panel titles off: the paper caption names the panels, and A, B ... tie
# them to it. Set True to bring them back for a standalone render.
SHOW_TITLES = False


def style(ax, title, ylabel, xlabel="Epoch", keep_title=False):
    """``keep_title`` forces the title through SHOW_TITLES, for the panels
    whose title is the only thing naming them."""
    ax.grid(False)
    if SHOW_TITLES or keep_title:
        ax.set_title(title, fontsize=13, pad=14, loc="left")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_labels(axes, start=0, loc="outside", fmt="{}", fontsize=12,
                 weight="bold"):
    """
    Stamp A, B, C ... across a row or grid of panels, left to right.

    Panels switched off with axis("off") are skipped. Pass the axes rather than fig.axes
    when the figure carries a colourbar. ``loc="outside"`` sits the letter above the top
    left corner; ``loc="inside"`` puts it within the frame.
    """
    from string import ascii_uppercase

    letters = iter(ascii_uppercase[start:])
    stamped = []
    for ax in np.atleast_1d(axes).ravel():
        # Axes3D keeps axison False and tracks its own frame in _axis3don, so
        # the 2D flag would read every 3D panel as blank and skip it
        is_3d = hasattr(ax, "text2D")
        drawn = getattr(ax, "_axis3don", True) if is_3d else getattr(ax, "axison", True)
        if not drawn or not ax.get_visible():
            continue
        x, y, va = (0.025, 0.97, "top") if loc == "inside" else (0.0, 1.03, "bottom")
        # Axes3D draws flat text through text2D, its ax.text wants three coords
        placer = getattr(ax, "text2D", ax.text)
        stamped.append(placer(x, y, fmt.format(next(letters)),
                              transform=ax.transAxes, ha="left", va=va,
                              fontsize=fontsize, fontweight=weight))
    return stamped


def plot_dim_activity(activity, threshold, title, ylabel, filename, color=OKABE_ITO[4]):
    """Activity spectrum of one model, dimensions sorted most to least active."""
    values = np.sort(activity.numpy())[::-1]
    dims = np.arange(1, values.size + 1)
    n_dead = count_dead(activity, threshold)

    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(dims, values, color=color, width=0.8)
    ax.axhline(threshold, color=OKABE_ITO[5], linewidth=1.5, linestyle="--")
    threshold_label(ax, f"dead < {threshold:g}", y=threshold, color=OKABE_ITO[5],
                    side="right")

    # symlog keeps the collapsed dimensions visible, they sit at (or below) zero
    ax.set_yscale("symlog", linthresh=threshold)
    style(ax, f"{title}  -  {n_dead}/{values.size} dead", ylabel,
          xlabel="Latent dimension (sorted)")
    fig.subplots_adjust(right=0.86)
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return n_dead


# The prior is isotropic, so nothing holds the code to the coordinate axes and a
# per axis test misses a collapse onto a tilted subspace. The covariance
# eigenvalues do not care about the frame. They also have to be taken on the
# mean code: a collapsed dimension has mu -> 0 and std -> 1, so a spectrum of
# the sampled code finds prior noise sitting in it and calls it alive.

EIGEN_TOL = 1e-2            # a direction is live above this share of the top eigenvalue
COLLAPSE_FRACTION = 0.5     # under half the width live counts as a collapsed run

RING_MMD = oi.DIVERGENCE["MMD"]
RING_KL = oi.DIVERGENCE["KL"]
RING_LIVE = oi.DARK_GRAY
THRESHOLD = oi.THRESHOLD
MARKERS = {"MMD": "o", "KL": "^"}
HIGHLIGHT = oi.BLACK


def latent_spectrum(Z):
    """Eigenvalues of the pooled code covariance, largest first."""
    Zc = Z - Z.mean(dim=0, keepdim=True)
    cov = Zc.T @ Zc / max(Zc.size(0) - 1, 1)
    return torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)      # (d_z,)


def active_directions(lam, tol=EIGEN_TOL):
    return int((lam > tol * lam[0]).sum())


def effective_rank(Yk):
    """Participation ratio of one recipe's code spectrum."""
    lam = torch.linalg.svdvals(Yk).pow(2)                       # (batch, min(n_comp, d_z))
    return lam.sum(-1).pow(2) / lam.pow(2).sum(-1).clamp_min(1e-12)


def cost_fit(pred, target, mask, cost_std=1.0):
    """
    Per-component fit of a standardized ln(cost) prediction, over priced components.

    ``r2`` is against predicting the mean. ``mae_decades`` puts the absolute error
    back into orders of magnitude of USD/L, the unit a price is read in.
    """
    pred, target, mask = (torch.as_tensor(t, dtype=torch.float).reshape(-1)
                          for t in (pred, target, mask))
    keep = mask > 0
    if keep.sum() < 2:
        return {"r2": None, "mae_decades": None}
    p, t = pred[keep], target[keep]
    sse = (p - t).pow(2).sum()
    sst = (t - t.mean()).pow(2).sum().clamp_min(1e-12)
    return {
        "r2": float(1 - sse / sst),
        "mae_decades": float((p - t).abs().mean() * cost_std / np.log(10)),
    }


def recipe_cost_rank(pred_totals, true_totals):
    """Spearman between predicted and true ln total recipe cost; None under 3 recipes."""
    from scipy.stats import spearmanr
    if len(pred_totals) < 3:
        return None
    rho = spearmanr(pred_totals, true_totals).correlation
    return None if rho is None or np.isnan(rho) else float(rho)


def cost_probe(Z_train, y_train, Z_eval, y_eval, ridge=1e-2):
    """
    Held-out R^2 of a ridge probe from the frozen code to standardized ln(cost).

    Divergence and head agnostic, so it says whether cost is linearly readable
    off the latent in an arm whose own cost head never trained.
    """
    Z_train, Z_eval = Z_train.double(), Z_eval.double()
    y_train, y_eval = y_train.double(), y_eval.double()
    if Z_train.size(0) < 2 or Z_eval.size(0) < 2:
        return None
    mu, sd = Z_train.mean(0), Z_train.std(0).clamp_min(1e-8)
    A = torch.cat([(Z_train - mu) / sd, torch.ones(Z_train.size(0), 1, dtype=Z_train.dtype)], 1)
    B = torch.cat([(Z_eval - mu) / sd, torch.ones(Z_eval.size(0), 1, dtype=Z_eval.dtype)], 1)
    reg = ridge * Z_train.size(0) * torch.eye(A.size(1), dtype=A.dtype)
    reg[-1, -1] = 0.0                                           # intercept is not shrunk
    w = torch.linalg.solve(A.T @ A + reg, A.T @ y_train)
    resid = y_eval - B @ w
    sst = (y_eval - y_eval.mean()).pow(2).sum().clamp_min(1e-12)
    return float(1 - resid.pow(2).sum() / sst)


def mean_code(run, x1, x2, tau=0.1, gated=True):
    """The code with the sampling noise dropped, forward() always samples."""
    model = run.model
    Y = model.f(model.encoder(x1, x2))                          # (1, n_comp, d_z)
    if run.ablate_svd:
        Yk = Y
    else:
        mu, Uk = model.truncated_svd(Y)
        Yk = torch.matmul(Uk, mu)                               # eps dropped

    if not gated:
        return Yk
    gate = gumbel_sigmoid(model.gate_logit(Yk), tau, sample=False)
    return Yk * gate


# nmse alone hid a
# total identity collapse (F1): the metric that caught it names every decoded
# slot and prices the decoded recipe. Pool SLOTS, not recipes, and split by
# dose band and by database-vs-poison -- a recipe with two protein slots must
# not weigh the same as one with twelve, and the poison recipes must not
# inflate identity accuracy.

DOSE_BANDS = ((0.0, 0.1), (0.1, float("inf")))      # mg/L; F1's low/high split


def recon_tables(run, store):
    """Per modality: (names, stacked standardized component vectors) in vec_lookup's space."""
    rows = {"pubchem": {}, "uniprot": {}}
    for item in run.dataset:
        for comp in item["components"]:
            name, source = comp["component"], comp["component_source"]
            if name in rows[source]:
                continue
            base = comp["component_vec"].to(dtype=torch.float32, device=run.device)
            if source == "pubchem":
                base = (base - store.targ_mean) / (store.targ_std + 1e-8)
            else:
                base = store.std_uniprot(base)
            rows[source][name] = base
    return {s: (list(r), torch.stack(list(r.values()))) for s, r in rows.items()}


def name_slots(store, x_hat, n_scalars, names, vecs):
    """Decoded slot names (nearest component) and concentrations, one recipe, one modality."""
    base = x_hat[0, :, :-n_scalars]
    idx = torch.cdist(base, vecs).argmin(-1)
    conc = torch.expm1(store.dec_conc(x_hat[0, :, -n_scalars]))
    return [names[i] for i in idx.tolist()], conc.tolist()


def recon_score(run, store, item, table, prices):
    """Decode one recipe: per-slot rows and its priced ln-cost error. None if one-sided."""
    import predict
    x1v = [store.make_x1_vec(c, run.device) for c in item["components"] if c["component_source"] == "pubchem"]
    x2v = [store.make_x2_vec(c, run.device) for c in item["components"] if c["component_source"] == "uniprot"]
    if not x1v or not x2v:
        return None
    x1, x2 = torch.stack(x1v).unsqueeze(0), torch.stack(x2v).unsqueeze(0)
    with torch.no_grad():
        Zc = mean_code(run, x1, x2, 0.1)
        x1_hat, x2_hat = run.model.decoder(Zc, x1.size(1), x2.size(1))
    rows, decoded = [], []
    for source, x_hat in (("pubchem", x1_hat), ("uniprot", x2_hat)):
        truth = [c for c in item["components"] if c["component_source"] == source]
        names, concs = name_slots(store, x_hat, store.n_scalars, *table[source])
        for n, c, t in zip(names, concs, truth):
            rows.append({"source": source, "hit": float(n == t["component"]), "alive": float(c > 0),
                         "conc_dec": (abs(np.log10(max(c, 1e-6)) - np.log10(t["concentration"]))
                                      if t.get("concentration") and c > 0 else None)})
            if c > 0:
                decoded.append({"component": n, "concentration": c})
    true_cost = sum(c.get("cost") or 0 for c in item["components"])
    rec_cost, _ = predict.realized_cost(decoded, prices)
    return {"slots": rows, "ln_true": float(np.log(max(true_cost, 1e-6))),
            "ln_recon": float(np.log(max(rec_cost, 1e-6)))}


def _slot_metrics(rows):
    if not rows:
        return {"n": 0, "id_acc": None, "alive": None, "conc_dec": None}
    errs = [r["conc_dec"] for r in rows if r["conc_dec"] is not None]
    return {"n": len(rows), "id_acc": float(np.mean([r["hit"] for r in rows])),
            "alive": float(np.mean([r["alive"] for r in rows])),
            "conc_dec": float(np.median(errs)) if errs else None}


def _median_dose(item):
    concs = [c["concentration"] for c in item["components"]
             if c["component_source"] == "uniprot" and c.get("concentration")]
    return float(np.median(concs)) if concs else 0.0


def recon_usefulness(run, store, items, table=None, prices=None):
    """Pooled slot metrics over ``items``, split by modality, dose band and database-vs-poison."""
    import predict
    table = recon_tables(run, store) if table is None else table
    prices = predict.load_prices() if prices is None else prices
    tagged, costs = [], []
    for item in items:
        r = recon_score(run, store, item, table, prices)
        if r is None:
            continue
        tagged += [(s, item) for s in r["slots"]]
        costs.append(abs(r["ln_recon"] - r["ln_true"]))

    band = lambda item: next(f"{lo:g}-{hi:g} mg/L" for lo, hi in DOSE_BANDS
                             if lo <= _median_dose(item) < hi)
    bands = sorted({band(i) for _, i in tagged})
    return {
        "all": _slot_metrics([s for s, _ in tagged]),
        "by_source": {src: _slot_metrics([s for s, _ in tagged if s["source"] == src])
                      for src in ("pubchem", "uniprot")},
        "by_dose": {b: _slot_metrics([s for s, i in tagged if band(i) == b]) for b in bands},
        "by_class": {name: _slot_metrics([s for s, i in tagged if pred(i)])
                     for name, pred in (("database", lambda i: i.get("proliferation", 1) != 0),
                                        ("poison", lambda i: i.get("proliferation", 1) == 0))},
        "cost_err_median": float(np.median(costs)) if costs else None,
        "n_recipes": len(costs),
    }



PREDICT_OBJECTIVES = {
    "pathway": {"pathway_w": 2.0, "uniprot_w": 0.0, "anchor_w": 1e-3, "fidelity_w": 1.0,
                "cost_w": 0.0, "decoded_w": 0.0},
    "pathway + proxy": {"pathway_w": 2.0, "uniprot_w": 1e-1, "anchor_w": 1e-3, "fidelity_w": 1.0,
                        "cost_w": 0.0, "decoded_w": 0.0},
    "pathway + latent": {"pathway_w": 2.0, "uniprot_w": 0.0, "anchor_w": 1e-3, "fidelity_w": 1.0,
                         "cost_w": 1.0, "decoded_w": 0.0},
    "pathway + decoded": {"pathway_w": 2.0, "uniprot_w": 0.0, "anchor_w": 1e-3, "fidelity_w": 1.0,
                          "cost_w": 0.0, "decoded_w": 1.0},
}


def predict_context(run, store, item):
    """predict.encode_base for an arbitrary recipe (encode_base only does the dataset's base)."""
    import predict
    device, model = run.device, run.model
    x1v = [store.make_x1_vec(c, device) for c in item["components"] if c["component_source"] == "pubchem"]
    x2v = [store.make_x2_vec(c, device) for c in item["components"] if c["component_source"] == "uniprot"]
    if not x1v or not x2v:
        return None
    x1, x2 = torch.stack(x1v).unsqueeze(0), torch.stack(x2v).unsqueeze(0)
    with torch.no_grad():
        alpha, Uk = model.truncated_svd(model.f(model.encoder(x1, x2)))
    return {"base": item, "x1": x1, "x2": x2, "Uk": Uk, "a_z0": alpha.detach(),
            "uniprot_names": [c["component"] for c in item["components"] if c["component_source"] == "uniprot"],
            "conc_index": -store.n_scalars, "cost_mean": store.cost_mean, "cost_std": store.cost_std,
            "decoded_log_cost": store.decoded_log_cost,
            "targets": torch.tensor([predict.TARGETS], dtype=torch.float, device=device)}


def predict_objective_summary(records):
    """Per arm and objective: ln realized USD/L against the pathway-only reference."""
    # predict_eval/ also holds other ADR artifacts; keep only real predict-eval records.
    keys = ("checkpoint", "recipe", "objective", "realized_cost", "pathway", "readout", "true_cost")
    records = [r for r in records if all(k in r for k in keys)]
    lnr = lambda c: float(np.log(max(c, 1e-6)))
    ref = {(r["checkpoint"], r["recipe"]): r for r in records if r["objective"] == "pathway"}
    for arm in sorted({r.get("arm", "?") for r in records}):
        rows = [r for r in records if r.get("arm", "?") == arm and r["objective"] == "pathway"]
        print(f"\n{arm}: {len({r['checkpoint'] for r in rows})} checkpoints, "
              f"median ln USD/L of the input recipes {np.median([lnr(r['true_cost']) for r in rows]):.3f}")
        _objective_table([r for r in records if r.get("arm", "?") == arm], ref, lnr)


def _objective_table(records, ref, lnr):
    print(f"{'objective':<20} {'ln USD/L real':>14} {'vs pathway':>11} {'mean':>7} {'cheaper':>8} "
          f"{'pathway BCE':>12} {'prolif':>7} {'surv':>6} {'diff':>6} {'n':>4}")
    for name in PREDICT_OBJECTIVES:
        # A row is only comparable if its pathway reference was loaded; a partial or
        # older file can leave a checkpoint/recipe without one.
        rows = [r for r in records if r["objective"] == name
                and (r["checkpoint"], r["recipe"]) in ref]
        if not rows:
            continue
        d = [lnr(r["realized_cost"]) - lnr(ref[(r["checkpoint"], r["recipe"])]["realized_cost"])
             for r in rows]
        print(f"{name:<20} {np.median([lnr(r['realized_cost']) for r in rows]):14.3f} "
              f"{np.median(d):+11.3f} {np.mean(d):+7.2f} {np.mean(np.array(d) < 0):8.2f} "
              f"{np.median([r['pathway'] for r in rows]):12.4f} "
              f"{np.median([r['readout'][0] for r in rows]):7.3f} "
              f"{np.median([r['readout'][1] for r in rows]):6.3f} "
              f"{np.median([r['readout'][2] for r in rows]):6.3f} {len(rows):4d}")


# Fill carries the surviving width, ring carries whether the run collapsed and
# under which divergence, marker area carries whatever axis is left over. Every
# figure that talks about collapse uses the same three, so they read together.

def jitter(coords, amount=0.11):
    """
    Spread points that share a lattice coordinate.

    Each co-located group is fanned around a small circle, deterministically, so a
    figure redraws the same way.
    """
    coords = [np.asarray(c, dtype=float).copy() for c in coords]
    groups = {}
    for i, key in enumerate(zip(*[np.round(c, 6) for c in coords])):
        groups.setdefault(key, []).append(i)

    for members in groups.values():
        if len(members) < 2:
            continue
        angles = np.linspace(0, 2 * np.pi, len(members), endpoint=False)
        for angle, i in zip(angles, members):
            coords[0][i] += amount * np.cos(angle)
            coords[1][i] += amount * np.sin(angle)
    return coords


def ringed(rows, ring_what):
    """
    Which runs get a ring.

    Collapse is usually the common outcome, so "live" marks the handful worth finding;
    pass "collapsed" to mark failures instead, or "none" to drop rings.
    """
    if ring_what == "none":
        return []
    if ring_what == "collapsed":
        return [r for r in rows if r["collapsed"]]
    return [r for r in rows if not r["collapsed"]]


def ring_color(divergence):
    return RING_MMD if divergence == "MMD" else RING_KL


def collapse_rings(rows):
    return [ring_color(r["divergence"]) if r["collapsed"] else RING_LIVE
            for r in rows]


def by_divergence(rows):
    """Split rows into (divergence, marker, group), skipping empty groups."""
    for divergence, marker in MARKERS.items():
        group = [r for r in rows if r["divergence"] == divergence]
        if group:
            yield divergence, marker, group


def best_of(rows, key, mode="min"):
    pick = min if mode == "min" else max
    return pick(rows, key=lambda r: r[key])


def drop_lines_3d(ax, point, color=HIGHLIGHT):
    """Dashed guides from a point out to the three axis walls."""
    x, y, z = point
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    style = dict(color=color, linestyle=(0, (5, 4)), linewidth=0.8, alpha=0.65, zorder=1)
    ax.plot([xlim[0], x], [y, y], [z, z], **style)
    ax.plot([x, x], [ylim[1], y], [z, z], **style)
    ax.plot([x, x], [y, y], [zlim[0], z], **style)
    ax.set_xlim(xlim), ax.set_ylim(ylim), ax.set_zlim(zlim)


def _edge_annotate(ax, text, x, y, pad=9, color=HIGHLIGHT):
    """Offset a label away from whichever axis edge it is closest to."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    if ax.get_xscale() == "log":
        fx = (np.log10(x) - np.log10(x0)) / (np.log10(x1) - np.log10(x0))
    else:
        fx = (x - x0) / (x1 - x0)
    fy = (y - y0) / (y1 - y0)

    dx, ha = (-pad, "right") if fx > 0.72 else (pad, "left")
    dy, va = (-pad, "top") if fy > 0.82 else (pad, "bottom")
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5, color=color, ha=ha, va=va, zorder=5)


def drop_lines_2d(ax, x, y, color=HIGHLIGHT):
    """Dashed guides from a point out to both axes."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    style = dict(color=color, linestyle=(0, (5, 4)), linewidth=0.8, alpha=0.65, zorder=1)
    ax.plot([xlim[0], x], [y, y], **style)
    ax.plot([x, x], [ylim[0], y], **style)
    ax.set_xlim(xlim), ax.set_ylim(ylim)


def marker_sizes(values, lo=None, hi=None, base=40, span=150):
    """Log spaced marker areas, for an axis with no spatial axis left."""
    values = np.asarray(values, dtype=float)
    lo = float(values.min()) if lo is None else float(lo)
    hi = float(values.max()) if hi is None else float(hi)
    reach = max(np.log2(hi) - np.log2(lo), 1e-9)
    return base + span * (np.log2(values) - np.log2(lo)) / reach


def ring_legend(d_z, fraction=COLLAPSE_FRACTION, highlight_label=None,
                ring_what="live"):
    base = dict(linestyle="none", markerfacecolor=oi.GRAY, markersize=9)
    handles = [
        Line2D([], [], marker="o", markeredgecolor=RING_LIVE, markeredgewidth=1.0,
               label="MMD run", **base),
        Line2D([], [], marker="^", markeredgecolor=RING_LIVE, markeredgewidth=1.0,
               label="KL run", **base),
    ]
    if ring_what != "none":
        word = "collapsed" if ring_what == "collapsed" else "surviving"
        cut = f"under {fraction:.0%}" if ring_what == "collapsed" else f"over {fraction:.0%}"
        handles += [
            Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=RING_MMD,
                   markeredgewidth=1.8, markersize=13, linestyle="none",
                   label=f"{word} MMD ({cut} of {d_z})"),
            Line2D([], [], marker="^", markerfacecolor="none", markeredgecolor=RING_KL,
                   markeredgewidth=1.8, markersize=13, linestyle="none",
                   label=f"{word} KL ({cut} of {d_z})"),
        ]
    if highlight_label:
        handles.append(Line2D([], [], color=HIGHLIGHT, linestyle=(0, (4, 3)),
                              linewidth=1.2, label=highlight_label))
    return handles


def threshold_label(ax, text, y=None, x=None, color=oi.ANNOTATION, side="left", above=True):
    """
    Caption a threshold line without colliding with it or the frame.

    Sits the text clear of the line on an opaque patch and anchors it in axes
    fraction along the free direction, so it cannot run off an edge or land on
    top of the dashes.
    """
    box = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.8)
    common = dict(color=color, fontsize=8.5, bbox=box, zorder=6, annotation_clip=False)

    if y is not None:
        at, dx, ha = (0.015, 4, "left") if side == "left" else (0.985, -4, "right")
        ax.annotate(text, xy=(at, y), xycoords=("axes fraction", "data"),
                    xytext=(dx, 6 if above else -6), textcoords="offset points",
                    ha=ha, va="bottom" if above else "top", **common)
    else:
        ax.annotate(text, xy=(x, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="bottom", **common)


def log3d_axis(ax, axis, values, label):
    """mplot3d has no log scale, so plot log10 and label the decades."""
    lo, hi = np.floor(np.log10(min(values))), np.ceil(np.log10(max(values)))
    ticks = np.arange(lo, hi + 1)
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"set_{axis}ticklabels")([f"$10^{{{int(t)}}}$" for t in ticks], fontsize=8)
    getattr(ax, f"set_{axis}label")(label, fontsize=10, labelpad=6)


def style_3d(ax, title, keep_title=False):
    if SHOW_TITLES or keep_title:
        ax.set_title(title, fontsize=12, pad=6, loc="left")
    ax.view_init(elev=20, azim=-58)
    ax.set_box_aspect((1, 1, 0.85))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
        pane._axinfo["grid"].update(color=oi.GRAY, linewidth=0.4)
    ax.tick_params(labelsize=8, pad=1)


def plot_latent_scatter3d(results, axes, d_z, size_key="tsvd_k", size_range=None,
                          split_key="svd", split_titles=("without tSVD", "with tSVD"),
                          highlight_key="div_loss", ring_what="live",
                          filename="latent_scatter3d.png"):
    """
    One point per run on three log axes, split into two panels.

    ``axes`` is three (key, label) pairs for x, y and z. Fill is the active direction
    count on a 0..d_z cividis scale, marker shape is the divergence, an outer ring marks
    a collapsed run, and marker area is ``size_key``. The smallest ``highlight_key`` in
    each divergence gets dashed guides out to the axes.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    norm = Normalize(vmin=0, vmax=d_z)
    lo, hi = (None, None) if size_range is None else size_range
    keys = [key for key, _ in axes]

    fig = plt.figure(figsize=(14, 6.6))
    scatter = None

    for i, (title, want) in enumerate(zip(split_titles, (False, True))):
        rows = [r for r in results if bool(r[split_key]) == want]
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        if not rows:
            continue

        for divergence, marker, group in by_divergence(rows):
            coords = jitter([np.log10([r[key] for r in group]) for key in keys])
            sizes = marker_sizes([r[size_key] for r in group], lo, hi)

            # Ring first and larger, so it sits on background rather than on the
            # fill, where no single colour contrasts with all of cividis
            marked = ringed(group, ring_what)
            if marked:
                index = {id(r): j for j, r in enumerate(group)}
                picks = [index[id(r)] for r in marked]
                ax.scatter(*[c[picks] for c in coords],
                           s=marker_sizes([r[size_key] for r in marked], lo, hi,
                                          base=90, span=300),
                           marker=marker, facecolors="none",
                           edgecolors=ring_color(divergence), linewidths=1.5,
                           depthshade=False, zorder=2)

            scatter = ax.scatter(*coords, c=[r["active"] for r in group], cmap=oi.SEQUENTIAL,
                                 norm=norm, s=sizes, marker=marker,
                                 edgecolors=RING_LIVE, linewidths=0.7, alpha=0.95,
                                 depthshade=False, zorder=3)

        for name, (key, label) in zip("xyz", axes):
            log3d_axis(ax, name, [r[key] for r in rows], label)

        n_dead = sum(r["collapsed"] for r in rows)
        style_3d(ax, f"{title}   -   {n_dead}/{len(rows)} collapsed")

        if highlight_key:
            for divergence, _, group in by_divergence(rows):
                best = best_of(group, highlight_key)
                drop_lines_3d(ax, [np.log10(best[key]) for key in keys])

    if scatter is not None:
        bar = fig.colorbar(scatter, ax=fig.axes, fraction=0.018, pad=0.06)
        bar.set_label("active units", fontsize=10)
        bar.ax.tick_params(labelsize=9)

    label = f"best per divergence (smallest {highlight_key})" if highlight_key else None
    fig.legend(handles=ring_legend(d_z, highlight_label=label, ring_what=ring_what), loc="lower center",
               ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.46, 0.0))
    panel_labels([a for a in fig.axes if hasattr(a, "text2D")])
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


def plot_tradeoff(results, d_z, driver=("div_beta", "div_beta"), size_key="tsvd_k",
                  size_range=None, highlight_key="div_loss", ring_what="live",
                  filename="latent_tradeoff.png"):
    """
    What the surviving width costs, and which knob buys it.

    Left, reconstruction against live directions with the Pareto front picked
    out. Right, ``driver`` against live directions with the collapse threshold
    drawn in. Same colour, shape and ring language as the 3D view.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    norm = Normalize(vmin=0, vmax=d_z)
    lo, hi = (None, None) if size_range is None else size_range
    driver_key, driver_label = driver

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))
    panels = ((ax1, "active", "recon"), (ax2, driver_key, "active"))
    scatter = None

    for ax, xkey, ykey in panels:
        for divergence, marker, group in by_divergence(results):
            sizes = marker_sizes([r[size_key] for r in group], lo, hi)

            marked = ringed(group, ring_what)
            if marked:
                ax.scatter([r[xkey] for r in marked], [r[ykey] for r in marked],
                           s=marker_sizes([r[size_key] for r in marked], lo, hi,
                                          base=90, span=300),
                           marker=marker, facecolors="none",
                           edgecolors=ring_color(divergence), linewidths=1.5, zorder=2)

            scatter = ax.scatter([r[xkey] for r in group], [r[ykey] for r in group],
                                 c=[r["active"] for r in group], cmap=oi.SEQUENTIAL,
                                 norm=norm, s=sizes, marker=marker,
                                 edgecolors=RING_LIVE, linewidths=0.7, alpha=0.95,
                                 zorder=3)

    # Pareto optimal: nothing else is both livelier and better reconstructed
    front = [r for r in results
             if not any(o["active"] >= r["active"] and o["recon"] < r["recon"]
                        for o in results)]
    ax1.scatter([r["active"] for r in front], [r["recon"] for r in front],
                s=marker_sizes([r[size_key] for r in front], lo, hi, base=175, span=430),
                facecolors="none", edgecolors=oi.BLACK, linewidths=1.0,
                linestyle=":", zorder=4, label="Pareto front")
    # Runs cluster at low active, so the right of this panel is the empty half
    ax1.legend(frameon=False, fontsize=9, loc="upper right", markerscale=0.3)
    ax1.set_xlim(0, d_z)
    style(ax1, "Reconstruction against surviving width",
          "reconstruction MSE", xlabel=f"active units of {d_z}")

    ax2.set_xscale("log")
    ax2.set_ylim(0, d_z)
    ax2.axhline(COLLAPSE_FRACTION * d_z, color=THRESHOLD, linewidth=1.2, linestyle="--")
    threshold_label(ax2, f"collapse threshold, {COLLAPSE_FRACTION:.0%}",
                    y=COLLAPSE_FRACTION * d_z)
    style(ax2, f"{driver_label} against surviving width",
          f"active units of {d_z}", xlabel=driver_label)

    if highlight_key:
        for divergence, _, group in by_divergence(results):
            best = best_of(group, highlight_key)
            drop_lines_2d(ax1, best["active"], best["recon"])
            drop_lines_2d(ax2, best[driver_key], best["active"])
            for ax, xkey, ykey in panels:
                _edge_annotate(ax, f"best {divergence}", best[xkey], best[ykey])

    bar = fig.colorbar(scatter, ax=(ax1, ax2), fraction=0.025, pad=0.03)
    bar.set_label("active units", fontsize=10)
    bar.ax.tick_params(labelsize=9)

    label = f"best per divergence (smallest {highlight_key})" if highlight_key else None
    fig.legend(handles=ring_legend(d_z, highlight_label=label, ring_what=ring_what), loc="lower center",
               ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.45, -0.06))
    panel_labels((ax1, ax2))
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


VARIANT_FALLBACK = (oi.BLUE, oi.SKY, oi.ORANGE_DARK, oi.ORANGE)


def variant_color(label, i=0):
    return oi.VARIANT.get(label, VARIANT_FALLBACK[i % len(VARIANT_FALLBACK)])


def plot_weight_marginals(results, params, d_z, label_key="label",
                          filename="latent_marginals.png"):
    """
    One panel per hyperparameter: its value against the surviving width.

    ``params`` is a sequence of (key, label) pairs. Variants are dodged sideways within
    each lattice column so nothing overlaps.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = sorted({r[label_key] for r in results})
    colors = {name: variant_color(name, i) for i, name in enumerate(labels)}

    fig, axes = plt.subplots(1, len(params), figsize=(3.6 * len(params), 4.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (key, label) in zip(axes, params):
        ticks = sorted({r[key] for r in results})
        # Dodge each variant into its own slot inside the lattice column
        gap = np.diff(np.log10(ticks)).min() if len(ticks) > 1 else 1.0
        step = gap * 0.55 / max(len(labels), 1)

        for i, name in enumerate(labels):
            rows = [r for r in results if r[label_key] == name]
            if not rows:
                continue
            offset = (i - (len(labels) - 1) / 2) * step
            x = 10 ** (np.log10([r[key] for r in rows]) + offset)
            marker = MARKERS[rows[0]["divergence"]]
            ax.scatter(x, [r["active"] for r in rows], s=34, marker=marker,
                       facecolors=colors[name], edgecolors=oi.DARK_GRAY, linewidths=0.6,
                       alpha=0.9, label=name if ax is axes[0] else None, zorder=3)

        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=8, rotation=45)
        ax.set_ylim(0, d_z)
        ax.axhline(COLLAPSE_FRACTION * d_z, color=THRESHOLD, linewidth=1.1, linestyle="--",
                   zorder=1)
        # The title normally names the parameter and an xlabel would collide
        # with the legend underneath, so it is empty. With titles off the name
        # has nowhere else to go and the four panels are anonymous, so it moves
        # down here and the legend is pushed clear in the savefig block.
        style(ax, label, f"active units of {d_z}" if ax is axes[0] else "",
              xlabel="" if SHOW_TITLES else label)

    threshold_label(axes[0], f"collapse threshold, {COLLAPSE_FRACTION:.0%}",
                    y=COLLAPSE_FRACTION * d_z)
    fig.legend(loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.14 if SHOW_TITLES else -0.22))
    panel_labels(axes)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


# The manifold plane is the top two eigenvectors of the mean code covariance,
# the directions that actually carry signal. Coordinates are standardized
# first: the codes here run about 15x the width of the unit prior, so a grid
# laid on raw prior quantiles would sit entirely inside the blob and find no
# holes by construction. Standardizing asks the useful question instead, "given
# the spread this model does use, is it filled in".

# 90% of a 2D standard normal lies inside this radius: P(R^2 <= r^2) = 1 - exp(-r^2/2)
PRIOR_RING_RADIUS = float(np.sqrt(-2.0 * np.log(1 - 0.90)))

HOLE_PERCENTILE = 5.0       # a query point below this percentile of code density is a hole
GRID_PER_AXIS = 17          # query points per axis, equal prior mass apart
# Kernel widths swept in plot_hole_bandwidth, as multiples of Scott's rule. The
# hole count is a KDE readout, so the smoothing is as load bearing as the
# density cut.
HOLE_BW_SCALES = np.geomspace(0.5, 2.0, 7)
COVERED_MARK = "white"
HOLE_MARK = oi.VERMILLION


def prior_quantile_axis(n=GRID_PER_AXIS):
    """
    Query positions carrying equal prior probability mass.

    Splits N(0,1) into n slices of mass 1/n and takes each midpoint, so every
    query point speaks for the same amount of prior. Uniform spacing would
    over-sample the tails, where the prior says almost nothing should live.
    """
    quantiles = (np.arange(n) + 0.5) / n
    return np.sqrt(2.0) * torch.erfinv(torch.tensor(2 * quantiles - 1.0)).numpy()


def project_active(Z, n_dims=2):
    """
    Project codes onto their most active directions and standardize.

    Returns (projection, eigenvalues, scales). The projection is unit variance
    per axis, so the prior it is compared against is the standard normal.
    """
    Z = np.asarray(Z, dtype=float)
    centred = Z - Z.mean(axis=0, keepdims=True)
    cov = centred.T @ centred / max(len(centred) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    projection = centred @ vectors[:, :n_dims]
    scales = projection.std(axis=0)
    return projection / np.where(scales > 0, scales, 1.0), values, scales


def project_raw(Z, dims):
    """
    Standardized projection onto named ORIGINAL coordinates.

    Same output contract as project_active minus the rotation, so find_holes can
    be run on the frame top_variance_dims would have picked and the projection
    choice becomes a measurement instead of an argument.
    """
    Z = np.asarray(Z, dtype=float)[:, list(dims)]
    centred = Z - Z.mean(axis=0, keepdims=True)
    scales = centred.std(axis=0)
    return centred / np.where(scales > 0, scales, 1.0), scales


def variance_share(Z, eigenvalues, dims):
    """
    Share of the code's spread carried by the top 2 PCs and by the named raw
    coordinates. lam.sum() is the covariance trace, which equals the summed per
    axis variances at ddof=1, so both fractions run off one denominator.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    per_axis = np.asarray(Z, dtype=float).var(axis=0, ddof=1)
    total = max(lam.sum(), 1e-12)
    return float(lam[:2].sum() / total), float(per_axis[list(dims)].sum() / total)


def find_holes(projection, n=GRID_PER_AXIS, percentile=HOLE_PERCENTILE, bw_method=None):
    """
    Equal prior mass query points that the aggregate posterior never reaches.

    A point is a hole when the code density there falls below the ``percentile`` of the
    density measured at the real codes; anchoring the cut to the data keeps it comparable
    across models whose densities differ by orders of magnitude.

    ``bw_method`` is passed to gaussian_kde. None keeps Scott's rule, which the reported
    number uses; hole_bandwidth_curve sweeps it.
    """
    from scipy.stats import gaussian_kde

    axis = prior_quantile_axis(n)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    kde = gaussian_kde(projection.T, bw_method=bw_method)
    at_grid = kde(grid.T)
    at_codes = kde(projection.T)
    cutoff = np.percentile(at_codes, percentile)

    holes = at_grid < cutoff
    return {
        "grid": grid,
        "density_at_grid": at_grid,
        "density_at_codes": at_codes,
        "cutoff": float(cutoff),
        "holes": holes,
        "hole_fraction": float(holes.mean()),
        # The kernel sd actually used, so the figure can say what it smoothed by
        "bandwidth": float(np.sqrt(kde.covariance[0, 0])),
        "bw_factor": float(kde.factor),
        "kde": kde,
    }


def hole_curve(result, percentiles=np.linspace(1, 25, 25)):
    """Hole fraction as the density cut moves, so one threshold is not load bearing."""
    return np.array([
        (result["density_at_grid"] < np.percentile(result["density_at_codes"], p)).mean()
        for p in percentiles
    ])


def hole_bandwidth_curve(projection, n=GRID_PER_AXIS, percentile=HOLE_PERCENTILE,
                         multiples=HOLE_BW_SCALES):
    """
    Hole fraction as the kernel width sweeps, in multiples of Scott's rule.

    hole_curve moves the density cut, this moves the smoothing that produced the
    density in the first place. Unlike hole_curve it cannot reuse the stored
    densities, since every bandwidth is a different KDE, so it refits per point.
    """
    from scipy.stats import gaussian_kde

    base = gaussian_kde(np.asarray(projection).T).factor
    return np.array([
        find_holes(projection, n, percentile, bw_method=float(m * base))["hole_fraction"]
        for m in multiples
    ])


def plot_latent_manifold(characterizations, n=GRID_PER_AXIS,
                         filename="latent_manifold.png", hole_key="holes",
                         dim_labels=("$z_1$", "$z_2$")):
    """
    The aggregate posterior on its two most active directions, probed on an
    equal prior mass grid. White + is a query point standing on q(z) mass, red X
    is one falling in a hole. The caption carries the hole fraction.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = (len(characterizations) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 5.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    axis = prior_quantile_axis(n)
    grid_reach = float(np.abs(axis).max())
    # Wide enough that the prior ring at r=2.146 clears the outermost query
    # points and still sits inside the frame
    span = grid_reach * 1.5
    dense = np.linspace(-span, span, 220)
    mx, my = np.meshgrid(dense, dense)

    for ax, item in zip(axes, characterizations):
        holes = item[hole_key]
        field = holes["kde"](np.column_stack([mx.ravel(), my.ravel()]).T).reshape(mx.shape)

        ax.imshow(field, origin="lower", extent=(-span, span, -span, span),
                  cmap=oi.SEQUENTIAL, aspect="equal")
        ax.contour(mx, my, field, levels=4, colors="white", linewidths=0.7, alpha=0.65)

        # The reference the query grid is laid against: after standardizing, the
        # scale matched prior is N(0, I) here, and 90% of a 2D standard normal
        # sits inside r = sqrt(-2 ln 0.1). Drawn so the grid is not the only
        # thing standing in for the prior.
        ring = plt.Circle((0, 0), PRIOR_RING_RADIUS, fill=False, color="white",
                          linewidth=1.8, linestyle=(0, (6, 4)), zorder=4)
        ring.set_path_effects([path_effects.withStroke(linewidth=3.2,
                                                       foreground=oi.BLACK)])
        ax.add_patch(ring)

        grid, mask = holes["grid"], holes["holes"]
        ax.scatter(grid[~mask, 0], grid[~mask, 1], marker="+", c=COVERED_MARK,
                   s=44, linewidths=1.1)
        ax.scatter(grid[mask, 0], grid[mask, 1], marker="x", c=HOLE_MARK,
                   s=52, linewidths=1.8)

        ax.set_xlim(-span, span), ax.set_ylim(-span, span)
        # The n/N count pushed the longer variant names into the panel letter,
        # and it is redundant: the percentage carries it and N is the same grid
        # in every panel
        style(ax, f"{item['label']}   -   {holes['hole_fraction']:.0%} holes",
              dim_labels[1], xlabel=dim_labels[0])

    for ax in axes[len(characterizations):]:
        ax.axis("off")

    handles = [
        Line2D([], [], marker="+", color=COVERED_MARK, linestyle="none", markersize=9,
               markeredgewidth=1.6, label="query point on q(z) mass"),
        Line2D([], [], marker="x", color=HOLE_MARK, linestyle="none", markersize=9,
               markeredgewidth=2.0, label="query point in a hole"),
        Line2D([], [], color="white", linestyle=(0, (6, 4)), linewidth=1.8,
               label="prior 90% mass"),
    ]
    fig.subplots_adjust(hspace=0.3, wspace=0.18)

    # Below the grid of panels, so it covers no query point at all. Anchored on
    # the leftmost panel's frame rather than the figure edge, otherwise it juts
    # out past the block of plots by the width of the y axis labels. The covered
    # mark is white, so it keeps a dark ground out here too.
    left = min(ax.get_position().x0 for ax in axes[:len(characterizations)])
    legend = fig.legend(handles=handles, loc="lower left", ncol=3, frameon=True,
                        fontsize=8.5, bbox_to_anchor=(left, -0.012),
                        facecolor=oi.DARK_GRAY, edgecolor="none", labelcolor="white",
                        borderpad=0.5, handletextpad=0.5, columnspacing=1.4,
                        borderaxespad=0.0)     # else it pads in from the anchor
    legend.get_frame().set_alpha(0.95)
    panel_labels(axes[:len(characterizations)])
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


def plot_spectrum_comparison(characterizations, d_z, filename="latent_spectrum.png"):
    """Eigenvalue spectra of the aggregate posterior, one line per variant."""
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))

    for i, item in enumerate(characterizations):
        lam = np.asarray(item["eigenvalues"])
        share = lam / max(lam.sum(), 1e-12)
        color = variant_color(item["label"], i)
        dims = np.arange(1, lam.size + 1)

        ax1.plot(dims, np.maximum(lam / max(lam[0], 1e-12), 1e-8), color=color,
                 linewidth=2, marker=MARKERS[item["divergence"]], markersize=4,
                 label=f"{item['label']}  ({item['active']} live)")
        ax2.plot(dims, np.cumsum(share), color=color, linewidth=2,
                 marker=MARKERS[item["divergence"]], markersize=4, label=item["label"])

    ax1.axhline(EIGEN_TOL, color=THRESHOLD, linewidth=1.2, linestyle="--")
    threshold_label(ax1, f"live above {EIGEN_TOL:g} of the top", y=EIGEN_TOL, side="right")
    ax1.set_yscale("log")
    style(ax1, "Covariance spectrum, normalized to the top eigenvalue",
          "eigenvalue share", xlabel="direction (sorted)")

    ax2.axhline(0.95, color=THRESHOLD, linewidth=1.2, linestyle="--")
    threshold_label(ax2, "95% of variance", y=0.95, side="right", above=False)
    ax2.set_ylim(0, 1.02)
    style(ax2, "Cumulative variance", "share of total variance",
          xlabel="direction (sorted)")

    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    panel_labels((ax1, ax2))
    fig.subplots_adjust(wspace=0.22)
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


def plot_reconstruction(characterizations, filename="latent_reconstruction.png"):
    """
    What the code still encodes.

    Left, per recipe reconstruction error as an ECDF. Right, the split by modality, each
    divided by the mean square of its own targets (x1 carries ~200x the target scale of
    x2, so raw MSE cannot show balance). The dashed line at 1 is the error of
    predicting the mean.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    width = 0.36

    for i, item in enumerate(characterizations):
        errors = np.asarray(item["recon_per_recipe"])
        color = variant_color(item["label"], i)
        ax1.ecdf(errors, color=color, linewidth=2,
                 label=f"{item['label']}  (median {np.median(errors):.2f})")

        for j, key in enumerate(("nmse_x1", "nmse_x2")):
            ax2.bar(j + (i - 1.5) * width / 2, item[key], width / 2, color=color,
                    label=item["label"] if j == 0 else None)

    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.02)
    style(ax1, "Per recipe reconstruction error", "cumulative fraction of recipes",
          xlabel="reconstruction MSE")
    ax1.legend(frameon=False, fontsize=9, loc="upper left")

    ax2.axhline(1.0, color="0.55", linestyle="--", linewidth=1, zorder=0)
    # Headroom above the y=1 line, else the legend sits on top of it and the
    # dashes read as a strikethrough across the first entry
    ax2.set_ylim(0, 1.45)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["PubChem (x1)", "UniProt (x2)"])
    style(ax2, "Error by modality (normalized)", "MSE / mean square of target", xlabel="")
    ax2.legend(frameon=False, fontsize=9, loc="upper left")

    panel_labels((ax1, ax2))
    fig.subplots_adjust(wspace=0.22)
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


def plot_prior_marginals(characterizations, n_dims=2, filename="latent_prior_fit.png"):
    """
    How far the aggregate posterior drifts from the prior it was pulled toward.

    One panel per active direction, with the standard normal drawn behind and
    the KS statistic printed.

    The axis is standardized, so every curve is forced to unit width and what is
    being compared is SHAPE, not scale. A code fifteen times too wide scores
    about 0.007 here and 0.43 un-standardized. Read this next to scale_ratio,
    which carries the width that was divided out.
    """
    from scipy.stats import gaussian_kde, kstest

    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, n_dims, figsize=(6.2 * n_dims, 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    grid = np.linspace(-4, 4, 300)
    normal = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)

    for d, ax in enumerate(axes):
        ax.plot(grid, normal, color=oi.DARK_GRAY, linewidth=1.6, linestyle="--",
                label="prior N(0,1)", zorder=4)
        for i, item in enumerate(characterizations):
            values = np.asarray(item["projection"])[:, d]
            color = variant_color(item["label"], i)
            # A binned histogram of a few hundred codes is mostly bin noise, and
            # four of them overlaid are unreadable. The KDE shows the shape the
            # KS statistic is actually testing.
            density = gaussian_kde(values)(grid)
            ax.plot(grid, density, color=color, linewidth=1.8,
                    label=f"{item['label']}  KS={kstest(values, 'norm').statistic:.3f}")
            ax.fill_between(grid, density, color=color, alpha=0.10, linewidth=0)

        style(ax, f"Direction {d + 1}", "density" if d == 0 else "",
              xlabel=f"$z_{d + 1}$ (standardized)")
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Headroom for the legend, which otherwise lands on the tallest mode. Set
    # on one axes only, sharey carries it to the rest.
    peak = max((line.get_ydata().max() for panel in axes for line in panel.lines),
               default=1.0)
    axes[0].set_ylim(0, peak * 1.5)

    panel_labels(axes)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


def _hole_curve_panel(ax, characterizations, percentiles):
    """Hole fraction against the density cut, drawn into an existing axes."""
    for i, item in enumerate(characterizations):
        curve = hole_curve(item["holes"], percentiles)
        ax.plot(percentiles, curve, color=variant_color(item["label"], i),
                linewidth=2, marker=MARKERS[item["divergence"]], markersize=4,
                label=item["label"])

    ax.axvline(HOLE_PERCENTILE, color=THRESHOLD, linewidth=1.2, linestyle="--")
    threshold_label(ax, f"reported at {HOLE_PERCENTILE:g}%", x=HOLE_PERCENTILE)
    ax.set_ylim(0, 1)
    style(ax, "Hole fraction against the density cut",
          "fraction of query points in holes",
          xlabel="density percentile used as the cut (%)")
    return ax


def _hole_bandwidth_panel(ax, characterizations, multiples):
    """Hole fraction against the kernel width, drawn into an existing axes."""
    for i, item in enumerate(characterizations):
        curve = hole_bandwidth_curve(item["projection"], multiples=multiples)
        ax.plot(multiples, curve, color=variant_color(item["label"], i),
                linewidth=2, marker=MARKERS[item["divergence"]], markersize=4,
                label=item["label"])

    ax.axvline(1.0, color=THRESHOLD, linewidth=1.2, linestyle="--")
    threshold_label(ax, "reported at Scott's rule", x=1.0)
    ax.set_xscale("log")
    ax.set_xticks(multiples)
    ax.set_xticklabels([f"{m:.2g}x" for m in multiples])
    ax.minorticks_off()                 # log minor ticks clutter a 7 point axis
    ax.set_ylim(0, 1)
    style(ax, "Hole fraction against the kernel width",
          "fraction of query points in holes",
          xlabel="KDE bandwidth (multiples of Scott's rule)")
    return ax


def plot_hole_curve(characterizations, percentiles=None, filename="latent_hole_curve.png"):
    """
    Hole fraction as the density cut sweeps. Curves that stay apart are a real
    ordering, curves that cross are not.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    percentiles = np.linspace(1, 25, 25) if percentiles is None else percentiles

    fig, ax = plt.subplots(figsize=(8.5, 5))
    _hole_curve_panel(ax, characterizations, percentiles)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


def plot_hole_bandwidth(characterizations, multiples=None,
                        filename="latent_hole_bandwidth.png"):
    """
    Hole fraction against the KDE kernel width.

    The cut is a percentile of the code density, which moves with the bandwidth too, so
    read the SEPARATION between curves rather than their level.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    multiples = HOLE_BW_SCALES if multiples is None else multiples

    fig, ax = plt.subplots(figsize=(8.5, 5))
    _hole_bandwidth_panel(ax, characterizations, multiples)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


def plot_hole_robustness(characterizations, percentiles=None, multiples=None,
                         filename="latent_hole_robustness.png"):
    """
    Both robustness checks on the hole fraction, side by side: A moves the density
    cut, B moves the KDE bandwidth. Shared y, one legend.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    percentiles = np.linspace(1, 25, 25) if percentiles is None else percentiles
    multiples = HOLE_BW_SCALES if multiples is None else multiples

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True)
    _hole_curve_panel(ax1, characterizations, percentiles)
    _hole_bandwidth_panel(ax2, characterizations, multiples)
    ax2.set_ylabel("")                  # sharey, the left label speaks for both
    ax1.legend(frameon=False, fontsize=9, loc="lower right")

    panel_labels((ax1, ax2))
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


def plot_projection_comparison(characterizations,
                               filename="latent_projection_choice.png"):
    """
    What the projection choice costs.

    Left: how much of the code's spread each candidate plane actually carries.
    Right: the hole fraction each one reports. Colour here separates PROJECTION,
    not variant, so it deliberately stays off the variant colors.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = [item["label"] for item in characterizations]
    x = np.arange(len(labels))
    pca_color, raw_color = oi.GREEN, oi.PURPLE

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, keys, names, title, ylabel in (
        (ax1, ("var_top2_pca", "var_top2_raw"), ("top-2 PCs", "top-2 raw dims"),
         "Spread captured by the plotted plane", "share of total code variance"),
        (ax2, ("hole_fraction", "raw_hole_fraction"),
         ("holes, PCA plane", "holes, raw plane"),
         "Hole fraction under each projection", "fraction of query points in holes"),
    ):
        ax.bar(x - 0.2, [item[keys[0]] for item in characterizations], 0.4,
               color=pca_color, label=names[0])
        ax.bar(x + 0.2, [item[keys[1]] for item in characterizations], 0.4,
               color=raw_color, label=names[1])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        style(ax, title, ylabel, xlabel="")
        ax.legend(frameon=False, fontsize=9)

    panel_labels((ax1, ax2))
    fig.subplots_adjust(wspace=0.2)
    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


SUMMARY_METRICS = (
    ("active", "live directions", "high"),
    ("pr", "participation ratio", "high"),
    ("hole_fraction", "hole fraction", "low"),
    ("ks", "KS vs prior shape", "low"),
    ("recon_median", "median recon MSE", "low"),
    ("scale_ratio", "code scale / prior", "low"),
)


def plot_summary(characterizations, filename="latent_summary.png"):
    """
    Every scalar side by side, each column rescaled to its own range.

    Colour is the position of the raw value between the column min and max, so a bigger
    number is always a brighter cell. It does not encode goodness; the column header
    says which end is good.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = [item["label"] for item in characterizations]
    raw = np.array([[float(item[key]) for key, _, _ in SUMMARY_METRICS]
                    for item in characterizations])

    # Straight column-wise rescale to [0,1]. No direction flip, so cell colour
    # and cell number always move together.
    scaled = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        column = raw[:, j]
        lo, hi = column.min(), column.max()
        scaled[:, j] = ((column - lo) / (hi - lo) if hi > lo
                        else np.full_like(column, 0.5))

    fig, ax = plt.subplots(figsize=(1.9 * len(SUMMARY_METRICS) + 3.2, 1.0 * len(labels) + 2.4))
    image = ax.imshow(scaled, cmap=oi.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            value = raw[i, j]
            text = f"{value:.0f}" if abs(value) >= 100 else f"{value:.3g}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9.5,
                    color="white" if scaled[i, j] < 0.55 else oi.BLACK)

    ax.set_xticks(range(len(SUMMARY_METRICS)))
    ax.set_xticklabels([f"{name}\n({better} is better)"
                        for _, name, better in SUMMARY_METRICS], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.tick_params(length=0)
    # Frame the matrix at the same weight the axis based plots use for a spine
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color(oi.BLACK)

    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("column min to max", fontsize=9.5)
    bar.set_ticks([0.0, 0.5, 1.0])
    bar.ax.tick_params(labelsize=9)
    bar.outline.set_linewidth(1)
    bar.outline.set_edgecolor(oi.BLACK)

    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


# The code is per component, so a cluster of codes is a group of ingredients the
# encoder placed together. Decoding a centroid therefore reads out what a region
# of the latent space means, which is the interpretability handle.
#
# These panels use the most variable ORIGINAL dimensions, not principal
# directions, so the axis labels name coordinates the decoder actually consumes
# and the per dimension centroid table lines up with them. The rotation
# invariant view is the manifold plot, which is where collapse is judged.

def top_variance_dims(Z, n_dims=4):
    """Indices of the most variable original coordinates, largest first."""
    variance = np.asarray(Z).var(axis=0)
    return np.argsort(variance)[::-1][:n_dims]


def cluster_styles(labels):
    """Color and marker per cluster, so color is never the only cue."""
    unique = np.unique(labels)
    return {c: oi.CLUSTER_STYLES[i % len(oi.CLUSTER_STYLES)] for i, c in enumerate(unique)}


def _corner_panels(item, axes, n_dims=4, point_size=5, label_size=10):
    """
    Draw one variant's corner plot into an existing (n-1, n-1) axes grid.

    Split out so a single variant and the pooled grid share one definition of
    what a corner block looks like.
    """
    dims = list(item["top_dims"][:n_dims])
    Z = np.asarray(item["codes"])[:, dims]
    labels = np.asarray(item["cluster_labels"])
    lookup = cluster_styles(labels)
    n = len(dims)

    for i in range(n - 1):
        for j in range(n - 1):
            ax = axes[i][j]
            if j > i:                                   # upper triangle is redundant
                ax.axis("off")
                continue

            for cluster, (color, marker) in lookup.items():
                pick = labels == cluster
                ax.scatter(Z[pick, j], Z[pick, i + 1], color=color, marker=marker,
                           s=point_size * 2, alpha=0.75, linewidths=0)
            ax.tick_params(labelsize=7.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1)
                spine.set_color(oi.BLACK)

            if j == 0:
                ax.set_ylabel(f"$z_{{{dims[i + 1]}}}$", fontsize=label_size)
            else:
                ax.set_yticklabels([])
            if i == n - 2:
                ax.set_xlabel(f"$z_{{{dims[j]}}}$", fontsize=label_size)
            else:
                ax.set_xticklabels([])

    # Legend goes in the freed upper triangle, which no panel is using
    handles = [Line2D([], [], marker=marker, linestyle="none", markersize=6,
                      markerfacecolor=color, markeredgecolor="none",
                      label=f"cluster {label}")
               for label, (color, marker) in lookup.items()]
    axes[0][n - 2].legend(handles=handles, loc="center", frameon=False, fontsize=8,
                          ncol=2 if len(handles) > 6 else 1,
                          title=f"{len(handles)} clusters", title_fontsize=9)
    return axes


def plot_latent_clusters(item, n_dims=4, filename=None, point_size=5):
    """
    Corner plot of the most active original dimensions, coloured by cluster.

    Lower triangle only, since the upper half is the same pairs transposed.
    ``item`` needs ``codes``, ``top_dims`` and ``cluster_labels``.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    n = min(n_dims, len(item["top_dims"]))
    fig, axes = plt.subplots(n - 1, n - 1, figsize=(2.9 * (n - 1), 2.9 * (n - 1)),
                             squeeze=False)
    _corner_panels(item, axes, n_dims, point_size)

    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    if filename:
        fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig


def plot_latent_clusters_grid(characterizations, n_dims=4, point_size=4,
                              filename="latent_clusters.png"):
    """
    Every variant's corner plot on one canvas, lettered A onward.

    Each variant keeps its own top_dims, cluster count and palette, so the four
    blocks are NOT a shared coordinate system: the letter and the caption are
    what tie a block to its variant. Subfigures rather than one flat grid, so
    each block keeps its own tight corner spacing and its own legend.
    """
    from string import ascii_uppercase

    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    n = n_dims - 1
    rows = (len(characterizations) + 1) // 2
    fig = plt.figure(figsize=(2.5 * n * 2 + 1.0, 2.5 * n * rows + 0.6))
    subfigs = np.atleast_1d(fig.subfigures(rows, 2)).ravel()

    for letter, subfig, item in zip(ascii_uppercase, subfigs, characterizations):
        axes = subfig.subplots(n, n, squeeze=False)
        _corner_panels(item, axes, n_dims, point_size, label_size=9)
        subfig.subplots_adjust(wspace=0.08, hspace=0.08, top=0.93)
        subfig.text(0.0, 0.995, letter, ha="left", va="top", fontsize=12, fontweight="bold")

    for subfig in subfigs[len(characterizations):]:
        subfig.set_visible(False)

    fig.savefig(config.FIGURE_DIR / filename, dpi=150, bbox_inches="tight")

    return fig
