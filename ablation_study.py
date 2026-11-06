"""
Ablation study for MVRRAE_GV to determine each component worth, over N seeds.

Every arm turns off exactly one component and replaces it with a defined
substitute, so a drop in a metric is attributable to the component and not to
the hole left behind:

    baseline    everything on
    no_svd      reduced-rank reparameterization -> per-component reparameterization
    no_mmd      MMD against a prior sample      -> closed-form KL
    no_rgcn     pathway R-GCN readout           -> mean-pool + MLP of matched width
    no_gate     Gumbel gate                     -> identity, every component passes
    no_cost     latent cost readout             -> head left untrained (delta = 0)

Weights come from the hipo.py sweep via config.py and are not rebalanced per arm.
The exceptions are GATE_GAMMA, dropped in `no_gate` because the substitute gate is
constant and its penalty carries no gradient, and COST_DELTA, which `no_cost` turns off.

`cost_probe_r2` is the cost row to read across arms: a ridge probe from the frozen
training-split codes to ln(cost), defined in `no_cost` arms whose own head never
trained. `cost_r2` and `recipe_cost_rho` exist only where `use_cost` is on (ADR-005).
"compute" below is FLOPs and parameters, not money.

`gate_auc` is the row to read first. Under the SVD, `Yk = Uk @ a_z` mixes the whole
recipe, so a gate can score well by recognizing a poison recipe rather than a poison
molecule; `no_svd` removes the mixing. A gap between the two is the leak.

The sixth factor is the dataset: every arm also exists on the graded negatives, named
with a `+graded` suffix and built on `data/database_graded.json`. Both levels share a
config_id, so the arm label is what tells them apart.

Run:
    python ablation_study.py --list-arms          # the 64 arms and their factors
    python ablation_study.py --params             # live/total counts, delta vs full
    python ablation_study.py                      # 64 arms x 3 seeds
    python ablation_study.py --arms baseline+graded --fold 3 --seeds 0 1 --resume
    python ablation_study.py --table              # reprint from the log, no training
    python ablation_study.py --evaluate artifacts/ablation__baseline__*.pth
"""

import argparse
import contextlib
import glob
import io
import itertools
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src import benchmark, config
from src.ablation import (
    AblationSpec,
    _build,
    checkpoint_path,
    evaluate_checkpoint,
    load_rows,
    run_spec,
)

__all__ = [
    "ARMS",
    "current",
    "figures",
    "main",
    "plot_arms",
    "plot_capacity",
    "plot_deltas",
    "plot_effects",
    "report",
    "table",
]

RUN_LOG = config.ARTIFACT_DIR / "ablation_runs.jsonl"

FACTORS = (("rgcn", "use_rgcn"), ("svd", "use_svd"),
           ("mmd", "use_mmd"), ("gate", "use_gate"), ("cost", "use_cost"))
LEVELS = ("legacy", "graded")
# The all-on arm of each level, the reference the per-arm deltas are taken against.
FULL_ARM = {"legacy": "baseline", "graded": "baseline+graded"}
# Read as a boolean for the effect contrasts, so the main effect of `negatives` is
# legacy minus graded, the same off-minus-on direction as the other five factors.
EFFECT_FACTORS = FACTORS + (("negatives", "negatives"),)


def _label(flags, level="legacy"):
    off = [f"no_{name}" for name, field in FACTORS if not flags[field]]
    label = "+".join(off) if off else "baseline"
    return f"{label}+graded" if level != "legacy" else label


def _label_level(label):
    return "graded" if label.endswith("+graded") else "legacy"


def _level(row):
    """Which dataset a log row came from; older rows carry it only in the label."""
    return row.get("negatives") or _label_level(row["label"])


def _cells():
    arms = []
    for level in LEVELS:
        for combo in itertools.product((True, False), repeat=len(FACTORS)):
            flags = {field: on for (_, field), on in zip(FACTORS, combo)}
            arms.append(AblationSpec(label=_label(flags, level), negatives=level, **flags))
    return tuple(sorted(arms, key=lambda a: (a.negatives != "legacy", a.label != "baseline",
                                             a.label.count("+"), a.label)))


ARMS = _cells()

# The one-factor-at-a-time subset, for the per-arm bar figure; the factorial
# belongs in plot_effects anyway. Both levels, since a graded arm beside its
# legacy twin is what the sixth factor is for.
OAT_LEGACY = ("baseline", "no_svd", "no_mmd", "no_rgcn", "no_gate", "no_cost")
OAT_LABELS = OAT_LEGACY + tuple(f"{label}+graded" for label in OAT_LEGACY)

# Printed in this order; the rest stay in the log.
HEADLINE = (
    ("gate_auc", "gate AUC", "{:>8.3f}"),
    ("recon", "recon", "{:>9.3f}"),
    ("nmse_x1", "nmse x1", "{:>8.3f}"),
    ("nmse_x2", "nmse x2", "{:>8.3f}"),
    ("active", "live", "{:>6.1f}"),
    ("pr", "PR", "{:>6.2f}"),
    ("gate_open", "gate", "{:>6.2f}"),
    ("div_loss", "div", "{:>9.4f}"),
    ("cost_probe_r2", "probe R2", "{:>10.3f}"),
    ("cost_r2", "cost R2", "{:>10.3f}"),
    ("cost_mae_dec", "cost MAE", "{:>10.3f}"),
    ("recipe_cost_rho", "rcp rho", "{:>10.3f}"),
)

COST = (
    ("params_live", "params live", "{:>12,.0f}"),
    ("flops_per_example", "FLOPs/recipe", "{:>13,.0f}"),
    ("wall_clock_s", "wall (s)", "{:>9.1f}"),
)


def _fold_rows(rows, fold):
    """Rows from one grouped fold; logs written before folds existed count as fold 0."""
    return [row for row in rows if row.get("fold", 0) == fold]


def _report_rows(rows, fold):
    return rows if fold is None else _fold_rows(rows, fold)


def current(rows, warn=False):
    """
    Rows from the newest config_id per arm.

    The log is append-only and an arm spans several config_ids once a spec default moves;
    mixing them would average two experiments and call it a seed spread.
    """
    newest = {}
    for row in rows:
        stamp = (row["timestamp"], row["config_id"])
        if row["label"] not in newest or stamp > newest[row["label"]]:
            newest[row["label"]] = stamp

    keep = {label: config_id for label, (_, config_id) in newest.items()}
    if warn:
        stale = sorted({(r["label"], r["config_id"]) for r in rows
                        if r["config_id"] != keep[r["label"]]})
        for label, config_id in stale:
            print(f"  ignoring stale {label} rows from config {config_id}, "
                  f"superseded by {keep[label]}")

    return [row for row in rows if row["config_id"] == keep[row["label"]]]


def _reference(key, rows, split):
    """
    (value, label) for a metric's reference line, or None.

    `active` is against the collapse threshold rather than zero, and that
    threshold is a fraction of the latent width, so it comes from the data.
    """
    if key == "active":
        widths = {row["value"] for row in current(rows)
                  if row["split"] == split and row["metric"] == "d_z"}
        if not widths:
            return None
        d_z = max(widths)
        return benchmark.COLLAPSE_FRACTION * d_z, f"collapse ({d_z:.0f} wide)"
    return REFERENCE.get(key)


def _run_key(row):
    """(fold, seed): one run. A seed repeats across the five grouped folds."""
    return (row.get("fold", 0), row["seed"])


def _by_arm(rows, split):
    """{label: {metric: {(fold, seed): value}}} for one split."""
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in current(rows):
        if row["split"] == split:
            grouped[row["label"]][row["metric"]][_run_key(row)] = row["value"]

    # gate_auc is signed: below 0.5 means the gate closes on toxic components,
    # which is the useful direction
    return grouped


def _vals(by_run):
    """Values in run order, for the summaries that do not care which run."""
    return [by_run[run] for run in sorted(by_run)] if by_run else []


def _costs(rows):
    """{label: {cost field: value}}, one per arm; costs do not vary by run."""
    out = {}
    for row in current(rows):
        out.setdefault(row["label"], {key: row[key] for key, _, _ in COST})
    return out


def table(rows, split, arms=None):
    """Mean +/- sd over runs, one line per arm."""
    grouped = _by_arm(rows, split)
    labels = [a for a in (arms or [spec.label for spec in ARMS]) if a in grouped]
    if not labels:
        print(f"nothing logged for split={split}")
        return

    counts = {len(v) for m in grouped.values() for v in m.values()}
    n_runs = max(counts)
    # The multi-factor labels run to ~29 characters; a fixed width silently
    # pushes every column out of line.
    width = max(len(label) for label in labels) + 2

    def block(reduce, note):
        header = f"{'arm':<{width}}" + "".join(f"{name:>10}" for _, name, _ in HEADLINE)
        print(f"\n{note}\n{header}\n" + "-" * len(header))
        for label in labels:
            line = f"{label:<{width}}"
            for key, _, fmt in HEADLINE:
                values = _vals(grouped[label].get(key, {}))
                line += fmt.format(float(reduce(values))) if values else f"{'--':>8}"
            print(line)

    spread = "" if len(counts) == 1 else f"  WARNING: arms hold {sorted(counts)} runs"
    block(np.mean, f"{split}  mean over {n_runs} run{'s' if n_runs > 1 else ''}{spread}")
    # Spread matters more than the mean at 121 usable recipes.
    block(np.std, f"{split}  standard deviation over runs")


def cost_table(rows):
    """What each arm costs, so a drop cannot be read as lost capacity."""
    costs = _costs(rows)
    labels = [spec.label for spec in ARMS if spec.label in costs]
    if not labels:
        return

    width = max(len(label) for label in labels) + 2
    header = f"{'arm':<{width}}" + "".join(f"{name:>15}" for _, name, _ in COST)
    print(f"\ncompute\n{header}\n" + "-" * len(header))
    for label in labels:
        line = f"{label:<{width}}"
        for key, _, fmt in COST:
            line += fmt.format(costs[label][key])
        print(line)


def deltas(rows, split, metric="gate_auc"):
    """Each arm against the full arm of its own dataset, with a paired-by-seed spread."""
    grouped = _by_arm(rows, split)
    for level in LEVELS:
        reference = FULL_ARM[level]
        if metric not in grouped.get(reference, {}):
            continue

        base = np.mean(_vals(grouped[reference][metric]))
        print(f"\n{metric} vs {reference} ({split})")
        for spec in ARMS:
            if spec.negatives != level or spec.label == reference:
                continue
            values = _vals(grouped.get(spec.label, {}).get(metric, {}))
            if not values:
                continue
            delta = float(np.mean(values)) - base
            print(f"  {spec.label:<30} {np.mean(values):7.3f}   {delta:+7.3f}")


def report(rows, split="test", arms=None):
    rows = current(rows, warn=True)
    table(rows, split, arms)
    cost_table(rows)
    deltas(rows, split)
    deltas(rows, split, "cost_probe_r2")
    deltas(rows, split, "recon")


# div_loss is deliberately absent because MMD value
# and a KL value are not the same quantity
FIGURE_METRICS = (
    ("gate_auc", "gate AUC, toxic vs rest"),
    ("recon", "reconstruction (MSE)"),
    ("nmse_x1", "nmse x1"),
    ("nmse_x2", "nmse x2"),
    ("active", "live directions"),
    ("pr", "participation ratio"),
    ("eff_rank", "effective rank"),
    ("gate_open", "mean gate"),
    ("cost_probe_r2", "cost probe R2 (frozen code)"),
    ("cost_r2", "cost head R2"),
    ("recipe_cost_rho", "recipe cost Spearman"),
)

REFERENCE = {
    "gate_auc": (0.5, "chance"),
    "nmse_x1": (1.0, "trivial"),
    "nmse_x2": (1.0, "trivial"),
    "gate_open": (1.0, "fully open"),
    "cost_probe_r2": (0.0, "mean predictor"),
    "cost_r2": (0.0, "mean predictor"),
    "recipe_cost_rho": (0.0, "no rank agreement"),
}

DERIVED = {}

GATE_DEPENDENT = ("gate_auc", "gate_open")
# Only defined where the cost head trained, so the cost factor is dropped from
# their effects the way the gate factor is dropped from the gate metrics.
COST_DEPENDENT = ("cost_r2", "cost_mae_dec", "recipe_cost_rho")

# One for discrimination, reconstruction, latent width
# Effects are drawn for each of these when --metric is not given.
EFFECT_METRICS = (
    ("gate_auc", "gate AUC, toxic vs rest"),
    ("recon", "reconstruction (MSE)"),
    ("active", "live directions"),
    ("cost_probe_r2", "cost probe R2"),
)

CAPACITY_METRICS = (
    ("gate_auc", "gate AUC, toxic vs rest"),
    ("recon", "reconstruction (MSE)"),
    ("active", "live directions"),
)

BASELINE_COLOR = benchmark.OKABE_ITO[4]
ARM_COLOR = benchmark.OKABE_ITO[0]
CHANCE_COLOR = benchmark.OKABE_ITO[5]

# Seeds are told apart by marker shape
# Spread is black and plain; the mean is the only thing carrying colour, so the
# eye lands on it first and the arms stay distinguishable at a glance.
SPREAD_COLOR = "#1A1A1A"
# One hue per arm, held stable across every figure so a colour means the same
# arm wherever it appears. OAT arms come first, so they keep the hues the
# capacity panels already use; the multi-factor cells follow.
ARM_ORDER = OAT_LABELS + tuple(spec.label for spec in ARMS
                               if spec.label not in OAT_LABELS)


def _arm_colour(label):
    index = ARM_ORDER.index(label) if label in ARM_ORDER else 0
    return benchmark.OKABE_ITO[index % len(benchmark.OKABE_ITO)]
MEAN_SIZE = 95

SEED_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
SEED_EDGE = "#1A1A1A"
SEED_FACE = "#FFFFFF"
SEED_SIZE = 58


def _paired(rows, split, metric):
    """{label: {(fold, seed): value}} so arms can be differenced run by run."""
    source, transform = DERIVED.get(metric, (metric, lambda v: v))
    out = defaultdict(dict)
    for row in current(rows):
        if row["split"] == split and row["metric"] == source:
            out[row["label"]][_run_key(row)] = transform(row["value"])
    return out


def _arms_present(grouped, arms=None):
    wanted = arms or [spec.label for spec in ARMS]
    return [label for label in wanted if label in grouped]


def _marker(run, runs):
    """Shape per run, so one run keeps its shape across every panel."""
    if len(runs) > len(SEED_MARKERS):
        return SEED_MARKERS[0]
    return SEED_MARKERS[sorted(runs).index(run)]


def _seed_dots(ax, index, by_run, runs, width=0.13):
    """Runs on the arm's center line"""
    ordered = sorted(by_run, key=lambda run: by_run[run])
    values = [by_run[run] for run in ordered]

    def place(run, x, y):
        ax.scatter(x, y, marker=_marker(run, runs), s=SEED_SIZE,
                   facecolors=SEED_FACE, edgecolors=SEED_EDGE,
                   linewidths=1.2, zorder=4)

    # No spread at all (no_gate's AUC is 0.5 by construction) is one point, not
    # N identical markers smeared sideways.
    if len(set(values)) == 1:
        place(ordered[0], index, values[0])
        return

    # Group runs of values close enough that their markers would overlap.
    tol = (max(values) - min(values)) * 0.05
    clusters, run = [], [0]
    for k in range(1, len(values)):
        if values[k] - values[k - 1] <= tol:
            run.append(k)
        else:
            clusters.append(run)
            run = [k]
    clusters.append(run)

    for run in clusters:
        if len(run) == 1:
            place(ordered[run[0]], index, values[run[0]])
            continue
        step = min(width, width * 2 / (len(run) - 1))
        start = -step * (len(run) - 1) / 2
        for k, idx in enumerate(run):
            place(ordered[idx], index + start + k * step, values[idx])


# Below this many runs the quartiles are the sample, so the box says nothing a
# reader cannot get from the points.
MIN_SEEDS_FOR_BOX = 5


def _candle(ax, index, values, colour, width=0.5, box=False, vertical=True,
            edge="white", label=None):
    lo, hi = float(np.min(values)), float(np.max(values))
    centre = float(np.mean(values))
    cap = width * 0.3

    def line(along, across, **kw):
        pair = (along, across) if vertical else (across, along)
        ax.plot(*pair, **kw)

    line([index, index], [lo, hi], color=SPREAD_COLOR, linewidth=1.3, zorder=2,
         solid_capstyle="round")
    for tip in (lo, hi):
        line([index - cap, index + cap], [tip, tip], color=SPREAD_COLOR,
             linewidth=1.3, zorder=2, solid_capstyle="round")

    if box and hi > lo:
        q1, q3 = np.percentile(values, [25, 75])
        corner = (index - width / 2, q1) if vertical else (q1, index - width / 2)
        size = (width, max(q3 - q1, 1e-12)) if vertical else (max(q3 - q1, 1e-12), width)
        ax.add_patch(plt.Rectangle(corner, *size, facecolor=colour, alpha=0.25,
                                   edgecolor="none", zorder=2))

    point = (index, centre) if vertical else (centre, index)
    ax.scatter(*point, s=MEAN_SIZE, color=colour, zorder=5,
               edgecolors=edge, linewidths=1.3, label=label)
    return centre


# Below this many seeds the quartiles are the sample, so the box says nothing a
# reader cannot get from the points.
MIN_SEEDS_FOR_BOX = 5


def plot_arms(rows, split="test", arms=None, filename=None, points=False):
    """
    One panel per metric. Drawn as median, interquartile body and min-max whiskers.
    """
    grouped = _by_arm(rows, split)
    labels = _arms_present(grouped, arms or OAT_LABELS)
    if not labels:
        return None

    panels = [(key, name) for key, name in FIGURE_METRICS
              if any(grouped[label].get(key) for label in labels)]
    runs = sorted({run for label in labels for m in grouped[label].values() for run in m})
    ncols = min(4, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(labels))

    boxed = len(runs) >= MIN_SEEDS_FOR_BOX

    for ax, (key, name) in zip(axes, panels):
        medians = {}
        for i, label in enumerate(labels):
            values = _vals(grouped[label][key])
            if not values:
                continue
            colour = _arm_colour(label)
            medians[label] = _candle(ax, i, values, colour, box=boxed,
                                     edge=SPREAD_COLOR)
            if points:
                _seed_dots(ax, i, grouped[label][key], runs)

        if "baseline" in medians:
            ax.axhline(medians["baseline"], color=_arm_colour("baseline"),
                       linewidth=1, linestyle="--", zorder=1)
        reference = _reference(key, rows, split)
        if reference is not None:
            value, note = reference
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, value), max(hi, value))
            ax.axhline(value, color=CHANCE_COLOR, linewidth=1.4, zorder=1)
            ax.annotate(note, xy=(0.995, value), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points",
                        color=CHANCE_COLOR, fontsize=8, ha="right")

        benchmark.style(ax, name, "", xlabel="")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_xlim(-0.7, len(labels) - 0.3)
        ax.margins(y=0.12)

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    # The shape legend only means something when the runs are drawn.
    if points and len(runs) <= len(SEED_MARKERS):
        handles = [Line2D([], [], marker=_marker(run, runs), linestyle="none",
                          markerfacecolor=SEED_FACE, markeredgecolor=SEED_EDGE,
                          markeredgewidth=1.2, markersize=7,
                          label=f"fold {run[0]} seed {run[1]}")
                   for run in runs]
        fig.legend(handles=handles, frameon=False, fontsize=9, ncol=len(runs),
                   loc="upper right", bbox_to_anchor=(0.995, 0.995),
                   handletextpad=0.3, columnspacing=1.4)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save(fig, filename or _figure(f"ablation_arms__{split}"))


def plot_deltas(rows, split="test", metric="gate_auc", arms=None, filename=None):
    """Each arm minus the full arm of its dataset, differenced within a run."""
    paired = _paired(rows, split, metric)
    if not any(label in paired for label in FULL_ARM.values()):
        return None

    labels = [label for label in _arms_present(paired, arms or OAT_LABELS)
              if label not in FULL_ARM.values()]
    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 0.9 * len(labels) + 2.4))

    for i, label in enumerate(labels):
        reference = FULL_ARM[_label_level(label)]
        if reference not in paired:
            continue
        base = paired[reference]
        shared = sorted(set(base) & set(paired[label]))
        deltas = [paired[label][run] - base[run] for run in shared]
        if not deltas:
            continue
        mean = _candle(ax, i, deltas, _arm_colour(label), width=0.5,
                       box=len(deltas) >= MIN_SEEDS_FOR_BOX, vertical=False,
                       edge=SPREAD_COLOR)

        # In a column outside the axes; a negative bar's label would otherwise
        # land on the y tick labels.
        ax.annotate(f"{mean:+.3f}", xy=(1.02, i), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=9, clip_on=False)

    ax.axvline(0, color=BASELINE_COLOR, linewidth=1.4, zorder=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    benchmark.style(ax, "", "", xlabel=f"Δ{metric} (paired by fold and seed)")
    fig.tight_layout()
    return _save(fig, filename or _figure(f"ablation_delta_{metric}__{split}"))


def _dodge(positions, fraction=0.035):
    """Nudge arms that share an x apart, so a shared capacity is still legible."""
    span = (max(positions.values()) - min(positions.values())) or 1.0
    groups = defaultdict(list)
    for label, x in positions.items():
        groups[x].append(label)

    out = {}
    for x, members in groups.items():
        step = span * fraction
        start = -step * (len(members) - 1) / 2
        for k, label in enumerate(sorted(members)):
            out[label] = start + k * step
    return out


def plot_capacity(rows, split="test", metrics=None, arms=None, filename=None):
    """Every headline metric against the capacity that produced it."""
    grouped = _by_arm(rows, split)
    costs = _costs(rows)
    metrics = metrics or CAPACITY_METRICS
    labels = [label for label in _arms_present(grouped, arms or OAT_LABELS)
              if label in costs]
    panels = [(key, name) for key, name in metrics
              if any(grouped[label].get(key) for label in labels)]
    if not labels or not panels:
        return None

    origin = costs.get("baseline", costs[labels[0]])["params_live"]
    offsets = _dodge({label: costs[label]["params_live"] - origin for label in labels})
    places = {label: costs[label]["params_live"] - origin + offsets[label]
              for label in labels}
    # The x axis is parameter counts, so the cap width has to be in those units
    # rather than the index units the per-arm figures use.
    span = (max(places.values()) - min(places.values())) or 1.0
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.8))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, name) in zip(axes, panels):
        for i, label in enumerate(labels):
            by_run = grouped[label].get(key, {})
            if not by_run:
                continue
            # Same candle as the per-arm and effect figures: black spread,
            # Okabe-Ito mean, dark outline.
            _candle(ax, places[label], _vals(by_run), _arm_colour(label),
                    width=span * 0.12, box=len(by_run) >= MIN_SEEDS_FOR_BOX,
                    edge=SPREAD_COLOR, label=label if ax is axes[0] else None)

        ax.axvline(0, color=BASELINE_COLOR, linewidth=1, linestyle="--", zorder=1)
        ax.annotate("baseline", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -10), textcoords="offset points",
                    color=BASELINE_COLOR, fontsize=8)
        reference = _reference(key, rows, split)
        if reference is not None:
            value, note = reference
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, value), max(hi, value))
            ax.axhline(value, color=CHANCE_COLOR, linewidth=1.4, zorder=1)
            ax.annotate(note, xy=(0.995, value), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points",
                        color=CHANCE_COLOR, fontsize=8, ha="right")

        benchmark.style(ax, name, "", xlabel="change in gradient-carrying parameters")
        ax.ticklabel_format(axis="x", style="plain")
        ax.margins(x=0.22, y=0.18)

    spread = max(costs[l]["params_live"] for l in labels) - \
        min(costs[l]["params_live"] for l in labels)
    # Inside the plot area: above it collides with the panel title, and the
    # span is the figure's point rather than a decoration.
    axes[0].annotate(f"arms span {spread:,} parameters ({spread / origin:.2%})",
                     xy=(0.03, 0.03), xycoords="axes fraction", fontsize=9)

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=min(len(labels), 6), fontsize=9,
               handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, filename or _figure(f"ablation_capacity__{split}"))


def _design(rows, split, metric):
    """[(flags, (fold, seed), value)] for one metric, off the logged flag columns."""
    source, transform = DERIVED.get(metric, (metric, lambda v: v))
    out = []
    for row in current(rows):
        if row["split"] == split and row["metric"] == source:
            flags = {field: row.get(field, True) for _, field in FACTORS}
            flags["negatives"] = _level(row) == "graded"
            out.append((flags, _run_key(row), transform(row["value"])))
    return out


def _effects(rows, split, metric):
    """
    Main effects and two-way interactions, computed within each run.

    A factor's effect is the mean with it off minus the mean with it on,
    averaged over every setting of the other five. An interaction is half the
    difference between one factor's effect at the two settings of another, the
    usual 2^k contrast. Working within a run keeps the spread paired.
    """
    design = _design(rows, split, metric)
    if not design:
        return {}, {}, []

    drop_gate = metric in GATE_DEPENDENT
    drop_cost = metric in COST_DEPENDENT
    factors = [f for f in EFFECT_FACTORS if not (drop_gate and f[1] == "use_gate")
               and not (drop_cost and f[1] == "use_cost")]
    runs = sorted({run for _, run, _ in design})
    main, inter = defaultdict(dict), defaultdict(dict)
    complete, partial = [], []

    for run in runs:
        cells = [(fl, v) for fl, key, v in design if key == run
                 and (not drop_gate or fl["use_gate"])
                 and (not drop_cost or fl["use_cost"])]

        # A contrast is only a main effect if the other factors are balanced
        # across it. On a partial design the same arithmetic returns a number
        # that silently confounds the missing corners, so drop that run and
        # keep the ones that are complete.
        corners = {tuple(fl[field] for _, field in factors) for fl, _ in cells}
        if len(corners) < 2 ** len(factors):
            partial.append((run, len(corners)))
            continue
        complete.append(run)

        def effect(field, restrict=None):
            off = [v for fl, v in cells if not fl[field] and (restrict is None or restrict(fl))]
            on = [v for fl, v in cells if fl[field] and (restrict is None or restrict(fl))]
            return float(np.mean(off) - np.mean(on)) if off and on else None

        for name, field in factors:
            value = effect(field)
            if value is not None:
                main[name][run] = value

        for (n1, f1), (n2, f2) in itertools.combinations(factors, 2):
            at_off = effect(f1, lambda fl, g=f2: not fl[g])
            at_on = effect(f1, lambda fl, g=f2: fl[g])
            if at_off is not None and at_on is not None:
                inter[f"{n1} x {n2}"][run] = 0.5 * (at_off - at_on)

    if partial:
        print(f"  plot_effects: {metric} needs all {2 ** len(factors)} corners; "
              f"skipped {len(partial)} incomplete run(s) "
              f"{[f'f{r[0]}s{r[1]}:{n}' for r, n in partial]}, kept {len(complete)}")
    return dict(main), dict(inter), complete


def _effect_panel(ax, effects, runs, title, xlabel):
    names = list(effects)
    if not names:
        return
    boxed = len(runs) >= MIN_SEEDS_FOR_BOX
    for i, name in enumerate(names):
        # One Okabe-Ito hue per effect, so the rows are told apart by colour as
        # well as position. The pale yellow needs a dark outline on white.
        colour = benchmark.OKABE_ITO[i % len(benchmark.OKABE_ITO)]
        # Whether the range straddles the zero line is the whole inference
        # here, so the spread is the mark rather than a dressing on a mean bar.
        med = _candle(ax, i, _vals(effects[name]), colour, width=0.5,
                      box=boxed, vertical=False, edge=SPREAD_COLOR)
        # In a column just outside the axes: a label at a negative end would
        # otherwise land on the y tick labels.
        ax.annotate(f"{med:+.3f}", xy=(1.02, i), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=9, clip_on=False)

    # Zero is the reference the panel is read against: an effect whose range
    # crosses it is not distinguishable from no effect.
    ax.axvline(0, color="blue", linewidth=1.4, linestyle="--", zorder=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.margins(x=0.10, y=0.18)
    benchmark.style(ax, title, "", xlabel=xlabel)


def plot_effects(rows, split="test", metric="gate_auc", filename=None):
    """
    The factorial view for what each component is worth, averaged over the rest.

    The right panel says whether that average hides a dependence between two components.
    """
    main, inter, runs = _effects(rows, split, metric)
    if not main:
        return None

    fig, axes = plt.subplots(
        1, 2, figsize=(15, max(2.8, 0.66 * max(len(main), len(inter)) + 2.4)))
    note = ("  (gate-on cells only)" if metric in GATE_DEPENDENT
            else "  (cost-on cells only)" if metric in COST_DEPENDENT else "")
    _effect_panel(axes[0], main, runs, f"main effect of removing{note}",
                  f"change in {metric}")
    _effect_panel(axes[1], inter, runs, "two-way interaction",
                  f"half-difference in {metric}")

    # The shape legend only means something when the runs are drawn.
    fig.tight_layout(rect=(0, 0, 0.97, 0.97), w_pad=5.0)
    return _save(fig, filename or _figure(f"ablation_effects_{metric}__{split}"))


def _figure(stem):
    return f"{stem}.png"


def _save(fig, filename):
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURE_DIR / filename
    if path.exists():
        plt.close(fig)
        raise SystemExit(f"{path} already exists; refusing to overwrite it")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  wrote {path}")
    return path


def figures(rows, split="test", metric=None, arms=None):
    """Every figure the study draws"""
    print("figures:")
    plot_arms(rows, split, arms)
    plot_deltas(rows, split, metric or "gate_auc", arms)
    plot_capacity(rows, split, arms=arms)
    wanted = EFFECT_METRICS if metric is None else ((metric, None),)
    for key, _ in wanted:
        plot_effects(rows, split, key)


def _flags_key(spec):
    return (spec.use_svd, spec.use_mmd, spec.use_rgcn, spec.use_gate, spec.use_cost)


def list_arms():
    width = max(len(spec.label) for spec in ARMS) + 2
    header = f"{'arm':<{width}}" + "".join(f"{name:>6}" for name, _ in FACTORS) + f"{'dataset':>9}"
    print(f"{len(ARMS)} arms\n{header}\n{'-' * len(header)}")
    for spec in ARMS:
        print(f"{spec.label:<{width}}" + "".join(
            f"{'on' if getattr(spec, field) else 'off':>6}" for _, field in FACTORS)
            + f"{spec.negatives:>9}")


def params(arms=None, readout=None):
    """Live and total parameter counts per arm, with the offset from its full arm."""
    selected = [spec for spec in ARMS if arms is None or spec.label in arms]
    if not selected:
        raise SystemExit(f"no arm matched {arms}")
    if readout:
        selected = [replace(spec, readout=readout) for spec in selected]
    full_spec = replace(AblationSpec(), readout=readout or "softmax")

    counts, readouts = {}, {}
    # negatives picks a dataset and nothing in the model
    with contextlib.redirect_stdout(io.StringIO()):
        for spec in selected + [full_spec]:
            key = _flags_key(spec)
            if key in counts:
                continue
            model = _build(replace(spec, negatives="legacy")).model
            counts[key] = (sum(p.numel() for p in model.parameters() if p.requires_grad),
                           sum(p.numel() for p in model.parameters()))
            readouts[key] = sum(p.numel() for p in model.pathway_graph.parameters())

    full = counts[_flags_key(full_spec)][0]
    full_readout = readouts[_flags_key(full_spec)]
    width = max(len(spec.label) for spec in selected) + 2
    print(f"{'arm':<{width}}{'dataset':>9}{'live':>14}{'total':>14}"
          f"{'live vs full':>15}{'%':>9}")
    worst = 0.0
    for spec in selected:
        live, total = counts[_flags_key(spec)]
        delta = live - full
        share = 100.0 * delta / full
        print(f"{spec.label:<{width}}{spec.negatives:>9}{live:>14,}{total:>14,}"
              f"{delta:>+15,}{share:>+8.2f}%")
        if not spec.use_rgcn:
            worst = max(worst, abs(share))

    # Sizes the substitute readout to the RGCN it replaces
    # the whole-model live gap also carries the frozen graph node rows.
    matched = [abs(readouts[_flags_key(spec)] - full_readout) / full_readout
               for spec in selected if not spec.use_rgcn]
    worst_matched = 100.0 * max(matched, default=0.0)
    print(f"\nmatched readout (no_rgcn): worst {worst_matched:.3f}% of the RGCN it "
          f"replaces; whole-model live {worst:.2f}% (AC-A2 allows 1%) -> "
          f"{'PASS' if max(worst_matched, worst) <= 1.0 else 'FAIL'}")


def _run_study(selected, seeds, dataset_path, log, resume, save, verbose):
    """Every arm at every seed, skipping the runs the log already holds.

    config_id excludes negatives and fold, so both have to be in the resume key;
    without them a graded arm is skipped as its legacy twin's duplicate.
    """
    done = {(row["config_id"], row["label"], row["seed"], row.get("fold", 0))
            for row in load_rows(log)} if resume else set()
    planned = [replace(spec, seed=seed) for spec in selected for seed in seeds]

    for i, spec in enumerate(planned, start=1):
        key = (spec.config_id, spec.label, spec.seed, spec.fold)
        if key in done:
            if verbose:
                print(f"[{i}/{len(planned)}] {spec.label} seed={spec.seed} "
                      f"fold={spec.fold}  already logged, skipping")
            continue
        checkpoint = checkpoint_path(spec)
        if save and checkpoint.exists():
            raise SystemExit(f"{checkpoint} already exists; refusing to overwrite it "
                             f"(archive it, or pass --no-save)")
        if verbose:
            print(f"[{i}/{len(planned)}] {spec.label} seed={spec.seed} "
                  f"fold={spec.fold}  ({spec.config_id})")
        run_spec(spec, dataset_path, save, log, verbose)
        done.add(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--arms", nargs="*", default=None,
                        help=f"subset of {[spec.label for spec in ARMS]}")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                        help="one training run per arm per seed")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--data", default=None,
                        help="alternative preprocessed recipes, for an out-of-distribution row; "
                             "graded arms always read data/database_graded.json")
    parser.add_argument("--log", default=None,
                        help=f"run log (default {RUN_LOG})")
    parser.add_argument("--out", default=None, help="alias for --log")
    parser.add_argument("--readout", default=None, choices=("softmax", "additive"),
                        help="pathway readout for every arm (default: the spec default); "
                             "one study run is one readout")
    parser.add_argument("--split", default="test", choices=("val", "test"),
                        help="split to print; val is the one to select on")
    parser.add_argument("--fold", type=int, default=None,
                        help="grouped fold to train on; 0 is the held-out test fold "
                             "(default: 0 to train, the checkpoint's own fold to evaluate). "
                             "The report covers every fold in the log unless this names one")
    parser.add_argument("--resume", action="store_true",
                        help="append to an existing log, skipping the runs it already holds")
    parser.add_argument("--no-resume", action="store_true",
                        help="rerun arms already present in the log; with --resume only")
    parser.add_argument("--no-save", action="store_true", help="skip checkpoints")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--list-arms", dest="list_arms", action="store_true",
                        help="print every arm and its factor settings, no training")
    parser.add_argument("--params", action="store_true",
                        help="print live and total parameter counts per arm and the matched "
                             "readout offset against the full arm, no training")
    parser.add_argument("--table", action="store_true",
                        help="reprint the log and exit, no training")
    parser.add_argument("--figures", action="store_true",
                        help="draw the arm, delta and capacity figures into artifacts/figures")
    parser.add_argument("--metric", default=None,
                        help="restrict the delta and effect figures to one metric "
                             "(default: effects are drawn for each of EFFECT_METRICS)")
    parser.add_argument("--evaluate", nargs="*", default=None,
                        help="score saved checkpoints instead of training")
    args = parser.parse_args()

    if args.list_arms:
        list_arms()
        return
    if args.params:
        params(args.arms, args.readout)
        return
    if args.log and args.out and args.log != args.out:
        raise SystemExit("--log and --out name one file; pass only one of them")

    log = Path(args.out or args.log or RUN_LOG)
    fold = 0 if args.fold is None else args.fold

    if args.table:
        rows = _report_rows(load_rows(log), args.fold)
        report(rows, args.split, args.arms)
        if args.figures:
            figures(rows, args.split, args.metric, args.arms)
        return

    if args.evaluate is not None:
        paths = [p for pattern in args.evaluate for p in sorted(glob.glob(pattern))]
        if not paths:
            raise SystemExit("no checkpoints matched")
        for path in paths:
            spec, metrics = evaluate_checkpoint(path, args.split, args.data, fold=args.fold)
            print(f"\n{spec.label}  seed={spec.seed}  {spec.config_id}  "
                  f"fold={spec.fold}  ({args.split})")
            for key, name, fmt in HEADLINE:
                if metrics.get(key) is not None:
                    print(f"  {name:<10}" + fmt.format(metrics[key]))
        return

    selected = [spec for spec in ARMS if args.arms is None or spec.label in args.arms]
    if not selected:
        raise SystemExit(f"no arm matched {args.arms}")
    if args.data is not None and any(spec.negatives != "legacy" for spec in selected):
        raise SystemExit("--data would override the dataset of every arm, and graded "
                         "arms must read data/database_graded.json; select legacy arms only")

    if log.exists() and log.stat().st_size and not args.resume:
        raise SystemExit(f"{log} already exists; refusing to overwrite it "
                         f"(pass --resume to append, or choose a new --log)")

    overrides = {"epochs": args.epochs, "fold": fold}
    if args.readout:
        overrides["readout"] = args.readout
    selected = [replace(spec, **overrides) for spec in selected]
    print(f"{len(selected)} arm(s) x {len(args.seeds)} seed(s) -> {log}  fold {fold}")
    _run_study(selected, args.seeds, args.data, log,
               resume=args.resume and not args.no_resume,
               save=not args.no_save, verbose=not args.quiet)

    rows = _report_rows(load_rows(log), args.fold)
    report(rows, args.split, args.arms)
    if args.figures:
        figures(rows, args.split, args.metric, args.arms)


if __name__ == "__main__":
    main()
