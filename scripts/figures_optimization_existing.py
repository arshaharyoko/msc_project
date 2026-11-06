import csv
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import predict
from scripts import okabe_ito as oi

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "figures/optimization"
RECIPE = ROOT / "artifacts" / "recipes" / "best_prot_576.398"
SEEDS = range(10)
ARMS = {"raw": "Raw", "anch": "Anchored", "prot": "Anchored, protein floor"}
COLOR = dict(oi.OPT_ARM)
THRESH = "constraint_thresholds__ablation__selected__728320fe947b__s{}__graded__roundtrip-v2.json"

NAMES = {
    "P01133": "EGF",
    "Q8TAT2": "FGFBP3",
    "P04085": "PDGF-A",
    "P05019": "IGF-1",
    "[O-][Se](=O)[O-].[Na+].[Na+]": "Sodium selenite",
    "CC(=O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CCC4=CC(=O)CC[C@]34C)C": "Progesterone",
    "Cl[Cr](Cl)Cl": "Chromium chloride",
    "[O-]S(=O)(=O)[O-].[Ni+2]": "Nickel sulfate",
    "[O-]S(=O)(=O)[O-].[Cu+2]": "Copper sulfate",
    "Cl[Sn]Cl": "Tin chloride",
    "C(C(=O)[O-])C(CC(=O)[O-])(C(=O)[O-])O.[Fe+3]": "Ferric citrate",
    "[NH4+].[O-][V](=O)=O": "Ammonium vanadate",
    "[O-]S(=O)(=O)[O-].[Mn+2]": "Manganese sulfate",
    "[O-]S(=O)(=O)[O-].[Zn+2]": "Zinc sulfate",
    "[NH4+].[NH4+].[NH4+].[NH4+].[NH4+].[NH4+].O.O.O.O.[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[Mo].[Mo].[Mo].[Mo].[Mo].[Mo].[Mo]": "Ammonium molybdate",
}

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def label(component):
    return NAMES.get(component, component[:12])


def load_arm(arm):
    return [json.load(open(ROOT / "artifacts" / "optimizer" / f"latent_{arm}__s{k}.json")) for k in SEEDS]


def letter(ax, text):
    ax.text(0, 1.03, text, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def panel_dose(ax):
    floors = predict.protein_floors()
    rows = list(csv.DictReader(open(f"{RECIPE}.csv")))
    changed, added = [], []
    for r in rows:
        dose = float(r["concentration_mg_per_l"])
        if r["base_mg_per_l"] == "":
            added.append((label(r["component"]), dose, None, None))
            continue
        base = float(r["base_mg_per_l"])
        if abs(dose / base - 1) > 0.01:
            floor = floors.get(r["component"]) if r["source"] == "uniprot" else None
            changed.append((label(r["component"]), dose, base, floor))
    changed.sort(key=lambda t: t[1] / t[2])
    items = changed + added
    x = np.arange(len(items))
    for i, (_, dose, base, floor) in enumerate(items):
        if base is not None:
            ax.plot([i, i], [base, dose], color="0.6", lw=0.8, zorder=1)
            ax.scatter(i, base, s=22, facecolor="white", edgecolor="black", lw=0.8, zorder=3)
        if floor is not None:
            ax.plot([i - 0.32, i + 0.32], [floor, floor], color=oi.VERMILLION, lw=1.6, zorder=2)
        ax.scatter(i, dose, s=22, color=oi.BLUE, zorder=4)
    ax.axvline(len(changed) - 0.5, color="0.5", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in items], rotation=50, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Dose (mg per liter)")
    ax.set_xlim(-0.7, len(items) - 0.3)
    ax.set_ylim(8e-5, 4)
    ax.text(len(changed) / 2 - 0.2, 0.99, "changed by more than 1 percent", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=5.5, color="0.3")
    ax.text((len(changed) + len(items)) / 2 - 0.5, 0.99, "no stated base dose", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=5.5, color="0.3")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="white", mec="black", ms=4, label="Base"),
        Line2D([], [], marker="o", ls="", color=oi.BLUE, ms=4, label="Best recipe"),
        Line2D([], [], color=oi.VERMILLION, lw=1.6, label="Protein floor"),
    ], loc="lower right", frameon=False, handletextpad=0.3)
    return changed, added


def panel_cost(ax, data, legend=True):
    base_cost = data["anch"][0]["runs"][0]["realized_cost_base"]
    ax.axhline(base_cost, color="0.4", lw=0.7, ls="--")
    ax.text(0.0115, base_cost, "Base", va="bottom", fontsize=5.5, color="0.3")
    weights = [r["cost_weight"] for r in data["anch"][0]["runs"]]
    for arm in ("anch", "prot"):
        feas_by_w = [[] for _ in weights]
        for d in data[arm]:
            cost = [r["realized_cost"] for r in d["runs"]]
            ok = [r["round_trip"]["feasible"] for r in d["runs"]]
            ax.plot(weights, cost, color=COLOR[arm], lw=0.4, alpha=0.55, zorder=2)
            for w, c, f in zip(weights, cost, ok):
                ax.scatter(w, c, s=6, facecolor=COLOR[arm] if f else "white", edgecolor=COLOR[arm], lw=0.5, zorder=3)
            for j, (c, f) in enumerate(zip(cost, ok)):
                if f:
                    feas_by_w[j].append(c)
        med = [(w, np.median(v)) for w, v in zip(weights, feas_by_w) if v]
        ax.plot(*zip(*med), color=COLOR[arm], lw=1.8, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Cost weight")
    ax.set_ylabel("Realized cost (USD per liter)")
    ax.text(0.97, 0.97, "Raw: 0 of 70 runs feasible", transform=ax.transAxes, ha="right", va="top", fontsize=6,
            color=oi.ANNOTATION)
    if legend:
        ax.legend(handles=[
            Line2D([], [], color=COLOR["anch"], lw=1.8, label="Anchored"),
            Line2D([], [], color=COLOR["prot"], lw=1.8, label="Anchored, protein floor"),
            Line2D([], [], marker="o", ls="", mfc="white", mec="0.2", ms=3, label="Infeasible run"),
        ], loc="upper left", bbox_to_anchor=(0.0, 0.97), frameon=False, handlelength=1.4)


def panel_counts(ax, data):
    width = 0.38
    seeds = np.arange(10)
    for off, arm in ((-width / 2, "anch"), (width / 2, "prot")):
        n = [sum(r["round_trip"]["feasible"] for r in d["runs"]) for d in data[arm]]
        ax.bar(seeds + off, n, width, color=COLOR[arm], label=ARMS[arm])
    ax.set_xticks(seeds)
    ax.set_ylim(0, 8.6)
    ax.set_yticks(range(0, 8, 1))
    ax.set_xlabel("Seed")
    ax.set_ylabel("Feasible runs of 7")
    ax.text(0.02, 0.97, "Raw: 0 of 7 for every seed", transform=ax.transAxes, va="top", fontsize=6,
            color=oi.ANNOTATION)


def panel_pareto(ax, data, legend=True):
    r95 = [json.load(open(ROOT / "artifacts" / THRESH.format(k)))["realism_95"] for k in SEEDS]
    for arm in ("raw", "anch", "prot"):
        for d in data[arm]:
            for r in d["runs"]:
                f = r["round_trip"]["feasible"]
                ax.scatter(r["round_trip"]["R"], r["realized_cost"], s=7,
                           facecolor=COLOR[arm] if f else "white", edgecolor=COLOR[arm], lw=0.5, zorder=3)
    pts = sorted((r["round_trip"]["R"], r["realized_cost"]) for arm in ("anch", "prot") for d in data[arm]
                 for r in d["runs"] if r["round_trip"]["feasible"])
    front, best = [], np.inf
    for R, c in pts:
        if c < best:
            front.append((R, c))
            best = c
    ax.step(*zip(*front), where="post", color="0.15", lw=0.8, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 5])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_xlim(0.3, 11)
    for v in r95:
        ax.plot(v, 0, marker="^", ms=3.5, color=oi.VERMILLION, clip_on=False, zorder=5,
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("R, distance from the base (unitless)")
    ax.set_ylabel("Realized cost (USD per liter)")
    if legend:
        ax.legend(handles=[
            Line2D([], [], marker="o", ls="", color=COLOR["raw"], ms=3, label="Raw"),
            Line2D([], [], marker="o", ls="", color=COLOR["anch"], ms=3, label="Anchored"),
            Line2D([], [], marker="o", ls="", color=COLOR["prot"], ms=3, label="Protein floor"),
            Line2D([], [], marker="o", ls="", mfc="white", mec="0.2", ms=3, label="Infeasible"),
            Line2D([], [], color="0.15", lw=0.8, label="Feasible front"),
            Line2D([], [], marker="^", ls="", color=oi.VERMILLION, ms=3.5, label="R95 of each seed"),
        ], loc="center right", bbox_to_anchor=(1.04, 0.36), frameon=False, handletextpad=0.3, labelspacing=0.25)
    return r95


def panel_step(ax, data):
    rng = np.random.default_rng(0)
    for i, arm in enumerate(("raw", "anch", "prot")):
        steps, feas = [], []
        for d in data[arm]:
            for r in d["runs"]:
                steps.append(r["step_norm"])
                feas.append(r["round_trip"]["feasible"])
        x = i + rng.uniform(-0.28, 0.28, len(steps))
        steps, feas = np.array(steps), np.array(feas)
        ax.scatter(x[feas], steps[feas], s=5, color=COLOR[arm], lw=0, zorder=3)
        ax.scatter(x[~feas], steps[~feas], s=6, facecolor="white", edgecolor=COLOR[arm], lw=0.5, zorder=3)
        ax.plot([i - 0.38, i + 0.38], [np.median(steps)] * 2, color="black", lw=1.2, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Raw", "Anchored", "Protein\nfloor"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Latent step norm (unitless)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {arm: load_arm(arm) for arm in ARMS}
    fig = plt.figure(figsize=(7.0, 5.2))
    gs = GridSpec(2, 6, figure=fig, left=0.075, right=0.99, top=0.95, bottom=0.115, wspace=1.5, hspace=0.62)
    axes = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6]), fig.add_subplot(gs[1, 0:2]),
            fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])]
    changed, added = panel_dose(axes[0])
    panel_cost(axes[1], data)
    panel_counts(axes[2], data)
    r95 = panel_pareto(axes[3], data)
    panel_step(axes[4], data)
    for ax, t in zip(axes, "ABCDE"):
        letter(ax, t)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(OUT / "optimization_overview.pdf")
    fig.savefig(OUT / "optimization_overview.png", dpi=300)
    print("changed", [(c[0], c[1], c[2]) for c in changed])
    print("added", [(a[0], a[1]) for a in added])
    print("r95", r95)


if __name__ == "__main__":
    main()
