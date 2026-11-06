import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures_optimization_existing as ex
from scripts import okabe_ito as oi

ROOT, OUT, SEEDS, COLOR = ex.ROOT, ex.OUT, ex.SEEDS, ex.COLOR
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
SCALES = ["1", "2", "5", "base"]
FREE, PROT = COLOR["anch"], COLOR["prot"]
ARM_COLOR = {"free": FREE, "prot": PROT}
_base_names = {}


def base_names(source):
    if source not in _base_names:
        db = json.load(open(ROOT / "data" / "database_graded.json"))
        _base_names[source] = {c["component"] for c in next(x for x in db if x["source"] == source)["components"]}
    return _base_names[source]


def files(kind, value, arm=None):
    if kind == "floor":
        return [ROOT / "artifacts" / "optimizer_sweeps" / f"latent_floor{value}__s{k}.json" for k in SEEDS]
    if value == 1.0:
        name = {"free": "anch", "prot": "prot"}[arm]
        return [ROOT / "artifacts" / "optimizer" / f"latent_{name}__s{k}.json" for k in SEEDS]
    return [ROOT / "artifacts" / "optimizer_sweeps" / f"latent_alpha{value:g}_{arm}__s{k}.json" for k in SEEDS]


def runs_of(kind, value, arm=None):
    out = []
    for path in files(kind, value, arm):
        d = json.load(open(path))
        names = base_names(d["base_source"])
        for r in d["runs"]:
            added = sum(1 for i in r["ingredients"] if i["component"] not in names)
            out.append({
                "feasible": bool(r["round_trip"]["feasible"]),
                "cost": r["realized_cost"],
                "retained": bool(r["protein_retained"]),
                "removed": len(r["components_removed"]),
                "added": added,
                "base_cost": r["realized_cost_base"],
            })
    return out


def stats(runs):
    feas = [r for r in runs if r["feasible"]]
    cost = np.array([r["cost"] for r in feas])
    moved = np.array([r["removed"] + r["added"] for r in feas])
    return {
        "n": len(runs),
        "feasible": len(feas),
        "cost": cost,
        "retained": sum(r["retained"] for r in feas),
        "removed": np.array([r["removed"] for r in feas]),
        "added": np.array([r["added"] for r in feas]),
        "moved": moved,
        "base_cost": runs[0]["base_cost"],
    }


def panel_graded(fig, spec, table):
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, height_ratios=[1, 1.35], hspace=0.12)
    top = fig.add_subplot(inner[0])
    bot = fig.add_subplot(inner[1], sharex=top)
    for arm in ("free", "prot"):
        st = [table[("alpha", a, arm)] for a in ALPHAS]
        frac = [s["feasible"] / s["n"] for s in st]
        top.plot(ALPHAS, frac, marker="o", ms=3, lw=1.2, color=ARM_COLOR[arm])
        xs = [a for a, s in zip(ALPHAS, st) if s["feasible"]]
        med = [np.median(s["cost"]) for s in st if s["feasible"]]
        lo = [np.percentile(s["cost"], 5) for s in st if s["feasible"]]
        hi = [np.percentile(s["cost"], 95) for s in st if s["feasible"]]
        bot.fill_between(xs, lo, hi, color=ARM_COLOR[arm], alpha=0.22, lw=0)
        bot.plot(xs, med, marker="o", ms=3, lw=1.2, color=ARM_COLOR[arm])
    for ax in (top, bot):
        ax.axvspan(0.5, 0.75, color="0.88", lw=0, zorder=0)
    top.text(0.625, 0.5, "cliff", ha="center", va="center", fontsize=6, color="0.25", rotation=90)
    top.set_ylabel("Fraction\nfeasible")
    top.set_ylim(-0.04, 1.04)
    top.set_yticks([0, 0.5, 1])
    plt.setp(top.get_xticklabels(), visible=False)
    bot.set_ylabel("USD per L")
    bot.set_xlabel("Anchor strength")
    bot.set_xticks(ALPHAS)
    bot.set_xticklabels(["0.1", "0.25", "0.5", "0.75", "0.9", "1"])
    bot.set_xlim(0.05, 1.05)
    return top, bot


def panel_floor(ax, table):
    base_cost = table[("floor", "1")]["base_cost"]
    xs = np.arange(len(SCALES))
    for i, s in enumerate(SCALES):
        st = table[("floor", s)]
        lo, hi, med = st["cost"].min(), st["cost"].max(), np.median(st["cost"])
        ax.plot([i, i], [lo, hi], color=PROT, lw=1.2)
        for v in (lo, hi):
            ax.plot([i - 0.08, i + 0.08], [v, v], color=PROT, lw=1.2)
        ax.scatter(i, med, s=18, color=PROT, zorder=3)
        ax.text(i, 0.03, f"{st['feasible']} of {st['n']}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6)
    ax.axhline(base_cost, color="0.4", lw=0.7, ls="--")
    ax.text(len(SCALES) - 0.55, base_cost * 1.03, f"Base {base_cost:.2f}", ha="right", va="bottom", fontsize=5.5,
            color="0.3")
    ax.set_xticks(xs)
    ax.set_xticklabels(["1", "2", "5", "Base dose"])
    ax.set_xlim(-0.5, len(SCALES) - 0.5)
    ax.set_yscale("log")
    ax.set_ylim(400, 2600)
    ax.set_yticks([600, 800, 1000, 1500, 2000])
    ax.set_yticklabels(["600", "800", "1000", "1500", "2000"])
    ax.minorticks_off()
    ax.set_xlabel("Floor scale")
    ax.set_ylabel("Cost (USD per liter)")


def panel_moved(ax, table):
    for arm, off in (("free", -0.012), ("prot", 0.012)):
        st = [table[("alpha", a, arm)] for a in ALPHAS]
        xs = [a + off for a, s in zip(ALPHAS, st) if s["feasible"]]
        med = np.array([np.median(s["moved"]) for s in st if s["feasible"]])
        lo = np.array([s["moved"].min() for s in st if s["feasible"]])
        hi = np.array([s["moved"].max() for s in st if s["feasible"]])
        ax.errorbar(xs, med, yerr=[med - lo, hi - med], color=ARM_COLOR[arm], marker="o", ms=3, lw=1.0,
                    elinewidth=0.8, capsize=1.5)
    ax.axvspan(0.5, 0.75, color="0.88", lw=0, zorder=0)
    ax.set_xticks(ALPHAS)
    ax.set_xticklabels(["0.1", "0.25", "0.5", "0.75", "0.9", "1"])
    ax.set_xlim(0.05, 1.05)
    ax.set_xlabel("Anchor strength")
    ax.set_ylabel("Components removed\nor added")
    ax.set_ylim(bottom=-1)


def build_table():
    table = {}
    for a in ALPHAS:
        for arm in ("free", "prot"):
            table[("alpha", a, arm)] = stats(runs_of("alpha", a, arm))
    for s in SCALES:
        table[("floor", s)] = stats(runs_of("floor", s))
    return table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {arm: ex.load_arm(arm) for arm in ex.ARMS}
    table = build_table()
    fig = plt.figure(figsize=(7.0, 5.2))
    gs = fig.add_gridspec(1, 12, left=0.075, right=0.99, top=0.955, bottom=0.58, wspace=3.4)
    gs2 = fig.add_gridspec(1, 3, left=0.095, right=0.99, top=0.40, bottom=0.115, wspace=0.5)
    axA = fig.add_subplot(gs[0, 0:5])
    axB = fig.add_subplot(gs[0, 5:9])
    axC = fig.add_subplot(gs[0, 9:12])
    axE = fig.add_subplot(gs2[0, 1])
    axF = fig.add_subplot(gs2[0, 2])
    ex.panel_dose(axA)
    axA.set_ylim(8e-5, 30)
    axA.texts[0].set_text("changed by more\nthan 1 percent")
    axA.texts[1].set_text("no stated\nbase dose")
    ex.panel_cost(axB, data, legend=False)
    ex.panel_counts(axC, data)
    axB.set_ylim(0, 2400)
    axC.set_ylim(0, 10)
    axC.texts[0].set_text("Raw: 0 of 7\nin every seed")
    top, bot = panel_graded(fig, gs2[0, 0], table)
    panel_floor(axE, table)
    panel_moved(axF, table)
    for ax, t in zip((axA, axB, axC, top, axE, axF), "ABCDEF"):
        ex.letter(ax, t)
    for ax in (axA, axB, axC, top, bot, axE, axF):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.legend(handles=[
        Line2D([], [], color=FREE, lw=1.6, marker="o", ms=3, label="No protein floor"),
        Line2D([], [], color=PROT, lw=1.6, marker="o", ms=3, label="Protein floor"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="0.2", ms=3, label="Infeasible run"),
        Line2D([], [], color="0.4", lw=0.8, ls="--", label="Base cost"),
    ], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 0.0), handlelength=1.6)
    fig.savefig(OUT / "optimization_main.pdf")
    fig.savefig(OUT / "optimization_main.png", dpi=300)
    return table


def supplement():
    data = {arm: ex.load_arm(arm) for arm in ex.ARMS}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"left": 0.085, "right": 0.99, "top": 0.93,
                                                                          "bottom": 0.27, "wspace": 0.32})
    ex.panel_pareto(ax1, data, legend=False)
    ex.panel_step(ax2, data)
    for ax, t in zip((ax1, ax2), "AB"):
        ex.letter(ax, t)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=COLOR["raw"], ms=3, label="Raw"),
        Line2D([], [], marker="o", ls="", color=COLOR["anch"], ms=3, label="Anchored"),
        Line2D([], [], marker="o", ls="", color=COLOR["prot"], ms=3, label="Protein floor"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="0.2", ms=3, label="Infeasible"),
        Line2D([], [], color="0.15", lw=0.8, label="Feasible front (A)"),
        Line2D([], [], marker="^", ls="", color=oi.VERMILLION, ms=3.5, label="R95 of each seed (A)"),
        Line2D([], [], color="black", lw=1.2, label="Median of the arm (B)"),
    ], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 0.0), handletextpad=0.4)
    fig.savefig(OUT / "supplement_pareto_displacement.pdf")
    fig.savefig(OUT / "supplement_pareto_displacement.png", dpi=300)


if __name__ == "__main__":
    supplement()
    t = main()
    for k, v in t.items():
        print(k, v["feasible"], v["n"], round(float(np.median(v["cost"])), 2) if v["feasible"] else "-")
