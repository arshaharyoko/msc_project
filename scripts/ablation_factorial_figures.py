#!/usr/bin/env python3
"""Figures for the factorial ablation: arm summary, effect panels, gate AUC deltas, parameters against effects."""

from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_factorial as af
from scripts import okabe_ito as oi
from src import benchmark

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figures/ablation"
RES = json.loads((ROOT / "artifacts/ablation_analysis/factorial.json").read_text())
BLUE, ORANGE, GREEN, VERM, PURPLE, BLACK = oi.BLUE, oi.ORANGE, oi.GREEN, oi.VERMILLION, oi.PURPLE, oi.BLACK
LEVEL = list(oi.LEVELS)
TITLE = {"active": "live latent directions", "recon": "reconstruction error", "gate_auc": "gate AUC"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def arm_name(bits):
    s = "".join(f[0].upper() if f != "cost" else "C" for f, b in zip(af.FACTORS5, bits) if b)
    return s or "base"


def arm_table(df, metric):
    rows = {}
    for neg in ("graded", "legacy"):
        sub = df[df.negatives == neg]
        for _, r in sub.iterrows():
            bits = tuple(int(not r[af.COL[f]]) for f in af.FACTORS5)
            rows.setdefault(bits, {}).setdefault(neg, []).append(r.get(metric, np.nan))
    return rows


def fig_summary(df):
    mets = ["active", "recon", "gate_auc"]
    tabs = {m: arm_table(df, m) for m in mets}
    bits_all = sorted(tabs["active"], key=lambda b: (sum(b), b))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 10), sharey=True)
    for ax, m in zip(axes, mets):
        for y, b in enumerate(bits_all[::-1]):
            col = LEVEL[sum(b)]
            vals = np.array([v for v in tabs[m][b]["graded"] if not np.isnan(v)])
            if len(vals):
                ax.boxplot([vals], positions=[y], vert=False, widths=0.6, whis=(0, 100), showfliers=False,
                           boxprops=dict(color=col), whiskerprops=dict(color=col), capprops=dict(color=col),
                           medianprops=dict(color=col, lw=1.5))
                ax.scatter(vals, np.full(len(vals), y) + np.linspace(-0.18, 0.18, len(vals)), s=9, color=col, alpha=0.8, zorder=3)
            leg = [v for v in tabs[m][b]["legacy"] if not np.isnan(v)]
            leg = sorted(set(round(v, 9) for v in leg))
            if leg:
                ax.scatter(leg, np.full(len(leg), y), marker="D", s=18, facecolor="white", edgecolor=BLACK, zorder=4)
        ax.set_xlabel(TITLE[m])
        ax.set_yticks(range(len(bits_all)))
        ax.set_yticklabels([arm_name(b) for b in bits_all[::-1]])
        ax.grid(axis="x", alpha=0.25)
    axes[2].axvline(0.5, color=oi.GRAY, lw=0.8, ls=":")
    h = [plt.Line2D([], [], marker="o", ls="", color=LEVEL[k], label=f"{k} removed") for k in range(6)]
    h.append(plt.Line2D([], [], marker="D", ls="", mfc="white", mec=BLACK, label="legacy negatives, one marker per seed"))
    fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    benchmark.panel_labels(axes)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "ablation_summary")


def interval_plot(ax, labels, graded, legacy):
    y = np.arange(len(labels))[::-1]
    for yy, k in zip(y, labels):
        e = graded.get(k)
        if e and e["est"] is not None:
            c = BLUE if e["excludes_zero"] else oi.GRAY
            ax.plot([e["lo"], e["hi"]], [yy + 0.12] * 2, color=c, lw=2)
            ax.scatter([e["est"]], [yy + 0.12], color=c, s=26, zorder=3)
        l = legacy.get(k)
        if l and l["est"] is not None:
            ax.scatter([l["est"]], [yy], marker="D", s=26, facecolor="white", edgecolor=BLACK, zorder=4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="x", alpha=0.25)


def unit_effects(metric):
    """Per-unit paired effects behind the bootstrap estimates, same pairing as af.run_new."""
    df = af.load(ROOT / "artifacts/ablation_runs.jsonl")
    out = {}
    for tag, sub, factors in (("A", df[df.negatives == "graded"], af.FACTORS5),
                              ("C", df, af.FACTORS5 + ("negatives",))):
        _, t = af.tensor(sub, factors, ["seed", "fold"], [metric])
        Y, fs = af.restrict(t[metric], factors, metric)
        main, inter = af.per_unit(Y)
        out[tag] = ({fs[i]: v for i, v in main.items()},
                    {f"{fs[i]}:{fs[j]}": v for (i, j), v in inter.items()})
    return out


def box_rows(ax, rows, values, legacy):
    y = np.arange(len(rows))[::-1]
    for yy, k in zip(y, rows):
        v = values[k]
        v = v[~np.isnan(v)]
        ax.boxplot([v], positions=[yy + 0.12], vert=False, widths=0.3, whis=(0, 100), showfliers=False,
                   showmeans=True, patch_artist=True,
                   boxprops=dict(facecolor=oi.SKY, edgecolor=BLUE, alpha=0.6),
                   whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
                   medianprops=dict(color=BLUE, lw=1.2),
                   meanprops=dict(marker="o", markerfacecolor=BLACK, markeredgecolor=BLACK, markersize=4))
        l = legacy.get(k)
        if l and l["est"] is not None:
            ax.scatter([l["est"]], [yy - 0.2], marker="D", s=26, facecolor="white", edgecolor=BLACK, zorder=4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", alpha=0.25)


def fig_effects(metric, name):
    A, B, C = (RES[k]["metrics"][metric] for k in ("A_graded", "B_legacy", "C_negatives"))
    ue = unit_effects(metric)
    am, ai = ue["A"]
    cm, _ = ue["C"]
    for k, v in am.items():
        assert np.isclose(np.nanmean(v), A["main"][k]["est"])
    for k, v in ai.items():
        assert np.isclose(np.nanmean(v), A["interaction"][k]["est"])
    assert np.isclose(np.nanmean(cm["negatives"]), C["main"]["negatives"]["est"])
    values = dict(am)
    values["negatives"] = cm["negatives"]
    rows = list(A["main"]) + ["negatives"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    box_rows(a1, rows, values, B["main"])
    a1.set_xlabel("paired difference, removed minus retained")
    pairs = list(A["interaction"])
    box_rows(a2, [p.replace(":", " x ") for p in pairs], {p.replace(":", " x "): ai[p] for p in pairs},
             {p.replace(":", " x "): v for p, v in B["interaction"].items()})
    a2.set_xlabel("half difference of differences")
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=oi.SKY, edgecolor=BLUE, alpha=0.6, label="graded units, IQR"),
         plt.Line2D([], [], color=BLUE, label="range"),
         plt.Line2D([], [], marker="o", ls="", color=BLACK, ms=4, label="mean"),
         plt.Line2D([], [], marker="D", ls="", mfc="white", mec=BLACK, label="legacy, 2 seeds")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
    benchmark.panel_labels([a1, a2])
    fig.tight_layout()
    save(fig, name)


def fig_gate_deltas():
    g, l = RES["A_graded"]["gate_auc_single"], RES["B_legacy"]["gate_auc_single"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, t in zip(axes, ("single", "single_abs_dev"), ("delta gate AUC", "delta of distance from chance")):
        labels = list(g[key])
        interval_plot(ax, [f"no {k}" for k in labels], {f"no {k}": v for k, v in g[key].items()}, {f"no {k}": v for k, v in l[key].items()})
        ax.set_xlabel(f"{t}, single removal minus baseline")
    h = [plt.Line2D([], [], color=BLUE, lw=2, label="graded, interval excludes 0"),
         plt.Line2D([], [], color=oi.GRAY, lw=2, label="graded, interval includes 0"),
         plt.Line2D([], [], marker="D", ls="", mfc="white", mec=BLACK, label="legacy, 2 seeds")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    benchmark.panel_labels(axes)
    fig.tight_layout()
    save(fig, "gate_auc_deltas")


def fig_params(df):
    graded = df[df.negatives == "graded"]
    base_p = graded[(graded.use_svd) & graded.use_mmd & graded.use_rgcn & graded.use_gate & graded.use_cost].params_live.iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cols = dict(zip(af.FACTORS5, [BLUE, ORANGE, GREEN, VERM, PURPLE]))
    for ax, m in zip(axes, ("gate_auc", "recon", "active")):
        units, t = af.tensor(graded, af.FACTORS5, ["seed", "fold"], [m])
        Y = t[m]
        Y = Y[~np.all(np.isnan(Y.reshape(Y.shape[0], -1)), axis=1)]
        base = Y[(slice(None),) + (0,) * 5]
        for i, f in enumerate(af.FACTORS5):
            sel = [0] * 5
            sel[i] = 1
            e = af.summarise(Y[(slice(None),) + tuple(sel)] - base, True)
            sub = graded[(~graded[af.COL[f]]) & np.all([graded[af.COL[o]] for o in af.FACTORS5 if o != f], axis=0)]
            x = sub.params_live.iloc[0] - base_p + (i - 2) * 200
            ax.errorbar([x], [e["est"]], yerr=[[e["est"] - e["lo"]], [e["hi"] - e["est"]]], fmt="o", color=cols[f], capsize=3, label=f"no {f}")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color=oi.GRAY, lw=0.6, ls=":")
        ax.set_xlabel("change in live parameters vs baseline\n(points dodged by up to 400 for legibility)")
        ax.set_ylabel(f"delta {TITLE[m]}")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    benchmark.panel_labels(axes)
    fig.tight_layout()
    save(fig, "params_vs_effects")


def main():
    df = af.load(ROOT / "artifacts/ablation_runs.jsonl")
    fig_summary(df)
    fig_effects("active", "ablation_effects_active")
    fig_effects("recon", "ablation_effects_recon")
    fig_effects("gate_auc", "ablation_effects_gate_auc")
    fig_gate_deltas()
    fig_params(df)


if __name__ == "__main__":
    main()
