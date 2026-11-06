import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hipo
from scripts import okabe_ito as oi
from src import benchmark, config

ART = Path("artifacts")
OUT = Path("paper/his_lsc")
TABLE = Path("paper/supplementaries/sweep_criteria_table.tex")
STAGES = ("screen", "refine")


def load(stage):
    return json.load(open(ART / f"hipo_{stage}.json"))["results"]


def cost_delta_figure(screen, chosen):
    deltas = sorted({r["cost_delta"] for r in screen})
    panels = (("cost_r2", "held-out cost R$^2$ (head)", benchmark.OKABE_ITO[5]),
              ("recon", "held-out reconstruction (MSE)", benchmark.OKABE_ITO[4]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (key, label, colour) in zip(axes, panels):
        vals = [[r[key] for r in screen if r["cost_delta"] == d and r[key] is not None]
                for d in deltas]
        med = [np.median(v) for v in vals]
        ax.fill_between(deltas, [np.percentile(v, 25) for v in vals],
                        [np.percentile(v, 75) for v in vals],
                        color=colour, alpha=0.18, linewidth=0)
        ax.plot(deltas, med, color=colour, linewidth=2, marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=2)
        ax.axvline(chosen, color=oi.ANNOTATION, linewidth=1, linestyle="--")
        ax.annotate(f"chosen {chosen:g}", xy=(chosen, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(-4, 6), textcoords="offset points", ha="right", fontsize=8,
                    color=oi.ANNOTATION)
        ax.set_xscale("log")
        ax.set_xticks(deltas)
        ax.set_xticklabels([f"{d:g}" for d in deltas])
        benchmark.style(ax, label, label, xlabel="COST_DELTA")
    benchmark.panel_labels(axes)
    fig.tight_layout()
    fig.savefig(OUT / "hipo_cost_delta_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def sweep_figures(rows):
    config.FIGURE_DIR = OUT
    hipo.plot_hyperparameter_space(rows, filename="his_hyperparameter_space.png")
    benchmark.plot_tradeoff(rows, d_z=hipo.LATENT_DIM,
                            size_range=(hipo.TSVD_K.min(), hipo.TSVD_K.max()),
                            filename="his_tradeoff.png")
    fig = benchmark.plot_weight_marginals(
        rows,
        params=(("pathway_alpha", "PATHWAY_ALPHA"), ("div_beta", "div beta"),
                ("gate_gamma", "GATE_GAMMA"), ("tsvd_k", "TSVD_K"),
                ("cost_delta", "COST_DELTA")),
        d_z=hipo.LATENT_DIM, filename="his_marginals.png")
    note = next(t for t in fig.axes[0].texts if t.get_text().startswith("collapse threshold"))
    note.remove()
    benchmark.threshold_label(fig.axes[0], note.get_text(), y=note.xy[1], side="right")
    fig.savefig(OUT / "his_marginals.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def exponent(x):
    return f"$10^{{{int(round(np.log10(x)))}}}$"


def divergence(x):
    if x == 0:
        return "0"
    if abs(x) >= 1e-3:
        return f"{x:.4f}"
    mantissa, power = f"{x:.1e}".split("e")
    return f"${mantissa}\\times10^{{{int(power)}}}$"


def table_rows(rows):
    out = []
    for title, key in (("Selected on smallest divergence", lambda r: (r["div_loss"], r["recon"])),
                       ("Selected on surviving width", lambda r: (-r["active"], r["recon"]))):
        out.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for label in ("MMD", "MMD + tSVD", "KL", "KL + tSVD"):
            r = min((x for x in rows if x["label"] == label), key=key)
            name = label.replace("+", "$+$")
            out.append(
                f"{name} & {exponent(r['pathway_alpha'])} & {exponent(r['div_beta'])} & "
                f"{exponent(r['gate_gamma'])} & {r['tsvd_k']} & {exponent(r['recon_w2'])} & "
                f"{exponent(r['cost_delta'])} & {r['active']}/32 & {r['recon']:.3f} & {divergence(r['div_loss'])} \\\\")
        if title.endswith("divergence"):
            out.append("\\midrule")
    return out


def caption(rows):
    tiny = {lab: sum(abs(r["div_loss"]) < 1e-6 for r in rows if r["label"] == lab)
            for lab in ("MMD", "MMD + tSVD")}
    each = sum(r["label"] == "MMD" for r in rows)
    return (
        "Best configuration per variant under two selection criteria over the screen and "
        "refine stages, with ties in width broken by the smaller reconstruction error. "
        "Notation as in Table~\\ref{tab:grid} and Table~\\ref{tab:sweep-best}. "
        "Divergence values are not comparable between the MMD and KL rows. The MMD "
        f"divergence is below $10^{{-6}}$ in absolute value in ${tiny['MMD']}$ of ${each}$ MMD runs "
        f"and ${tiny['MMD + tSVD']}$ of ${each}$ MMD$+$tSVD runs, so the smallest value in those rows "
        "sits at numerical zero.")


def write_table(rows):
    head = [
        "\\begin{table}[h]",
        f"\\caption{{{caption(rows)}}}",
        "\\label{tab:sweep-criteria}",
        "\\small",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lrrrrrrrrr}",
        "\\toprule",
        "\\textbf{Variant} & $\\boldsymbol{\\alpha}$ & $\\boldsymbol{\\beta}$ & $\\boldsymbol{\\gamma}$ & "
        "$\\boldsymbol{k}$ & $\\boldsymbol{w_2}$ & $\\boldsymbol{\\delta}$ & \\textbf{live} & "
        "\\textbf{recon} & \\textbf{div} \\\\",
        "\\midrule",
    ]
    tail = ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    TABLE.write_text("\n".join(head + table_rows(rows) + tail) + "\n")


def marginal_ranges(rows):
    keys = ("pathway_alpha", "div_beta", "gate_gamma", "tsvd_k")
    lines = ["# sweep marginals", "", "Mean surviving width per level over the screen and refine runs.", ""]
    for label in ("MMD", "MMD + tSVD", "KL", "KL + tSVD"):
        sub = [r for r in rows if r["label"] == label]
        for key in keys:
            means = {k: np.mean([r["active"] for r in sub if r[key] == k])
                     for k in sorted({r[key] for r in sub})}
            spread = max(means.values()) - min(means.values())
            levels = ", ".join(f"{k:g}: {v:.1f}" for k, v in means.items())
            lines.append(f"- {label}, {key}: {levels} (range {spread:.1f})")
    collapsed = {bool(v): sum(r["active"] >= 16 for r in rows if bool(r["svd"]) == bool(v)) for v in (0, 1)}
    lines += ["", f"runs with at least 16 live units, without tSVD: {collapsed[False]}, with tSVD: {collapsed[True]}"]
    (ART / "adr/logs/sweep_marginals.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    screen, refine = load("screen"), load("refine")
    selection = json.load(open(ART / "hipo_selection.json"))
    cost_delta_figure(screen, selection["shared"]["cost_delta"])
    sweep_figures(screen + refine)
    write_table(screen + refine)
    marginal_ranges(screen + refine)


if __name__ == "__main__":
    main()
