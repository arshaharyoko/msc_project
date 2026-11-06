import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures_optimization_main as fm

OUT = fm.OUT
ARM_TEXT = {"free": "no protein floor", "prot": "protein floor"}


def cells(st, protected):
    if not st["feasible"]:
        return [f"{st['feasible']}", "--", "--", "n/a", "--"]
    kept = f"{st['retained']} of {st['feasible']}" if protected else "n/a"
    return [
        f"{st['feasible']}",
        f"{np.median(st['cost']):.1f}",
        f"{st['cost'].min():.1f}",
        kept,
        f"{np.median(st['removed']):g}/{np.median(st['added']):g}",
    ]


def main():
    table = fm.build_table()
    head = ["Setting", "Feasible of 70", "Median cost", "Minimum cost", "At or above floors", "Removed/added"]
    lines = [
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lccccc}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in head) + " \\\\",
        "\\midrule",
        "\\multicolumn{6}{@{}l}{\\textit{Anchor strength}} \\\\",
    ]
    for a in (0.75, 0.9, 1.0):
        for arm in ("free", "prot"):
            row = cells(table[("alpha", a, arm)], arm == "prot")
            lines.append(" & ".join([f"{a:g}, {ARM_TEXT[arm]}"] + row) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{6}{@{}l}{\\textit{Floor scale, protein floor}} \\\\")
    for s in fm.SCALES:
        row = cells(table[("floor", s)], True)
        name = "base dose" if s == "base" else s
        lines.append(" & ".join([name] + row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", ""]
    (OUT / "optimization_table.tex").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
