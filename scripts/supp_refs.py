"""Supplementary Fig., Table and Note numbers for main.tex.

The supplement is compiled apart from main.tex, so \\ref cannot reach it. The numbers
come from the order of figure and table environments in the files that
paper/supplementaries/Supplementary.tex includes. `--write` regenerates
paper/supp_refs.tex, `--check` exits 1 if that file differs from a fresh computation.
"""

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"
ROOT = PAPER / "supplementaries" / "Supplementary.tex"
OUT = PAPER / "supp_refs.tex"
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def strip_comments(text):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in text.splitlines())


def expand(path, seen=()):
    if path in seen:
        raise SystemExit(f"input cycle at {path}")
    text = strip_comments(path.read_text())

    def sub(m):
        name = m.group(1)
        for base in (path.parent, PAPER / "supplementaries", PAPER):
            cand = base / (name if name.endswith(".tex") else name + ".tex")
            if cand.is_file():
                return expand(cand, seen + (path,))
        raise SystemExit(f"{path}: cannot resolve \\input{{{name}}}")

    return re.sub(r"\\input\{([^}]+)\}", sub, text)


def compute():
    text = expand(ROOT)
    text = text.split(r"\begin{thebibliography}")[0]
    items = []
    counts = {"figure": 0, "table": 0}
    token = re.compile(
        r"\\begin\{(?:sideways)?(figure|table)\*?\}|\\end\{(?:sideways)?(figure|table)\*?\}"
        r"|\\section\*\{Supplementary Note (\d+)[^}]*\}|\\label\{([^}]+)\}")
    current = None
    note = None
    for m in token.finditer(text):
        if m.group(1):
            kind = m.group(1)
            counts[kind] += 1
            current = (kind, counts[kind])
        elif m.group(2):
            current = None
        elif m.group(3):
            note = int(m.group(3))
        else:
            label = m.group(4)
            if current is not None and label.split(":")[0] in ("fig", "tab"):
                items.append((label, current[0], current[1]))
                current = None
            elif note is not None and label.startswith("subsec:") and note not in {n for _, k, n in items if k == "note"}:
                items.append((label, "note", note))
    return items


def macro_name(label, kind):
    stem = re.sub(r"[^a-z]", "", label.split(":", 1)[1].lower())
    return {"figure": "sfig", "table": "stab"}[kind] + stem


def render():
    items = compute()
    lines = []
    names = {}
    for label, kind, n in items:
        if kind == "note":
            name = "snote" + WORDS[n]
            full = f"Supplementary Note~{n}"
            lines.append((name, full))
            names[name] = label
            continue
        name = macro_name(label, kind)
        if name in names:
            raise SystemExit(f"macro collision: {name} for {label} and {names[name]}")
        names[name] = label
        word = "Fig." if kind == "figure" else "Table"
        lines.append((name, f"Supplementary {word}~{n}"))
        nname = "n" + name[1:]
        names[nname] = label
        lines.append((nname, str(n)))
    return "".join(f"\\newcommand{{\\{n}}}{{{v}\\xspace}}\n" for n, v in lines)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    new = render()
    if mode == "--write":
        OUT.write_text(new)
        print(f"wrote {OUT} ({new.count(chr(10))} lines)")
        return 0
    if not OUT.is_file() or OUT.read_text() != new:
        print(f"{OUT} is stale, run scripts/supp_refs.py --write", file=sys.stderr)
        return 1
    print("supp_refs.tex is current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
