#!/usr/bin/env python3
"""Check that a style pass lost no substance from the .tex sources.

Extracts, per file, the multiset of every number, every citation key and every
label/reference key.  A snapshot records the state; a comparison reports the
difference and fails unless `artifacts/adr/paper_changelog.md` carries a row
that mentions the token and the file.

    check_deslop.py snapshot before.json
    check_deslop.py compare before.json [after.json]

`compare` snapshots the tree when `after.json` is omitted.  Exit status is 1
when any difference is unexplained.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
CHANGELOG = ROOT / "artifacts" / "adr" / "paper_changelog.md"

CITE_RE = re.compile(r"\\(?:no)?cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
REF_RE = re.compile(
    r"\\(?:label|ref|eqref|autoref|pageref|Cref|cref|nameref)\*?\{([^}]*)\}"
)
NUMBER_RE = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?")

# Arguments of these commands are keys, not prose; they are counted separately.
KEYED_RE = re.compile(
    r"\\(?:no)?cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{[^}]*\}"
    r"|\\(?:label|ref|eqref|autoref|pageref|Cref|cref|nameref)\*?\{[^}]*\}"
    r"|\\input\{[^}]*\}|\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}"
    r"|\\begin\{[^}]*\}|\\end\{[^}]*\}|\\bibliography\{[^}]*\}"
    r"|\\bibliographystyle\{[^}]*\}"
)


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def normalise_number(token: str) -> str:
    sign = ""
    if token[0] in "+-":
        sign, token = token[0], token[1:]
    token = token.replace(",", "")
    return sign + token


def collect(text: str) -> dict[str, Counter]:
    text = strip_comments(text)
    cites: Counter[str] = Counter()
    for match in CITE_RE.finditer(text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                cites[key] += 1
    refs: Counter[str] = Counter()
    for match in REF_RE.finditer(text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                refs[key] += 1
    numbers: Counter[str] = Counter(
        normalise_number(m.group()) for m in NUMBER_RE.finditer(KEYED_RE.sub(" ", text))
    )
    return {"numbers": numbers, "cites": cites, "refs": refs}


def snapshot(files: list[Path]) -> dict[str, dict[str, dict[str, int]]]:
    return {
        str(path.relative_to(ROOT)): {
            kind: dict(counter) for kind, counter in collect(path.read_text(encoding="utf-8")).items()
        }
        for path in files
    }


def paper_files() -> list[Path]:
    return sorted(PAPER.rglob("*.tex"))


def load_changelog(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def explained(changelog: str, token: str, rel: str) -> bool:
    name = Path(rel).name
    needle = token.lstrip("+-")
    for line in changelog.splitlines():
        if needle in line and (name in line or rel in line):
            return True
    return False


def compare(before: dict, after: dict, changelog: str) -> list[str]:
    unexplained: list[str] = []
    for rel in sorted(set(before) | set(after)):
        old = before.get(rel, {})
        new = after.get(rel, {})
        for kind in ("numbers", "cites", "refs"):
            old_c = Counter(old.get(kind, {}))
            new_c = Counter(new.get(kind, {}))
            for token in sorted(set(old_c) | set(new_c)):
                delta = new_c[token] - old_c[token]
                if delta == 0:
                    continue
                direction = "added" if delta > 0 else "lost"
                line = f"{rel}: {kind} {token!r} {direction} x{abs(delta)}"
                if explained(changelog, token, rel):
                    print(f"ok    {line} (changelog row)")
                else:
                    print(f"FAIL  {line}")
                    unexplained.append(line)
    return unexplained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("snapshot", "compare"))
    parser.add_argument("before", type=Path, help="snapshot file to write, or to compare against")
    parser.add_argument("after", type=Path, nargs="?", help="snapshot to compare with the tree")
    parser.add_argument("--changelog", type=Path, default=CHANGELOG)
    parser.add_argument("--paper", type=Path, default=PAPER)
    args = parser.parse_args()

    files = sorted(args.paper.rglob("*.tex"))
    if args.mode == "snapshot":
        data = snapshot(files)
        args.before.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        total = sum(sum(sum(c.values()) for c in entry.values()) for entry in data.values())
        print(f"{len(data)} files, {total} tokens -> {args.before}")
        return 0

    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = (
        json.loads(args.after.read_text(encoding="utf-8"))
        if args.after is not None
        else snapshot(files)
    )
    unexplained = compare(before, after, load_changelog(args.changelog))
    if unexplained:
        print(f"\n{len(unexplained)} unexplained difference(s)")
        return 1
    print("\nparity: every difference is covered by a changelog row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
