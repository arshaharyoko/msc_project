#!/usr/bin/env python3
"""Comment-hygiene checker for ADR-026, AC-E1 to AC-E4.
Usage: python scripts/check_comments.py [--baseline] [--json] [paths ...]
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

COMMENT_PCT_MAX = 12.0
DOC_PCT_MAX = 55.0
DOC_LINES_MAX = 6
HEADER_LIBRARY_MAX = 5
HEADER_ENTRY_MAX = 15
BASELINE_COMMENT_PCT = 10.2

_COMMENT_PATTERNS = (
    ("step scaffolding", re.compile(r"^#\s*Step\s+\d", re.IGNORECASE)),
    ("numbered scaffolding", re.compile(r"^#\s*\d+\.(?:\s|$)")),
    ("section banner", re.compile(r"^#\s*-{2,}")),
    ("TODO(agent)", re.compile(r"TODO\s*\(\s*agent", re.IGNORECASE)),
    ("note to self", re.compile(r"^#\s*NOTE:\s*I\b")),
)
_MARKER_PATTERN = re.compile(r"\b(?:Claude|agent|session\s+3)\b", re.IGNORECASE)
_ARGS_HEADER = re.compile(
    r"^\s*(?:Args|Arguments|Parameters|Params|Returns|Yields)\s*:\s*$",
    re.IGNORECASE,
)
_PARAM_LINE = re.compile(r"^\s*\w+\s*(?:\([^)]*\))?\s*:")
_MAIN_PATTERN = re.compile(
    r"^[ \t]*if\s+__name__\s*==\s*['\"]__main__['\"]", re.MULTILINE
)

_BLOCK_START = re.compile(
    r"^(?:if|elif|for|while|with|try|except|finally|else|def|class|async\s+def)\b"
)


@dataclass
class Hit:
    line: int
    pattern: str
    text: str


@dataclass
class Report:
    path: Path
    kind: str = "python"
    code_lines: int = 0
    comment_lines: int = 0
    definitions: int = 0
    documented: int = 0
    longest_docstring: int = 0
    header_lines: int = 0
    runnable: bool = False
    hits: list[Hit] = field(default_factory=list)
    error: str | None = None

    @property
    def comment_pct(self) -> float | None:
        if self.code_lines == 0:
            return None
        return 100.0 * self.comment_lines / self.code_lines

    @property
    def doc_pct(self) -> float:
        if self.definitions == 0:
            return 0.0
        return 100.0 * self.documented / self.definitions


def _comment_tokens(text: str, lines: list[str]) -> list[tuple[int, int, str]]:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return [
            (tok.start[0], tok.start[1], tok.string)
            for tok in tokens
            if tok.type == tokenize.COMMENT
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # A file that cannot be tokenized still deserves a line-based estimate.
        fallback = []
        for lineno, line in enumerate(lines, 1):
            pos = line.find("#")
            if pos >= 0:
                fallback.append((lineno, pos, line[pos:]))
        return fallback


def _looks_like_code(body: str) -> bool:
    body = body.strip()
    if not body:
        return False
    if body in {"else:", "try:", "finally:"} or body.startswith("@"):
        return True
    try:
        tree = ast.parse(body)
    except (SyntaxError, ValueError):
        return bool(_BLOCK_START.match(body)) and bool(
            re.search(r"[=(\[]|self\.", body)
        )
    if not tree.body or len(tree.body) != 1:
        return False
    stmt = tree.body[0]
    if isinstance(stmt, ast.Expr):
        value = stmt.value
        if isinstance(value, ast.BinOp):
            return not isinstance(value.left, (ast.Tuple, ast.List)) and not isinstance(
                value.right, (ast.Tuple, ast.List)
            )
        return isinstance(value, (ast.Call, ast.Attribute, ast.Subscript))
    if isinstance(stmt, ast.Assign):
        return not isinstance(stmt.value, ast.Name)
    if isinstance(stmt, ast.AnnAssign) and stmt.value is None:
        return False
    return True


def _scan_comment(lineno: int, token: str, hits: list[Hit]) -> None:
    for name, pattern in _COMMENT_PATTERNS:
        if pattern.search(token):
            hits.append(Hit(lineno, name, token.strip()))
    if _MARKER_PATTERN.search(token):
        hits.append(Hit(lineno, "agent/session marker", token.strip()))
    if _looks_like_code(token[1:]):
        hits.append(Hit(lineno, "commented-out code", token.strip()))


def _scan_docstring(
    lines: list[str], start: int, end: int, hits: list[Hit]
) -> None:
    for lineno in range(start, min(end, len(lines)) + 1):
        line = lines[lineno - 1]
        if _MARKER_PATTERN.search(line):
            hits.append(Hit(lineno, "agent/session marker", line.strip()))
        if not _ARGS_HEADER.match(line):
            continue
        probe = lineno + 1
        while probe <= end and probe <= len(lines) and not lines[probe - 1].strip():
            probe += 1
        if probe <= end and probe <= len(lines) and _PARAM_LINE.match(lines[probe - 1]):
            hits.append(Hit(lineno, "Args/Returns block", line.strip()))


def _dedupe(hits: list[Hit]) -> list[Hit]:
    seen = set()
    unique = []
    for hit in hits:
        key = (hit.line, hit.pattern)
        if key not in seen:
            seen.add(key)
            unique.append(hit)
    return sorted(unique, key=lambda h: (h.line, h.pattern))


def _first_docstring_span(node: ast.AST) -> tuple[int, int, str] | None:
    body = getattr(node, "body", [])
    if not body:
        return None
    first = body[0]
    if not isinstance(first, ast.Expr):
        return None
    value = first.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    end = getattr(first, "end_lineno", None) or first.lineno
    return first.lineno, end, value.value


def _docstring_length(lines: list[str], start: int, end: int) -> int:
    if end <= start or end > len(lines):
        return 1
    opener = lines[start - 1].strip().lstrip("rRbBuUfF")
    closer = lines[end - 1].strip()
    # The ADR baseline counts content lines, so a delimiter on its own does not count.
    quotes = ('"""', "'''")
    return max(
        0,
        end - start + 1
        - int(opener in quotes)
        - int(closer in quotes),
    )


def _leading_comment_block(lines: list[str]) -> int:
    last = 0
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("#"):
            last = index + 1
        elif stripped:
            break
        index += 1
    return last


def _header_lines(
    lines: list[str], module_doc: tuple[int, int, str] | None
) -> int:
    shebang = 1 if lines and lines[0].startswith("#!") else 0
    block = _leading_comment_block(lines)
    if module_doc is not None:
        doc_length = _docstring_length(lines, module_doc[0], module_doc[1])
        return max(block, shebang + doc_length)
    return max(block, shebang)


def scan_python(path: Path) -> Report:
    report = Report(path, "python")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        report.error = f"read error: {exc}"
        return report

    lines = text.splitlines()
    comments = _comment_tokens(text, lines)
    pure_comment_lines = set()
    for lineno, col, _ in comments:
        prefix = lines[lineno - 1][:col] if 0 < lineno <= len(lines) else ""
        if not prefix.strip():
            pure_comment_lines.add(lineno)
    report.comment_lines = len({lineno for lineno, _, _ in comments})
    report.code_lines = sum(
        1
        for lineno, line in enumerate(lines, 1)
        if line.strip() and lineno not in pure_comment_lines
    )
    report.runnable = bool(lines and lines[0].startswith("#!")) or bool(
        _MAIN_PATTERN.search(text)
    )

    hits: list[Hit] = []
    for lineno, _, token in comments:
        _scan_comment(lineno, token, hits)

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        report.error = f"syntax error at line {exc.lineno}: {exc.msg}"
        report.header_lines = _header_lines(lines, None)
        report.hits = _dedupe(hits)
        return report
    except ValueError as exc:
        report.error = f"parse error: {exc}"
        report.header_lines = _header_lines(lines, None)
        report.hits = _dedupe(hits)
        return report

    module_doc = _first_docstring_span(tree)
    doc_ranges = []
    if module_doc is not None:
        doc_ranges.append(module_doc)

    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    report.definitions = len(definitions)
    for node in definitions:
        span = _first_docstring_span(node)
        if span is None:
            continue
        report.documented += 1
        report.longest_docstring = max(
            report.longest_docstring,
            _docstring_length(lines, span[0], span[1]),
        )
        doc_ranges.append(span)

    for start, end, _ in doc_ranges:
        _scan_docstring(lines, start, end, hits)

    report.header_lines = _header_lines(lines, module_doc)
    report.hits = _dedupe(hits)
    return report


def scan_tex(path: Path) -> Report:
    report = Report(path, "tex")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        report.error = f"read error: {exc}"
        return report

    hits = []
    for lineno, line in enumerate(lines, 1):
        if line.lstrip().startswith("%"):
            hits.append(Hit(lineno, "tex % comment", line.strip()))
    report.hits = hits
    report.comment_lines = len(hits)
    report.code_lines = sum(1 for line in lines if line.strip()) - len(hits)
    return report


def evaluate(report: Report) -> dict[str, dict[str, str]]:
    criteria: dict[str, dict[str, str]] = {}
    if report.kind != "python":
        criteria["E1"] = {"status": "n/a", "detail": "not a Python module"}
        criteria["E2"] = {"status": "n/a", "detail": "not a Python module"}
        criteria["E3"] = {"status": "n/a", "detail": "not a Python module"}
        if report.error:
            criteria["E4"] = {"status": "fail", "detail": report.error}
        elif report.hits:
            criteria["E4"] = {
                "status": "fail",
                "detail": f"{len(report.hits)} banned pattern(s)",
            }
        else:
            criteria["E4"] = {"status": "pass", "detail": "no banned patterns"}
        return criteria

    if report.code_lines == 0:
        e1_pass = report.comment_lines == 0
        e1_detail = (
            "empty file"
            if report.comment_lines == 0
            else f"{report.comment_lines} comment lines with no code lines"
        )
    else:
        e1_pass = report.comment_lines * 100 <= COMMENT_PCT_MAX * report.code_lines
        e1_detail = f"comment ratio {report.comment_pct:.1f}% <= 12.0%"
    criteria["E1"] = {
        "status": "pass" if e1_pass else "fail",
        "detail": e1_detail,
    }

    doc_pct = report.doc_pct
    e2_pass = (
        report.definitions == 0
        or report.documented * 100 <= DOC_PCT_MAX * report.definitions
    ) and report.longest_docstring <= DOC_LINES_MAX
    if report.error and report.definitions == 0:
        e2_pass = False
        e2_detail = report.error
    else:
        e2_detail = (
            f"docstrings {doc_pct:.1f}% <= 55.0%, "
            f"longest {report.longest_docstring} <= 6"
        )
    criteria["E2"] = {
        "status": "pass" if e2_pass else "fail",
        "detail": e2_detail,
    }

    header_ceiling = HEADER_ENTRY_MAX if report.runnable else HEADER_LIBRARY_MAX
    criteria["E3"] = {
        "status": "pass" if report.header_lines <= header_ceiling else "fail",
        "detail": (
            f"header {report.header_lines} <= {header_ceiling} "
            f"({'entry point' if report.runnable else 'library'})"
        ),
    }

    if report.error:
        criteria["E4"] = {"status": "fail", "detail": report.error}
    elif report.hits:
        criteria["E4"] = {
            "status": "fail",
            "detail": f"{len(report.hits)} banned pattern(s)",
        }
    else:
        criteria["E4"] = {"status": "pass", "detail": "no banned patterns"}
    return criteria


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _iter_tree(root: Path) -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in {"__pycache__", ".git"} and not name.startswith(".")
        )
        for name in sorted(filenames):
            if name.endswith((".py", ".tex")):
                found.append(Path(dirpath) / name)
    return found


def collect(paths: list[str]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    errors: list[str] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(_iter_tree(path))
        elif path.is_file():
            if path.suffix.lower() not in {".py", ".tex"}:
                errors.append(f"{raw}: unsupported extension")
                continue
            files.append(path)
        else:
            errors.append(f"{raw}: no such file or directory")

    unique: dict[str, Path] = {}
    for path in files:
        unique.setdefault(str(path.resolve()), path)
    return sorted(unique.values(), key=lambda p: str(p)), errors


def default_paths() -> list[str]:
    paths = []
    src = REPO_ROOT / "src"
    if src.is_dir():
        paths.append(str(src))
    paths.extend(str(p) for p in sorted(REPO_ROOT.glob("*.py")))
    return paths


def report_text(report: Report, criteria: dict[str, dict[str, str]]) -> list[str]:
    lines = [_display_path(report.path)]
    ratio = report.comment_pct
    ratio_text = f"{ratio:.1f}%" if ratio is not None else "n/a"
    lines.append(
        f"  code {report.code_lines}  comments {report.comment_lines}  "
        f"ratio {ratio_text}"
    )
    lines.append(
        f"  defs {report.definitions}  documented {report.documented} "
        f"({report.doc_pct:.1f}%)  longest docstring {report.longest_docstring}"
    )
    lines.append(
        f"  header {report.header_lines} "
        f"({'entry point' if report.runnable else 'library'})"
    )
    for name in ("E1", "E2", "E3", "E4"):
        item = criteria[name]
        lines.append(f"  {name} {item['status'].upper():4} {item['detail']}")
    for hit in report.hits:
        text = hit.text if len(hit.text) <= 100 else hit.text[:97] + "..."
        lines.append(f"    line {hit.line}: {hit.pattern}: {text}")
    if report.error:
        lines.append(f"  error: {report.error}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check comment hygiene against ADR-026 (AC-E1 to AC-E4)."
    )
    parser.add_argument("paths", nargs="*", help="files or directories to check")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="walk the whole repository tree instead of src/ and top-level scripts",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.paths:
        targets = args.paths
    elif args.baseline:
        targets = [str(REPO_ROOT)]
    else:
        targets = default_paths()

    files, errors = collect(targets)
    results = []
    failed_files = 0
    counts = {
        "E1": {"pass": 0, "fail": 0, "n/a": 0},
        "E2": {"pass": 0, "fail": 0, "n/a": 0},
        "E3": {"pass": 0, "fail": 0, "n/a": 0},
        "E4": {"pass": 0, "fail": 0, "n/a": 0},
    }

    for path in files:
        report = scan_tex(path) if path.suffix.lower() == ".tex" else scan_python(path)
        criteria = evaluate(report)
        file_failed = any(item["status"] == "fail" for item in criteria.values())
        if file_failed:
            failed_files += 1
        for name, item in criteria.items():
            counts[name][item["status"]] += 1
        results.append((report, criteria))

    total_code = sum(r.code_lines for r, _ in results if r.kind == "python")
    total_comments = sum(r.comment_lines for r, _ in results if r.kind == "python")
    aggregate = 100.0 * total_comments / total_code if total_code else 0.0
    exit_code = 1 if failed_files or errors else 0

    if args.json:
        payload = {
            "files": [
                {
                    "path": _display_path(report.path),
                    "kind": report.kind,
                    "code_lines": report.code_lines,
                    "comment_lines": report.comment_lines,
                    "comment_pct": (
                        round(report.comment_pct, 2)
                        if report.comment_pct is not None
                        else None
                    ),
                    "definitions": report.definitions,
                    "documented": report.documented,
                    "doc_pct": round(report.doc_pct, 2),
                    "longest_docstring": report.longest_docstring,
                    "header_lines": report.header_lines,
                    "runnable": report.runnable,
                    "criteria": criteria,
                    "banned": [
                        {"line": hit.line, "pattern": hit.pattern, "text": hit.text}
                        for hit in report.hits
                    ],
                    "error": report.error,
                }
                for report, criteria in results
            ],
            "errors": errors,
            "summary": {
                "files": len(results),
                "passed": len(results) - failed_files,
                "failed": failed_files,
                "criteria": counts,
                "aggregate_comment_pct": round(aggregate, 2),
                "baseline_comment_pct": BASELINE_COMMENT_PCT,
                "exit_code": exit_code,
            },
        }
        print(json.dumps(payload, indent=2))
        return exit_code

    for report, criteria in results:
        print("\n".join(report_text(report, criteria)))

    if errors:
        print()
        for error in errors:
            print(f"error: {error}")

    print()
    print(
        f"Summary: {len(results)} files, {len(results) - failed_files} pass, "
        f"{failed_files} fail"
    )
    for name in ("E1", "E2", "E3", "E4"):
        item = counts[name]
        print(
            f"  {name}: {item['pass']} pass, {item['fail']} fail, "
            f"{item['n/a']} n/a"
        )
    print(
        f"  aggregate comment ratio {aggregate:.1f}% "
        f"(ADR-026 baseline {BASELINE_COMMENT_PCT:.1f}%, informational)"
    )
    print(f"Exit: {'FAIL' if exit_code else 'PASS'}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
