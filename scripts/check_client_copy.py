#!/usr/bin/env python3
"""check_client_copy: fail when authored client-facing text carries banned punctuation or AI-slop.

STYLE RULE for every client-facing surface, and for any future prompt that produces prose a
reader sees (import STYLE_RULE and put it in the shared system instruction):

    Write in concise, natural professional English. Do not use an em dash or a middle dot.
    Avoid generic AI/consulting language, rhetorical sales CTAs, unnecessary adjectives,
    inflated transitions and formulaic prose. Prefer direct factual sentences and ordinary
    punctuation. Preserve all substantive detail, qualifications and source accuracy.

What is scanned (the surfaces a user or API caller reads):
  frontend/**            html/js/css with comments stripped
  backend/routes/*.py    string literals, docstrings excluded
  backend/models/provenance.py, backend/models/inference.py, backend/analytics/*.py,
  backend/main.py, backend/constants.py, experiments/publish_metrics.py (generates README/RESULTS)
  README.md, RESULTS.md  whole file (public docs, including the generated blocks)

Not scanned, on purpose: code comments and docstrings (nobody but a developer reads them),
docs/ and experiments/reports/ (internal analysis, not served), AI_STATE/SPECS/CHANGELOG,
tests/, data/, models/, .venv. Reproduced external text (a quoted title, regulatory wording)
goes in VERBATIM_ALLOW or carries the inline marker `copy-check: allow` on its line.

Usage: python -m scripts.check_client_copy   (exit 1 on findings; also run by tests/test_client_copy.py)
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]

BANNED_CHARS = {"·": "middle dot U+00B7", "—": "em dash U+2014"}
SLOP_PATTERNS = [
    r"\bunlock(?:s|ed|ing)?\b", r"\bdrive impact\b", r"\bseamless(?:ly)?\b", r"\bholistic\b",
    r"\bactionable (?:insights?|intelligence)\b", r"\bempower(?:s|ed|ing)?\b", r"\bgame-changing\b",
    r"\bcutting-edge\b", r"\bbest-in-class\b", r"\bsupercharge\b", r"\brevolutioni[sz]e\b",
    r"\bat the intersection of\b", r"\bin today's rapidly\b", r"\bkey takeaways\b",
    r"\bworth a quick chat\b", r"\bworth exploring\b", r"\bwould you be open to\b",
    r"\bleverage synerg", r"\bstate-of-the-art\b", r"\bthe cleanest first step\b",
    r"\bnatural cadence\b", r"\bif it earns its place\b",
]
SLOP = re.compile("|".join(SLOP_PATTERNS), re.IGNORECASE)

STYLE_RULE = (
    "Write in concise, natural professional English. Do not use an em dash (—) or a middle "
    "dot/interpunct (·). Avoid generic AI/consulting language, rhetorical sales CTAs, "
    "unnecessary adjectives, inflated transitions and formulaic prose. Prefer direct factual "
    "sentences and ordinary punctuation. Preserve all substantive detail, qualifications and "
    "source accuracy."
)

FRONTEND_GLOBS = ["frontend/**/*.html", "frontend/**/*.js", "frontend/**/*.css"]
PY_STRING_GLOBS = ["backend/routes/*.py", "backend/models/provenance.py", "backend/models/inference.py",
                   "backend/analytics/*.py", "backend/main.py", "backend/constants.py",
                   "experiments/publish_metrics.py"]
MARKDOWN_GLOBS = ["README.md", "RESULTS.md"]
ALLOW_MARKER = "copy-check: allow"
#: Path globs (relative to ROOT) that reproduce external text verbatim and are exempt.
VERBATIM_ALLOW: List[str] = []

Finding = Tuple[str, int, str, str]   # (path, line, what, snippet)


def _allowed(rel: str) -> bool:
    return any(Path(rel).match(g) for g in VERBATIM_ALLOW)


def _check_line(rel: str, lineno: int, text: str, out: List[Finding]) -> None:
    if ALLOW_MARKER in text:
        return
    for ch, name in BANNED_CHARS.items():
        if ch in text:
            out.append((rel, lineno, name, text.strip()[:100]))
    m = SLOP.search(text)
    if m:
        out.append((rel, lineno, f"AI-slop phrase '{m.group(0)}'", text.strip()[:100]))


def _strip_block_comments(text: str) -> str:
    # Keep newlines so line numbers survive.
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))
    text = re.sub(r"/\*.*?\*/", blank, text, flags=re.S)
    return re.sub(r"<!--.*?-->", blank, text, flags=re.S)


def scan_text(path: Path, rel: str, out: List[Finding]) -> None:
    """Frontend files: block comments stripped; whole-line `//` comments skipped."""
    text = _strip_block_comments(path.read_text(encoding="utf-8"))
    for i, line in enumerate(text.splitlines(), 1):
        if path.suffix == ".js" and line.lstrip().startswith("//"):
            continue
        _check_line(rel, i, line, out)


def scan_markdown(path: Path, rel: str, out: List[Finding]) -> None:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _check_line(rel, i, line, out)


def scan_python(path: Path, rel: str, out: List[Finding]) -> None:
    """Every string literal that is not a docstring, f-string parts included."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    # A bare string statement (a docstring, or the module text that follows a
    # `from __future__` import) is never assigned or passed anywhere, so no reader sees it.
    docstrings = {id(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                  and isinstance(node.value.value, str)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            src_line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
            if ALLOW_MARKER in src_line:
                continue
            _check_line(rel, node.lineno, node.value, out)


def findings(root: Path = ROOT) -> List[str]:
    out: List[Finding] = []
    for globs, fn in ((FRONTEND_GLOBS, scan_text), (PY_STRING_GLOBS, scan_python), (MARKDOWN_GLOBS, scan_markdown)):
        for g in globs:
            for p in sorted(root.glob(g)):
                rel = p.relative_to(root).as_posix()
                if _allowed(rel):
                    continue
                fn(p, rel, out)
    return [f"{rel}:{ln}: {what}: {snip}" for rel, ln, what, snip in out]


def main() -> int:
    hits = findings()
    for h in hits:
        print(h)
    if hits:
        print(f"check_client_copy: {len(hits)} finding(s)", file=sys.stderr)
        return 1
    print("check_client_copy: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
