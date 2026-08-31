"""House style gate (Katie, 2026-08-31): no em dash (U+2014) and no middle dot (U+00B7) in any
text the app or the generator authors for readers. Verbatim external material is exempt;
code comments are not reader-facing. Guarded here: every string literal in publish_metrics,
the markdown it renders, and the docs authored under this rule."""
import ast
from pathlib import Path

import pytest

from experiments import publish_metrics as pm

ROOT = Path(__file__).resolve().parents[1]
BANNED = {"—": "em dash", "·": "middle dot"}


def _offenders(text: str):
    return sorted({name for ch, name in BANNED.items() if ch in text})


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def test_publish_metrics_string_literals_are_clean():
    """Every string that can reach a reader (markdown, JSON text, status lines). Docstrings
    are developer-facing and excluded."""
    tree = ast.parse((ROOT / "experiments" / "publish_metrics.py").read_text())
    docs = _docstring_nodes(tree)
    bad = [(n.lineno, _offenders(n.value)) for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and id(n) not in docs and _offenders(n.value)]
    assert not bad, bad


def test_rendered_markdown_is_clean():
    rows = pm.load_rows(ROOT / "tests" / "fixtures" / "ledger_parity.jsonl")
    rows[1]["ipcw_applied"] = True; rows[4]["ipcw_applied"] = True
    pub = pm.build(rows)
    for text in (pm.markdown(pub), pm.markdown_rate(pub), pub["fold"], pub["rate"]["what"], pub["rate_head"]["what"]):
        assert not _offenders(text), text[:120]


@pytest.mark.parametrize("doc", ["docs/VERSION_HISTORY.md"])
def test_authored_docs_are_clean(doc):
    assert not _offenders((ROOT / doc).read_text()), doc
