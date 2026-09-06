"""House style gate (Katie, 2026-08-31): no em dash (U+2014), no middle dot (U+00B7) and no
AI-slop phrasing in any text the app or the generator authors for readers. Verbatim external
material is exempt, and code comments plus docstrings are not reader-facing.

Two layers, kept deliberately separate:

* The AST layer walks `publish_metrics` string literals, the markdown it renders, and the
  authored docs. It is self-contained and cannot be switched off by a missing script.
* The scanner layer runs `scripts/check_client_copy.py` across the frontend, the backend
  routes/models/analytics and the top-level markdown, and is measured with mutations, because
  a checker that cannot fail is indistinguishable from one that works.
"""
import ast
from pathlib import Path

import pytest

from experiments import publish_metrics as pm
from scripts import check_client_copy as ccc

ROOT = Path(__file__).resolve().parents[1]
BANNED = {"\u2014": "em dash", "\u00b7": "middle dot"}
EM, DOT = "\u2014", "\u00b7"


# --- AST layer ------------------------------------------------------------

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


# --- scanner layer --------------------------------------------------------

def test_client_facing_text_is_clean():
    hits = ccc.findings()
    assert not hits, "\n".join(hits)


def test_checker_catches_planted_violations(tmp_path):
    (tmp_path / "frontend").mkdir()
    (tmp_path / "backend" / "routes").mkdir(parents=True)
    (tmp_path / "frontend" / "x.html").write_text(
        f"<p>Phase 2 {DOT} live</p>\n<!-- a comment {EM} ignored -->\n", encoding="utf-8")
    (tmp_path / "backend" / "routes" / "r.py").write_text(
        f'"""doc {EM} ignored"""\nNOTE = "modelled {EM} not observed"\n'
        f'MSG = "Worth a quick chat?"\nOK = "fine"  # {EM} comment ignored\n', encoding="utf-8")
    (tmp_path / "README.md").write_text(f"Data {DOT} Code\n", encoding="utf-8")
    hits = ccc.findings(tmp_path)
    assert any(h.startswith("frontend/x.html:1:") and "middle dot" in h for h in hits)
    assert not any("x.html:2:" in h for h in hits), "HTML comment must be ignored"
    assert any(h.startswith("backend/routes/r.py:2:") and "em dash" in h for h in hits)
    assert any("r.py:3:" in h and "AI-slop" in h for h in hits)
    assert not any("r.py:1:" in h or "r.py:4:" in h for h in hits), "docstring and comment must be ignored"
    assert any(h.startswith("README.md:1:") for h in hits)
    assert len(hits) == 4, hits


def test_inline_allow_marker_exempts_verbatim_text(tmp_path):
    (tmp_path / "README.md").write_text(
        f'Title: "Endpoints {EM} a review" (copy-check: allow, quoted title)\n', encoding="utf-8")
    assert ccc.findings(tmp_path) == []


def test_style_rule_names_the_banned_characters():
    assert EM in ccc.STYLE_RULE and DOT in ccc.STYLE_RULE
    assert "concise" in ccc.STYLE_RULE and "Preserve all substantive detail" in ccc.STYLE_RULE


def test_endpoint_classifier_prompt_emits_labels_not_prose():
    """The only LLM prompt in the repo. It must never produce reader-facing prose, and its
    own text must follow the punctuation rule so nothing leaks into a label or a log."""
    src = (ROOT / "scripts" / "classify_endpoints_llm.py").read_text(encoding="utf-8")
    start = src.index("PROMPT_HEAD = ")
    prompt = src[start: src.index('"""', src.index('"""', start) + 3) + 3]
    assert "STRICT JSON only, no prose" in prompt
    assert EM not in prompt and DOT not in prompt
