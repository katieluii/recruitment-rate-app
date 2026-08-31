"""Client-facing copy carries no em dash, no middle dot and no AI-slop phrasing.

The gate is measured with mutations (a planted violation must be caught), because a checker
that cannot fail is indistinguishable from one that works.
"""
from pathlib import Path

from scripts import check_client_copy as ccc

ROOT = Path(__file__).resolve().parents[1]
EM, DOT = "—", "·"


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
