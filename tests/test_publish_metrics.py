"""publish_metrics gates, each driven against tests/fixtures/ledger_parity.jsonl.

The fixture is the artifact each gate must REJECT: a shipped-config duration row whose
IPCW weights did not fire, a newer rate-head baseline row that would shadow the duration
baseline if `latest` ignored target, and a rate row that fails its coverage gate.
"""
from pathlib import Path

import pytest

from experiments import publish_metrics as pm

FIXTURE = Path(__file__).parent / "fixtures" / "ledger_parity.jsonl"


def rows():
    return pm.load_rows(FIXTURE)


def test_parity_gate_refuses_unweighted_duration_row():
    with pytest.raises(pm.ParityError) as e:
        pm.build(rows())
    msg = str(e.value)
    assert "P2" in msg and "row 2" in msg and "ipcw_applied=False" in msg


def test_parity_gate_refuses_missing_ipcw_flag():
    r = rows()
    del r[1]["ipcw_applied"]
    with pytest.raises(pm.ParityError):
        pm.build(r)


def test_latest_filters_by_target():
    # Row 3 is the NEWER ta_median row but under the rate target; the duration baseline
    # must still resolve to row 1.
    r = rows()
    assert pm.latest(r, "ta_median", "P2")["ledger_row"] == 1
    assert pm.latest(r, "ta_median", "P2", target=pm.RATE)["ledger_row"] == 3


def test_publishes_weighted_row_and_rate_block():
    r = rows()
    r[1]["ipcw_applied"] = True
    r[1]["interval_nominal"] = 0.85
    pub = pm.build(r)
    p2 = pub["phases"]["P2"]
    assert p2["ipcw_applied"] is True and p2["ledger_row"] == 2
    assert p2["baseline_mae_months"] == 12.0  # not the rate baseline's 4.0
    rate = pub["rate"]["phases"]["P2"]
    assert rate["mae"] == 2.2 and rate["baseline_mae"] == 4.0
    assert rate["all_gates_pass"] is False
    assert pub["gates_failing"] == ["P2 rate"]
    for ph in ("P1HV", "P1", "P3"):
        assert pub["phases"][ph]["status"].startswith("NOT MEASURED")
    md = pm.markdown(pub)
    assert "0.80 (0.85)" in md  # achieved (nominal) — P1HV's 0.85 band must not read as 0.80
    md_rate = pm.markdown_rate(pub)
    assert "**FAIL** (coverage 0.6 outside 0.75–0.9)" in md_rate


def test_fill_docs_replaces_both_blocks(tmp_path):
    r = rows()
    r[1]["ipcw_applied"] = True
    pub = pm.build(r)
    doc = tmp_path / "X.md"
    doc.write_text("a\n<!-- published_metrics:start -->\nOLD\n<!-- published_metrics:end -->\n"
                   "b\n<!-- published_metrics_rate:start -->\nOLD\n<!-- published_metrics_rate:end -->\n")
    assert pm.fill_docs(pub, docs=[doc]) == ["X.md"]
    text = doc.read_text()
    assert "OLD" not in text and "| P2 | 9.00 mo |" in text and "| P2 | 2.20 |" in text
    assert pm.fill_docs(pub, docs=[doc], write=False) == []  # idempotent


def _nominal_of(config: str) -> float:
    return 0.85 if "cov85" in config else 0.80


def test_shipped_configs_name_the_trainer_targets():
    """publish_metrics.SHIPPED / RATE_SHIPPED and trainer.COVERAGE_TARGET /
    RATE_COVERAGE_TARGET must describe the same band, or the published coverage is
    measured against a target the served artifact was not calibrated to."""
    from backend.models import trainer

    for ph, cfg in pm.SHIPPED.items():
        assert _nominal_of(cfg) == trainer.COVERAGE_TARGET.get(ph, 0.80), (ph, cfg)
        assert cfg.endswith("_ipcw"), f"{ph}: shipped duration config must be an IPCW-parity config"
    for ph, cfg in pm.RATE_SHIPPED.items():
        assert _nominal_of(cfg) == trainer.RATE_COVERAGE_TARGET.get(ph, 0.80), (ph, cfg)


def test_shipped_target_mismatch_is_caught(monkeypatch):
    """The rejecting case for the test above: drift one side and the check must fail."""
    from backend.models import trainer

    monkeypatch.setitem(trainer.RATE_COVERAGE_TARGET, "P1HV", 0.80)
    with pytest.raises(AssertionError):
        test_shipped_configs_name_the_trainer_targets()
