from __future__ import annotations
"""Append-only experiment ledger.

One complete JSON record per call, written with O_APPEND so concurrent runs
cannot lose each other's rows. Do NOT convert this to read-modify-write — that
reintroduces the lost-update race this format exists to avoid.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LEDGER_PATH = Path(__file__).parent / "ledger.jsonl"


def append(record: dict[str, Any], path: Path = LEDGER_PATH) -> None:
    """Atomically append one experiment record."""
    record = {"ts": datetime.now().isoformat(timespec="seconds"), **record}
    line = json.dumps(record, default=str, sort_keys=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
    log.info("Ledger += %s/%s", record.get("config"), record.get("phase"))


def read_all(path: Path = LEDGER_PATH) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("Skipping malformed ledger line: %.80s", line)
    return rows
