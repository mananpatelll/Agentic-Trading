# src/journal.py
import csv
import json
from datetime import datetime
from pathlib import Path

JOURNAL_DIR = Path("data/journal")


def log_decision(record: dict):
    """Append any record to today's JSONL. Caller owns the structure;
    'ts' is stamped automatically."""
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now().isoformat(timespec="seconds"), **record}
    path = JOURNAL_DIR / f"{datetime.now():%Y-%m-%d}_decisions.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n\n\n")
