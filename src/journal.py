# src/journal.py
import csv
import json
from datetime import datetime
from pathlib import Path

JOURNAL_DIR = Path("data/journal")


def log_decision(symbol: str, scan_signals: str, snapshot: dict, outlook: dict):
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "scan_signals": scan_signals,
        "market_snapshot": snapshot,     # what the agent SAW
        "technical_outlook": outlook,    # what it DECIDED
    }
    path = JOURNAL_DIR / f"{datetime.now():%Y-%m-%d}_decisions.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_summary(symbol: str, outlook: dict):
    path = JOURNAL_DIR / f"{datetime.now():%Y-%m-%d}_summary.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "symbol", "direction",
                       "confidence", "setup", "reasons"])
        w.writerow([
            datetime.now().strftime("%H:%M"), symbol,
            outlook["direction"], outlook["confidence"],
            outlook.get("setup_type") or "", " | ".join(
                outlook["key_reasons"]),
        ])
