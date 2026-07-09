import sys
import glob
import uuid
from datetime import datetime
import pandas as pd
from langchain_core.messages import HumanMessage
from graph import build_graph
from journal import log_decision, log_summary


def scan_csv() -> str:
    files = sorted(glob.glob("data/scans/*_scan.csv"))
    if not files:
        sys.exit("No scan file found. Run the scanner first")
    return files[-1]  # Returns latest file


def load_candidates(path: str) -> list[dict]:
    df = pd.read_csv(path)
    return df.to_dict("records")


def run():
    app = build_graph()
    candidates = load_candidates(scan_csv())
    print(F"Loaded {len(candidates)} candidates from today's scan")
    for c in candidates:
        print(f"Analyzing {c}")
        symbol = c["symbol"]
        result = app.invoke({"symbol": symbol})
        log_decision(
            symbol, c["signals"], result["market_snapshot"], result["technical_outlook"])
        log_summary(symbol, result["technical_outlook"])
        print(f"{symbol}: {result['technical_outlook']['direction']} "
              f"({result['technical_outlook']['confidence']})")


if __name__ == "__main__":
    run()
