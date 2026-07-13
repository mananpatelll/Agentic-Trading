import sys
import glob
import pandas as pd
from graph import build_graph
from journal import log_decision
from agents.market_agent import analyze_market


def scan_csv() -> str:
    files = sorted(glob.glob("data/scans/*_scan.csv"))
    if not files:
        sys.exit("No scan file found. Run the scanner first")
    return files[-1]  # Returns latest file


def load_candidates(path: str) -> list[dict]:
    df = pd.read_csv(path)
    return df.to_dict("records")


def run():
    market_outlook = analyze_market().model_dump()
    print(
        f"\n MARKET : {market_outlook['regime']} ({market_outlook['confidence']}) \n")
    app = build_graph()
    candidates = load_candidates(scan_csv())
    print(F"Loaded {len(candidates)} candidates from today's scan")
    for c in candidates:
        print(f"Analyzing {c}")
        symbol = c["symbol"]
        result = app.invoke(
            {"symbol": symbol, "market_outlook": market_outlook})
        log_decision(result)


if __name__ == "__main__":
    run()
