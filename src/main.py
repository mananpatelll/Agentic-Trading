import sys
import glob
import pandas as pd
from datetime import datetime
from langgraph.types import Command

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
        config = {"configurable": {
            "thread_id": f"{symbol}-{datetime.now():%Y%m%d}"}}
        result = app.invoke(
            {"symbol": symbol, "market_outlook": market_outlook}, config)
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(f"\n=== PROPOSED: {payload['symbol']} ===")
            print(payload["proposal"])
            print("technical:", payload["technical"]
                  ["direction"], payload["technical"]["confidence"])
            print("news:", payload["news"]["direction"],
                  "| regime:", payload["regime"])
            choice = input("approve / reject: ").strip().lower()
            while choice not in ("approve", "reject"):      # validate OUTSIDE the graph
                choice = input("type 'approve' or 'reject': ").strip().lower()

            result = app.invoke(Command(resume=choice),
                                config)   # same thread_id
        log_decision(result)


if __name__ == "__main__":
    run()
