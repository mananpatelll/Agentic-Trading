import sys
import glob
from typing import Any
import pandas as pd
from datetime import datetime
from langgraph.types import Command

from graph import build_graph
from journal import log_decision
from agents.market_agent import get_market_outlook


def scan_csv() -> str:
    files = sorted(glob.glob("data/scans/*_scan.csv"))
    if not files:
        sys.exit("No scan file found. Run the scanner first")
    return files[-1]  # Returns latest file


def load_candidates(path: str) -> list[dict]:
    df = pd.read_csv(path)
    return df.to_dict("records")


def thread_status(app, config: dict) -> tuple[str, Any]:
    """ Fresh = Never run
        Paused == awaiting approval
        done == completed for today
    """
    snap = app.get_state(config)

    if snap.created_at is None:
        return "fresh", None  # checkpoint never created
    if not snap.next:
        return "done", snap  # finished exeuction
    if "approval" in snap.next:  # paused at approval node
        return "awaiting_approval", snap
    return "interrupted", snap


def display_proposal(state: dict[str, Any]) -> None:
    """Render pending proposal from graph state"""
    print(f"\n=== PROPOSED: {state['symbol']} ===")
    print(state["proposal"])
    print("risk:", state["risk_gate"]["reason"])
    t, n = state["technical_outlook"], state["news_outlook"]
    print("technical:", t["direction"], t["confidence"])
    print("news:", n["direction"], "| regime:",
          state["market_outlook"]["regime"])


def get_decision() -> str:
    choice = input("approve/ reject: ").strip().lower()
    while choice not in ("approve", "reject"):
        choice = input("type 'approve' or 'reject': ").strip().lower()
    return choice


def run() -> None:
    market_outlook = None
    app = build_graph()
    candidates = load_candidates(scan_csv())
    print(F"Loaded {len(candidates)} candidates from today's scan")

    for c in candidates:

        symbol = c["symbol"]
        config = {"configurable": {
            "thread_id": f"{symbol}-{datetime.now():%Y%m%d}"}}
        status, snap = thread_status(app, config)

        if status == "done":
            print(
                f" {symbol}: Already analyzed today, Do you want to analyze again?")
            choice = input("yes/no: ").strip().lower()
            while choice not in ("yes", "no"):
                choice = input("types 'yes' or 'no' ").strip().lower()
            if choice == "no":
                print(f"skipping {symbol}")
                continue
            else:
                print("re-analyzing with a fresh thread")
                config = {"configurable": {
                    "thread_id": f"{symbol}-{datetime.now():%Y%m%d-%H%M%S}"}}
                status = "fresh"
        elif status == "awaiting_approval":  # If graph paused due to pending human approval
            display_proposal(snap.values)
            result = app.invoke(Command(resume=get_decision()), config)
            log_decision(result)
            continue

        if status in ("interrupted", "fresh"):
            if market_outlook is None:
                market_outlook = get_market_outlook()
                print(
                    f"\nMARKET: {market_outlook['regime']} ({market_outlook['confidence']})\n")

            if status == "interrupted":
                print(f"{symbol}: previous run died mid-execution, re-running")
            else:
                print(f"Fresh Analyzing {symbol}")

            result = app.invoke(
                {"symbol": symbol, "market_outlook": market_outlook},
                config)

        if "__interrupt__" in result:  # Fresh run that paused for approval
            display_proposal(app.get_state(config).values)
            result = app.invoke(Command(resume=get_decision()),
                                config)   # same thread_id
        log_decision(result)


if __name__ == "__main__":
    run()
