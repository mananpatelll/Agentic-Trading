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


def thread_config(symbol: str, unique: bool = False) -> dict:
    """One thread per symbol per day, so re running the same scan resumes rather than starting over. 'unique' appends the time to force a brand-new thread when the user chooses to re-analyze an already finished candidate."""
    if unique:
        stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    else:
        stamp = f"{datetime.now():%Y%m%d}"
    return {"configurable": {"thread_id": f"{symbol}-{stamp}"}}


def ask_yes_no(question: str) -> bool:
    choice = input(f"{question} [yes/no]: ").strip().lower()
    while choice not in ("yes", "no"):
        choice = input("types 'yes' or 'no' : ").strip().lower()
    return choice == "yes"


def preflight(app, candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Phase 0 - settle every human decision before any concurrency starts.

    Returns (to_analyze, to_review). Touches no network and no LLM: it only
    reads checkpoint state and asks the user questions.
    """
    to_analyze: list[dict] = []
    to_review: list[dict] = []

    for c in candidates:
        symbol = c["symbol"]
        config = thread_config(symbol)
        status, _ = thread_status(app, config)

        if status == "done":
            if not ask_yes_no(f"{symbol}: Already analyzed today, Do you want to analyze again?"):
                print(f"skipping {symbol}")
                continue
            # A finished thread can not re-run, so give it a new id and status
            print("re-analyzing with a fresh thread")
            config = thread_config(symbol, unique=True)
            status = "fresh"

        if status == "awaiting_approval":
            to_review.append({"symbol": symbol, "config": config})
        else:  # fresh | interrupted
            if status == "interrupted":
                print(f"{symbol}: previous run died mid-execution, re-running")
            to_analyze.append({"symbol": symbol, "config": config})

    return to_analyze, to_review


def run() -> None:
    app = build_graph()
    candidates = load_candidates(scan_csv())
    print(F"Loaded {len(candidates)} candidates from today's scan")

    # ---- Phase 0: pre-flight -------------------------------------------
    to_analyze, to_review = preflight(app, candidates)

    if not to_analyze and not to_review:
        print("Nothing to do.")
        return

    market_outlook = None
    if to_analyze:
        # Fetched exactly once, before any worker starts. Every worker is
        # handed this same dict, so nothing lazily initializes shared state.
        market_outlook = get_market_outlook()
        print(
            f"\nMARKET: {market_outlook['regime']} ({market_outlook['confidence']})\n")

    print(
        f"analyzing {len(to_analyze)} | {len(to_review)} already awaiting approval")

    # ---- Phase 1: analysis (sequential for now) --------
    for item in to_analyze:
        print(f"Analyzing {item['symbol']}")
        result = app.invoke(
            {"symbol": item["symbol"], "market_outlook": market_outlook},
            item["config"])

        if "__interrupt__" in result:   # paused for approval, hand to Phase 2
            to_review.append(item)
        else:                           # no_trade or risk-gate rejected
            log_decision(result)

    # ---- Phase 2: review queue -----------------------------------------
    for item in sorted(to_review, key=lambda i: i["symbol"]):
        display_proposal(app.get_state(item["config"]).values)
        result = app.invoke(Command(resume=get_decision()), item["config"])
        log_decision(result)


if __name__ == "__main__":
    run()
