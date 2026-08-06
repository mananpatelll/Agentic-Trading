import sys
import glob
import time
import logging
from typing import Any
import pandas as pd
from datetime import datetime
from langgraph.types import Command
from llm import TRANSIENT_API_ERRORS

from graph import build_graph
from journal import log_decision
from agents.market_agent import get_market_outlook

from concurrent.futures import ThreadPoolExecutor, as_completed
from load_config import load_config

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


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
                log.info("skipping : %s", symbol)
                continue
            # A finished thread can not re-run, so give it a new id and status
            log.info("re-analyzing with a fresh thread")
            config = thread_config(symbol, unique=True)
            status = "fresh"

        if status == "awaiting_approval":
            to_review.append({"symbol": symbol, "config": config})
        else:  # fresh | interrupted
            if status == "interrupted":
                log.info(
                    " %s : previous run died mid-execution, re-running", symbol)
            to_analyze.append({"symbol": symbol, "config": config})

    return to_analyze, to_review


def analyze_candidate(app, item: dict, market_outlook: dict,
                      attempts: int = 3) -> dict:
    """Runs in a worker thread. One candidate start to finish.

    Retries on 429. The waits are tens of seconds on purpose: a TPM limit
    only clears when the rolling one-minute window rolls over, and the
    OpenAI SDK's own backoff is sub-second - sized for request-rate spikes,
    not for an exhausted token budget.
    """
    payload = {"symbol": item["symbol"], "market_outlook": market_outlook}

    for attempt in range(1, attempts + 1):
        try:
            return app.invoke(payload, item["config"])
        except TRANSIENT_API_ERRORS:
            if attempt == attempts:
                raise
            wait = 30 * attempt
            log.warning("%s | rate limited, retry %d%d in %ds",
                        item["symbol"], attempt, attempts, wait)
            time.sleep(wait)
            # Resume from the checkpoint rather than restarting. Nodes that
            # already succeeded must not re-run and spend their tokens twice,
            # which would only deepen the shortfall we're waiting out.
            payload = None


def run() -> None:
    app = build_graph()
    candidates = load_candidates(scan_csv())
    log.info("loaded %d candidates", (len(candidates)))

    # ---- Phase 0: pre-flight -------------------------------------------
    to_analyze, to_review = preflight(app, candidates)

    if not to_analyze and not to_review:
        log.info("nothing to do")
        return

    market_outlook = None
    if to_analyze:
        # Fetched exactly once, before any worker starts. Every worker is
        # handed this same dict, so nothing lazily initializes shared state.
        market_outlook = get_market_outlook()
        log.info("market : %s (%s)",
                 market_outlook["regime"], market_outlook['confidence'])

    log.info("analyzing %d | %d already waiting for approval",
             len(to_analyze), len(to_review))

    # ---- Phase 1: concurrent analysis ----------------------------------
    workers = load_config().get("run", {}).get("max_workers", 3)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cand") as pool:
        # Map each future back to its candidate, so a failure is attributable
        futures = {
            pool.submit(analyze_candidate, app, item, market_outlook): item
            for item in to_analyze
        }
        # as_completed yields futures in finishing order, not submission order
        # THis loop body runs on the main thread, which is why to_review and log_decision need no lock.
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                log.exception("%s : analysis failed - %s", item['symbol'], e)
                log_decision({"symbol": item["symbol"],
                              "status": "failed", "error": str(e)})
                continue

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
