"""Orchestration in main.py: routing, retry, and the logging contract.

Every bug this file has had was in orchestration rather than analysis - a
result handler indented out of its loop, a rate limit recorded as a trading
decision, a retry that restarted instead of resuming. These tests pin the
behaviour those bugs broke.

Nothing here touches the network: the compiled graph is stubbed, so a whole
run costs milliseconds and no tokens.
"""

from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError, RateLimitError

from src import main

# --- thread states, as thread_status() reads them -------------------------
FRESH = SimpleNamespace(created_at=None, next=())  # Fresh threads are empty
# Finished threads are created but no value in next
DONE = SimpleNamespace(created_at="t", next=())
# Waiting for approval are parked at interrupt
PARKED = SimpleNamespace(created_at="t", next=("approval",))
# Died mid exeuction shows next node name
DIED = SimpleNamespace(created_at="t", next=("trader_agent",))


def rate_limit_error() -> RateLimitError:
    return RateLimitError(
        "tpm exhausted",
        response=httpx.Response(
            429, request=httpx.Request("POST", "http://x")),
        body=None,
    )


def bad_request_error() -> BadRequestError:
    return BadRequestError(
        "bad prompt",
        response=httpx.Response(
            400, request=httpx.Request("POST", "http://x")),
        body=None,
    )


class FakeApp:
    """Stands in for the compiled graph, keyed by symbol.

    `states` drives get_state (what preflight inspects); `outcomes` drives
    invoke - a dict is returned, an Exception is raised.
    """

    def __init__(self, states=None, outcomes=None):
        self.states = states or {}
        self.outcomes = outcomes or {}

    @staticmethod
    def _symbol(config: dict) -> str:
        return config["configurable"]["thread_id"].split("-")[0]

    def get_state(self, config):
        snap = self.states.get(self._symbol(config), FRESH)
        return SimpleNamespace(created_at=snap.created_at, next=snap.next,
                               values={"symbol": self._symbol(config)})

    def invoke(self, payload, config):
        outcome = self.outcomes.get(self._symbol(config),
                                    {"symbol": self._symbol(config)})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# --- preflight: routing ---------------------------------------------------

def test_routes_each_thread_state_to_the_right_queue():
    app = FakeApp(states={"FRESH": FRESH, "DIED": DIED, "PARKED": PARKED})
    candidates = [{"symbol": s} for s in ("FRESH", "DIED", "PARKED")]

    to_analyze, to_review = main.preflight(app, candidates)

    # a thread that died mid-run still needs analysing; one already parked at
    # the approval node does not - its work is done, only a human is missing
    assert sorted(i["symbol"] for i in to_analyze) == ["DIED", "FRESH"]
    assert [i["symbol"] for i in to_review] == ["PARKED"]


def test_reanalysing_a_finished_thread_uses_a_new_id(monkeypatch):
    """A completed thread cannot be re-run, so it needs a fresh lineage."""
    monkeypatch.setattr("builtins.input", lambda *_: "yes")
    app = FakeApp(states={"DONE": DONE})

    to_analyze, to_review = main.preflight(app, [{"symbol": "DONE"}])

    assert not to_review
    reused = main.thread_config("DONE")["configurable"]["thread_id"]
    assert to_analyze[0]["config"]["configurable"]["thread_id"] != reused


def test_declining_a_finished_thread_drops_it_entirely(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "no")
    app = FakeApp(states={"DONE": DONE})

    assert main.preflight(app, [{"symbol": "DONE"}]) == ([], [])


# --- analyze_candidate: retry --------------------------------------------

def test_retry_resumes_from_the_checkpoint_instead_of_restarting(monkeypatch):
    """The retry must pass None, not the original input.

    Re-passing the input restarts the graph from START, so nodes that already
    succeeded run again and spend their tokens a second time - deepening the
    exact shortfall the retry is waiting out.
    """
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    payloads = []

    class App:
        def invoke(self, payload, config):
            payloads.append(payload)
            if len(payloads) == 1:
                raise rate_limit_error()
            return {"ok": True}

    item = {"symbol": "AAPL", "config": {
        "configurable": {"thread_id": "AAPL-x"}}}
    result = main.analyze_candidate(App(), item, {"regime": "risk_on"})

    assert result == {"ok": True}
    assert payloads[0]["symbol"] == "AAPL"  # first attempt carries the input
    assert payloads[1] is None              # retry resumes


def test_retry_waits_escalate_in_tens_of_seconds(monkeypatch):
    """A TPM limit clears when the rolling minute rolls over, so sub-second
    backoff cannot outlast it."""
    waits = []
    monkeypatch.setattr(main.time, "sleep", waits.append)

    class App:
        def invoke(self, payload, config):
            raise rate_limit_error()

    item = {"symbol": "AAPL", "config": {
        "configurable": {"thread_id": "AAPL-x"}}}
    with pytest.raises(RateLimitError):
        main.analyze_candidate(App(), item, {}, attempts=3)

    assert waits == [30, 60]


def test_permanent_errors_are_not_retried(monkeypatch):
    """A 400 will never succeed. Retrying only burns 90s and fails anyway."""
    waits = []
    monkeypatch.setattr(main.time, "sleep", waits.append)
    attempts = []

    class App:
        def invoke(self, payload, config):
            attempts.append(payload)
            raise bad_request_error()

    item = {"symbol": "AAPL", "config": {
        "configurable": {"thread_id": "AAPL-x"}}}
    with pytest.raises(BadRequestError):
        main.analyze_candidate(App(), item, {})

    assert len(attempts) == 1
    assert waits == []


# --- run: the logging contract -------------------------------------------

@pytest.fixture
def journal(monkeypatch):
    """Stubs everything run() reaches for and captures what it journals."""
    written = []
    monkeypatch.setattr(main, "scan_csv", lambda: "scan.csv")
    monkeypatch.setattr(main, "get_market_outlook",
                        lambda: {"regime": "mixed", "confidence": "medium"})
    monkeypatch.setattr(main, "log_decision", written.append)
    monkeypatch.setattr(main, "display_proposal", lambda _: None)
    monkeypatch.setattr(main, "get_decision", lambda: "approve")
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    return written


def _wire(monkeypatch, app, symbols):
    monkeypatch.setattr(main, "build_graph", lambda: app)
    monkeypatch.setattr(main, "load_candidates",
                        lambda _: [{"symbol": s} for s in symbols])


def test_every_candidate_produces_exactly_one_record(journal, monkeypatch):
    """The result handler once sat outside its loop, so it ran once on
    leftover variables: 14 candidates in, 1 record out."""
    outcomes = {
        "AAA": {"symbol": "AAA", "proposal": {"action": "no_trade"}},
        "BBB": {"symbol": "BBB", "__interrupt__": [1]},
        "CCC": {"symbol": "CCC", "proposal": {"action": "no_trade"}},
        "DDD": {"symbol": "DDD", "__interrupt__": [1]},
        "EEE": RuntimeError("boom"),
    }
    _wire(monkeypatch, FakeApp(outcomes=outcomes), outcomes)

    main.run()

    assert len(journal) == len(outcomes)
    assert {r["symbol"] for r in journal} == set(outcomes)


def test_a_crash_is_recorded_as_failed_not_as_a_decision(journal, monkeypatch):
    """A rate limit is not a no_trade. Recording it as one puts a fiction in
    the journal that later reads as a real call."""
    _wire(monkeypatch, FakeApp(outcomes={"EEE": RuntimeError("429 rate limited")}),
          ["EEE"])

    main.run()

    assert len(journal) == 1
    assert journal[0]["status"] == "failed"
    assert "429" in journal[0]["error"]
    assert "proposal" not in journal[0]


def test_one_failure_does_not_sink_the_other_candidates(journal, monkeypatch):
    outcomes = {
        "AAA": {"symbol": "AAA", "proposal": {"action": "no_trade"}},
        "BOOM": RuntimeError("boom"),
        "CCC": {"symbol": "CCC", "proposal": {"action": "no_trade"}},
    }
    _wire(monkeypatch, FakeApp(outcomes=outcomes), outcomes)

    main.run()

    assert {r["symbol"] for r in journal} == {"AAA", "BOOM", "CCC"}


def test_review_order_is_deterministic(journal, monkeypatch):
    """Completion order under the pool is nondeterministic; review order must
    not be, or the same scan presents differently on every run."""
    reviewed = []
    symbols = ["DDD", "AAA", "CCC", "BBB"]
    outcomes = {s: {"symbol": s, "__interrupt__": [1]} for s in symbols}
    _wire(monkeypatch, FakeApp(outcomes=outcomes), symbols)
    monkeypatch.setattr(main, "display_proposal",
                        lambda state: reviewed.append(state["symbol"]))

    main.run()

    assert reviewed == sorted(symbols)


def test_nothing_to_do_skips_the_market_analysis(monkeypatch):
    """get_market_outlook costs several searches and an LLM call. It must not
    run when every candidate was declined."""
    called = []
    monkeypatch.setattr(main, "scan_csv", lambda: "scan.csv")
    monkeypatch.setattr(main, "get_market_outlook", lambda: called.append(1))
    monkeypatch.setattr(main, "log_decision", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda *_: "no")
    _wire(monkeypatch, FakeApp(states={"DONE": DONE}), ["DONE"])

    main.run()

    assert called == []
