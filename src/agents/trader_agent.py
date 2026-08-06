import logging
from typing import Literal, Optional
from pydantic import BaseModel, Field
from risk_gate import run_risk_gate
from langchain_core.tools import tool
from langchain.agents import create_agent
from llm import get_model, TRANSIENT_API_ERRORS

log = logging.getLogger(__name__)

model = get_model("trader")


@tool
def risk_gate(symbol: str, entry: float, stop: float, target: float, action: str) -> str:
    """"
    Validate a proposed trade against risk rules.
    Parameters:
    - symbol : the name of symbol
    - entry: the proposed entry price
    - stop: the stop loss price
    - target: the profit target price
    - action: 'buy' or 'sell_short'
    Returns:
    - "PASS" if all checks pass
    - "FAIL: <detailed reason>" if any check fails, explaining what's wrong and how to fix it.
    """
    proposal = {"entry": entry, "stop": stop,
                "target": target, "action": action}
    result = run_risk_gate(state={"symbol": symbol, "proposal": proposal})
    rg = result["risk_gate"]
    if rg["passed"]:
        return "Risk Gate PASSED"
    else:
        return f"FAIL: {rg['reason']}"


class Proposal(BaseModel):
    action: Literal["buy", "sell_short", "no_trade"]
    entry: Optional[float] = Field(
        default=None, description="Proposed entry price. Null if no_trade.")
    stop: Optional[float] = Field(
        default=None, description="Stop loss at a technical level from the data. Null if no_trade.")
    target: Optional[float] = Field(
        default=None, description="Profit target. Null if no_trade.")
    rationale: str = Field(
        description="2-4 sentences: the synthesis that led here.")
    evidence_cited: list[str] = Field(
        description="Which specific claims from the analysts' outlooks support this. "
                    "Each item must reference an actual stated reason, not a new observation.")


def format_desk_view(state: dict) -> str:
    s, t, n, m = (state["market_snapshot"], state["technical_outlook"],
                  state["news_outlook"], state["market_outlook"])
    return (
        f"SYMBOL: {state['symbol']})\n\n"
        f"MARKET REGIME (desk-wide context): {m['regime']} ({m['confidence']} confidence)\n"
        f"  {m['summary']}\n\n"
        f"TECHNICAL ANALYST: {t['direction']} ({t['confidence']})"
        f"{', setup: ' + t['setup_type'] if t.get('setup_type') else ''}\n"
        + "".join(f"  - {r}\n" for r in t["key_reasons"]) +
        f"\nNEWS ANALYST: {n['direction']} ({n['confidence']}), material_news={n['material_news']}\n"
        + "".join(f"  - {r}\n" for r in n["key_reasons"]) +
        f"\nKEY LEVELS: price ${s['price']:.2f} | SMA20 ${s['sma20']:.2f} | "
        f"SMA50 ${s['sma50']:.2f} | 52w high {s['pct_from_52w_high']:+.1f}% away"
    )


MANDATE = """You are the head short term trader. holding time frame (5-7 days).

Your job is to synthesize the analysts' conclusions into exactly one action:
- buy
- sell_short
- no_trade

Rules:
1. The market regime determines whether long or short trades are appropriate.
2. The technical outlook is the primary directional signal.
3. News adjusts conviction. Material news that clearly invalidates the technical thesis should result in no_trade.
4. Choose buy or sell_short when the evidence is sufficiently aligned and confidence is at least medium.
5. Choose no_trade only when the evidence is conflicting, confidence is too low, or risk controls prohibit a trade.
6. Entry should be near the current price.
7. Stops must be placed using technical levels from the provided data.
8. Target must provide at least a  reward-to-risk ratio.
9. Never invent evidence. Cite only analyst statements.

RISK‑GATE TOOL (MANDATORY):

Before finalising a buy or sell_short, you MUST call the tool `risk_gate` with your planned entry, stop, target, and action.
- If the tool returns "PASS": you may output your final decision.
- If the tool returns a message starting with "FAIL": carefully read the reason, adjust your prices (entry, stop, target) to fix the problem, and call the tool again.
- Do NOT finalise a trade without a successful tool call.
- Do NOT guess the risk rules – rely only on the tool results.


"""

trader_agent = create_agent(
    model=model,
    tools=[risk_gate],
    response_format=Proposal,
    system_prompt=MANDATE,
)


def propose(state: dict) -> Proposal:
    prompt = f"\n\n{format_desk_view(state)}\n\nWhat is your decision:"
    return trader_agent.invoke({"messages": [{"role": "user", "content": prompt}]})


def trader_node(state: dict) -> dict:
    log.info("making trading decisions for %s", state["symbol"])
    try:
        proposal = propose(state)
        proposal_dict = proposal["structured_response"].model_dump()
        log.info("Trader agent decision for %s : %s", state["symbol"],
                 proposal["structured_response"].action)
    except Exception as e:
        if isinstance(e, TRANSIENT_API_ERRORS):
            # Don't handle it. A rate limit is not a trading decision - let it
            # reach the retry in main.py, which waits out the window and
            # resumes from the checkpoint.
            log.warning("%s : trader hit %s, propagating",
                        state['symbol'], type(e).__name__)
            raise
        log.exception("Error in trader agent %s", e)
        proposal = Proposal(action="no_trade",
                            rationale=f"trader error: {e}", evidence_cited=[])
        proposal_dict = proposal.model_dump()
    return {"proposal": proposal_dict}
