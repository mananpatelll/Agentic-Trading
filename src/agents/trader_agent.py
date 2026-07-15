from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0)


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
8. Target must provide at least a 1.5:1 reward-to-risk ratio.
9. Never invent evidence. Cite only analyst statements."""


def propose(state: dict) -> Proposal:
    prompt = f"{MANDATE}\n\n{format_desk_view(state)}\n\nYour decision:"
    return model.with_structured_output(Proposal).invoke(prompt)


def trader_node(state: dict) -> dict:
    try:
        proposal = propose(state)
    except Exception as e:
        proposal = Proposal(action="no_trade",
                            rationale=f"trader error: {e}", evidence_cited=[])
    return {"proposal": proposal.model_dump()}
