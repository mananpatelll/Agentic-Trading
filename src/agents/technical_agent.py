from dotenv import load_dotenv
from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()
model = ChatOpenAI(model="gpt-4o", max_retries=3)


class Technical_outlook(BaseModel):
    direction: Literal["long", "short", "neutral"]
    confidence: Literal["low", "medium", "high"]
    key_reasons: list[str] = Field(
        description="2-4 reasons, Each must reference a specific value from the provided data"
    )
    setup_type: Optional[str] = Field(
        default=None,
        description="If a recognizable setup is present (e.g. 'breakout near 52w high', 'oversold mean-reversion'), name it. Otherwise null."
    )


PROMPT = """You are a technical analyst on a swing-trading desk (1-5 day holding period).

MANDATE — read carefully:
- You analyze provided PRICE and other technical data and decide your direction.
- Every reason you give MUST cite a specific number from the data below. A reason
  that doesn't reference the provided values is invalid.
- "neutral" is a fully acceptable answer. If the picture is mixed or unremarkable,
  say neutral with low/medium confidence rather than manufacturing a view.
- State what would invalidate your view as a concrete price level or condition.

DATA:
{snapshot}

Give your technical outlook for a 1-5 day swing horizon."""


def format_snapshot(snap: dict) -> str:
    """Formates the data"""
    return (
        f"Price: ${snap['price']:.2f}\n"
        f"RSI(14): {snap['rsi14']:.1f}\n"
        f"SMA20: ${snap['sma20']:.2f}\n"
        f"SMA50: ${snap['sma50']:.2f}\n"
        f"Volume: {snap['vol_ratio']:.1f}x the 20-day average\n"
        f"Distance from 52-week high: {snap['pct_from_52w_high']:+.1f}%\n"
        f"Distance from 52-week low: {snap['pct_from_52w_low']:+.1f}%\n"
    )


def analyze(snapshot: dict) -> Technical_outlook:
    prompt = PROMPT.format(snapshot=format_snapshot(snapshot))
    return model.with_structured_output(Technical_outlook).invoke(prompt)


def technical_node(state: dict) -> dict:
    outlook = analyze(state["market_snapshot"])
    return {"technical_outlook": outlook.model_dump()}
