from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", max_retries=3)
search_tool = TavilySearch(max_results=10, topic="news",
                           time_range="week")  # recency locked


class MarketOutlook(BaseModel):
    regime: Literal["risk_on", "risk_off", "mixed", "none"]
    confidence: Literal["low", "medium", "high", "none"]
    top_headlines: list[str] = Field(
        description="top most market-relevant headlines found, each with its date and "
                    "one clause on why it matters.")
    key_risks: list[str] = Field(
        description="Near-term risks to the tape (events, macro, geopolitics).")
    summary: str = Field(
        description="2-3 sentence market outlook for the coming week.")


MANDATE = """You are the market-context analyst on a swing-trading desk (1-5 day horizon).
Assess the OVERALL tape: macro/economic news, geopolitics, index behavior, major events.
You do NOT analyze individual stocks.

RULES:
- Use the search tool multiple times covering different angles (macro/economy, geopolitics,
  market conditions/indices, upcoming events).
- Check each result's publication date in its metadata; ignore anything older than ~7 days
  regardless of relevance.
- Base everything ONLY on search results — cite headline + date for every claim.
- "mixed" with honest confidence is better than a manufactured strong view.
- All fields are required, if nothing found explain why it failed  in summary and write none in key risk, confidence and regime None
"""

market_agent = create_agent(model=model, tools=[search_tool],
                            response_format=MarketOutlook, system_prompt=MANDATE)


def analyze_market() -> MarketOutlook:
    msg = (f"Today's date is {datetime.now().date().isoformat()}.\n"
           f"Assess current market conditions for the coming trading week.")
    result = market_agent.invoke({"messages": [{"role": "user", "content": msg}]},
                                 config={"recursion_limit": 10})
    return result["structured_response"]
