import logging
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from llm import get_model


log = logging.getLogger(__name__)

search_tool = TavilySearch(max_results=5)
model = get_model("news")
log.info("model for news agent : %s", model.model_name)


class NewsOutlook(BaseModel):
    direction: Literal["long", "short", "neutral"]
    confidence: Literal["low", "medium", "high"]
    material_news: bool = Field(
        description="True only if the provided headlines contain genuinely relevant news.")
    key_reasons: list[str] = Field(
        description="1-3 reasons. Each MUST quote or reference a specific provided headline "
                    "and its date. No reason may rely on anything not in the provided articles.")


MANDATE = f"""You are a news analyst on a swing-trading desk (1-5 day holding period).

RULES:
- Base your outlook ONLY on the baseline headlines provided and on your search results.
  Nothing you remember about this company from training exists. If neither source shows
  material news, say so: neutral, material_news=false.
- Use the search tool again as much time as you need, and only if the baseline feed leaves a real gap
  (a vague headline worth verifying, or checking for macro/events coverage).
- Every reason must cite a specific headline or search result.
"""

news_agent = create_agent(
    model=model,
    tools=[search_tool],
    response_format=NewsOutlook,
    system_prompt=MANDATE,
)


def news_node(state: dict) -> dict:
    log.info("analyzing news for : %s", state['symbol'])
    msg = (
        f"Assess the news picture for {state["symbol"]} on a 1-5 day horizon.")
    result = news_agent.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        # hard cap under the prompt's soft cap
        config={"recursion_limit": 8},
    )
    return {"news_outlook": result["structured_response"].model_dump()}
