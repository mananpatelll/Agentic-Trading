from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from agents.technical_agent import technical_node
from agents.news_agent import news_node
from agents.trader_agent import trader_node
from indicators import *
from scanner import fetch_daily_bars, load_config
from risk_gate import run_risk_gate


class AgentState(TypedDict):
    symbol: dict
    market_snapshot: dict
    market_outlook: dict
    technical_outlook: dict
    news_outlook: dict
    events_outlook: dict
    proposal: dict
    risk_check: dict


cfg = load_config()


def load_data(state: AgentState) -> AgentState:
    symbol = state["symbol"]
    df = fetch_daily_bars(symbol)
    d = df.xs(symbol, level="symbol")
    close, vol = d.close, d.volume
    snapshot = {
        "price":             float(close.iloc[-1]),
        "rsi14":             float(rsi(close).iloc[-1]),
        "sma20":             float(sma(close, 20).iloc[-1]),
        "sma50":             float(sma(close, 50).iloc[-1]),
        "vol_ratio":         float(volume_ratio(vol, 20)),
        "pct_from_52w_high": pct_from_52w_high(close),
        "pct_from_52w_low":  pct_from_52w_low(close),
    }
    return {"market_snapshot": snapshot}


def risk_gate_router(state: AgentState) -> str:
    print("At the router")
    print(
        f"{"risk_gate" if state["proposal"]["action"] != "no_trade" else "skip"}")
    return "risk_gate" if state["proposal"]["action"] != "no_trade" else "skip"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load_data", load_data)
    graph.add_node("technical", technical_node)
    graph.add_node("news_agent", news_node)
    graph.add_node("trader_agent", trader_node)
    graph.add_node("risk_gate", run_risk_gate)

    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "technical")
    graph.add_edge("technical", "news_agent")
    graph.add_edge("news_agent", "trader_agent")
    graph.add_conditional_edges("trader_agent", risk_gate_router, {
                                "risk_gate": "risk_gate", "skip": END})
    graph.add_edge("risk_gate", END)
    return graph.compile()
