from typing import TypedDict, Annotated
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
from agents.technical_agent import technical_node
from agents.news_agent import news_node
from agents.trader_agent import trader_node
from indicators import *
from scanner import fetch_daily_bars, load_config
from risk_gate import run_risk_gate

conn = sqlite3.connect("data/checkpoint.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)


class AgentState(TypedDict):
    symbol: dict
    market_snapshot: dict
    market_outlook: dict
    technical_outlook: dict
    news_outlook: dict
    events_outlook: dict
    proposal: dict
    risk_gate: dict
    decision: dict
    status: dict


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
    return "risk_gate" if state["proposal"]["action"] != "no_trade" else "skip"


def approval_node(state: AgentState) -> dict:
    decision = interrupt("awaiting_human_approval")
    return {"decision": decision}


def approval_route(state: AgentState) -> str:
    # goes to human approval if passed by the risk gate
    return "approval" if state["risk_gate"]["passed"] else "rejected"


def decision_route(state: AgentState) -> str:
    "goes to trade execution if approved by the human"
    return "execute" if state["decision"] == "approve" else "rejected"


def rejected_node(state: AgentState) -> dict:
    if not state["risk_gate"]["passed"]:  # If rejected by risk gate update the status
        return {"status": "risk_gate_rejected", "executed": False}
    return {"status": "human_rejected", "executed": False}


def execute_node(state: AgentState) -> dict:
    # ONLY reachable through human approval by graph structure, not by a check
    # False until real execution
    return {"status": "human_approved", "executed": False}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load_data", load_data)
    graph.add_node("technical", technical_node)
    graph.add_node("news_agent", news_node)
    graph.add_node("trader_agent", trader_node)
    graph.add_node("risk_gate", run_risk_gate)
    graph.add_node("approval", approval_node)
    graph.add_node("rejected", rejected_node)
    graph.add_node("execute", execute_node)

    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "technical")
    graph.add_edge("load_data", "news_agent")
    graph.add_edge("technical", "trader_agent")
    graph.add_edge("news_agent", "trader_agent")
    graph.add_conditional_edges("trader_agent", risk_gate_router, {
                                "risk_gate": "risk_gate", "skip": END})
    graph.add_conditional_edges("risk_gate", approval_route, {
        "approval": "approval", "rejected": "rejected"})
    graph.add_conditional_edges("approval", decision_route, {
                                "execute": "execute", "rejected": "rejected"})
    graph.add_edge("execute", END)
    graph.add_edge("rejected", END)
    return graph.compile(checkpointer=checkpointer)
