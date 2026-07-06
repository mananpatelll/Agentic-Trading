import os
import yaml
from datetime import datetime, timedelta
import pandas as pd
from indicators import *
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

load_dotenv()
ALPACA_API_KEY = os.getenv("APCA-API-KEY-ID")
ALPACA_SECRET_KEY = os.getenv("APCA-API-SECRET-KEY")
output_dir = "data/scans"

os.makedirs(output_dir, exist_ok=True)


def load_config(path="config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["scanner"]


def fetch_daily_bars(symbols: list[str], lookback_days: int = 420) -> pd.DataFrame:
    """fetch pricing data using alpaca api 
    lookback_days = 420 calendar days = 290 trading days - enough for 52w high low calculations
    """
    client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=lookback_days),
        adjustment=Adjustment.ALL       # ← split + dividend adjusted

    )
    bars = client.get_stock_bars(req)
    return bars.df  # return as DataFrame


def scan(universe: list[str], cfg: dict) -> pd.DataFrame:
    df = fetch_daily_bars(universe)
    candidates = []  # List to store scanned stocks
    for symbol in universe:
        try:
            d = df.xs(symbol, level="symbol")
        except KeyError:
            continue
        if len(d) < 260:
            continue

        # if avg volume is low skip this stock
        if avg_dollar_volume(d.close, d.volume) < cfg["min_dollar_volume"]:
            continue

        signals = []
        r = rsi(d.close).iloc[-1]
        if r <= cfg["rsi_oversold"]:
            print(f"{symbol} satisfys rsi_oversold condition : {r}")
            signals.append(f"RSI oversold ({r:.0f})")
        if r >= cfg["rsi_overbought"]:
            print(f"{symbol} satisfys rsi_overbought condition : {r}")
            signals.append(f"RSI overbought ({r:.0f})")
        if d.close.iloc[-1] > sma(d.close, 50).iloc[-1]:
            signals.append("Price above 50 day SMA")
        if volume_spike(d.volume, mult=cfg["vol_spike_mult"]):
            signals.append("Volume spike (2x 20 day avg)")

        high_dist = round(pct_from_52w_high(d.close), 1)
        low_dist = round(pct_from_52w_low(d.close), 1)

        if len(signals) >= cfg["min_signals"]:  # Score threshold
            candidates.append({"symbol": symbol, "close": d.close.iloc[-1], "distance_from_52week_high": high_dist, "distance_from_52week_low": low_dist,
                               "score": len(signals), "signals": " | ". join(signals)
                               })
    out = (pd.DataFrame(candidates)).sort_values("score", ascending=False)
    # .head(cfg["top_k"]) -> if want to filter top few stocks

    out.to_csv(
        f"data/scans/{datetime.now():%Y-%m-%d}_scan.csv", index=False)
    return out


def load_universe(path="config/sp500.csv") -> list[str]:
    return pd.read_csv(path)["symbol"].to_list()


if __name__ == "__main__":
    cfg = load_config()
    out = scan(load_universe(), cfg)
    print("scanned succesfully")
