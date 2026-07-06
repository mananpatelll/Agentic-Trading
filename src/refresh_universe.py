# src/refresh_universe.py
"""Fetch current S&P 500 constituents from Wikipedia and cache to CSV.
Run manually ~monthly."""
import pandas as pd
import requests
from io import StringIO

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT_PATH = "config/sp500.csv"


def refresh_universe() -> pd.DataFrame:
    resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    df = pd.read_html(StringIO(resp.text))[0]
    # header has been "Symbol" or "Ticker symbol" across revisions — handle both
    sym_col = "Symbol" if "Symbol" in df.columns else "Ticker symbol"
    df = df.rename(columns={sym_col: "symbol", "Security": "name",
                            "GICS Sector": "sector"})[["symbol", "name", "sector"]]

    df["symbol"] = df["symbol"].str.strip()

    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} symbols to {OUT_PATH}")
    return df


if __name__ == "__main__":
    refresh_universe()
