import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI values for given stock"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period,
                        adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period,
                        adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def sma(close: pd.Series, period: int) -> pd.Series:
    """calculate SMA"""
    pass


def near_52(close: pd.Series) -> float:
    """calculate 52week high """
    pass


def low_52w(close: pd.Series) -> float:
    """calculate 52 week low"""
    pass
