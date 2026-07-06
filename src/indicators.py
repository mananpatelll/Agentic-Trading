import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (0-100)."""
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
    """Simple moving average over `period` bars."""
    return close.rolling(period).mean()


def volume_spike(volume: pd.Series, period: int = 20, mult: float = 2.0) -> bool:
    """True if latest volume >= `mult` * average volume of previous `period` bars."""
    avg = volume.rolling(period).mean().shift(1)
    return bool(volume.iloc[-1] >= mult * avg.iloc[-1])


def pct_from_52w_high(close: pd.Series) -> float:
    """% distance from 252-day high (negative = below high)."""
    high_52w = close.rolling(252, min_periods=200).max()
    # e.g. -3.2 = 3.2 % below
    return float((close.iloc[-1] / high_52w.iloc[-1] - 1) * 100)


def pct_from_52w_low(close: pd.Series) -> float:
    """% distance from 252-day low (positive = above low)."""
    low_52w = close.rolling(252, min_periods=200).min()
    return float((close.iloc[-1] / low_52w.iloc[-1] - 1) * 100)


def avg_dollar_volume(close: pd.Series, volume: pd.Series, period: int = 20) -> float:
    """Average daily dollar volume (close * volume) over `period` days."""
    return float((close * volume).rolling(period).mean().iloc[-1])
