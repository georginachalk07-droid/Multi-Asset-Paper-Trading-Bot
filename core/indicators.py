"""Shared technical indicators.

All functions take/return pandas objects aligned to the input index, with
NaN for warm-up periods, so strategies can simply skip bars where the
indicators are not yet defined.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (pandas default smoothing, span = period)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def zscore(series: pd.Series, period: int) -> pd.Series:
    """Z-score of the series against its own rolling SMA and standard deviation."""
    mean = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    # Guard against zero variance (flat price) producing infinities.
    return (series - mean) / std.replace(0.0, np.nan)


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range from a DataFrame with high/low/close columns."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range using Wilder's smoothing (the conventional ATR)."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rolling_high(series: pd.Series, period: int) -> pd.Series:
    """Highest value of the PRIOR `period` bars (excludes the current bar)."""
    return series.shift(1).rolling(period).max()
