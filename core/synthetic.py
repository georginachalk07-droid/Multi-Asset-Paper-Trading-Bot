"""Synthetic OHLCV generator — pipeline validation ONLY.

Used to exercise the full backtest stack (indicators, strategies, risk layer,
engine, reporting) when Alpaca API keys are not available. Output is seeded
and deterministic. Results on synthetic data say NOTHING about real-market
performance and every report generated from it is labelled accordingly.
"""

import numpy as np
import pandas as pd

# Rough starting prices so position sizing operates at realistic magnitudes.
START_PRICES = {"SPY": 520.0, "QQQ": 450.0, "BTC/USD": 65000.0, "GLD": 215.0, "USO": 75.0}


def generate(symbol: str, timeframe: str, start: str, end: str, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed if seed is not None else abs(hash(symbol)) % 2**32)
    index = _bar_index(symbol, timeframe, start, end)
    n = len(index)

    # Regime-switching drift + volatility clustering, with a mean-reverting
    # intraday component so each strategy type has structure to find.
    regime_len = max(50, n // 24)
    drift = np.repeat(rng.normal(0.0, 0.0004, size=n // regime_len + 1), regime_len)[:n]
    vol = np.abs(np.repeat(rng.normal(0.004, 0.0015, size=n // regime_len + 1), regime_len)[:n]) + 0.001

    noise = rng.normal(0.0, 1.0, n)
    ou = np.zeros(n)
    for t in range(1, n):  # mean-reverting overlay
        ou[t] = 0.95 * ou[t - 1] + 0.3 * noise[t]
    log_price = np.cumsum(drift + vol * noise) + 0.002 * ou
    close = START_PRICES.get(symbol, 100.0) * np.exp(log_price)

    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, 0.5, n)) * vol * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    base_volume = 1e6 if "/" not in symbol else 500.0
    volume = base_volume * np.exp(rng.normal(0.0, 0.5, n))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _bar_index(symbol: str, timeframe: str, start: str, end: str) -> pd.DatetimeIndex:
    """Bar START timestamps (UTC) respecting each market's trading calendar."""
    if "/" in symbol:  # crypto: 24/7
        freq = {"1Hour": "1h", "4Hour": "4h", "15Min": "15min"}[timeframe]
        return pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")

    # Equities: weekday US regular session, stamped in exchange time.
    days = pd.bdate_range(start, end, tz="America/New_York")
    session_times = {
        "15Min": pd.timedelta_range("9h30min", "15h45min", freq="15min"),
        "1Hour": pd.timedelta_range("9h30min", "15h30min", freq="1h"),
        "4Hour": pd.timedelta_range("9h30min", "13h30min", freq="4h"),
    }[timeframe]
    stamps = [day + offset for day in days for offset in session_times]
    return pd.DatetimeIndex(stamps).tz_convert("UTC")
