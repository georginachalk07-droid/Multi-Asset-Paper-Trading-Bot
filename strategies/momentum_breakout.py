"""Strategy 2: Momentum breakout on BTC/USD, 1-hour candles, long-only, 24/7.

- Enter when the close breaks above the highest high of the prior 24 hourly
  candles AND volume is above its 24-period average.
- Exit on a trailing stop 2x ATR(14) below the highest close since entry, or
  a close below the 12-period EMA — whichever comes first.

Stop reconciliation (flagged in the README): the risk layer mandates the
INITIAL hard stop at 1.5x ATR for sizing, which is tighter than the 2x ATR
trail. The trail therefore only takes over once price has risen enough that
(highest close - 2x ATR) is above the initial stop; the engine never widens
a stop.
"""

import pandas as pd

from core import indicators
from core.risk import OpenPosition
from strategies.base import Signal, Strategy


class MomentumBreakout(Strategy):
    name = "momentum_breakout"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["breakout_level"] = indicators.rolling_high(df["high"], self.params["breakout_lookback"])
        df["volume_avg"] = indicators.sma(df["volume"], self.params["volume_avg_period"])
        df["atr"] = indicators.atr(df, 14)
        df["exit_ema"] = indicators.ema(df["close"], self.params["exit_ema_period"])
        return df

    def entry_signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        row = df.iloc[i]
        if pd.isna(row["breakout_level"]) or pd.isna(row["volume_avg"]) or pd.isna(row["atr"]):
            return None
        if row["close"] > row["breakout_level"] and row["volume"] > row["volume_avg"]:
            return Signal(
                "enter_long",
                f"close {row['close']:.0f} broke 24h high {row['breakout_level']:.0f} "
                f"on above-average volume",
            )
        return None

    def exit_signal(self, df: pd.DataFrame, i: int, position: OpenPosition) -> Signal | None:
        row = df.iloc[i]
        if pd.isna(row["exit_ema"]):
            return None
        if row["close"] < row["exit_ema"]:
            return Signal("exit", f"close {row['close']:.0f} below EMA{self.params['exit_ema_period']}")
        return None

    def update_stop(self, df: pd.DataFrame, i: int, position: OpenPosition) -> float | None:
        """Trail at 2x ATR below the highest close since entry (long-only)."""
        row = df.iloc[i]
        if pd.isna(row["atr"]):
            return None
        position.highest_close = max(position.highest_close, row["close"])
        return position.highest_close - self.params["trail_atr_multiple"] * row["atr"]
