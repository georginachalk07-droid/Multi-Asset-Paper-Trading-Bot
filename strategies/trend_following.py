"""Strategy 3: Trend following on GLD and USO, 4-hour candles, long-only.

- Enter when the 20-period EMA crosses above the 50-period EMA with the
  close above both.
- Exit when the 20 EMA crosses back below the 50 EMA, or on the hard stop.

Deliberately slow: few trades are expected and that is by design.
"""

import pandas as pd

from core import indicators
from core.risk import OpenPosition
from strategies.base import Signal, Strategy


class TrendFollowing(Strategy):
    name = "trend_following"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = indicators.ema(df["close"], self.params["fast_ema"])
        df["ema_slow"] = indicators.ema(df["close"], self.params["slow_ema"])
        df["atr"] = indicators.atr(df, 14)
        return df

    def _crossed_up(self, df: pd.DataFrame, i: int) -> bool:
        if i == 0:
            return False
        prev, curr = df.iloc[i - 1], df.iloc[i]
        if pd.isna(prev["ema_slow"]) or pd.isna(curr["ema_slow"]):
            return False
        return prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]

    def _crossed_down(self, df: pd.DataFrame, i: int) -> bool:
        if i == 0:
            return False
        prev, curr = df.iloc[i - 1], df.iloc[i]
        if pd.isna(prev["ema_slow"]) or pd.isna(curr["ema_slow"]):
            return False
        return prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]

    def entry_signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        row = df.iloc[i]
        if self._crossed_up(df, i) and row["close"] > row["ema_fast"] and row["close"] > row["ema_slow"]:
            return Signal(
                "enter_long",
                f"EMA{self.params['fast_ema']} crossed above EMA{self.params['slow_ema']} "
                f"with close above both",
            )
        return None

    def exit_signal(self, df: pd.DataFrame, i: int, position: OpenPosition) -> Signal | None:
        if self._crossed_down(df, i):
            return Signal(
                "exit",
                f"EMA{self.params['fast_ema']} crossed below EMA{self.params['slow_ema']}",
            )
        return None
