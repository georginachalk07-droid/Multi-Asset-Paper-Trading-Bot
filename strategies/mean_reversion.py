"""Strategy 1: Mean reversion on SPY and QQQ, 15-minute candles.

- Enter long when the z-score of close vs its 20-period SMA drops below -2.0;
  enter short above +2.0 (configurable; Alpaca paper accounts allow ETF shorts).
- Exit when the z-score reverts through 0, at the session-flat cutoff, or on
  the hard stop (handled by the risk layer).
- Regime filter: no entries when ATR(14) exceeds 1.5x its own 50-period
  average — do not fade strong trends or news candles.
- Entries only during the US regular session, flat 10 minutes before the
  close. The window is enforced in exchange time (America/New_York) rather
  than UK clock time, because UK and US daylight saving change on different
  dates and a fixed UK window would drift by an hour twice a year.
"""

from datetime import time

import pandas as pd

from core import indicators
from core.risk import LONG, OpenPosition
from strategies.base import Signal, Strategy

EXCHANGE_TZ = "America/New_York"


def _parse_hhmm(value: str) -> time:
    hours, minutes = value.split(":")
    return time(int(hours), int(minutes))


class MeanReversion(Strategy):
    name = "mean_reversion"

    def __init__(self, symbol: str, timeframe: str, params: dict):
        super().__init__(symbol, timeframe, params)
        self.session_open = _parse_hhmm(params["session_open_et"])
        self.session_flat = _parse_hhmm(params["session_flat_et"])

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["zscore"] = indicators.zscore(df["close"], self.params["sma_period"])
        df["atr"] = indicators.atr(df, self.params["regime_atr_period"])
        df["atr_avg"] = indicators.sma(df["atr"], self.params["regime_atr_avg_period"])
        # Bar END time in exchange-local terms drives the session window.
        bar_end = df.index + pd.Timedelta(minutes=15)
        df["bar_end_et"] = bar_end.tz_convert(EXCHANGE_TZ)
        return df

    # ------------------------------------------------------------------
    def _in_entry_window(self, df: pd.DataFrame, i: int) -> bool:
        end_et = df["bar_end_et"].iloc[i]
        return self.session_open < end_et.time() <= self.session_flat

    def _past_flat_cutoff(self, df: pd.DataFrame, i: int) -> bool:
        end_et = df["bar_end_et"].iloc[i]
        return end_et.time() > self.session_flat or end_et.time() <= self.session_open

    # ------------------------------------------------------------------
    def entry_signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        z = df["zscore"].iloc[i]
        atr_now = df["atr"].iloc[i]
        atr_avg = df["atr_avg"].iloc[i]
        if pd.isna(z) or pd.isna(atr_avg):
            return None
        if not self._in_entry_window(df, i):
            return None
        # Regime filter: stand aside when volatility is elevated.
        if atr_now > self.params["regime_atr_max_ratio"] * atr_avg:
            return None

        threshold = self.params["zscore_entry"]
        if z < -threshold:
            return Signal("enter_long", f"z-score {z:.2f} < -{threshold}")
        if z > threshold and self.params.get("allow_short", False):
            return Signal("enter_short", f"z-score {z:.2f} > +{threshold}")
        return None

    def exit_signal(self, df: pd.DataFrame, i: int, position: OpenPosition) -> Signal | None:
        if self._past_flat_cutoff(df, i):
            return Signal("exit", "session flat cutoff (10 min before close)")
        z = df["zscore"].iloc[i]
        if pd.isna(z):
            return None
        if position.side == LONG and z >= 0:
            return Signal("exit", f"z-score reverted to {z:.2f}")
        if position.side != LONG and z <= 0:
            return Signal("exit", f"z-score reverted to {z:.2f}")
        return None
