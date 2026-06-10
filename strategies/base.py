"""Strategy interface.

Strategies are pure signal generators. They see candles and an optional open
position, and say what they would like to do. They NEVER size positions,
place stops, or override risk rules — that is the risk layer's job. The same
classes drive both the backtester and the live loop.

Bar-indexing convention: a signal is evaluated on the CLOSE of bar i, using
only data up to and including bar i. Index timestamps are bar start times.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from core.risk import OpenPosition


@dataclass
class Signal:
    action: str          # 'enter_long' | 'enter_short' | 'exit'
    reason: str


class Strategy(ABC):
    """One strategy instance trades exactly one symbol on one timeframe."""

    name: str = "base"

    def __init__(self, symbol: str, timeframe: str, params: dict):
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns (must include 'atr' for position sizing)."""

    @abstractmethod
    def entry_signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        """Entry decision at the close of bar i, when flat."""

    @abstractmethod
    def exit_signal(self, df: pd.DataFrame, i: int, position: OpenPosition) -> Signal | None:
        """Exit decision at the close of bar i, when holding `position`."""

    def update_stop(self, df: pd.DataFrame, i: int, position: OpenPosition) -> float | None:
        """Return a new stop price if the strategy trails its stop, else None.

        The risk layer only ever tightens stops; a returned value further from
        price than the current stop is ignored by the engine.
        """
        return None
