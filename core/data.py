"""Market data layer: candle fetching and on-disk caching.

Handles both halves of the Alpaca data API (stocks and crypto) and returns a
uniform DataFrame: UTC DatetimeIndex named 'timestamp' with columns
open/high/low/close/volume. Bar timestamps are the bar START time, matching
Alpaca's convention; consumers that need the close time add the bar duration.

Data limitations (documented per the brief):
- Equities on the free plan use the IEX feed, which is a single-venue subset
  of consolidated volume. Prices track the market closely but volume is lower
  than SIP. Set data.stock_feed to 'sip' in config.yaml if you subscribe.
- Alpaca crypto history is reliable from roughly 2021 onwards, comfortably
  covering a 2-year backtest for BTC/USD.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from core.config import PROJECT_ROOT, is_crypto

TIMEFRAME_MINUTES = {"15Min": 15, "1Hour": 60, "4Hour": 240}


def timeframe_delta(timeframe: str) -> timedelta:
    return timedelta(minutes=TIMEFRAME_MINUTES[timeframe])


def _to_alpaca_timeframe(timeframe: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    mapping = {
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
    }
    return mapping[timeframe]


class DataFeed:
    """Fetches candles from Alpaca with a simple CSV cache for backtests."""

    def __init__(self, data_config: dict):
        load_dotenv()
        self.feed = data_config.get("stock_feed", "iex")
        self.cache_dir = PROJECT_ROOT / data_config.get("cache_dir", "data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._stock_client = None
        self._crypto_client = None

    # ------------------------------------------------------------------
    # Lazy clients (so unit tests and synthetic runs never need API keys)
    # ------------------------------------------------------------------
    def _stocks(self):
        if self._stock_client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._stock_client = StockHistoricalDataClient(
                os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
            )
        return self._stock_client

    def _crypto(self):
        if self._crypto_client is None:
            from alpaca.data.historical import CryptoHistoricalDataClient

            self._crypto_client = CryptoHistoricalDataClient(
                os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
            )
        return self._crypto_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def historical_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime, use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch candles for [start, end), cached on disk for repeat backtests."""
        cache_file = self._cache_path(symbol, timeframe, start, end)
        if use_cache and cache_file.exists():
            return self._load_cache(cache_file)

        df = self._fetch(symbol, timeframe, start, end)
        if use_cache and not df.empty:
            df.to_csv(cache_file)
        return df

    def recent_bars(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Most recent `limit` candles, uncached (live mode and connection checks)."""
        # Generous window: equities only print bars in market hours.
        minutes = TIMEFRAME_MINUTES[timeframe] * limit * 6
        start = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        df = self._fetch(symbol, timeframe, start, datetime.now(timezone.utc))
        return df.tail(limit)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fetch(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        if is_crypto(symbol):
            from alpaca.data.requests import CryptoBarsRequest

            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=_to_alpaca_timeframe(timeframe),
                start=start,
                end=end,
            )
            bars = self._crypto().get_crypto_bars(request)
        else:
            from alpaca.data.requests import StockBarsRequest

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=_to_alpaca_timeframe(timeframe),
                start=start,
                end=end,
                feed=self.feed,
            )
            bars = self._stocks().get_stock_bars(request)

        df = bars.df
        if df.empty:
            raise RuntimeError(f"Alpaca returned no bars for {symbol} {timeframe} {start}..{end}")
        # alpaca-py returns a (symbol, timestamp) MultiIndex; flatten it.
        df = df.reset_index(level="symbol", drop=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    def _cache_path(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
        safe_symbol = symbol.replace("/", "-")
        return self.cache_dir / (
            f"{safe_symbol}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        )

    @staticmethod
    def _load_cache(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
