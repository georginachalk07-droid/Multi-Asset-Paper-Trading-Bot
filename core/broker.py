"""Alpaca broker wrapper — PAPER TRADING ONLY.

Wraps alpaca-py so the rest of the codebase never touches the SDK directly.
The constructor refuses to run unless the trading endpoint is the paper URL.

Run `python -m core.broker` to verify the connection and pull a sample
candle for every instrument (build-order step 1 check).
"""

import os

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
)
from dotenv import load_dotenv

from core.config import is_crypto

PAPER_URL_FRAGMENT = "paper-api.alpaca.markets"


class PaperGuardError(RuntimeError):
    """Raised when anything other than the paper endpoint is detected."""


class Broker:
    """Thin, paper-only wrapper around the Alpaca trading API."""

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
                "Copy .env.example to .env and add your PAPER keys."
            )

        # paper=True forces the paper endpoint; the assertion below is a
        # belt-and-braces guard in case of SDK changes or monkey-patching.
        # alpaca-py stores the endpoint as a BaseURL enum, so unwrap .value.
        self.client = TradingClient(api_key, secret_key, paper=True)
        raw_url = getattr(self.client, "_base_url", "")
        base_url = str(getattr(raw_url, "value", raw_url))
        if PAPER_URL_FRAGMENT not in base_url:
            raise PaperGuardError(
                f"Refusing to run: trading endpoint '{base_url}' is not the "
                f"Alpaca paper URL. This bot is paper-only."
            )

    # ------------------------------------------------------------------
    # Account and positions
    # ------------------------------------------------------------------
    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def positions(self) -> list:
        return self.client.get_all_positions()

    def open_orders(self) -> list:
        return self.client.get_orders()

    # ------------------------------------------------------------------
    # Orders. Entries are market orders; stops are REAL broker-side orders,
    # never software-only (risk rule 1).
    # ------------------------------------------------------------------
    def submit_market(self, symbol: str, qty: float, side: int):
        order_side = OrderSide.BUY if side > 0 else OrderSide.SELL
        tif = TimeInForce.GTC if is_crypto(symbol) else TimeInForce.DAY
        request = MarketOrderRequest(symbol=symbol, qty=qty, side=order_side, time_in_force=tif)
        return self.client.submit_order(request)

    def submit_stop(self, symbol: str, qty: float, position_side: int, stop_price: float):
        """Place the protective stop for an open position.

        Alpaca rejects plain stop orders for crypto, so BTC uses a stop-limit
        with the limit set slightly through the stop to keep fill odds high.
        """
        exit_side = OrderSide.SELL if position_side > 0 else OrderSide.BUY
        if is_crypto(symbol):
            # 0.5% through the stop: behaves like a stop-market in practice.
            cushion = 0.995 if position_side > 0 else 1.005
            request = StopLimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=exit_side,
                time_in_force=TimeInForce.GTC,
                stop_price=round(stop_price, 2),
                limit_price=round(stop_price * cushion, 2),
            )
        else:
            request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=exit_side,
                time_in_force=TimeInForce.GTC,
                stop_price=round(stop_price, 2),
            )
        return self.client.submit_order(request)

    def cancel_order(self, order_id: str) -> None:
        self.client.cancel_order_by_id(order_id)

    def close_position(self, symbol: str):
        # Alpaca's REST path for crypto positions uses the slash-free form.
        return self.client.close_position(symbol.replace("/", ""))

    def close_all_positions(self):
        return self.client.close_all_positions(cancel_orders=True)


def _connection_check() -> None:
    """Confirm the paper connection and one candle pull per instrument."""
    from core.config import load_config
    from core.data import DataFeed

    config = load_config()
    broker = Broker()
    account = broker.client.get_account()
    print(f"Connected to PAPER account {account.account_number}: equity ${float(account.equity):,.2f}")

    feed = DataFeed(config["data"])
    instruments = [
        ("SPY", "15Min"), ("QQQ", "15Min"), ("BTC/USD", "1Hour"),
        ("GLD", "4Hour"), ("USO", "4Hour"),
    ]
    for symbol, timeframe in instruments:
        bars = feed.recent_bars(symbol, timeframe, limit=3)
        last = bars.iloc[-1]
        print(f"{symbol:8s} {timeframe:6s} last bar {bars.index[-1]}  close={last['close']:.2f}")


if __name__ == "__main__":
    _connection_check()
