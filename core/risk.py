"""Risk management — the non-negotiable layer.

Every rule from the project brief is enforced here, outside the strategy
modules, so a strategy bug can never bypass them:

1. Hard stop: 1% of equity per trade, placed as a real stop order.
2. ATR-based sizing: qty = (risk %) x equity / (stop_atr_multiple x ATR).
3. Correlation filter on the risk-on group (SPY, QQQ, BTC/USD).
4. Max concurrent positions.
5. Daily kill switch at -3% from the day's starting equity.
6. Max total open risk of 3% of equity.

The same RiskManager is used by the backtester and (later) the live loop.

Definition used for "open risk": the committed risk of a position, i.e.
qty x max(0, distance from entry price to current stop). When a trailing
stop moves past break-even the position no longer contributes open risk.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime

from core.config import is_crypto

LONG = 1
SHORT = -1


@dataclass
class OpenPosition:
    """Broker-agnostic snapshot of one open position, as the risk layer sees it."""

    symbol: str
    side: int            # LONG (+1) or SHORT (-1)
    qty: float
    entry_price: float
    stop_price: float
    entry_time: datetime | None = None
    strategy: str = ""
    # Mutable state used by trailing-stop strategies.
    highest_close: float = field(default=0.0)

    @property
    def open_risk(self) -> float:
        """Money lost if the current stop is hit, never below zero."""
        if self.side == LONG:
            return self.qty * max(0.0, self.entry_price - self.stop_price)
        return self.qty * max(0.0, self.stop_price - self.entry_price)


@dataclass
class SizedOrder:
    """Result of position sizing. The stop itself is derived from the actual
    fill price via RiskManager.stop_price() once the entry executes."""

    qty: float
    risk_amount: float


class RiskManager:
    """Stateless rule checks plus the stateful daily kill switch."""

    def __init__(self, risk_config: dict):
        self.risk_per_trade_pct = float(risk_config["risk_per_trade_pct"])
        self.stop_atr_multiple = float(risk_config["stop_atr_multiple"])
        self.max_concurrent_positions = int(risk_config["max_concurrent_positions"])
        self.max_total_open_risk_pct = float(risk_config["max_total_open_risk_pct"])
        self.daily_kill_switch_pct = float(risk_config["daily_kill_switch_pct"])
        self.risk_on_group = set(risk_config["risk_on_group"])
        self.risk_on_max_longs = int(risk_config["risk_on_max_longs"])
        self.max_position_notional_pct = float(risk_config["max_position_notional_pct"])

        # Kill-switch state. Once tripped, trading halts until manually reset.
        self.halted = False
        self.halt_reason = ""

    # ------------------------------------------------------------------
    # Position sizing (rules 1 and 2)
    # ------------------------------------------------------------------
    def size_position(self, symbol: str, equity: float, price: float, atr_value: float) -> SizedOrder | None:
        """Size a trade so a 1.5x ATR stop loses exactly risk_per_trade_pct of equity.

        Returns None (with no exception) when a valid size cannot be produced,
        e.g. ATR not yet defined, or an ETF position rounding down to zero shares.
        """
        if equity <= 0 or price <= 0 or atr_value is None or not atr_value > 0:
            return None

        stop_distance = self.stop_atr_multiple * atr_value
        risk_amount = equity * self.risk_per_trade_pct / 100.0
        qty = risk_amount / stop_distance

        # Notional safety valve: low-volatility instruments would otherwise be
        # sized into enormous positions; crypto must also fit available funds.
        max_notional = equity * self.max_position_notional_pct / 100.0
        if qty * price > max_notional:
            qty = max_notional / price

        if is_crypto(symbol):
            qty = math.floor(qty * 1e6) / 1e6  # Alpaca accepts fractional crypto
        else:
            # Alpaca does not accept stop orders on fractional equity shares,
            # so ETFs are sized in whole shares.
            qty = float(math.floor(qty))

        if qty <= 0:
            return None

        return SizedOrder(qty=qty, risk_amount=qty * stop_distance)

    def stop_price(self, side: int, entry_price: float, atr_value: float) -> float:
        """Initial hard stop: 1.5x ATR away from entry, against the position."""
        distance = self.stop_atr_multiple * atr_value
        return entry_price - distance if side == LONG else entry_price + distance

    # ------------------------------------------------------------------
    # Pre-trade checks (rules 3, 4, 6 and the kill switch)
    # ------------------------------------------------------------------
    def can_open(
        self,
        symbol: str,
        side: int,
        new_risk_amount: float,
        equity: float,
        open_positions: list[OpenPosition],
    ) -> tuple[bool, str]:
        """Return (allowed, reason). Reason explains every refusal for the journal."""
        if self.halted:
            return False, f"kill switch active: {self.halt_reason}"

        if any(p.symbol == symbol for p in open_positions):
            return False, f"already holding a position in {symbol}"

        if len(open_positions) >= self.max_concurrent_positions:
            return False, (
                f"max concurrent positions reached "
                f"({len(open_positions)}/{self.max_concurrent_positions})"
            )

        # Correlation filter: block the third long in the risk-on group.
        if side == LONG and symbol in self.risk_on_group:
            group_longs = [
                p.symbol
                for p in open_positions
                if p.symbol in self.risk_on_group and p.side == LONG
            ]
            if len(group_longs) >= self.risk_on_max_longs:
                return False, (
                    f"correlation filter: {len(group_longs)} risk-on longs already open "
                    f"({', '.join(sorted(group_longs))})"
                )

        # Total open risk cap.
        current_risk = sum(p.open_risk for p in open_positions)
        max_risk = equity * self.max_total_open_risk_pct / 100.0
        if current_risk + new_risk_amount > max_risk + 1e-9:
            return False, (
                f"total open risk cap: {current_risk + new_risk_amount:.2f} would exceed "
                f"{max_risk:.2f} ({self.max_total_open_risk_pct}% of equity)"
            )

        return True, "ok"

    # ------------------------------------------------------------------
    # Daily kill switch (rule 5)
    # ------------------------------------------------------------------
    def check_kill_switch(self, equity: float, day_start_equity: float) -> bool:
        """Trip the halt if equity has dropped daily_kill_switch_pct from the day's start.

        Returns True when the switch trips (caller must flatten all positions).
        """
        if day_start_equity <= 0:
            return False
        drawdown_pct = (day_start_equity - equity) / day_start_equity * 100.0
        if drawdown_pct >= self.daily_kill_switch_pct:
            self.halted = True
            self.halt_reason = (
                f"daily loss {drawdown_pct:.2f}% breached the "
                f"{self.daily_kill_switch_pct}% kill switch"
            )
            return True
        return False

    def reset_kill_switch(self) -> None:
        """Manual restart. In live mode this is a deliberate human action."""
        self.halted = False
        self.halt_reason = ""
