"""Portfolio backtest engine.

Why a hand-rolled bar-loop rather than backtrader or a vectorised backtest:
the portfolio-level risk rules (correlation filter, max concurrent positions,
total open-risk cap, daily kill switch) span instruments AND strategies on
mixed timeframes. Vectorised per-instrument backtests cannot model them, and
backtrader makes cross-strategy portfolio constraints awkward while adding an
unmaintained dependency. A chronological event loop over pandas DataFrames is
~30k bars for 2 years across all five instruments — fast, transparent, and it
exercises exactly the same RiskManager the live loop will use.

Execution model (deliberately conservative):
- Signals are evaluated on the bar close; fills happen at that close price
  with per-side slippage+fees applied against the trade.
- Protective stops are checked against the NEXT bars' lows/highs. A gap
  through the stop fills at the bar's open (worse than the stop), never at
  the stop itself.
- Stops only ever tighten; trailing logic cannot widen a stop.
- If the kill switch trips, everything is flattened at the last seen price
  and trading halts. The backtest auto-resumes the next day to keep the
  simulation going; in live mode the restart is manual (per the brief).
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from core.config import cost_pct_for
from core.data import timeframe_delta
from core.risk import LONG, SHORT, OpenPosition, RiskManager
from strategies.base import Strategy


@dataclass
class Trade:
    symbol: str
    strategy: str
    side: int
    qty: float
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    pnl: float
    r_multiple: float
    exit_reason: str


@dataclass
class BlockedSignal:
    time: datetime
    symbol: str
    strategy: str
    action: str
    reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade]
    blocked: list[BlockedSignal]
    kill_switch_events: list[datetime]
    initial_equity: float

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else self.initial_equity


class BacktestEngine:
    def __init__(
        self,
        strategies: list[Strategy],
        data: dict[str, pd.DataFrame],
        risk: RiskManager,
        costs_config: dict,
        initial_equity: float,
    ):
        self.strategies = {s.symbol: s for s in strategies}
        self.data = {sym: self.strategies[sym].prepare(df) for sym, df in data.items()}
        self.risk = risk
        self.costs = costs_config
        self.initial_equity = initial_equity

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        events = self._build_event_queue()

        cash = self.initial_equity
        positions: dict[str, OpenPosition] = {}
        initial_risks: dict[str, float] = {}   # committed risk at entry, for R multiples
        last_price: dict[str, float] = {}
        trades: list[Trade] = []
        blocked: list[BlockedSignal] = []
        kill_events: list[datetime] = []
        curve_times: list[datetime] = []
        curve_values: list[float] = []
        current_day = None
        day_start_equity = self.initial_equity

        def mark_equity() -> float:
            value = cash
            for pos in positions.values():
                value += pos.side * pos.qty * last_price[pos.symbol]
            return value

        def close_position(symbol: str, ts: datetime, raw_price: float, reason: str) -> None:
            nonlocal cash
            pos = positions.pop(symbol)
            cost = cost_pct_for(symbol, self.costs)
            # Exiting a long sells (price shaded down); exiting a short buys (up).
            fill = raw_price * (1 - cost) if pos.side == LONG else raw_price * (1 + cost)
            cash += pos.side * pos.qty * fill
            pnl = pos.side * pos.qty * (fill - pos.entry_price)
            risk0 = initial_risks.pop(symbol)
            trades.append(
                Trade(
                    symbol=symbol, strategy=pos.strategy, side=pos.side, qty=pos.qty,
                    entry_time=pos.entry_time, entry_price=pos.entry_price,
                    exit_time=ts, exit_price=fill, pnl=pnl,
                    r_multiple=pnl / risk0 if risk0 > 0 else 0.0,
                    exit_reason=reason,
                )
            )

        for ts, symbol, i in events:
            df = self.data[symbol]
            strategy = self.strategies[symbol]
            bar = df.iloc[i]
            last_price[symbol] = float(bar["close"])

            # ---- new trading day: reset kill-switch baseline -------------
            if current_day != ts.date():
                current_day = ts.date()
                day_start_equity = mark_equity()
                if self.risk.halted:
                    self.risk.reset_kill_switch()  # auto-resume in backtest only

            # ---- manage an open position in this symbol -----------------
            if symbol in positions:
                pos = positions[symbol]
                stop_hit = (
                    bar["low"] <= pos.stop_price if pos.side == LONG
                    else bar["high"] >= pos.stop_price
                )
                if stop_hit:
                    # Gap through the stop fills at the open, not the stop.
                    if pos.side == LONG:
                        raw = min(float(bar["open"]), pos.stop_price)
                    else:
                        raw = max(float(bar["open"]), pos.stop_price)
                    close_position(symbol, ts, raw, "stop hit")
                else:
                    signal = strategy.exit_signal(df, i, pos)
                    if signal:
                        close_position(symbol, ts, float(bar["close"]), signal.reason)
                    else:
                        new_stop = strategy.update_stop(df, i, pos)
                        if new_stop is not None:
                            # Tighten only — never widen a stop.
                            if pos.side == LONG:
                                pos.stop_price = max(pos.stop_price, new_stop)
                            else:
                                pos.stop_price = min(pos.stop_price, new_stop)

            # ---- daily kill switch --------------------------------------
            equity = mark_equity()
            if not self.risk.halted and self.risk.check_kill_switch(equity, day_start_equity):
                kill_events.append(ts)
                for sym in list(positions.keys()):
                    close_position(sym, ts, last_price[sym], "kill switch flatten")
                equity = mark_equity()

            # ---- entries (only if flat at the START of this event) ------
            # Signals are still evaluated while halted so the kill-switch
            # refusal is logged, per "log every blocked signal".
            if symbol not in positions:
                signal = strategy.entry_signal(df, i)
                if signal:
                    side = LONG if signal.action == "enter_long" else SHORT
                    atr_value = float(bar["atr"]) if pd.notna(bar["atr"]) else None
                    sized = self.risk.size_position(symbol, equity, float(bar["close"]), atr_value)
                    if sized is None:
                        blocked.append(BlockedSignal(ts, symbol, strategy.name, signal.action,
                                                     "position sizing produced no valid quantity"))
                    else:
                        allowed, reason = self.risk.can_open(
                            symbol, side, sized.risk_amount, equity, list(positions.values())
                        )
                        if not allowed:
                            blocked.append(BlockedSignal(ts, symbol, strategy.name, signal.action, reason))
                        else:
                            cost = cost_pct_for(symbol, self.costs)
                            raw = float(bar["close"])
                            fill = raw * (1 + cost) if side == LONG else raw * (1 - cost)
                            stop = self.risk.stop_price(side, fill, atr_value)
                            cash -= side * sized.qty * fill
                            pos = OpenPosition(
                                symbol=symbol, side=side, qty=sized.qty,
                                entry_price=fill, stop_price=stop, entry_time=ts,
                                strategy=strategy.name, highest_close=raw,
                            )
                            positions[symbol] = pos
                            initial_risks[symbol] = pos.open_risk

            curve_times.append(ts)
            curve_values.append(mark_equity())

        # ---- end of data: flatten whatever is still open -----------------
        if events:
            final_ts = events[-1][0]
            for sym in list(positions.keys()):
                close_position(sym, final_ts, last_price[sym], "end of backtest")
            curve_times.append(final_ts)
            curve_values.append(mark_equity())

        curve = pd.Series(curve_values, index=pd.DatetimeIndex(curve_times), name="equity")
        curve = curve[~curve.index.duplicated(keep="last")]
        return BacktestResult(
            equity_curve=curve, trades=trades, blocked=blocked,
            kill_switch_events=kill_events, initial_equity=self.initial_equity,
        )

    # ------------------------------------------------------------------
    def _build_event_queue(self) -> list[tuple]:
        """All (bar_end_time, symbol, row_index) tuples in chronological order.

        Bars are processed at their END time, which is when the close is known.
        """
        events = []
        for symbol, df in self.data.items():
            delta = timeframe_delta(self.strategies[symbol].timeframe)
            for i, ts in enumerate(df.index):
                events.append((ts + delta, symbol, i))
        events.sort(key=lambda e: (e[0], e[1]))
        return events
