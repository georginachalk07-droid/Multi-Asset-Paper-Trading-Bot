"""Backtest engine behaviour tests using a deterministic toy strategy."""

import pandas as pd
import pytest

from core.engine import BacktestEngine
from core.risk import LONG, OpenPosition, RiskManager
from strategies.base import Signal, Strategy

RISK_CONFIG = {
    "risk_per_trade_pct": 1.0,
    "stop_atr_multiple": 1.5,
    "atr_period": 14,
    "max_concurrent_positions": 4,
    "max_total_open_risk_pct": 3.0,
    "daily_kill_switch_pct": 3.0,
    "risk_on_group": ["SPY", "QQQ", "BTC/USD"],
    "risk_on_max_longs": 2,
    "max_position_notional_pct": 50.0,
}

COSTS = {"etf_cost_pct_per_side": 0.0, "crypto_cost_pct_per_side": 0.0}


class BuyBarTwo(Strategy):
    """Enters long at bar 2 and never signals an exit (stop must do the work)."""

    name = "toy"

    def prepare(self, df):
        df = df.copy()
        df["atr"] = 2.0
        return df

    def entry_signal(self, df, i):
        return Signal("enter_long", "scripted entry") if i == 2 else None

    def exit_signal(self, df, i, position):
        return None


def make_df(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2025-01-06 14:30", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=index,
    )


def run_engine(closes, strategy_cls=BuyBarTwo):
    df = make_df(closes)
    strategy = strategy_cls("SPY", "1Hour", {})
    engine = BacktestEngine([strategy], {"SPY": df}, RiskManager(RISK_CONFIG), COSTS, 100_000.0)
    return engine.run()


def test_stop_is_hit_at_stop_price():
    # Entry at 100 (bar 2 close); stop = 100 - 3 = 97. Bar at 96 pierces it.
    result = run_engine([100, 100, 100, 99, 96, 96, 96])
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop hit"
    # Gap: bar open (96) is below the stop (97), so the fill is the open.
    assert trade.exit_price == pytest.approx(96.0)


def test_stop_loss_close_to_one_percent():
    result = run_engine([100, 100, 100, 99, 97.2, 96.9])
    trade = result.trades[0]
    # Stop at 97, intrabar low 96.7 <= 97, no gap (open 97.2 > 97): fill at 97.
    assert trade.exit_price == pytest.approx(97.0)
    assert trade.pnl == pytest.approx(-1000.0, rel=0.01)  # 1% of 100k


def test_open_position_flattened_at_end():
    result = run_engine([100, 100, 100, 101, 102])
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "end of backtest"
    assert result.trades[0].pnl > 0


def test_kill_switch_flattens_and_halts():
    class BuyEveryBar(BuyBarTwo):
        def entry_signal(self, df, i):
            return Signal("enter_long", "scripted") if i >= 2 else None

    # A brutal collapse: equity loss far beyond 3% in one day.
    result = run_engine([100, 100, 100, 90, 85, 85, 85], strategy_cls=BuyEveryBar)
    assert len(result.kill_switch_events) >= 1
    # After the halt, the scripted re-entries must be blocked and logged.
    assert any("kill switch" in b.reason for b in result.blocked)


def test_trailing_stop_only_tightens():
    class TrailWide(BuyBarTwo):
        def update_stop(self, df, i, position):
            return 50.0  # absurdly wide trail must be ignored

    result = run_engine([100, 100, 100, 99, 96.5, 96.5], strategy_cls=TrailWide)
    trade = result.trades[0]
    assert trade.exit_reason == "stop hit"          # original 97 stop still active
    assert trade.exit_price == pytest.approx(96.5)  # gap fill at open below stop


def test_equity_curve_starts_at_initial_equity():
    result = run_engine([100, 100, 100, 100, 100])
    assert result.equity_curve.iloc[0] == pytest.approx(100_000.0)
