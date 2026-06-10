"""Unit tests for the risk layer — the most important code in the project."""

import pytest

from core.risk import LONG, SHORT, OpenPosition, RiskManager

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


@pytest.fixture
def risk() -> RiskManager:
    return RiskManager(RISK_CONFIG)


def make_position(symbol="SPY", side=LONG, qty=100.0, entry=500.0, stop=494.0) -> OpenPosition:
    return OpenPosition(symbol=symbol, side=side, qty=qty, entry_price=entry, stop_price=stop)


# ----------------------------------------------------------------------
# Rule 2: ATR-based position sizing
# ----------------------------------------------------------------------
class TestSizing:
    def test_etf_sizing_maths(self, risk):
        # 1% of 100k = 1000 risk; stop distance 1.5 x 2.0 = 3.0; qty = 333.33 -> 333 shares
        sized = risk.size_position("SPY", equity=100_000, price=100.0, atr_value=2.0)
        assert sized.qty == 333
        assert sized.risk_amount == pytest.approx(333 * 3.0)

    def test_etf_whole_shares_only(self, risk):
        sized = risk.size_position("SPY", equity=100_000, price=100.0, atr_value=2.0)
        assert sized.qty == int(sized.qty)

    def test_crypto_allows_fractional(self, risk):
        # 1000 risk / (1.5 x 800) = 0.833333 BTC
        sized = risk.size_position("BTC/USD", equity=100_000, price=60_000.0, atr_value=800.0)
        assert sized.qty == pytest.approx(0.833333, abs=1e-6)

    def test_volatile_instrument_gets_smaller_size(self, risk):
        quiet = risk.size_position("SPY", 100_000, 100.0, atr_value=2.0)
        volatile = risk.size_position("SPY", 100_000, 100.0, atr_value=4.0)
        assert volatile.qty < quiet.qty
        # ...but the risked money stays (approximately) constant.
        assert volatile.risk_amount == pytest.approx(quiet.risk_amount, rel=0.01)

    def test_notional_cap_limits_quiet_instruments(self, risk):
        # ATR 0.1 would imply 6666 shares = $2.7M notional; cap = 50% of equity.
        sized = risk.size_position("SPY", 100_000, 400.0, atr_value=0.1)
        assert sized.qty * 400.0 <= 50_000

    def test_zero_or_missing_atr_refused(self, risk):
        assert risk.size_position("SPY", 100_000, 400.0, atr_value=0.0) is None
        assert risk.size_position("SPY", 100_000, 400.0, atr_value=None) is None

    def test_tiny_equity_rounds_to_zero_shares(self, risk):
        # Risk budget too small for even one share's stop distance.
        assert risk.size_position("SPY", equity=100, price=400.0, atr_value=2.0) is None


# ----------------------------------------------------------------------
# Rule 1: hard stop placement
# ----------------------------------------------------------------------
class TestStops:
    def test_long_stop_below_entry(self, risk):
        assert risk.stop_price(LONG, entry_price=100.0, atr_value=2.0) == pytest.approx(97.0)

    def test_short_stop_above_entry(self, risk):
        assert risk.stop_price(SHORT, entry_price=100.0, atr_value=2.0) == pytest.approx(103.0)

    def test_stop_loss_equals_one_percent_of_equity(self, risk):
        equity = 100_000
        sized = risk.size_position("BTC/USD", equity, price=60_000.0, atr_value=800.0)
        stop = risk.stop_price(LONG, 60_000.0, 800.0)
        loss_at_stop = sized.qty * (60_000.0 - stop)
        assert loss_at_stop == pytest.approx(equity * 0.01, rel=1e-4)


# ----------------------------------------------------------------------
# Rule 3: correlation filter on the risk-on group
# ----------------------------------------------------------------------
class TestCorrelationFilter:
    def test_third_risk_on_long_blocked(self, risk):
        open_positions = [make_position("SPY"), make_position("BTC/USD")]
        allowed, reason = risk.can_open("QQQ", LONG, 1000.0, 100_000, open_positions)
        assert not allowed
        assert "correlation filter" in reason

    def test_second_risk_on_long_allowed(self, risk):
        allowed, _ = risk.can_open("QQQ", LONG, 1000.0, 100_000, [make_position("SPY")])
        assert allowed

    def test_short_in_group_not_blocked(self, risk):
        open_positions = [make_position("SPY"), make_position("BTC/USD")]
        allowed, _ = risk.can_open("QQQ", SHORT, 1000.0, 100_000, open_positions)
        assert allowed

    def test_non_group_symbol_not_blocked(self, risk):
        open_positions = [make_position("SPY"), make_position("BTC/USD")]
        allowed, _ = risk.can_open("GLD", LONG, 1000.0, 100_000, open_positions)
        assert allowed

    def test_group_shorts_do_not_count_towards_limit(self, risk):
        open_positions = [make_position("SPY", side=SHORT, stop=506.0), make_position("BTC/USD")]
        allowed, _ = risk.can_open("QQQ", LONG, 1000.0, 100_000, open_positions)
        assert allowed


# ----------------------------------------------------------------------
# Rule 4: max concurrent positions
# ----------------------------------------------------------------------
class TestMaxPositions:
    def test_fifth_position_blocked(self, risk):
        # Four positions whose stops have trailed to break-even (zero open risk),
        # so only the position-count rule can block the fifth.
        open_positions = [
            make_position(sym, stop=500.0) for sym in ("SPY", "QQQ", "GLD", "USO")
        ]
        allowed, reason = risk.can_open("BTC/USD", LONG, 1000.0, 100_000, open_positions)
        assert not allowed
        assert "max concurrent positions" in reason

    def test_duplicate_symbol_blocked(self, risk):
        allowed, reason = risk.can_open("SPY", LONG, 1000.0, 100_000, [make_position("SPY")])
        assert not allowed
        assert "already holding" in reason


# ----------------------------------------------------------------------
# Rule 6: total open risk cap
# ----------------------------------------------------------------------
class TestTotalOpenRisk:
    def test_open_risk_computation(self):
        pos = make_position(qty=100, entry=500.0, stop=494.0)
        assert pos.open_risk == pytest.approx(600.0)

    def test_trailed_stop_reduces_open_risk_to_zero(self):
        pos = make_position(qty=100, entry=500.0, stop=505.0)  # stop above entry
        assert pos.open_risk == 0.0

    def test_short_open_risk(self):
        pos = make_position(side=SHORT, qty=100, entry=500.0, stop=506.0)
        assert pos.open_risk == pytest.approx(600.0)

    def test_fourth_full_risk_position_blocked(self, risk):
        # Three positions each carrying 1% (1000) of open risk = 3% total.
        open_positions = [
            make_position("SPY", qty=100, entry=500, stop=490),
            make_position("GLD", qty=200, entry=215, stop=210),
            make_position("USO", qty=500, entry=75, stop=73),
        ]
        total = sum(p.open_risk for p in open_positions)
        assert total == pytest.approx(3000.0)
        allowed, reason = risk.can_open("BTC/USD", LONG, 1000.0, 100_000, open_positions)
        assert not allowed
        assert "total open risk" in reason

    def test_allowed_again_once_stops_trail_in(self, risk):
        open_positions = [
            make_position("SPY", qty=100, entry=500, stop=502),  # trailed past entry
            make_position("GLD", qty=200, entry=215, stop=210),
            make_position("USO", qty=500, entry=75, stop=73),
        ]
        allowed, _ = risk.can_open("BTC/USD", LONG, 1000.0, 100_000, open_positions)
        assert allowed


# ----------------------------------------------------------------------
# Rule 5: daily kill switch
# ----------------------------------------------------------------------
class TestKillSwitch:
    def test_trips_at_exactly_three_percent(self, risk):
        assert risk.check_kill_switch(equity=97_000, day_start_equity=100_000)
        assert risk.halted

    def test_does_not_trip_above_threshold(self, risk):
        assert not risk.check_kill_switch(equity=97_100, day_start_equity=100_000)
        assert not risk.halted

    def test_blocks_all_entries_when_halted(self, risk):
        risk.check_kill_switch(97_000, 100_000)
        allowed, reason = risk.can_open("GLD", LONG, 1000.0, 97_000, [])
        assert not allowed
        assert "kill switch" in reason

    def test_manual_reset_restores_trading(self, risk):
        risk.check_kill_switch(97_000, 100_000)
        risk.reset_kill_switch()
        allowed, _ = risk.can_open("GLD", LONG, 1000.0, 97_000, [])
        assert allowed

    def test_gains_never_trip_it(self, risk):
        assert not risk.check_kill_switch(equity=105_000, day_start_equity=100_000)
