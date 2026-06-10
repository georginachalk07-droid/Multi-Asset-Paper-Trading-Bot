"""Performance metrics for backtest results."""

import numpy as np
import pandas as pd

from core.engine import BacktestResult

TRADING_DAYS_PER_YEAR = 252  # standard annualisation; BTC trades weekends so
                             # the Sharpe figure is a mild approximation


def summarise(result: BacktestResult) -> dict:
    curve = result.equity_curve
    trades = sorted(result.trades, key=lambda t: t.exit_time)
    initial = result.initial_equity
    final = result.final_equity

    total_return_pct = (final / initial - 1) * 100

    if len(curve) >= 2:
        years = max((curve.index[-1] - curve.index[0]).days, 1) / 365.25
        cagr_pct = ((final / initial) ** (1 / years) - 1) * 100 if final > 0 else -100.0
        running_max = curve.cummax()
        max_dd_pct = float(((curve - running_max) / running_max).min() * 100)
        daily = curve.resample("1D").last().ffill()
        rets = daily.pct_change().dropna()
        sharpe = (
            float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            if len(rets) > 1 and rets.std() > 0
            else 0.0
        )
    else:
        cagr_pct, max_dd_pct, sharpe = 0.0, 0.0, 0.0

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))

    streak = longest = 0
    for t in trades:
        streak = streak + 1 if t.pnl <= 0 else 0
        longest = max(longest, streak)

    return {
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "max_drawdown_pct": max_dd_pct,
        "win_rate_pct": len(wins) / len(trades) * 100 if trades else 0.0,
        "avg_win_r": float(np.mean([t.r_multiple for t in wins])) if wins else 0.0,
        "avg_loss_r": float(np.mean([t.r_multiple for t in losses])) if losses else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        "sharpe": sharpe,
        "n_trades": len(trades),
        "longest_losing_streak": longest,
        "n_blocked_signals": len(result.blocked),
        "n_kill_switch_events": len(result.kill_switch_events),
        "final_equity": final,
    }


def honesty_verdict(stats: dict) -> str:
    """Plain-spoken assessment, per the brief's honesty requirement."""
    if stats["n_trades"] == 0:
        return "NO TRADES — the strategy never triggered in this period. No conclusion can be drawn."
    if stats["total_return_pct"] <= 0 or stats["profit_factor"] < 1.0:
        return (
            "NEGATIVE after costs. This strategy LOST money over the test period "
            "with the assumed slippage and fees. Do not trade it as configured."
        )
    if stats["profit_factor"] < 1.15 or stats["sharpe"] < 0.5:
        return (
            "MARGINAL after costs. The edge, if any, is thin (profit factor "
            f"{stats['profit_factor']:.2f}, Sharpe {stats['sharpe']:.2f}) and could "
            "easily be noise. Treat with scepticism."
        )
    return (
        f"Positive after costs (profit factor {stats['profit_factor']:.2f}, "
        f"Sharpe {stats['sharpe']:.2f}). Past performance still guarantees nothing."
    )
