"""Backtesting entry point.

Usage:
    python backtest.py                          # all strategies + combined portfolio
    python backtest.py --strategy mean_reversion
    python backtest.py --start 2024-06-01 --end 2026-06-01
    python backtest.py --synthetic              # seeded fake data, pipeline check only

Each run writes a one-page markdown summary plus an equity-curve PNG to
reports/backtests/<timestamp>/, and prints the summary to stdout.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in scheduled/headless runs
import matplotlib.pyplot as plt
import pandas as pd

from core.config import PROJECT_ROOT, load_config
from core.engine import BacktestEngine, BacktestResult
from core.metrics import honesty_verdict, summarise
from core.risk import RiskManager
from strategies.base import Strategy
from strategies.mean_reversion import MeanReversion
from strategies.momentum_breakout import MomentumBreakout
from strategies.trend_following import TrendFollowing

STRATEGY_CLASSES = {
    "mean_reversion": MeanReversion,
    "momentum_breakout": MomentumBreakout,
    "trend_following": TrendFollowing,
}


def build_strategies(config: dict, names: list[str]) -> list[Strategy]:
    instances = []
    for name in names:
        params = config["strategies"][name]
        cls = STRATEGY_CLASSES[name]
        for symbol in params["symbols"]:
            instances.append(cls(symbol, params["timeframe"], params))
    return instances


def load_data(strategies: list[Strategy], config: dict, start: str, end: str, synthetic: bool) -> dict:
    data = {}
    if synthetic:
        from core.synthetic import generate

        for s in strategies:
            data[s.symbol] = generate(s.symbol, s.timeframe, start, end)
    else:
        from core.data import DataFeed

        feed = DataFeed(config["data"])
        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        for s in strategies:
            print(f"Fetching {s.symbol} {s.timeframe} bars from Alpaca...")
            data[s.symbol] = feed.historical_bars(s.symbol, s.timeframe, start_dt, end_dt)
    return data


def run_one(label: str, strategy_names: list[str], config: dict, args) -> tuple[BacktestResult, dict]:
    strategies = build_strategies(config, strategy_names)
    data = load_data(strategies, config, args.start, args.end, args.synthetic)
    risk = RiskManager(config["risk"])
    engine = BacktestEngine(
        strategies=strategies,
        data=data,
        risk=risk,
        costs_config=config["costs"],
        initial_equity=config["account"]["initial_equity"],
    )
    result = engine.run()
    stats = summarise(result)
    print(f"  {label}: {stats['n_trades']} trades, "
          f"return {stats['total_return_pct']:+.2f}%, max DD {stats['max_drawdown_pct']:.2f}%")
    return result, stats


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def plot_equity(result: BacktestResult, title: str, path: Path) -> None:
    curve = result.equity_curve
    running_max = curve.cummax()
    drawdown = (curve - running_max) / running_max * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(curve.index, curve.values, linewidth=1.0, color="#1f6feb")
    ax1.set_title(title)
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.3)
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#d1242f", alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def stats_table(stats: dict) -> str:
    rows = [
        ("Total return", f"{stats['total_return_pct']:+.2f}%"),
        ("CAGR", f"{stats['cagr_pct']:+.2f}%"),
        ("Max drawdown", f"{stats['max_drawdown_pct']:.2f}%"),
        ("Win rate", f"{stats['win_rate_pct']:.1f}%"),
        ("Average win", f"{stats['avg_win_r']:+.2f} R"),
        ("Average loss", f"{stats['avg_loss_r']:+.2f} R"),
        ("Profit factor", f"{stats['profit_factor']:.2f}"),
        ("Sharpe ratio", f"{stats['sharpe']:.2f}"),
        ("Trades", f"{stats['n_trades']}"),
        ("Longest losing streak", f"{stats['longest_losing_streak']}"),
        ("Blocked signals", f"{stats['n_blocked_signals']}"),
        ("Kill-switch events", f"{stats['n_kill_switch_events']}"),
        ("Final equity", f"${stats['final_equity']:,.2f}"),
    ]
    lines = ["| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def write_summary(runs: dict, config: dict, args, out_dir: Path) -> Path:
    lines = ["# Backtest summary", ""]
    lines.append(f"- Period: **{args.start} → {args.end}**")
    lines.append(f"- Initial equity: ${config['account']['initial_equity']:,.0f}")
    lines.append(
        f"- Costs modelled per side: {config['costs']['etf_cost_pct_per_side']}% ETFs, "
        f"{config['costs']['crypto_cost_pct_per_side']}% crypto (deliberately pessimistic)"
    )
    lines.append(f"- Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    if args.synthetic:
        lines.insert(1, "")
        lines.insert(
            2,
            "> ⚠️ **SYNTHETIC DATA RUN.** This run uses seeded random data to validate "
            "the pipeline only. The numbers below say NOTHING about real-market "
            "performance. Re-run without `--synthetic` once Alpaca keys are configured.",
        )
    lines.append("")

    for label, (result, stats) in runs.items():
        lines.append(f"## {label}")
        lines.append("")
        lines.append(stats_table(stats))
        lines.append("")
        lines.append(f"**Verdict:** {honesty_verdict(stats)}")
        lines.append("")
        png_name = f"equity_{label.lower().replace(' ', '_')}.png"
        lines.append(f"![equity curve]({png_name})")
        lines.append("")
        if result.blocked:
            lines.append(f"<details><summary>{len(result.blocked)} blocked signals</summary>")
            lines.append("")
            for b in result.blocked[:50]:
                lines.append(f"- `{b.time:%Y-%m-%d %H:%M}` {b.symbol} {b.action} — {b.reason}")
            if len(result.blocked) > 50:
                lines.append(f"- ... and {len(result.blocked) - 50} more")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    lines.append("---")
    lines.append(
        "*Parameters are the brief's defaults — no tuning was performed against this "
        "test period, so no walk-forward split was required. If any default is changed, "
        "re-run with a train/validate split per the honesty requirement.*"
    )
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the paper trading strategies.")
    parser.add_argument("--strategy", default="all",
                        choices=["all", *STRATEGY_CLASSES.keys()])
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (default from config.yaml)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default from config.yaml)")
    parser.add_argument("--synthetic", action="store_true",
                        help="use seeded synthetic data (pipeline validation only)")
    args = parser.parse_args()

    config = load_config()
    args.start = args.start or config["backtest"]["start"]
    args.end = args.end or config["backtest"]["end"]

    if args.strategy == "all":
        plan = {name.replace("_", " ").title(): [name] for name in STRATEGY_CLASSES}
        plan["Combined Portfolio"] = list(STRATEGY_CLASSES.keys())
    else:
        plan = {args.strategy.replace("_", " ").title(): [args.strategy]}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = "-synthetic" if args.synthetic else ""
    out_dir = PROJECT_ROOT / config["backtest"]["output_dir"] / f"{stamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for label, names in plan.items():
        print(f"Running backtest: {label} ({args.start} → {args.end})")
        result, stats = run_one(label, names, config, args)
        runs[label] = (result, stats)
        plot_equity(result, f"{label} — equity curve", out_dir / f"equity_{label.lower().replace(' ', '_')}.png")

    summary_path = write_summary(runs, config, args, out_dir)
    print("\n" + summary_path.read_text(encoding="utf-8"))
    print(f"\nReport written to {summary_path.parent}/", file=sys.stderr)


if __name__ == "__main__":
    main()
