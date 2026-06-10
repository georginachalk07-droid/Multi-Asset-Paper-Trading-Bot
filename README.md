# Multi-Asset Paper Trading Bot

Automated **paper-only** trading bot for an Alpaca paper account. Five
instruments, three strategies, ATR-based sizing, hard risk limits, a
correlation filter, and a full backtesting harness.

| Instrument | Ticker | Strategy | Timeframe |
|---|---|---|---|
| S&P 500 | SPY | Mean reversion | 15-min |
| Nasdaq 100 | QQQ | Mean reversion | 15-min |
| Bitcoin | BTC/USD | Momentum breakout | 1-hour |
| Gold | GLD | Trend following | 4-hour |
| Oil | USO | Trend following | 4-hour |

## Project status

Build-order steps **1–4 are complete**: broker wrapper, data layer, risk
module (unit-tested), all three strategies, and the individual + combined
portfolio backtester. Per the project ground rules, the live trading loop
(`run.py`) and daily reports (`report.py`) are **deliberately not built yet**
— backtest results are to be reviewed first.

The full pipeline has been validated end-to-end on seeded synthetic data.
**The real 2-year backtest has not been run yet** because it needs Alpaca API
keys (see Setup). Synthetic results are labelled as such and prove only that
the machinery works, not that the strategies do.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your Alpaca PAPER keys
```

Verify the paper connection and a candle pull for all five instruments:

```bash
python -m core.broker
```

## Commands

```bash
python -m pytest tests/                         # unit tests (risk logic etc.)
python backtest.py                              # all strategies + combined portfolio
python backtest.py --strategy mean_reversion    # one strategy only
python backtest.py --start 2024-06-01 --end 2026-06-01
python backtest.py --synthetic                  # no API keys needed; pipeline check only
```

Each backtest writes `summary.md` plus equity-curve PNGs to
`reports/backtests/<timestamp>/` and prints the summary to stdout.

## Risk management (enforced in `core/risk.py`, never in strategies)

1. **Hard stop, 1% of equity per trade** — placed as a real broker-side stop
   order at entry, never software-only.
2. **ATR sizing** — qty = (1% of equity) / (1.5 × ATR(14)); volatile
   instruments get smaller positions, risk per trade stays constant.
3. **Correlation filter** — SPY, QQQ and BTC/USD form a risk-on group; with
   two group longs open, the third is blocked and the refusal logged.
4. **Max 4 concurrent positions.**
5. **Daily kill switch** — equity −3% from the day's start flattens
   everything and halts until manually restarted.
6. **Max 3% total open risk** across all positions.

## Design decisions and justifications

- **Custom bar-loop backtester, not backtrader or pure vectorisation.** The
  portfolio rules (correlation filter, position caps, total-risk cap, kill
  switch) cut across instruments and strategies on mixed timeframes.
  Vectorised per-instrument backtests cannot model them; backtrader can be
  bent to it but is effectively unmaintained and would add a heavy dependency.
  Two years of data is only ~30k bars across all five instruments, so a
  transparent chronological loop in pandas is fast enough and — crucially —
  exercises the exact same `RiskManager` class the live loop will use.
- **Conservative fill model.** Signals act on the bar close at the close
  price, shaded by per-side costs (0.05% ETFs, 0.25% crypto). Stops are
  checked against subsequent bars' highs/lows; a gap through a stop fills at
  the bar's open (worse than the stop), never at the stop price.
- **matplotlib added to the stack** (flagging per the ground rules): the
  brief requires an equity-curve PNG per run and matplotlib is the minimal
  standard way to produce one. PyYAML (for `config.yaml`) and pytest (for
  `tests/`) are likewise implied by the brief.

## Ambiguities flagged (and how they were resolved)

1. **BTC stop conflict.** The risk layer mandates the initial stop at
   1.5 × ATR (it defines position size), but the breakout strategy specifies a
   2 × ATR trailing stop. At entry, "highest close − 2 × ATR" sits *below* the
   1.5 × ATR hard stop. Resolution: the hard 1.5 × ATR stop applies at entry;
   the 2 × ATR trail takes over only once it rises above it. Stops are never
   widened.
2. **Trading hours in exchange time, not UK time.** The brief says
   14:30–21:00 UK, but UK and US daylight saving change on different dates,
   so a fixed UK window drifts an hour twice a year. The session filter uses
   09:30–16:00 America/New_York (identical most of the year), flat from
   15:50 ET.
3. **"Total open risk" definition.** Implemented as committed risk: qty ×
   distance from entry to current stop, floored at zero. A position whose
   stop has trailed past break-even consumes no risk budget. Consequence:
   with 1% per trade and a 3% cap, a fourth position can only open once at
   least one stop has trailed to break-even — the 4-position cap and 3% cap
   interact by design.
4. **Notional safety cap (addition).** On quiet 15-minute bars, 1%-risk ATR
   sizing can demand several times account equity in notional (e.g. SPY with
   a 0.3% ATR). A single position is therefore capped at 50% of equity
   notional (`max_position_notional_pct`); risk per trade can only ever be
   *less* than 1% as a result. Without this, ETF positions would lean on
   margin and crypto buys could exceed available cash.
5. **Whole shares for ETFs.** Alpaca rejects stop orders on fractional
   equity quantities, so ETF positions are sized in whole shares (crypto is
   fractional). Rounding down means realised risk is marginally under 1%.
6. **Crypto stops are stop-limit orders.** Alpaca does not accept plain stop
   orders for crypto; BTC uses a stop-limit with the limit 0.5% through the
   stop, which behaves like a stop-market in practice.
7. **Kill-switch restart in backtests.** Live mode halts until a human
   restarts the bot, as specified. The backtest auto-resumes the next day —
   otherwise one bad day in year 1 would invalidate the rest of the test.
   Kill-switch trip counts are reported.
8. **ETF shorting enabled.** Alpaca paper accounts do support shorting
   ETFs, so the mean-reversion short side is implemented (toggle with
   `allow_short` in `config.yaml`).

## Data limitations

- **Equities:** the free Alpaca plan serves the IEX feed — prices track the
  consolidated market closely but volume is a single-venue subset. Set
  `data.stock_feed: sip` in `config.yaml` if you have a market-data
  subscription.
- **Crypto:** Alpaca BTC/USD history is reliable from ~2021, comfortably
  covering a 2-year backtest. Crypto data needs no special subscription.

## Structure

```
config.yaml            # every tunable parameter; nothing hard-coded
backtest.py            # backtesting entry point
core/
  broker.py            # Alpaca wrapper — refuses non-paper endpoints
  data.py              # candle fetching + CSV cache
  risk.py              # sizing, stops, correlation filter, kill switch
  engine.py            # portfolio backtest engine
  metrics.py           # performance stats + honesty verdicts
  journal.py           # SQLite trade journal
  indicators.py        # ATR, EMA, SMA, z-score, rolling high
  synthetic.py         # seeded fake data for pipeline validation
strategies/
  base.py              # strategy interface (signals only — no sizing, no stops)
  mean_reversion.py    # SPY/QQQ, 15-min
  momentum_breakout.py # BTC/USD, 1-hour
  trend_following.py   # GLD/USO, 4-hour
tests/                 # 40 unit tests; risk logic covered exhaustively
```

## Not built yet (by design)

- `run.py` — live paper loop with scheduler, crash-safe reconciliation
- `report.py` — 07:00 / 21:00 UK daily reports

These follow once the real-data backtest results have been reviewed, per the
brief's ground rules.
