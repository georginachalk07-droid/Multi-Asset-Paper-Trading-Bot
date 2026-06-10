"""SQLite trade journal.

Every order, fill, stop placement, blocked signal, exit and P&L event is
written here so the bot's behaviour can be audited after the fact. The same
journal backs the live loop and (optionally) backtests.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.config import PROJECT_ROOT

DEFAULT_DB_PATH = PROJECT_ROOT / "journal.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,              -- event time, ISO-8601 UTC
    event TEXT NOT NULL,           -- signal | blocked | order | fill | stop_set |
                                   -- stop_moved | exit | kill_switch | error
    symbol TEXT,
    strategy TEXT,
    side TEXT,                     -- long | short
    qty REAL,
    price REAL,
    stop_price REAL,
    pnl REAL,
    detail TEXT                    -- free-text reason / context
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts);
CREATE INDEX IF NOT EXISTS idx_events_event ON events (event);
"""


class Journal:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def log(
        self,
        event: str,
        symbol: str | None = None,
        strategy: str | None = None,
        side: str | None = None,
        qty: float | None = None,
        price: float | None = None,
        stop_price: float | None = None,
        pnl: float | None = None,
        detail: str | None = None,
        ts: datetime | None = None,
    ) -> None:
        ts = ts or datetime.now(timezone.utc)
        self.conn.execute(
            "INSERT INTO events (ts, event, symbol, strategy, side, qty, price, stop_price, pnl, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts.isoformat(), event, symbol, strategy, side, qty, price, stop_price, pnl, detail),
        )
        self.conn.commit()

    def events_between(self, start: datetime, end: datetime) -> list[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.execute(
            "SELECT * FROM events WHERE ts >= ? AND ts < ? ORDER BY ts",
            (start.isoformat(), end.isoformat()),
        )
        return cursor.fetchall()

    def close(self) -> None:
        self.conn.close()
