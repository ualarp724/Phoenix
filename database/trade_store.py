import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

DB_PATH = "database/trades.db"


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    side: str
    lot_size: float
    entry_price: float
    exit_price: float
    pnl: float


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                lot_size REAL,
                entry_price REAL,
                exit_price REAL,
                pnl REAL
            )
            """
        )


def insert_trade(trade: TradeRecord) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, side, lot_size, entry_price, exit_price, pnl) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                trade.timestamp.isoformat(),
                trade.symbol,
                trade.side,
                trade.lot_size,
                trade.entry_price,
                trade.exit_price,
                trade.pnl,
            ),
        )


def insert_many(trades: Iterable[TradeRecord]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT INTO trades (timestamp, symbol, side, lot_size, entry_price, exit_price, pnl) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    t.timestamp.isoformat(),
                    t.symbol,
                    t.side,
                    t.lot_size,
                    t.entry_price,
                    t.exit_price,
                    t.pnl,
                )
                for t in trades
            ],
        )
