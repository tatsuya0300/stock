"""SQLite 永続化レイヤ（FR-DATA-03/05）。

価格（prices）・売り可否スナップショット（shortability）・約定記録（fills）を保持する。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    code TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, turnover REAL,
    PRIMARY KEY (code, date)
);
CREATE TABLE IF NOT EXISTS shortability (
    code TEXT NOT NULL,
    date TEXT NOT NULL,
    is_margin_lendable INTEGER,     -- 1: 貸借銘柄, 0: 制度信用のみ, NULL: 不明
    short_restricted   INTEGER,     -- 1: 新規売り停止
    PRIMARY KEY (code, date)
);
CREATE TABLE IF NOT EXISTS fills (
    trade_date TEXT, code TEXT, side TEXT,
    qty INTEGER, price REAL, note TEXT
);
"""

_PRICE_COLS = ["code", "date", "open", "high", "low", "close", "volume", "turnover"]
_SHORT_COLS = ["code", "date", "is_margin_lendable", "short_restricted"]


class Storage:
    """SQLite ベースの永続化ストア。"""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ prices
    def upsert_prices(self, df: pd.DataFrame) -> None:
        """price を UPSERT する。df columns: code,date,open,high,low,close,volume,turnover"""
        if df is None or df.empty:
            return
        df = df[_PRICE_COLS].copy()
        df.to_sql("_stg_prices", self.conn, if_exists="replace", index=False)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO prices
            SELECT code,date,open,high,low,close,volume,turnover FROM _stg_prices;
            """
        )
        self.conn.execute("DROP TABLE _stg_prices;")
        self.conn.commit()

    def load_prices(self, codes: list[str], start: str, end: str) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame(columns=_PRICE_COLS)
        placeholders = ",".join("?" * len(codes))
        q = f"""
        SELECT * FROM prices
        WHERE code IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY code, date
        """
        return pd.read_sql(q, self.conn, params=[*codes, start, end])

    # ------------------------------------------------------------ shortability
    def upsert_shortability(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        df = df[_SHORT_COLS].copy()
        df.to_sql("_stg_short", self.conn, if_exists="replace", index=False)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO shortability
            SELECT code,date,is_margin_lendable,short_restricted FROM _stg_short;
            """
        )
        self.conn.execute("DROP TABLE _stg_short;")
        self.conn.commit()

    def load_shortability(self, codes: list[str], start: str, end: str) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame(columns=_SHORT_COLS)
        placeholders = ",".join("?" * len(codes))
        q = f"""
        SELECT * FROM shortability
        WHERE code IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY code, date
        """
        return pd.read_sql(q, self.conn, params=[*codes, start, end])

    # ------------------------------------------------------------------- fills
    def append_fill(self, trade_date: str, code: str, side: str,
                    qty: int, price: float, note: str = "") -> None:
        self.conn.execute(
            "INSERT INTO fills (trade_date, code, side, qty, price, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (trade_date, code, side, qty, price, note),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
