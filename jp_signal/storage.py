"""SQLite 永続化レイヤ（FR-DATA-03/05）。改訂版。

価格（prices）・売り可否スナップショット（shortability）・約定記録（fills）を保持する。

改訂点:
  - prices テーブルに adj_close 列を追加（分割・配当調整後終値）。
    リターン計算は adj_close、約定金額は生 open/close を使う。
  - upsert_prices / upsert_shortability を固定名ステージングテーブルから
    executemany + トランザクションに変更（cron 多重起動時の衝突・残留を回避）。
  - sqlite3.connect を check_same_thread=False + timeout + WAL でスレッド安全化。
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
    adj_close REAL,
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

_PRICE_COLS = [
    "code", "date", "open", "high", "low", "close",
    "adj_close", "volume", "turnover",
]
_SHORT_COLS = ["code", "date", "is_margin_lendable", "short_restricted"]


class Storage:
    """SQLite ベースの永続化ストア。"""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # cron 多重起動やスレッド利用に備え check_same_thread=False + タイムアウト
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ prices
    def upsert_prices(self, df: pd.DataFrame) -> None:
        """price を UPSERT する。ステージングテーブルを使わず executemany で行う。

        df columns: code,date,open,high,low,close,adj_close,volume,turnover
        adj_close が無い（旧スキーマ由来）場合は close で補完する。
        """
        if df is None or df.empty:
            return
        df = df.copy()
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        df = df[_PRICE_COLS]
        records = list(df.itertuples(index=False, name=None))
        placeholders = ",".join("?" * len(_PRICE_COLS))
        cols = ",".join(_PRICE_COLS)
        with self.conn:  # トランザクション自動コミット/ロールバック
            self.conn.executemany(
                f"INSERT OR REPLACE INTO prices ({cols}) VALUES ({placeholders})",
                records,
            )

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
        records = list(df.itertuples(index=False, name=None))
        placeholders = ",".join("?" * len(_SHORT_COLS))
        cols = ",".join(_SHORT_COLS)
        with self.conn:
            self.conn.executemany(
                f"INSERT OR REPLACE INTO shortability ({cols}) VALUES ({placeholders})",
                records,
            )

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
        with self.conn:
            self.conn.execute(
                "INSERT INTO fills (trade_date, code, side, qty, price, note) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (trade_date, code, side, qty, price, note),
            )

    def close(self) -> None:
        self.conn.close()
