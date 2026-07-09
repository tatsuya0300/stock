"""SQLite 永続化レイヤ（FR-DATA-03/05）。改訂版。

価格（prices）・売り可否スナップショット（shortability）・約定記録（fills）を保持する。

改訂点:
  - prices テーブルに adj_close 列を追加（分割・配当調整後終値）。
    リターン計算は adj_close、約定金額は生 open/close を使う。
  - prices に生 OHLC 列（open_raw, close_raw）を追加し、調整後（open/high/low/close）
    と生値を分離して保持する。
  - INSERT OR REPLACE → ON CONFLICT DO UPDATE に変更（PK を保持しつつ安全な UPSERT）。
  - orders / signals テーブルを追加し、発注指示とシグナルを監査可能に。
  - upsert_prices で旧スキーマ（adj_close 無し）から新スキーマへの互換性補完。
  - sqlite3.connect を check_same_thread=False + timeout + WAL でスレッド安全化。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA_V1 = """
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
    is_margin_lendable INTEGER,
    short_restricted   INTEGER,
    PRIMARY KEY (code, date)
);
CREATE TABLE IF NOT EXISTS fills (
    trade_date TEXT, code TEXT, side TEXT,
    qty INTEGER, price REAL, note TEXT
);
"""

SCHEMA_V2 = """
-- v2: raw OHLC columns + orders/signals audit tables + consistent UPSERT
ALTER TABLE prices ADD COLUMN open_raw REAL;
ALTER TABLE prices ADD COLUMN high_raw REAL;
ALTER TABLE prices ADD COLUMN low_raw REAL;
ALTER TABLE prices ADD COLUMN close_raw REAL;

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL,
    as_of_date  TEXT NOT NULL,
    code TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL DEFAULT 'MKT_OPEN',
    qty INTEGER NOT NULL,
    ref_price REAL,
    value_yen REAL,
    warn TEXT,
    shortable INTEGER DEFAULT 1,
    dry_run INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL,
    as_of_date  TEXT NOT NULL,
    code TEXT NOT NULL,
    side TEXT NOT NULL,
    score REAL,
    limit_price REAL
);
"""

_PRICE_COLS = [
    "code", "date", "open", "high", "low", "close",
    "adj_close", "volume", "turnover",
    "open_raw", "high_raw", "low_raw", "close_raw",
]
_SHORT_COLS = ["code", "date", "is_margin_lendable", "short_restricted"]

_PRICE_UPSERT_SQL = """
INSERT INTO prices ({cols}) VALUES ({placeholders})
ON CONFLICT(code, date) DO UPDATE SET
    open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
    adj_close=excluded.adj_close,
    volume=excluded.volume, turnover=excluded.turnover,
    open_raw=excluded.open_raw, high_raw=excluded.high_raw,
    low_raw=excluded.low_raw, close_raw=excluded.close_raw
"""

_SHORT_UPSERT_SQL = """
INSERT INTO shortability ({cols}) VALUES ({placeholders})
ON CONFLICT(code, date) DO UPDATE SET
    is_margin_lendable=excluded.is_margin_lendable,
    short_restricted=excluded.short_restricted
"""


def _detect_schema_version(conn: sqlite3.Connection) -> int:
    """prices テーブルに open_raw 列があれば v2、無ければ v1 とみなす。"""
    curs = conn.execute("PRAGMA table_info(prices)")
    cols = {row[1] for row in curs.fetchall()}
    return 2 if "open_raw" in cols else 1


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """v1 → v2 マイグレーション: raw OHLC 列を追加し、既存データを補完する。"""
    for stmt in SCHEMA_V2.split(";"):
        s = stmt.strip()
        if s:
            try:
                conn.execute(s)
            except sqlite3.OperationalError:
                pass  # 列が既に存在する場合など
    # 既存行で raw 列が NULL なら close/open で補完
    conn.execute(
        "UPDATE prices SET open_raw = open WHERE open_raw IS NULL"
    )
    conn.execute(
        "UPDATE prices SET close_raw = close WHERE close_raw IS NULL"
    )
    conn.execute(
        "UPDATE prices SET high_raw = high WHERE high_raw IS NULL"
    )
    conn.execute(
        "UPDATE prices SET low_raw = low WHERE low_raw IS NULL"
    )


class Storage:
    """SQLite ベースの永続化ストア。"""

    def __init__(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(p), check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(SCHEMA_V1)

        # 必要なら v2 にマイグレーション
        if _detect_schema_version(self.conn) < 2:
            _migrate_v1_to_v2(self.conn)

    # ------------------------------------------------------------------ prices
    def upsert_prices(self, df: pd.DataFrame) -> None:
        """price を UPSERT する。ON CONFLICT DO UPDATE で安全に更新。

        df columns: code,date,open,high,low,close,adj_close,volume,turnover
        adj_close が無い（旧スキーマ由来）場合は close で補完する。
        raw 列がない場合は adjust 列の値をコピーする（互換性）。
        """
        if df is None or df.empty:
            return
        df = df.copy()

        # 互換性補完
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        for raw_col, adj_col in [
            ("open_raw", "open"),
            ("high_raw", "high"),
            ("low_raw", "low"),
            ("close_raw", "close"),
        ]:
            if raw_col not in df.columns:
                df[raw_col] = df[adj_col] if adj_col in df.columns else None

        # _PRICE_COLS に含まれる列のみ抽出
        cols = [c for c in _PRICE_COLS if c in df.columns]
        df = df[cols]
        records = list(df.itertuples(index=False, name=None))
        placeholders = ",".join("?" * len(cols))
        cols_str = ",".join(cols)

        sql = _PRICE_UPSERT_SQL.format(cols=cols_str, placeholders=placeholders)
        with self.conn:
            self.conn.executemany(sql, records)

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
        cols_str = ",".join(_SHORT_COLS)
        sql = _SHORT_UPSERT_SQL.format(cols=cols_str, placeholders=placeholders)
        with self.conn:
            self.conn.executemany(sql, records)

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

    # -------------------------------------------------------------------- fills
    def append_fill(self, trade_date: str, code: str, side: str,
                    qty: int, price: float, note: str = "") -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO fills (trade_date, code, side, qty, price, note) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (trade_date, code, side, qty, price, note),
            )

    # ------------------------------------------------------------ audit tables
    def append_orders(
        self,
        orders: pd.DataFrame,
        as_of_date: str,
        dry_run: bool = False,
    ) -> None:
        """orders テーブルに発注指示を追記する。"""
        import datetime

        now = datetime.datetime.utcnow().isoformat()
        rows = []
        for _, o in orders.iterrows():
            rows.append((
                now, as_of_date,
                o.get("code", ""),
                o.get("side", ""),
                o.get("order_type", "MKT_OPEN"),
                int(o.get("qty", 0)),
                float(o.get("ref_price", 0.0)),
                float(o.get("value_yen", 0.0)),
                str(o.get("warn", "")),
                1 if o.get("shortable", True) else 0,
                1 if dry_run else 0,
            ))
        with self.conn:
            self.conn.executemany(
                "INSERT INTO orders "
                "(generated_at, as_of_date, code, side, order_type, qty, "
                " ref_price, value_yen, warn, shortable, dry_run) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def append_signals(
        self,
        signals: pd.DataFrame,
        as_of_date: str,
    ) -> None:
        """signals テーブルに生成シグナルを追記する。"""
        import datetime

        now = datetime.datetime.utcnow().isoformat()
        import numpy as np

        rows = []
        for _, s in signals.iterrows():
            limit_price = s.get("limit_price")
            if limit_price is None or (isinstance(limit_price, float) and np.isnan(limit_price)):
                limit_price = None
            rows.append((
                now, as_of_date,
                s.get("code", ""),
                s.get("side", ""),
                float(s.get("score", 0.0)) if s.get("score") is not None else None,
                float(limit_price) if limit_price is not None else None,
            ))
        with self.conn:
            self.conn.executemany(
                "INSERT INTO signals "
                "(generated_at, as_of_date, code, side, score, limit_price) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    # ------------------------------------------------------------------ close
    def close(self) -> None:
        self.conn.close()
