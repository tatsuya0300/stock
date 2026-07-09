"""SQLite 永続化レイヤ。

schema v3:
- raw OHLC と adjusted OHLC を分離
- signals/orders/fills を追加
- signals/orders に PRIMARY KEY を追加し ON CONFLICT DO UPDATE を有効化
- shortability テーブル
- INSERT OR REPLACE を避け、ON CONFLICT DO UPDATE を使用
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 3

PRICE_COLS = [
    "code",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "volume",
    "turnover",
]

SHORT_COLS = ["code", "date", "is_margin_lendable", "short_restricted"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prices (
    code TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_open REAL,
    adj_high REAL,
    adj_low REAL,
    adj_close REAL,
    volume REAL,
    turnover REAL,
    PRIMARY KEY (code, date)
);

CREATE TABLE IF NOT EXISTS shortability (
    code TEXT NOT NULL,
    date TEXT NOT NULL,
    is_margin_lendable INTEGER,
    short_restricted INTEGER,
    PRIMARY KEY (code, date)
);

CREATE TABLE IF NOT EXISTS signals (
    run_id TEXT NOT NULL,
    signal_asof_date TEXT NOT NULL,
    code TEXT NOT NULL,
    side TEXT NOT NULL,
    score REAL,
    model_name TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (signal_asof_date, code, side, model_name)
);

CREATE TABLE IF NOT EXISTS orders (
    run_id TEXT NOT NULL,
    order_date TEXT NOT NULL,
    signal_asof_date TEXT,
    code TEXT NOT NULL,
    name TEXT,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    qty INTEGER NOT NULL,
    ref_price REAL,
    value_yen REAL,
    shortable INTEGER,
    warn TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (order_date, code, side, order_type)
);

CREATE TABLE IF NOT EXISTS fills (
    run_id TEXT,
    trade_date TEXT,
    code TEXT,
    side TEXT,
    qty INTEGER,
    price REAL,
    note TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class Storage:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.set_metadata("schema_version", str(SCHEMA_VERSION))

    def _initial_setup(self) -> None:
        """完全新規DB用の初期セットアップ。"""
        self.conn.executescript(SCHEMA)

    def set_metadata(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def get_metadata(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return default if row is None else str(row[0])

    def _table_columns(self, table: str) -> set[str]:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}

    def _table_exists(self, table: str) -> bool:
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None

    def _migrate(self) -> None:
        """v1/v2 → v3 移行。"""
        if not self._table_exists("prices"):
            return

        cols = self._table_columns("prices")
        needed = {
            "adj_open": "REAL",
            "adj_high": "REAL",
            "adj_low": "REAL",
            "adj_close": "REAL",
        }
        with self.conn:
            for col, typ in needed.items():
                if col not in cols:
                    self.conn.execute(f"ALTER TABLE prices ADD COLUMN {col} {typ}")

            self.conn.execute("UPDATE prices SET adj_open = open WHERE adj_open IS NULL")
            self.conn.execute("UPDATE prices SET adj_high = high WHERE adj_high IS NULL")
            self.conn.execute("UPDATE prices SET adj_low = low WHERE adj_low IS NULL")
            self.conn.execute("UPDATE prices SET adj_close = close WHERE adj_close IS NULL")

    def upsert_prices(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return

        x = df.copy()
        for c in ["adj_open", "adj_high", "adj_low", "adj_close"]:
            if c not in x.columns:
                if c == "adj_close":
                    x[c] = x["close"]
                else:
                    x[c] = x[c.replace("adj_", "")]

        x["code"] = x["code"].astype(str).str.strip()
        x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")
        x = x[PRICE_COLS]
        records = list(x.itertuples(index=False, name=None))

        sql = """
        INSERT INTO prices (
            code, date, open, high, low, close,
            adj_open, adj_high, adj_low, adj_close,
            volume, turnover
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(code, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            adj_open = excluded.adj_open,
            adj_high = excluded.adj_high,
            adj_low = excluded.adj_low,
            adj_close = excluded.adj_close,
            volume = excluded.volume,
            turnover = excluded.turnover
        """
        with self.conn:
            self.conn.executemany(sql, records)

    def load_prices(self, codes: list[str], start: str, end: str) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame(columns=PRICE_COLS)
        codes = [str(c) for c in codes]
        placeholders = ",".join("?" * len(codes))
        q = f"""
        SELECT {", ".join(PRICE_COLS)}
        FROM prices
        WHERE code IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY code, date
        """
        return pd.read_sql(q, self.conn, params=[*codes, start, end])

    def upsert_shortability(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        x = df.copy()
        x["code"] = x["code"].astype(str).str.strip()
        x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")
        x = x[SHORT_COLS]
        records = list(x.itertuples(index=False, name=None))
        sql = """
        INSERT INTO shortability (
            code, date, is_margin_lendable, short_restricted
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(code, date) DO UPDATE SET
            is_margin_lendable = excluded.is_margin_lendable,
            short_restricted = excluded.short_restricted
        """
        with self.conn:
            self.conn.executemany(sql, records)

    def load_shortability(self, codes: list[str], start: str, end: str) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame(columns=SHORT_COLS)
        codes = [str(c) for c in codes]
        placeholders = ",".join("?" * len(codes))
        q = f"""
        SELECT {", ".join(SHORT_COLS)}
        FROM shortability
        WHERE code IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY code, date
        """
        return pd.read_sql(q, self.conn, params=[*codes, start, end])

    def append_signals(
        self,
        run_id: str,
        signals: pd.DataFrame,
        signal_asof_date: str,
        model_name: str = "",
    ) -> None:
        if signals is None or signals.empty:
            return

        required = {"code", "side"}
        missing = required - set(signals.columns)
        if missing:
            raise ValueError(f"signals missing columns: {sorted(missing)}")

        x = signals.copy()
        x["run_id"] = run_id
        x["signal_asof_date"] = signal_asof_date
        x["model_name"] = model_name or ""
        if "score" not in x.columns:
            x["score"] = None

        cols = [
            "run_id",
            "signal_asof_date",
            "code",
            "side",
            "score",
            "model_name",
        ]
        x["code"] = x["code"].astype(str).str.strip()
        x["side"] = x["side"].astype(str).str.upper()
        records = list(x[cols].itertuples(index=False, name=None))

        sql = """
        INSERT INTO signals (
            run_id, signal_asof_date, code, side, score, model_name
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(signal_asof_date, code, side, model_name) DO UPDATE SET
            run_id = excluded.run_id,
            score = excluded.score,
            created_at = CURRENT_TIMESTAMP
        """
        with self.conn:
            self.conn.executemany(sql, records)

    def append_orders(self, run_id: str, orders: pd.DataFrame) -> None:
        if orders is None or orders.empty:
            return

        x = orders.copy()
        x["run_id"] = run_id
        if "shortable" not in x.columns:
            x["shortable"] = False
        x["shortable"] = x["shortable"].fillna(False).astype(bool).astype(int)

        cols = [
            "run_id",
            "order_date",
            "signal_asof_date",
            "code",
            "name",
            "side",
            "order_type",
            "qty",
            "ref_price",
            "value_yen",
            "shortable",
            "warn",
        ]
        for c in cols:
            if c not in x.columns:
                x[c] = None

        x["code"] = x["code"].astype(str).str.strip()
        x["side"] = x["side"].astype(str).str.upper()
        records = list(x[cols].itertuples(index=False, name=None))

        sql = """
        INSERT INTO orders (
            run_id, order_date, signal_asof_date, code, name, side, order_type,
            qty, ref_price, value_yen, shortable, warn
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(order_date, code, side, order_type) DO UPDATE SET
            run_id = excluded.run_id,
            signal_asof_date = excluded.signal_asof_date,
            name = excluded.name,
            qty = excluded.qty,
            ref_price = excluded.ref_price,
            value_yen = excluded.value_yen,
            shortable = excluded.shortable,
            warn = excluded.warn,
            created_at = CURRENT_TIMESTAMP
        """
        with self.conn:
            self.conn.executemany(sql, records)

    def append_fill(
        self,
        trade_date: str,
        code: str,
        side: str,
        qty: int,
        price: float,
        note: str = "",
        run_id: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO fills (run_id, trade_date, code, side, qty, price, note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    trade_date,
                    str(code),
                    side.upper(),
                    int(qty),
                    float(price),
                    note,
                ),
            )

    def load_orders(
        self,
        order_date: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """orders を読み出す。order_date 優先、無ければ start/end。"""
        cols = [
            "run_id",
            "order_date",
            "signal_asof_date",
            "code",
            "name",
            "side",
            "order_type",
            "qty",
            "ref_price",
            "value_yen",
            "shortable",
            "warn",
            "created_at",
        ]
        if order_date:
            q = f"""
            SELECT {', '.join(cols)}
            FROM orders
            WHERE order_date = ?
            ORDER BY code, side
            """
            return pd.read_sql(q, self.conn, params=[order_date])

        if start and end:
            q = f"""
            SELECT {', '.join(cols)}
            FROM orders
            WHERE order_date BETWEEN ? AND ?
            ORDER BY order_date, code, side
            """
            return pd.read_sql(q, self.conn, params=[start, end])

        q = f"SELECT {', '.join(cols)} FROM orders ORDER BY order_date, code, side"
        return pd.read_sql(q, self.conn)

    def load_fills(
        self,
        trade_date: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """fills を読み出す。trade_date 優先、無ければ start/end。"""
        cols = [
            "run_id",
            "trade_date",
            "code",
            "side",
            "qty",
            "price",
            "note",
            "created_at",
        ]
        if trade_date:
            q = f"""
            SELECT {', '.join(cols)}
            FROM fills
            WHERE trade_date = ?
            ORDER BY code, side
            """
            return pd.read_sql(q, self.conn, params=[trade_date])
        if start and end:
            q = f"""
            SELECT {', '.join(cols)}
            FROM fills
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date, code
            """
            return pd.read_sql(q, self.conn, params=[start, end])
        q = f"SELECT {', '.join(cols)} FROM fills ORDER BY trade_date, code"
        return pd.read_sql(q, self.conn)

    def import_fills_csv(self, path: str | Path) -> int:
        """CSV から fills を一括取込。戻り値は取込件数。

        必須列: trade_date, code, side, qty, price
        任意列: note, run_id
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"fills CSV が見つかりません: {path}")

        df = pd.read_csv(p, dtype={"code": str})
        required = {"trade_date", "code", "side", "qty", "price"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"fills CSV に必須列が不足: {sorted(missing)}")

        if "note" not in df.columns:
            df["note"] = ""
        if "run_id" not in df.columns:
            df["run_id"] = None

        n = 0
        with self.conn:
            for _, r in df.iterrows():
                self.conn.execute(
                    """
                    INSERT INTO fills (run_id, trade_date, code, side, qty, price, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        None if pd.isna(r["run_id"]) else str(r["run_id"]),
                        str(pd.to_datetime(r["trade_date"]).date()),
                        str(r["code"]).strip(),
                        str(r["side"]).upper(),
                        int(r["qty"]),
                        float(r["price"]),
                        "" if pd.isna(r["note"]) else str(r["note"]),
                    ),
                )
                n += 1
        return n

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def max_price_date(self, codes: list[str] | None = None) -> str | None:
        """prices テーブルの最大 date を返す。無ければ None。"""
        if codes:
            codes = [str(c) for c in codes]
            placeholders = ",".join("?" * len(codes))
            q = f"SELECT MAX(date) FROM prices WHERE code IN ({placeholders})"
            row = self.conn.execute(q, codes).fetchone()
        else:
            row = self.conn.execute("SELECT MAX(date) FROM prices").fetchone()
        if row is None or row[0] is None:
            return None
        return str(row[0])
