"""SQLite 永続化レイヤ。

schema v4:
- raw OHLC と adjusted OHLC を分離
- signals/orders/fills を追加
- signals/orders に PRIMARY KEY を追加し ON CONFLICT DO UPDATE を有効化
- shortability テーブル
- INSERT OR REPLACE を避け、ON CONFLICT DO UPDATE を使用
- order_rejections テーブル (v3)
- price_observations テーブル（取得時刻・revision履歴）(v4)
- record_price_observations() / ingest_prices()
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 4


def _sha256_file(path: str | Path) -> str:
    """ファイルのSHA256を計算する。"""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    """現在UTC時刻をISO 8601形式で返す（マイクロ秒精度）。"""
    return datetime.now(UTC).isoformat()


def _price_payload_hash(row: dict) -> str:
    """PRICE_COLSからSHA256ペイロードハッシュを計算する。

    JSONシリアライズはソート済みキーで行い、ハッシュの一貫性を保証する。
    """
    payload = {k: row.get(k) for k in PRICE_COLS}
    serialized = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
    source_file_hash TEXT,
    source_row_number INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_file_hash, source_row_number)
);

CREATE TABLE IF NOT EXISTS order_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    rejection_date TEXT NOT NULL,
    stage TEXT NOT NULL,
    reason TEXT NOT NULL,
    code TEXT,
    name TEXT,
    side TEXT,
    score REAL,
    qty INTEGER,
    ref_price REAL,
    value_yen REAL,
    payload_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_order_rejections_date
ON order_rejections(rejection_date);

CREATE INDEX IF NOT EXISTS idx_order_rejections_run
ON order_rejections(run_id);

CREATE TABLE IF NOT EXISTS price_observations (
    code TEXT NOT NULL,
    date TEXT NOT NULL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    available_at TEXT,
    payload_hash TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(code, date, source, payload_hash)
);

CREATE INDEX IF NOT EXISTS idx_price_observations_lookup
ON price_observations(code, date, source);

CREATE TABLE IF NOT EXISTS shortability_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    code TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    available_at TEXT NOT NULL,

    source TEXT NOT NULL,
    short_type TEXT NOT NULL,

    is_shortable INTEGER NOT NULL,
    is_margin_lendable INTEGER,
    short_restricted INTEGER NOT NULL,

    stock_loan_fee_annual REAL,
    payload_hash TEXT NOT NULL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(payload_hash)
);

CREATE INDEX IF NOT EXISTS idx_shortability_observations_lookup
ON shortability_observations(
    code,
    short_type,
    available_at
);

CREATE INDEX IF NOT EXISTS idx_shortability_observations_effective
ON shortability_observations(
    code,
    effective_at
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

    def record_price_observations(
        self,
        df: pd.DataFrame,
        *,
        source: str,
        available_at: str | None = None,
    ) -> int:
        """価格データのリビジョン履歴を price_observations に記録する。

        各(code, date, source) のペイロード内容が前回と異なる場合のみ
        新しい行を挿入する（同一ハッシュならスキップ）。

        Returns:
            新規挿入行数。
        """
        if df is None or df.empty:
            return 0

        fetched_at = _utc_now_iso()
        x = df.copy()
        x["code"] = x["code"].astype(str).str.strip()
        x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")

        sql = """
        INSERT OR IGNORE INTO price_observations (
            code, date, source, fetched_at, available_at, payload_hash
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """

        records: list[tuple[str, str, str, str, str | None, str]] = []
        for _, row in x.iterrows():
            row_dict = row.to_dict()
            payload_hash = _price_payload_hash(row_dict)
            records.append(
                (
                    str(row_dict.get("code", "")),
                    str(row_dict.get("date", "")),
                    source,
                    fetched_at,
                    available_at,
                    payload_hash,
                )
            )

        with self.conn:
            self.conn.executemany(sql, records)
            n = self.conn.total_changes

        return n

    def ingest_prices(
        self,
        df: pd.DataFrame,
        *,
        source: str,
        available_at: str | None = None,
    ) -> None:
        """価格データの取込（リビジョン記録 + 最新投影の更新）を一括で行う。

        同一トランザクション内で以下を実行:
        1. record_price_observations() — リビジョン履歴の保存
        2. upsert_prices() — prices テーブルの最新値更新

        Args:
            df: 価格データフレーム（PRICE_COLS を含む）
            source: データソース名（例: "jquants", "yfinance"）
            available_at: ISO 8601 形式の利用可能時刻（None なら fetched_at と同じ）
        """
        if df is None or df.empty:
            return

        with self.conn:
            self.record_price_observations(df, source=source, available_at=available_at)
            self.upsert_prices(df)

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
            SELECT {", ".join(cols)}
            FROM orders
            WHERE order_date = ?
            ORDER BY code, side
            """
            return pd.read_sql(q, self.conn, params=[order_date])

        if start and end:
            q = f"""
            SELECT {", ".join(cols)}
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
            "source_file_hash",
            "source_row_number",
            "created_at",
        ]
        if trade_date:
            q = f"""
            SELECT {", ".join(cols)}
            FROM fills
            WHERE trade_date = ?
            ORDER BY code, side
            """
            return pd.read_sql(q, self.conn, params=[trade_date])
        if start and end:
            q = f"""
            SELECT {", ".join(cols)}
            FROM fills
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date, code
            """
            return pd.read_sql(q, self.conn, params=[start, end])
        q = f"SELECT {', '.join(cols)} FROM fills ORDER BY trade_date, code"
        return pd.read_sql(q, self.conn)

    def import_fills_csv(self, path: str | Path) -> int:
        """CSV から fills を一括取込。戻り値は新規取込件数。

        必須列: trade_date, code, side, qty, price
        任意列: note, run_id

        重複防止:
          source_file_hash + source_row_number を一意キーにする。
          同じCSVを再インポートしても二重計上しない。
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"fills CSV が見つかりません: {path}")

        file_hash = _sha256_file(p)

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
        sql = """
        INSERT INTO fills (
            run_id, trade_date, code, side, qty, price, note,
            source_file_hash, source_row_number
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_file_hash, source_row_number) DO NOTHING
        """

        with self.conn:
            for row_number, (_, r) in enumerate(df.iterrows(), start=2):
                cur = self.conn.execute(
                    sql,
                    (
                        None if pd.isna(r["run_id"]) else str(r["run_id"]),
                        str(pd.to_datetime(r["trade_date"]).date()),
                        str(r["code"]).strip(),
                        str(r["side"]).upper(),
                        int(r["qty"]),
                        float(r["price"]),
                        "" if pd.isna(r["note"]) else str(r["note"]),
                        file_hash,
                        row_number,
                    ),
                )
                n += int(cur.rowcount)

        return n

    def append_order_rejections(
        self,
        *,
        run_id: str,
        rejection_date: str,
        rejected: pd.DataFrame,
    ) -> None:
        if rejected is None or rejected.empty:
            return

        sql = """
        INSERT INTO order_rejections (
            run_id,
            rejection_date,
            stage,
            reason,
            code,
            name,
            side,
            score,
            qty,
            ref_price,
            value_yen,
            payload_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        records = []

        for _, row in rejected.iterrows():
            payload = {
                str(key): (
                    None if pd.isna(value) else value.item() if hasattr(value, "item") else value
                )
                for key, value in row.to_dict().items()
            }

            records.append(
                (
                    run_id,
                    rejection_date,
                    str(row.get("stage", "UNKNOWN")),
                    str(row.get("reason", "UNKNOWN")),
                    str(row.get("code", "")) if pd.notna(row.get("code")) else "",
                    str(row.get("name", "")) if pd.notna(row.get("name")) else "",
                    str(row.get("side", "")) if pd.notna(row.get("side")) else "",
                    (None if pd.isna(row.get("score")) else float(row["score"])),
                    (None if pd.isna(row.get("qty")) else int(row["qty"])),
                    (None if pd.isna(row.get("ref_price")) else float(row["ref_price"])),
                    (None if pd.isna(row.get("value_yen")) else float(row["value_yen"])),
                    json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True),
                )
            )

        with self.conn:
            self.conn.executemany(sql, records)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def insert_shortability_observations(
        self,
        frame: pd.DataFrame,
    ) -> int:
        """PIT shortability観測を保存する。"""
        from .shortability_pit import (
            normalize_shortability_observations,
        )

        x = normalize_shortability_observations(
            frame
        )

        if x.empty:
            return 0

        sql = """
        INSERT INTO shortability_observations (
            code,
            effective_at,
            fetched_at,
            available_at,
            source,
            short_type,
            is_shortable,
            is_margin_lendable,
            short_restricted,
            stock_loan_fee_annual,
            payload_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(payload_hash) DO NOTHING
        """

        inserted = 0

        with self.conn:
            for row in x.itertuples(index=False):
                cursor = self.conn.execute(
                    sql,
                    (
                        str(row.code),
                        pd.Timestamp(
                            row.effective_at
                        ).isoformat(),
                        pd.Timestamp(
                            row.fetched_at
                        ).isoformat(),
                        pd.Timestamp(
                            row.available_at
                        ).isoformat(),
                        str(row.source),
                        str(row.short_type),
                        int(row.is_shortable),
                        (
                            None
                            if pd.isna(
                                row.is_margin_lendable
                            )
                            else int(
                                row.is_margin_lendable
                            )
                        ),
                        int(row.short_restricted),
                        (
                            None
                            if pd.isna(
                                row.stock_loan_fee_annual
                            )
                            else float(
                                row.stock_loan_fee_annual
                            )
                        ),
                        str(row.payload_hash),
                    ),
                )

                inserted += max(
                    int(cursor.rowcount),
                    0,
                )

        return inserted

    def load_shortability_observations(
        self,
        codes: list[str],
        *,
        available_before: str | pd.Timestamp,
        available_after: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """指定時刻までに利用可能だったPIT観測を読み込む。"""
        columns = [
            "code",
            "effective_at",
            "fetched_at",
            "available_at",
            "source",
            "short_type",
            "is_shortable",
            "is_margin_lendable",
            "short_restricted",
            "stock_loan_fee_annual",
            "payload_hash",
        ]

        if not codes:
            return pd.DataFrame(columns=columns)

        normalized_codes = [
            str(code).strip()
            for code in codes
        ]

        before = pd.Timestamp(available_before)

        if before.tzinfo is None:
            before = before.tz_localize(
                "Asia/Tokyo"
            )

        before_utc = before.tz_convert(
            "UTC"
        ).isoformat()

        placeholders = ",".join(
            ["?"] * len(normalized_codes)
        )

        where = [
            f"code IN ({placeholders})",
            "available_at <= ?",
        ]
        params: list[str] = [
            *normalized_codes,
            before_utc,
        ]

        if available_after is not None:
            after = pd.Timestamp(available_after)

            if after.tzinfo is None:
                after = after.tz_localize(
                    "Asia/Tokyo"
                )

            after_utc = after.tz_convert(
                "UTC"
            ).isoformat()

            where.append(
                "available_at >= ?"
            )
            params.append(after_utc)

        query = f"""
        SELECT {", ".join(columns)}
        FROM shortability_observations
        WHERE {" AND ".join(where)}
        ORDER BY
            code,
            available_at,
            effective_at,
            fetched_at
        """

        frame = pd.read_sql(
            query,
            self.conn,
            params=params,
        )

        for column in [
            "effective_at",
            "fetched_at",
            "available_at",
        ]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(
                    frame[column],
                    errors="raise",
                    utc=True,
                )

        return frame

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
