"""price_observation_values テーブルと load_prices_asof のテスト。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from jp_signal.storage import (
    PRICE_COLS,
    Storage,
)


@pytest.fixture
def db_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return f.name


@pytest.fixture
def storage(db_path: str) -> Storage:
    s = Storage(db_path)
    yield s
    s.close()
    Path(db_path).unlink(missing_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────


def _make_price_row(
    code: str = "7203",
    date: str = "2024-01-04",
    close: float = 100.0,
    **overrides,
) -> dict:
    """PRICE_COLS を持つ1行の辞書を返す。"""
    row = {
        "code": code,
        "date": date,
        "open": close * 0.99 if close else None,
        "high": close * 1.02 if close else None,
        "low": close * 0.98 if close else None,
        "close": close,
        "adj_open": close * 0.99 if close else None,
        "adj_high": close * 1.02 if close else None,
        "adj_low": close * 0.98 if close else None,
        "adj_close": close,
        "volume": 1_000_000,
        "turnover": close * 1_000_000 if close else None,
    }
    row.update(overrides)
    return row


def _make_price_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── price_observation_values テーブルの存在 ──────────────────────────────


def test_price_observation_values_table_exists(storage: Storage):
    """price_observation_values テーブルが作成されていること。"""
    rows = storage.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='price_observation_values'"
    ).fetchall()
    assert len(rows) == 1


def test_price_observation_values_has_expected_columns(storage: Storage):
    """price_observation_values が全OHLCカラムを持つこと。"""
    cols = {
        r[1]
        for r in storage.conn.execute(
            "PRAGMA table_info(price_observation_values)"
        ).fetchall()
    }
    for c in [
        "id",
        "code",
        "date",
        "source",
        "fetched_at",
        "available_at",
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
        "payload_hash",
    ]:
        assert c in cols, f"column {c} missing from price_observation_values"


# ── ingest_prices による price_observation_values への書込み ────────────


def test_ingest_prices_inserts_into_price_observation_values(storage: Storage):
    """ingest_prices() が price_observation_values に行を挿入すること。"""
    df = _make_price_df([_make_price_row()])
    storage.ingest_prices(df, source="jquants")

    rows = storage.conn.execute(
        "SELECT code, date, source, close FROM price_observation_values"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "7203"
    assert rows[0][1] == "2024-01-04"
    assert rows[0][2] == "jquants"
    assert rows[0][3] == 100.0


def test_ingest_prices_inserts_adj_columns_filled(storage: Storage):
    """adj_* カラムが無い場合、raw カラムで補完されること。"""
    df = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000,
                "turnover": 100_000_000.0,
            }
        ]
    )
    storage.ingest_prices(df, source="jquants")

    row = storage.conn.execute(
        "SELECT adj_open, adj_high, adj_low, adj_close FROM price_observation_values"
    ).fetchone()
    assert row[0] == 100.0  # adj_open = open
    assert row[1] == 101.0  # adj_high = high
    assert row[2] == 99.0  # adj_low = low
    assert row[3] == 100.5  # adj_close = close


def test_ingest_prices_dedup_by_payload_hash_in_pov(storage: Storage):
    """price_observation_values で同一 payload_hash の行が重複しないこと。"""
    df = _make_price_df([_make_price_row()])
    storage.ingest_prices(df, source="jquants")
    storage.ingest_prices(df, source="jquants")

    count = storage.conn.execute(
        "SELECT COUNT(*) FROM price_observation_values"
    ).fetchone()[0]
    assert count == 1  # 重複しない


def test_ingest_prices_different_hash_inserts_new_pov_row(storage: Storage):
    """異なるデータは price_observation_values に別行として記録されること。"""
    df1 = _make_price_df([_make_price_row(close=100.0)])
    df2 = _make_price_df([_make_price_row(close=101.0)])

    storage.ingest_prices(df1, source="jquants")
    storage.ingest_prices(df2, source="jquants")

    count = storage.conn.execute(
        "SELECT COUNT(*) FROM price_observation_values"
    ).fetchone()[0]
    assert count == 2


# ── load_prices_asof ────────────────────────────────────────────────────


def test_load_prices_asof_returns_latest_before_asof(storage: Storage):
    """load_prices_asof() が asof 時点で利用可能な最新リビジョンを返すこと。"""
    # 1回目の取込 (close=100.0)
    df1 = _make_price_df([_make_price_row(close=100.0)])
    storage.ingest_prices(df1, source="jquants", available_at="2024-01-04T05:00:00+00:00")

    # 2回目の取込 (close=101.0 — 修正)
    df2 = _make_price_df([_make_price_row(close=101.0)])
    storage.ingest_prices(df2, source="jquants", available_at="2024-01-04T06:00:00+00:00")

    # asof=05:30 時点では 1回目のリビジョンが見える
    result = storage.load_prices_asof(
        asof_date="2024-01-04T05:30:00+00:00",
        codes=["7203"],
        start="2024-01-04",
        end="2024-01-04",
    )
    assert not result.empty
    assert float(result.iloc[0]["close"]) == 100.0

    # asof=06:30 時点では 2回目のリビジョンが見える
    result2 = storage.load_prices_asof(
        asof_date="2024-01-04T06:30:00+00:00",
        codes=["7203"],
        start="2024-01-04",
        end="2024-01-04",
    )
    assert not result2.empty
    assert float(result2.iloc[0]["close"]) == 101.0


def test_load_prices_asof_empty_codes(storage: Storage):
    """codes が空なら空の DataFrame を返すこと。"""
    result = storage.load_prices_asof(
        asof_date="2024-01-04T06:00:00+00:00",
        codes=[],
        start="2024-01-04",
        end="2024-01-04",
    )
    assert result.empty
    assert list(result.columns) == PRICE_COLS


def test_load_prices_asof_no_data_returns_empty(storage: Storage):
    """該当データが無ければ空の DataFrame を返すこと。"""
    result = storage.load_prices_asof(
        asof_date="2024-01-04T06:00:00+00:00",
        codes=["9999"],
        start="2024-01-04",
        end="2024-01-04",
    )
    assert result.empty


def test_load_prices_asof_respects_available_at(storage: Storage):
    """available_at が asof_date より後のデータは見えないこと。"""
    df = _make_price_df([_make_price_row(close=100.0)])
    storage.ingest_prices(df, source="jquants", available_at="2024-01-04T08:00:00+00:00")

    # asof が available_at より前 → データは見えない
    result = storage.load_prices_asof(
        asof_date="2024-01-04T07:00:00+00:00",
        codes=["7203"],
        start="2024-01-04",
        end="2024-01-04",
    )
    assert result.empty

    # asof が available_at より後 → データが見える
    result2 = storage.load_prices_asof(
        asof_date="2024-01-04T09:00:00+00:00",
        codes=["7203"],
        start="2024-01-04",
        end="2024-01-04",
    )
    assert not result2.empty


def test_load_prices_asof_multiple_codes(storage: Storage):
    """複数コードのポイントインタイム取得ができること。"""
    rows = [
        _make_price_row(code="7203", date="2024-01-04", close=100.0),
        _make_price_row(code="7203", date="2024-01-05", close=101.0),
        _make_price_row(code="9984", date="2024-01-04", close=5000.0),
        _make_price_row(code="9984", date="2024-01-05", close=5100.0),
    ]
    storage.ingest_prices(
        _make_price_df(rows),
        source="jquants",
        available_at="2024-01-05T06:00:00+00:00",
    )

    result = storage.load_prices_asof(
        asof_date="2024-01-05T07:00:00+00:00",
        codes=["7203", "9984"],
        start="2024-01-04",
        end="2024-01-05",
    )
    assert len(result) == 4  # 2 codes × 2 dates
    assert set(result["code"]) == {"7203", "9984"}


def test_load_prices_asof_returns_all_price_cols(storage: Storage):
    """load_prices_asof() が PRICE_COLS を全て含むこと。"""
    df = _make_price_df([_make_price_row()])
    storage.ingest_prices(df, source="jquants", available_at="2024-01-04T06:00:00+00:00")

    result = storage.load_prices_asof(
        asof_date="2024-01-04T07:00:00+00:00",
        codes=["7203"],
        start="2024-01-04",
        end="2024-01-04",
    )
    assert list(result.columns) == PRICE_COLS


def test_load_prices_asof_orders_by_code_date(storage: Storage):
    """結果が code, date 順にソートされていること。"""
    rows = [
        _make_price_row(code="9984", date="2024-01-05", close=5100.0),
        _make_price_row(code="7203", date="2024-01-05", close=101.0),
        _make_price_row(code="7203", date="2024-01-04", close=100.0),
    ]
    storage.ingest_prices(
        _make_price_df(rows),
        source="jquants",
        available_at="2024-01-05T06:00:00+00:00",
    )

    result = storage.load_prices_asof(
        asof_date="2024-01-05T07:00:00+00:00",
        codes=["7203", "9984"],
        start="2024-01-04",
        end="2024-01-05",
    )
    codes = list(result["code"])
    dates = list(result["date"])
    # 7203-01-04, 7203-01-05, 9984-01-05 の順
    assert codes[0] == "7203"
    assert dates[0] == "2024-01-04"
    assert codes[1] == "7203"
    assert dates[1] == "2024-01-05"
    assert codes[2] == "9984"
    assert dates[2] == "2024-01-05"


def test_load_prices_asof_date_range_filtering(storage: Storage):
    """start/end の日付範囲でフィルタリングされること。"""
    rows = [
        _make_price_row(code="7203", date="2024-01-03", close=99.0),
        _make_price_row(code="7203", date="2024-01-04", close=100.0),
        _make_price_row(code="7203", date="2024-01-05", close=101.0),
    ]
    storage.ingest_prices(
        _make_price_df(rows),
        source="jquants",
        available_at="2024-01-05T06:00:00+00:00",
    )

    result = storage.load_prices_asof(
        asof_date="2024-01-05T07:00:00+00:00",
        codes=["7203"],
        start="2024-01-04",
        end="2024-01-04",
    )
    assert len(result) == 1
    assert result.iloc[0]["date"] == "2024-01-04"


# ── 同一 available_at での複数リビジョン ──────────────────────────────


def test_load_prices_asof_multiple_revisions_same_available_at(storage: Storage):
    """同一 available_at に複数リビジョンがある場合、fetched_at が最新のものを返すこと。"""
    # available_at を同じに設定して2回取込
    df1 = _make_price_df([_make_price_row(close=100.0)])
    storage.ingest_prices(df1, source="jquants", available_at="2024-01-04T06:00:00+00:00")

    df2 = _make_price_df([_make_price_row(close=101.0)])
    storage.ingest_prices(df2, source="jquants", available_at="2024-01-04T06:00:00+00:00")

    result = storage.load_prices_asof(
        asof_date="2024-01-04T07:00:00+00:00",
        codes=["7203"],
        start="2024-01-04",
        end="2024-01-04",
    )
    assert not result.empty
    assert float(result.iloc[0]["close"]) == 101.0  # fetched_at が最新の方


# ── schema_version ──────────────────────────────────────────────────────


def test_schema_version_is_5(storage: Storage):
    """新規DBの schema_version が 5 であること。"""
    v = storage.get_metadata("schema_version")
    assert v == "5"
