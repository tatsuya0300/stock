"""price_observations テーブルと ingest_prices のテスト。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from jp_signal.storage import (
    Storage,
    _price_payload_hash,
    _utc_now_iso,
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


def test_utc_now_iso_format():
    """_utc_now_iso() が ISO 8601 形式の文字列を返すことを確認。"""
    result = _utc_now_iso()
    assert isinstance(result, str)
    assert "T" in result  # ISO 8601 contains T separator
    assert result.endswith("+00:00") or "+00:" in result


def test_price_payload_hash_deterministic():
    """同一データからは常に同じハッシュが得られること。"""
    row = {
        "code": "7203",
        "date": "2024-01-04",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "adj_open": 100.0,
        "adj_high": 101.0,
        "adj_low": 99.0,
        "adj_close": 100.5,
        "volume": 1_000_000,
        "turnover": 100_000_000.0,
    }
    h1 = _price_payload_hash(row)
    h2 = _price_payload_hash(row)
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex digest length


def test_price_payload_hash_different_data():
    """異なるデータからは異なるハッシュが得られること。"""
    row1 = {
        "code": "7203",
        "date": "2024-01-04",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "adj_open": 100.0,
        "adj_high": 101.0,
        "adj_low": 99.0,
        "adj_close": 100.5,
        "volume": 1_000_000,
        "turnover": 100_000_000.0,
    }
    row2 = row1.copy()
    row2["close"] = 101.0  # 変更
    assert _price_payload_hash(row1) != _price_payload_hash(row2)


def test_record_price_observations_inserts_new(storage: Storage):
    """新しいprice_observations行が挿入されること。"""
    df = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_open": 100.0,
                "adj_high": 101.0,
                "adj_low": 99.0,
                "adj_close": 100.5,
                "volume": 1_000_000,
                "turnover": 100_000_000.0,
            }
        ]
    )

    n = storage.record_price_observations(df, source="jquants", available_at="2024-01-04T08:00:00+00:00")
    assert n > 0

    rows = storage.conn.execute(
        "SELECT code, date, source, available_at FROM price_observations"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "7203"  # code
    assert rows[0][1] == "2024-01-04"  # date
    assert rows[0][2] == "jquants"  # source


def test_record_price_observations_dedup_by_hash(storage: Storage):
    """同一ハッシュの観測は重複挿入されないこと。"""
    df = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_open": 100.0,
                "adj_high": 101.0,
                "adj_low": 99.0,
                "adj_close": 100.5,
                "volume": 1_000_000,
                "turnover": 100_000_000.0,
            }
        ]
    )

    storage.record_price_observations(df, source="jquants")
    storage.record_price_observations(df, source="jquants")

    rows = storage.conn.execute(
        "SELECT COUNT(*) FROM price_observations"
    ).fetchone()
    assert rows[0] == 1  # 重複しない


def test_record_price_observations_different_hash_inserts_new_row(storage: Storage):
    """同一(code, date, source)でもハッシュが違えば新しい行が挿入されること。"""
    df1 = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_open": 100.0,
                "adj_high": 101.0,
                "adj_low": 99.0,
                "adj_close": 100.5,
                "volume": 1_000_000,
                "turnover": 100_000_000.0,
            }
        ]
    )
    df2 = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "open": 100.0,
                "high": 102.0,  # 修正値
                "low": 99.0,
                "close": 101.5,
                "adj_open": 100.0,
                "adj_high": 102.0,
                "adj_low": 99.0,
                "adj_close": 101.5,
                "volume": 1_000_000,
                "turnover": 100_000_000.0,
            }
        ]
    )

    storage.record_price_observations(df1, source="jquants")
    storage.record_price_observations(df2, source="jquants")

    rows = storage.conn.execute(
        "SELECT COUNT(*) FROM price_observations"
    ).fetchone()
    assert rows[0] == 2  # 2つの異なるリビジョン


def test_record_price_observations_empty_df(storage: Storage):
    """空のDataFrameは何も挿入しないこと。"""
    n = storage.record_price_observations(pd.DataFrame(), source="jquants")
    assert n == 0


def test_ingest_prices_updates_prices_and_records_observation(storage: Storage):
    """ingest_prices() は prices の更新と price_observations の記録を同時に行うこと。"""
    df = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "adj_open": 100.0,
                "adj_high": 101.0,
                "adj_low": 99.0,
                "adj_close": 100.5,
                "volume": 1_000_000,
                "turnover": 100_000_000.0,
            }
        ]
    )

    storage.ingest_prices(df, source="jquants")

    # prices テーブルが更新されている
    prices = storage.load_prices(["7203"], start="2024-01-04", end="2024-01-04")
    assert not prices.empty
    assert float(prices.iloc[0]["close"]) == 100.5

    # price_observations にも記録されている
    obs = storage.conn.execute(
        "SELECT COUNT(*) FROM price_observations WHERE code='7203' AND source='jquants'"
    ).fetchone()
    assert obs[0] == 1


def test_ingest_prices_empty_df(storage: Storage):
    """空のDataFrameでingest_prices()を呼んでもエラーにならないこと。"""
    storage.ingest_prices(pd.DataFrame(), source="jquants")  # 例外が発生しない
