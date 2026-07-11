"""PITユニバースの回帰テスト。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from jp_signal.universe import load_universe


def write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "universe.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_static_universe_rejects_duplicate_code(tmp_path: Path):
    path = write_csv(
        tmp_path,
        [
            {"code": "7203", "name": "Toyota"},
            {"code": "7203", "name": "Toyota duplicate"},
        ],
    )

    with pytest.raises(ValueError, match="重複コード"):
        load_universe(str(path))


def test_pit_universe_allows_non_overlapping_reentry(tmp_path: Path):
    path = write_csv(
        tmp_path,
        [
            {
                "code": "1234",
                "name": "Example old",
                "effective_from": "2018-01-01",
                "effective_to": "2020-12-31",
            },
            {
                "code": "1234",
                "name": "Example new",
                "effective_from": "2023-01-01",
                "effective_to": "",
            },
        ],
    )

    old = load_universe(str(path), as_of="2019-06-01")
    absent = load_universe(str(path), as_of="2022-01-01")
    new = load_universe(str(path), as_of="2024-01-01")
    all_codes = load_universe(str(path))

    assert old["code"].tolist() == ["1234"]
    assert old.iloc[0]["name"] == "Example old"

    assert absent.empty

    assert new["code"].tolist() == ["1234"]
    assert new.iloc[0]["name"] == "Example new"

    # as_of未指定では取得対象コードの集合を返す。
    assert all_codes["code"].tolist() == ["1234"]
    assert all_codes.iloc[0]["name"] == "Example new"


def test_pit_universe_rejects_overlapping_intervals(tmp_path: Path):
    path = write_csv(
        tmp_path,
        [
            {
                "code": "1234",
                "name": "Example",
                "effective_from": "2020-01-01",
                "effective_to": "2022-12-31",
            },
            {
                "code": "1234",
                "name": "Example",
                "effective_from": "2022-12-31",
                "effective_to": "2024-12-31",
            },
        ],
    )

    with pytest.raises(ValueError, match="有効期間が重複"):
        load_universe(str(path))


def test_pit_universe_rejects_invalid_range(tmp_path: Path):
    path = write_csv(
        tmp_path,
        [
            {
                "code": "1234",
                "name": "Example",
                "effective_from": "2024-01-01",
                "effective_to": "2023-12-31",
            }
        ],
    )

    with pytest.raises(ValueError, match="effective_from"):
        load_universe(str(path))


def test_effective_to_is_inclusive(tmp_path: Path):
    path = write_csv(
        tmp_path,
        [
            {
                "code": "1234",
                "name": "Example",
                "effective_from": "2020-01-01",
                "effective_to": "2023-12-31",
            }
        ],
    )

    result = load_universe(str(path), as_of="2023-12-31")

    assert result["code"].tolist() == ["1234"]
