"""Tests for backtest input snapshot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from jp_signal.snapshot import (
    _canonical_frame,
    _sha256_file,
    write_backtest_input_snapshot,
)


def _universe_csv(tmp_path: Path) -> Path:
    p = tmp_path / "universe.csv"
    p.write_text("code,name\n7203,Toyota\n", encoding="utf-8")
    return p


def _prices_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["7203", "7203"],
            "date": ["2024-01-04", "2024-01-05"],
            "close": [100.0, 102.0],
            "volume": [1000, 2000],
        }
    )


def test_writes_prices_csv(tmp_path):
    out = tmp_path / "snap"
    univ = _universe_csv(tmp_path)
    prices = _prices_df()

    meta = write_backtest_input_snapshot(
        output_dir=out,
        prices=prices,
        shortability=None,
        universe_file=univ,
    )

    assert meta["prices"]["file"] == "prices.csv"
    assert meta["prices"]["rows"] == 2
    assert meta["prices"]["sha256"] == _sha256_file(out / "prices.csv")


def test_writes_shortability_csv(tmp_path):
    out = tmp_path / "snap"
    univ = _universe_csv(tmp_path)
    prices = _prices_df()
    short = pd.DataFrame(
        {
            "code": ["7203"],
            "effective_at": pd.to_datetime(["2024-01-04"]),
            "fetched_at": pd.to_datetime(["2024-01-04"]),
            "source": ["jsf"],
            "short_type": ["system"],
            "is_shortable": [1],
            "short_restricted": [0],
        }
    )

    meta = write_backtest_input_snapshot(
        output_dir=out,
        prices=prices,
        shortability=short,
        universe_file=univ,
    )

    assert meta["shortability"]["file"] == "shortability.csv"
    assert meta["shortability"]["rows"] == 1
    assert meta["shortability"]["sha256"] == _sha256_file(out / "shortability.csv")


def test_shortability_is_empty_when_none(tmp_path):
    out = tmp_path / "snap"
    univ = _universe_csv(tmp_path)
    prices = _prices_df()

    meta = write_backtest_input_snapshot(
        output_dir=out,
        prices=prices,
        shortability=None,
        universe_file=univ,
    )

    assert meta["shortability"]["rows"] == 0
    assert meta["shortability"]["sha256"] == _sha256_file(out / "shortability.csv")


def test_copies_universe_file(tmp_path):
    out = tmp_path / "snap"
    univ = _universe_csv(tmp_path)
    prices = _prices_df()

    meta = write_backtest_input_snapshot(
        output_dir=out,
        prices=prices,
        shortability=None,
        universe_file=univ,
    )

    assert meta["universe"]["file"] == "universe.csv"
    assert meta["universe"]["sha256"] == _sha256_file(univ)


def test_universe_file_not_found(tmp_path):
    prices = _prices_df()

    with pytest.raises(FileNotFoundError):
        write_backtest_input_snapshot(
            output_dir=tmp_path / "snap",
            prices=prices,
            shortability=None,
            universe_file=tmp_path / "nonexistent.csv",
        )


def test_writes_snapshot_json(tmp_path):
    out = tmp_path / "snap"
    univ = _universe_csv(tmp_path)
    prices = _prices_df()

    meta = write_backtest_input_snapshot(
        output_dir=out,
        prices=prices,
        shortability=None,
        universe_file=univ,
    )

    assert (out / "snapshot.json").exists()
    assert "format_version" in meta
    assert meta["format_version"] == 1
    assert "metadata_sha256" in meta


def test_canonical_frame_sorts_columns():
    df = pd.DataFrame({"z": [1], "a": [2], "code": ["7203"]})
    result = _canonical_frame(df)
    assert list(result.columns) == ["a", "code", "z"]


def test_canonical_frame_sorts_rows():
    df = pd.DataFrame(
        {
            "code": ["7203", "7203", "7203"],
            "date": ["2024-01-05", "2024-01-04", "2024-01-06"],
        }
    )
    result = _canonical_frame(df)
    assert result["date"].tolist() == [
        "2024-01-04",
        "2024-01-05",
        "2024-01-06",
    ]


def test_empty_frame_returns_empty():
    result = _canonical_frame(pd.DataFrame())
    assert result.empty
