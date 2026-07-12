"""Tests for shortability_provider module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from jp_signal.shortability_provider import (
    CsvShortabilityDataSource,
    ShortabilityDataSource,
    refresh_shortability,
)


def test_shortability_data_source_is_abstract():
    with pytest.raises(TypeError):
        ShortabilityDataSource()  # type: ignore[abstract]


def test_csv_shortability_data_source_file_not_found():
    with pytest.raises(FileNotFoundError):
        CsvShortabilityDataSource("/nonexistent/path.csv")


def test_csv_shortability_data_source_empty_codes():
    csv_content = "code,shortable\n1001,true\n1002,false\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        ds = CsvShortabilityDataSource(path)
        result = ds.fetch([], pd.Timestamp("2026-07-12"))
        assert result.empty
        assert list(result.columns) == ["code", "shortable", "updated_at"]
    finally:
        Path(path).unlink(missing_ok=True)


def test_csv_shortability_data_source_fetch():
    csv_content = "code,shortable\n1001,true\n1002,false\n1003,true\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        ds = CsvShortabilityDataSource(path)
        result = ds.fetch(["1001", "1002"], pd.Timestamp("2026-07-12"))

        assert len(result) == 2
        assert set(result["code"]) == {"1001", "1002"}
        assert result[result["code"] == "1001"]["shortable"].iloc[0] == True
        assert result[result["code"] == "1002"]["shortable"].iloc[0] == False
    finally:
        Path(path).unlink(missing_ok=True)


def test_csv_shortability_data_source_with_updated_at():
    csv_content = "code,shortable,updated_at\n1001,true,2026-07-10\n1002,false,2026-07-11\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        ds = CsvShortabilityDataSource(path)
        result = ds.fetch(["1001"], pd.Timestamp("2026-07-12"))

        assert len(result) == 1
        assert result["updated_at"].iloc[0] == "2026-07-10"
    finally:
        Path(path).unlink(missing_ok=True)


def test_refresh_shortability():
    csv_content = "code,shortable\n1001,true\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        ds = CsvShortabilityDataSource(path)
        result = refresh_shortability(ds, ["1001"], pd.Timestamp("2026-07-12"))
        assert len(result) == 1
    finally:
        Path(path).unlink(missing_ok=True)
