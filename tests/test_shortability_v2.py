"""shortability_v2（PIT shortability）のテスト。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from jp_signal.shortability_v2 import (
    ShortabilityDecision,
    decide_shortability,
    load_shortability_csv,
    normalize_shortability_frame,
)


def _make_shortability_df() -> pd.DataFrame:
    """基本的な shortability DataFrame を作成する。"""
    return pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-04",
                "effective_at": "2024-01-04T15:00:00",
                "fetched_at": "2024-01-04T16:00:00",
                "is_margin_lendable": 1,
                "short_restricted": 0,
            },
            {
                "code": "7203",
                "date": "2024-01-05",
                "effective_at": "2024-01-05T15:00:00",
                "fetched_at": "2024-01-05T16:00:00",
                "is_margin_lendable": 1,
                "short_restricted": 0,
            },
            {
                "code": "6758",
                "date": "2024-01-04",
                "effective_at": "2024-01-04T15:00:00",
                "fetched_at": "2024-01-04T16:00:00",
                "is_margin_lendable": 1,
                "short_restricted": 1,
            },
        ]
    )


class TestNormalizeShortabilityFrame:
    def test_normalize_renames_columns(self):
        df = pd.DataFrame(
            [
                {
                    "code": "7203",
                    "date": "2024-01-04",
                    "is_margin_lendable": 1,
                    "short_restricted": 0,
                }
            ]
        )
        result = normalize_shortability_frame(df)
        assert "code" in result.columns
        assert "is_margin_lendable" in result.columns
        assert "short_restricted" in result.columns
        assert "effective_at" in result.columns
        assert "fetched_at" in result.columns
        assert result.iloc[0]["code"] == "7203"

    def test_normalize_empty_df(self):
        result = normalize_shortability_frame(pd.DataFrame())
        assert result.empty

    def test_normalize_with_japanese_columns(self):
        df = pd.DataFrame(
            [
                {
                    "コード": "7203",
                    "日付": "2024-01-04",
                    "貸付可能": 1,
                    "空売り制限": 0,
                }
            ]
        )
        result = normalize_shortability_frame(
            df,
            code_col="コード",
            date_col="日付",
            lendable_col="貸付可能",
            restricted_col="空売り制限",
        )
        assert result.iloc[0]["code"] == "7203"
        assert result.iloc[0]["is_margin_lendable"] == 1
        assert result.iloc[0]["short_restricted"] == 0

    def test_normalize_adds_effective_fetched_at(self):
        df = pd.DataFrame(
            [
                {
                    "code": "7203",
                    "date": "2024-01-04",
                    "is_margin_lendable": 1,
                    "short_restricted": 0,
                }
            ]
        )
        result = normalize_shortability_frame(df)
        assert result.iloc[0]["effective_at"] == "2024-01-04T00:00:00"


class TestDecideShortability:
    def test_shortable_ok(self):
        df = _make_shortability_df()
        decision = decide_shortability(df, "7203", "2024-01-05")
        assert decision.is_shortable is True
        assert decision.reason == "OK"

    def test_not_shortable_by_flags(self):
        df = _make_shortability_df()
        decision = decide_shortability(df, "6758", "2024-01-05")
        assert decision.is_shortable is False
        assert "NOT_SHORTABLE_BY_FLAGS" in decision.reason

    def test_no_data_returns_not_shortable(self):
        decision = decide_shortability(None, "7203", "2024-01-05")
        assert decision.is_shortable is False
        assert decision.reason == "NO_SHORTABILITY_DATA"

    def test_no_data_for_code(self):
        df = _make_shortability_df()
        decision = decide_shortability(df, "9999", "2024-01-05")
        assert decision.is_shortable is False
        assert decision.reason == "NO_SHORTABILITY_DATA_FOR_CODE"

    def test_stale_data_rejected(self):
        df = pd.DataFrame(
            [
                {
                    "code": "7203",
                    "date": "2024-01-01",
                    "effective_at": "2024-01-01T15:00:00",
                    "fetched_at": "2024-01-01T16:00:00",
                    "is_margin_lendable": 1,
                    "short_restricted": 0,
                }
            ]
        )
        decision = decide_shortability(df, "7203", "2024-01-10", max_age_days=4)
        assert decision.is_shortable is False
        assert "STALE_DATA_AGE" in decision.reason

    def test_fresh_data_within_max_age(self):
        df = pd.DataFrame(
            [
                {
                    "code": "7203",
                    "date": "2024-01-08",
                    "effective_at": "2024-01-08T15:00:00",
                    "fetched_at": "2024-01-08T16:00:00",
                    "is_margin_lendable": 1,
                    "short_restricted": 0,
                }
            ]
        )
        # 月曜日のデータを金曜日(as_of=01-05)に使うには古すぎる
        decision = decide_shortability(df, "7203", "2024-01-05", max_age_days=4)
        assert decision.is_shortable is False
        assert "NO_EFFECTIVE_BEFORE_AS_OF" in decision.reason

    def test_picks_latest_effective(self):
        df = pd.DataFrame(
            [
                {
                    "code": "7203",
                    "date": "2024-01-02",
                    "effective_at": "2024-01-02T15:00:00",
                    "fetched_at": "2024-01-02T16:00:00",
                    "is_margin_lendable": 1,
                    "short_restricted": 1,
                },
                {
                    "code": "7203",
                    "date": "2024-01-04",
                    "effective_at": "2024-01-04T15:00:00",
                    "fetched_at": "2024-01-04T16:00:00",
                    "is_margin_lendable": 1,
                    "short_restricted": 0,
                },
            ]
        )
        decision = decide_shortability(df, "7203", "2024-01-05")
        assert decision.is_shortable is True


class TestShortabilityDecision:
    def test_dataclass_frozen(self):
        d = ShortabilityDecision(code="7203", effective_at="2024-01-04T15:00:00", is_shortable=True, reason="OK")
        assert d.code == "7203"
        assert d.is_shortable is True


class TestLoadShortabilityCsv:
    def test_csv_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_shortability_csv("/nonexistent/path.csv")

    def test_load_valid_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("code,date,is_margin_lendable,short_restricted\n")
            f.write("7203,2024-01-04,1,0\n")
            f.write("6758,2024-01-04,1,1\n")
            csv_path = f.name

        try:
            df = load_shortability_csv(csv_path)
            assert len(df) == 2
            assert df.iloc[0]["code"] == "7203"
            assert df.iloc[0]["is_margin_lendable"] == 1
        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_load_japanese_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("コード,日付,貸付可能,空売り制限\n")
            f.write("7203,2024-01-04,1,0\n")
            f.write("6758,2024-01-04,1,1\n")
            csv_path = f.name

        try:
            df = load_shortability_csv(csv_path)
            assert len(df) == 2
            assert df.iloc[0]["code"] == "7203"
            assert df.iloc[0]["is_margin_lendable"] == 1
        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_missing_required_columns_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("code,date\n")
            f.write("7203,2024-01-04\n")
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="必須列"):
                load_shortability_csv(csv_path)
        finally:
            Path(csv_path).unlink(missing_ok=True)
