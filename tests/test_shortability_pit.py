"""Tests for point-in-time shortability observations."""

from __future__ import annotations

import pandas as pd
import pytest

from jp_signal.shortability_pit import (
    ShortabilityDecision,
    decide_shortability,
    load_shortability_observations_csv,
    normalize_shortability_observations,
)


def make_observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "7203",
                "effective_at": (
                    "2026-07-10T07:00:00+09:00"
                ),
                "fetched_at": (
                    "2026-07-10T07:05:00+09:00"
                ),
                "source": "test",
                "short_type": "system",
                "is_shortable": 1,
                "is_margin_lendable": 1,
                "short_restricted": 0,
                "stock_loan_fee_annual": None,
            }
        ]
    )


class TestNormalization:
    def test_normalization_calculates_available_at(self):
        result = normalize_shortability_observations(
            make_observations()
        )

        assert result.iloc[0][
            "available_at"
        ] == pd.Timestamp(
            "2026-07-10T07:05:00+09:00"
        ).tz_convert(
            "UTC"
        )

    def test_missing_required_column_raises(self):
        bad = make_observations().drop(
            columns=["effective_at"]
        )

        with pytest.raises(
            ValueError,
            match="missing columns",
        ):
            normalize_shortability_observations(
                bad
            )

    def test_empty_input_returns_empty(self):
        result = normalize_shortability_observations(
            pd.DataFrame()
        )

        assert result.empty

    def test_none_input_returns_empty(self):
        result = normalize_shortability_observations(
            None  # type: ignore[arg-type]
        )

        assert result.empty

    def test_invalid_short_type_raises(self):
        obs = make_observations()
        obs["short_type"] = "invalid"

        with pytest.raises(
            ValueError,
            match="invalid short_type",
        ):
            normalize_shortability_observations(
                obs
            )

    def test_system_requires_margin_lendable(self):
        obs = make_observations()
        obs["is_margin_lendable"] = None

        with pytest.raises(
            ValueError,
            match="system shortability requires",
        ):
            normalize_shortability_observations(
                obs
            )

    def test_general_allows_null_margin_lendable(self):
        obs = make_observations()
        obs["short_type"] = "general"
        obs["is_margin_lendable"] = None
        obs["stock_loan_fee_annual"] = 0.015

        result = normalize_shortability_observations(
            obs
        )

        assert len(result) == 1
        assert pd.isna(
            result.iloc[0]["is_margin_lendable"]
        )

    def test_negative_fee_raises(self):
        obs = make_observations()
        obs["stock_loan_fee_annual"] = -0.01

        with pytest.raises(
            ValueError,
            match="stock_loan_fee_annual",
        ):
            normalize_shortability_observations(
                obs
            )

    def test_payload_hash_is_deterministic(self):
        first = normalize_shortability_observations(
            make_observations()
        )
        second = normalize_shortability_observations(
            make_observations()
        )

        assert (
            first.iloc[0]["payload_hash"]
            == second.iloc[0]["payload_hash"]
        )


class TestDecideShortability:
    def test_observation_is_not_available_before_fetch(
        self,
    ):
        decision = decide_shortability(
            make_observations(),
            code="7203",
            as_of="2026-07-10T07:03:00+09:00",
            requested_short_type="system",
        )

        assert not decision.is_shortable
        assert decision.reason == (
            "NO_AVAILABLE_OBSERVATION"
        )

    def test_observation_is_available_after_fetch(self):
        decision = decide_shortability(
            make_observations(),
            code="7203",
            as_of="2026-07-10T08:15:00+09:00",
            requested_short_type="system",
        )

        assert decision.is_shortable
        assert decision.reason == "SHORTABLE"

    def test_future_effective_observation_not_used(
        self,
    ):
        observations = make_observations()
        observations.loc[
            0,
            "effective_at",
        ] = "2026-07-11T00:00:00+09:00"

        observations.loc[
            0,
            "fetched_at",
        ] = "2026-07-10T07:00:00+09:00"

        decision = decide_shortability(
            observations,
            code="7203",
            as_of="2026-07-10T08:15:00+09:00",
            requested_short_type="system",
        )

        assert not decision.is_shortable
        assert decision.reason == (
            "NO_AVAILABLE_OBSERVATION"
        )

    def test_restricted_stock_is_rejected(self):
        observations = make_observations()
        observations.loc[
            0,
            "short_restricted",
        ] = 1

        decision = decide_shortability(
            observations,
            code="7203",
            as_of="2026-07-10T08:15:00+09:00",
            requested_short_type="system",
        )

        assert not decision.is_shortable
        assert decision.reason == (
            "SHORT_RESTRICTED"
        )

    def test_system_short_requires_margin_lendable(
        self,
    ):
        observations = make_observations()
        observations.loc[
            0,
            "is_margin_lendable",
        ] = 0

        decision = decide_shortability(
            observations,
            code="7203",
            as_of="2026-07-10T08:15:00+09:00",
            requested_short_type="system",
        )

        assert not decision.is_shortable
        assert decision.reason == (
            "NOT_MARGIN_LENDABLE"
        )

    def test_general_short_does_not_require_margin_lendable(
        self,
    ):
        observations = make_observations()

        observations.loc[
            0,
            "short_type",
        ] = "general"
        observations.loc[
            0,
            "is_margin_lendable",
        ] = None
        observations.loc[
            0,
            "stock_loan_fee_annual",
        ] = 0.015

        decision = decide_shortability(
            observations,
            code="7203",
            as_of="2026-07-10T08:15:00+09:00",
            requested_short_type="general",
        )

        assert decision.is_shortable
        assert (
            decision.stock_loan_fee_annual
            == pytest.approx(0.015)
        )

    def test_stale_observation_is_rejected(self):
        decision = decide_shortability(
            make_observations(),
            code="7203",
            as_of="2026-07-20T08:15:00+09:00",
            requested_short_type="system",
            max_age=pd.Timedelta(days=4),
        )

        assert not decision.is_shortable
        assert decision.reason == (
            "STALE_OBSERVATION"
        )

    def test_invalid_short_type_raises(self):
        with pytest.raises(
            ValueError,
            match="invalid requested_short_type",
        ):
            decide_shortability(
                make_observations(),
                code="7203",
                as_of="2026-07-10T08:15:00+09:00",
                requested_short_type="invalid",  # type: ignore[arg-type]
            )

    def test_unknown_short_type_rejected(self):
        # 観測のshort_typeが'system'のため、'unknown'を要求しても該当なし
        decision = decide_shortability(
            make_observations(),
            code="7203",
            as_of="2026-07-10T08:15:00+09:00",
            requested_short_type="unknown",
        )

        assert not decision.is_shortable
        assert decision.reason == (
            "NO_AVAILABLE_OBSERVATION"
        )

    def test_none_observations_rejected(self):
        decision = decide_shortability(
            None,
            code="7203",
            as_of="2026-07-10T08:15:00+09:00",
            requested_short_type="system",
        )

        assert not decision.is_shortable
        assert decision.reason == (
            "NO_OBSERVATION"
        )

    def test_wrong_code_not_found(self):
        decision = decide_shortability(
            make_observations(),
            code="9999",
            as_of="2026-07-10T08:15:00+09:00",
            requested_short_type="system",
        )

        assert not decision.is_shortable
        assert decision.reason == (
            "NO_AVAILABLE_OBSERVATION"
        )


class TestShortabilityDecision:
    def test_to_dict_includes_all_fields(self):
        decision = ShortabilityDecision(
            code="7203",
            as_of=pd.Timestamp(
                "2026-07-10T08:15:00+09:00"
            ),
            is_shortable=True,
            short_type="system",
            reason="SHORTABLE",
            source="test",
            effective_at=pd.Timestamp(
                "2026-07-10T07:00:00+09:00"
            ),
            fetched_at=pd.Timestamp(
                "2026-07-10T07:05:00+09:00"
            ),
            available_at=pd.Timestamp(
                "2026-07-10T07:05:00+09:00"
            ),
            stock_loan_fee_annual=None,
        )

        d = decision.to_dict()

        assert d["code"] == "7203"
        assert d["is_shortable"] is True
        assert d["reason"] == "SHORTABLE"

    def test_decision_is_frozen(self):
        decision = ShortabilityDecision(
            code="7203",
            as_of=pd.Timestamp(
                "2026-07-10T08:15:00+09:00"
            ),
            is_shortable=False,
            short_type="system",
            reason="NO_OBSERVATION",
        )

        with pytest.raises(AttributeError):
            decision.is_shortable = True  # type: ignore[misc]


class TestStorage:
    def test_shortability_observation_is_idempotent(
        self,
        tmp_path,
    ):
        from jp_signal.storage import Storage

        db_path = tmp_path / "shortability.sqlite"
        observations = make_observations()

        with Storage(str(db_path)) as storage:
            first = (
                storage.insert_shortability_observations(
                    observations
                )
            )
            second = (
                storage.insert_shortability_observations(
                    observations
                )
            )

            loaded = (
                storage.load_shortability_observations(
                    ["7203"],
                    available_before=(
                        "2026-07-10T08:15:00+09:00"
                    ),
                )
            )

        assert first == 1
        assert second == 0
        assert len(loaded) == 1
        assert (
            loaded.iloc[0]["code"] == "7203"
        )

    def test_load_with_available_after_filter(
        self,
        tmp_path,
    ):
        from jp_signal.storage import Storage

        db_path = tmp_path / "shortability_filter.sqlite"
        observations = make_observations()

        with Storage(str(db_path)) as storage:
            storage.insert_shortability_observations(
                observations
            )

            # available_at is 2026-07-09T22:05:00Z (UTC)
            # which is 2026-07-10T07:05:00+09:00
            # So filter available_after=2026-07-09T21:00:00Z (UTC) should include it
            loaded = (
                storage.load_shortability_observations(
                    ["7203"],
                    available_before=(
                        "2026-07-10T08:15:00+09:00"
                    ),
                    available_after=(
                        "2026-07-10T06:00:00+09:00"
                    ),
                )
            )

            assert len(loaded) == 1

            # Should find nothing after a later cutoff
            empty = (
                storage.load_shortability_observations(
                    ["7203"],
                    available_before=(
                        "2026-07-10T08:15:00+09:00"
                    ),
                    available_after=(
                        "2026-07-10T08:00:00+09:00"
                    ),
                )
            )

            assert empty.empty


class TestCSVLoading:
    def test_load_nonexistent_csv(self):
        with pytest.raises(
            FileNotFoundError,
            match="shortability CSV not found",
        ):
            load_shortability_observations_csv(
                "/nonexistent/path.csv"
            )
