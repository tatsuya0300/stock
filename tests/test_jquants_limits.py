"""J-Quants limits module tests."""

import pytest

from jp_signal.jquants_limits import (
    JQUANTS_MAX_INTERVAL_SEC,
    JQUANTS_PLAN_MIN_INTERVAL_SEC,
    normalize_jquants_plan,
    resolve_jquants_sleep_sec,
)


class TestNormalizeJQuantsPlan:
    def test_free_plan(self):
        assert normalize_jquants_plan("free") == "free"

    def test_light_plan(self):
        assert normalize_jquants_plan("light") == "light"

    def test_standard_plan(self):
        assert normalize_jquants_plan("standard") == "standard"

    def test_premium_plan(self):
        assert normalize_jquants_plan("premium") == "premium"

    def test_case_insensitive(self):
        assert normalize_jquants_plan(" FREE ") == "free"
        assert normalize_jquants_plan("Light") == "light"
        assert normalize_jquants_plan("STANDARD") == "standard"

    def test_invalid_plan_raises(self):
        with pytest.raises(ValueError, match="plan"):
            normalize_jquants_plan("enterprise")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="plan"):
            normalize_jquants_plan("")


class TestResolveJQuantsSleepSec:
    def test_none_uses_plan_minimum_free(self):
        plan, sleep = resolve_jquants_sleep_sec(plan="free", sleep_sec=None)
        assert plan == "free"
        assert sleep == JQUANTS_PLAN_MIN_INTERVAL_SEC["free"]

    def test_none_uses_plan_minimum_light(self):
        plan, sleep = resolve_jquants_sleep_sec(plan="light", sleep_sec=None)
        assert plan == "light"
        assert sleep == JQUANTS_PLAN_MIN_INTERVAL_SEC["light"]

    def test_none_uses_plan_minimum_standard(self):
        plan, sleep = resolve_jquants_sleep_sec(plan="standard", sleep_sec=None)
        assert plan == "standard"
        assert sleep == JQUANTS_PLAN_MIN_INTERVAL_SEC["standard"]

    def test_none_uses_plan_minimum_premium(self):
        plan, sleep = resolve_jquants_sleep_sec(plan="premium", sleep_sec=None)
        assert plan == "premium"
        assert sleep == JQUANTS_PLAN_MIN_INTERVAL_SEC["premium"]

    def test_explicit_interval_accepted(self):
        plan, sleep = resolve_jquants_sleep_sec(plan="free", sleep_sec=30.0)
        assert plan == "free"
        assert sleep == 30.0

    def test_below_minimum_rejected(self):
        with pytest.raises(ValueError, match="sleep_sec"):
            resolve_jquants_sleep_sec(plan="free", sleep_sec=1.0)

    def test_exceeds_maximum_rejected(self):
        with pytest.raises(ValueError, match="sleep_sec"):
            resolve_jquants_sleep_sec(
                plan="free",
                sleep_sec=JQUANTS_MAX_INTERVAL_SEC + 1.0,
            )

    def test_boundary_maximum_accepted(self):
        plan, sleep = resolve_jquants_sleep_sec(
            plan="free",
            sleep_sec=JQUANTS_MAX_INTERVAL_SEC,
        )
        assert sleep == JQUANTS_MAX_INTERVAL_SEC

    def test_invalid_plan_raises(self):
        with pytest.raises(ValueError, match="plan"):
            resolve_jquants_sleep_sec(plan="enterprise", sleep_sec=None)

    def test_case_insensitive_plan(self):
        plan, sleep = resolve_jquants_sleep_sec(plan=" FREE ", sleep_sec=None)
        assert plan == "free"
        assert sleep == JQUANTS_PLAN_MIN_INTERVAL_SEC["free"]

    @pytest.mark.parametrize(
        "plan,expected_min",
        [
            ("free", 12.0),
            ("light", 1.0),
            ("standard", 0.5),
            ("premium", 0.2),
        ],
    )
    def test_all_plan_minimums(self, plan: str, expected_min: float):
        _, sleep = resolve_jquants_sleep_sec(plan=plan, sleep_sec=None)
        assert sleep == expected_min
