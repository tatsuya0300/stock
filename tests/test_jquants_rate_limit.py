"""J-Quants rate-limit tests.

J-Quants のレート制限は将来変更され得るため、マージ前に公式仕様を再確認してください。
公式: https://jpx-jquants.com/en/spec/rate-limit
"""

import pytest

from jp_signal.datasource import JQuantsSource


def test_default_sleep_sec_is_valid_for_free_plan():
    """free plan の最小間隔 (12s) に対してデフォルト sleep_sec=0.3 はエラーになるはず。"""
    with pytest.raises(ValueError, match="sleep_sec"):
        JQuantsSource("dummy-api-key")


def test_free_plan_minimum_sleep_sec():
    """free plan では 12.0s 以上が必要。"""
    source = JQuantsSource("dummy-api-key", sleep_sec=12.0, plan="free")
    assert source.sleep_sec == 12.0


def test_free_plan_below_minimum_rejected():
    """free plan で 12.0s 未満は拒否。"""
    with pytest.raises(ValueError, match="sleep_sec"):
        JQuantsSource("dummy-api-key", sleep_sec=11.0, plan="free")


def test_light_plan_minimum_sleep_sec():
    """light plan では 1.0s 以上が必要。"""
    source = JQuantsSource("dummy-api-key", sleep_sec=1.0, plan="light")
    assert source.sleep_sec == 1.0


def test_standard_plan_minimum_sleep_sec():
    """standard plan では 0.5s 以上が必要。"""
    source = JQuantsSource("dummy-api-key", sleep_sec=0.5, plan="standard")
    assert source.sleep_sec == 0.5


def test_premium_plan_minimum_sleep_sec():
    """premium plan では 0.2s 以上が必要。"""
    source = JQuantsSource("dummy-api-key", sleep_sec=0.2, plan="premium")
    assert source.sleep_sec == 0.2


def test_sleep_sec_cannot_exceed_120s():
    """sleep_sec は 120s を超えられない。"""
    with pytest.raises(ValueError, match="sleep_sec"):
        JQuantsSource("dummy-api-key", sleep_sec=121.0, plan="free")


def test_invalid_plan_rejected():
    """存在しない plan は拒否。"""
    with pytest.raises(ValueError, match="plan"):
        JQuantsSource("dummy-api-key", sleep_sec=12.0, plan="enterprise")
