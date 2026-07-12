"""J-Quants rate-limit tests.

J-Quants のレート制限は将来変更され得るため、マージ前に公式仕様を再確認してください。
公式: https://jpx-jquants.com/en/spec/rate-limit

Datasource 層のテストは jquants_limits モジュールを正しく利用していることを確認する。
実際の検証ロジックは tests/test_jquants_limits.py に集約されている。
"""

import pytest

from jp_signal.datasource import JQuantsSource
from jp_signal.jquants_limits import JQUANTS_PLAN_MIN_INTERVAL_SEC


def test_default_sleep_sec_uses_plan_minimum():
    """デフォルト sleep_sec=None は free plan の最小値 (12.0s) を使用する。"""
    source = JQuantsSource("dummy-api-key")
    assert source.plan == "free"
    assert source.sleep_sec == JQUANTS_PLAN_MIN_INTERVAL_SEC["free"]


@pytest.mark.parametrize(
    "plan,expected_min",
    [
        ("free", 12.0),
        ("light", 1.0),
        ("standard", 0.5),
        ("premium", 0.2),
    ],
)
def test_plan_minimum_sleep_sec(plan: str, expected_min: float):
    """各プランで sleep_sec=None がプラン最小値になる。"""
    source = JQuantsSource("dummy-api-key", plan=plan)
    assert source.plan == plan
    assert source.sleep_sec == expected_min


def test_explicit_sleep_sec_accepted():
    """明示的な sleep_sec 指定が受け入れられる。"""
    source = JQuantsSource("dummy-api-key", sleep_sec=30.0, plan="free")
    assert source.sleep_sec == 30.0


def test_below_minimum_rejected():
    """プラン最小値未満の sleep_sec は拒否。"""
    with pytest.raises(ValueError, match="sleep_sec"):
        JQuantsSource("dummy-api-key", sleep_sec=11.0, plan="free")


def test_sleep_sec_cannot_exceed_120s():
    """sleep_sec は 120s を超えられない。"""
    with pytest.raises(ValueError, match="sleep_sec"):
        JQuantsSource("dummy-api-key", sleep_sec=121.0, plan="free")


def test_invalid_plan_rejected():
    """存在しない plan は拒否。"""
    with pytest.raises(ValueError, match="plan"):
        JQuantsSource("dummy-api-key", sleep_sec=12.0, plan="enterprise")


def test_empty_api_key_rejected():
    """空の API Key は拒否。"""
    with pytest.raises(ValueError, match="API_KEY"):
        JQuantsSource("")


def test_plan_stored_as_normalized():
    """プラン名が正規化されて保存される。"""
    source = JQuantsSource("dummy-api-key", plan=" LIGHT ")
    assert source.plan == "light"
