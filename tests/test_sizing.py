"""compute_size の回帰テスト。"""

import pytest

from jp_signal.sizing import compute_size


def test_cap_limits_target_notional():
    qty, yen, _ = compute_size(
        prev_turnover=10_000_000.0,
        ref_price=1000.0,
        adv_ratio=0.001,
        adv_ratio_cap=0.002,
        target_notional=50_000.0,
        unit=100,
    )
    assert qty == 0
    assert yen == 0.0


def test_default_uses_adv_ratio():
    qty, yen, _ = compute_size(
        prev_turnover=100_000_000.0,
        ref_price=1000.0,
        adv_ratio=0.001,
        adv_ratio_cap=0.002,
        unit=100,
    )
    assert qty == 100
    assert yen == 100_000.0


def test_market_open_unit_cap_warns():
    _, _, warn = compute_size(
        prev_turnover=1_000_000_000.0,
        ref_price=100.0,
        adv_ratio=0.001,
        adv_ratio_cap=0.01,
        unit=100,
        market_open_unit_cap=50,
        is_market_open_order=True,
    )
    assert warn != ""


def test_market_open_unit_cap_can_be_enforced():
    qty, yen, warn = compute_size(
        prev_turnover=1_000_000_000.0,
        ref_price=100.0,
        adv_ratio=0.001,
        adv_ratio_cap=0.01,
        unit=100,
        market_open_unit_cap=50,
        is_market_open_order=True,
        enforce_market_open_unit_cap=True,
    )
    assert qty == 5000
    assert yen == 500_000.0
    assert "クリップ" in warn


def test_no_warn_when_not_market_open():
    _, _, warn = compute_size(
        prev_turnover=1_000_000_000.0,
        ref_price=100.0,
        adv_ratio=0.001,
        adv_ratio_cap=0.01,
        unit=100,
        market_open_unit_cap=50,
        is_market_open_order=False,
    )
    assert warn == ""


def test_zero_on_invalid_price():
    qty, yen, warn = compute_size(
        prev_turnover=1_000_000.0,
        ref_price=0.0,
        adv_ratio=0.001,
        adv_ratio_cap=0.002,
    )
    assert qty == 0
    assert yen == 0.0
    assert warn == ""


def test_adv_ratio_must_not_exceed_cap():
    with pytest.raises(ValueError):
        compute_size(
            prev_turnover=100_000_000.0,
            ref_price=1000.0,
            adv_ratio=0.003,
            adv_ratio_cap=0.002,
        )


def test_negative_target_notional_rejected():
    with pytest.raises(ValueError):
        compute_size(
            prev_turnover=100_000_000.0,
            ref_price=1000.0,
            adv_ratio=0.001,
            adv_ratio_cap=0.002,
            target_notional=-1.0,
        )


def test_invalid_unit_rejected():
    with pytest.raises(ValueError):
        compute_size(
            prev_turnover=100_000_000.0,
            ref_price=1000.0,
            adv_ratio=0.001,
            adv_ratio_cap=0.002,
            unit=0,
        )
