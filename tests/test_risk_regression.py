"""Risk regression tests for P0 fixes."""

import pandas as pd

from jp_signal.risk import RiskConfig, apply_order_risk_limits, risk_config_from_dict


def test_long_only_orders_are_rejected_for_long_short_strategy():
    """require_both_sides=True で BUY のみの注文は空になる。"""
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 1.0,
                "shortable": False,
            },
            {
                "code": "6758",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 0.8,
                "shortable": False,
            },
        ]
    )

    risk = RiskConfig(
        require_both_sides=True,
        max_net_exposure_yen=500_000,
    )

    result = apply_order_risk_limits(orders, risk)

    assert result.empty


def test_short_only_orders_are_rejected_for_long_short_strategy():
    """require_both_sides=True で SELL のみの注文は空になる。"""
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "SELL",
                "value_yen": 1_000_000,
                "score": 0.5,
                "shortable": True,
            },
            {
                "code": "6758",
                "side": "SELL",
                "value_yen": 2_000_000,
                "score": 0.4,
                "shortable": True,
            },
        ]
    )

    risk = RiskConfig(
        require_both_sides=True,
        max_net_exposure_yen=500_000,
    )

    result = apply_order_risk_limits(orders, risk)

    assert result.empty


def test_net_exposure_limit_trims_imbalanced_orders():
    """max_net_exposure_yen を超える場合、過剰な側から注文が削除される。"""
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 1.0,
                "shortable": True,
            },
            {
                "code": "6758",
                "side": "BUY",
                "value_yen": 2_000_000,
                "score": 0.9,
                "shortable": True,
            },
            {
                "code": "9984",
                "side": "SELL",
                "value_yen": 500_000,
                "score": 0.8,
                "shortable": True,
            },
        ]
    )

    risk = RiskConfig(
        max_net_exposure_yen=500_000,
        require_both_sides=True,
    )

    result = apply_order_risk_limits(orders, risk)

    if not result.empty:
        long_sum = result[result["side"] == "BUY"]["value_yen"].sum()
        short_sum = result[result["side"] == "SELL"]["value_yen"].sum()
        assert abs(long_sum - short_sum) <= risk.max_net_exposure_yen


def test_risk_config_from_dict_with_new_keys():
    """新しいキー require_both_sides / max_net_exposure_yen が反映される。"""
    d = {
        "max_orders_per_day": 5,
        "max_net_exposure_yen": 2_000_000,
        "require_both_sides": True,
    }

    cfg = risk_config_from_dict(d)

    assert cfg.max_orders_per_day == 5
    assert cfg.max_net_exposure_yen == 2_000_000
    assert cfg.require_both_sides is True


def test_risk_config_defaults():
    """デフォルト値が正しく設定される。"""
    cfg = risk_config_from_dict({})

    assert cfg.max_net_exposure_yen == 5_000_000
    assert cfg.require_both_sides is True


def test_invalid_side_filtered_out():
    """BUY/SELL 以外の side は除去される。"""
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 1.0,
                "shortable": True,
            },
            {
                "code": "6758",
                "side": "HOLD",
                "value_yen": 1_000_000,
                "score": 0.5,
                "shortable": True,
            },
        ]
    )

    risk = RiskConfig(require_both_sides=False)
    result = apply_order_risk_limits(orders, risk)

    assert "HOLD" not in result["side"].tolist()
