"""order_builder tests."""

import pandas as pd

from jp_signal.order_builder import is_shortable_asof, signals_to_orders
from jp_signal.risk import RiskConfig


def test_is_shortable_missing_is_false():
    assert is_shortable_asof(None, "7203", "2024-01-10") is False
    assert is_shortable_asof(pd.DataFrame(), "7203", "2024-01-10") is False


def test_is_shortable_no_lookahead():
    sh = pd.DataFrame(
        [
            {"code": "7203", "date": "2024-01-10", "is_margin_lendable": 1, "short_restricted": 0},
            {"code": "7203", "date": "2024-01-12", "is_margin_lendable": 1, "short_restricted": 0},
        ]
    )
    # 1/11 時点では 1/10 のみ見える
    assert is_shortable_asof(sh, "7203", "2024-01-11") is True
    # 制限あり
    sh2 = pd.DataFrame(
        [{"code": "7203", "date": "2024-01-10", "is_margin_lendable": 1, "short_restricted": 1}]
    )
    assert is_shortable_asof(sh2, "7203", "2024-01-11") is False


def test_signals_to_orders_includes_name_for_live():
    """live モードでも name が付与されることを確認（#6 回帰防止）。"""
    prices = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-09",
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 100,
                "adj_close": 100,
                "volume": 1e6,
                "turnover": 1e10,
            },
            {
                "code": "7203",
                "date": "2024-01-10",
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 105,
                "adj_close": 105,
                "volume": 1e6,
                "turnover": 1e10,
            },
        ]
    )
    sig = pd.DataFrame([{"code": "7203", "side": "BUY", "score": 1.0, "limit_price": None}])
    univ = pd.DataFrame([{"code": "7203", "name": "トヨタ"}])
    risk = RiskConfig(
        allow_short_without_confirmed_shortability=True,
        require_both_sides=False,
        max_net_exposure_yen=50_000_000,
    )
    orders = signals_to_orders(
        sig,
        prices,
        as_of="2024-01-11",
        sizing_cfg={
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "market_open_unit_cap": 50,
            "require_full_adv_history": False,
            "allow_single_day_turnover_fallback": True,
        },
        risk_cfg=risk,
        universe=univ,
        for_backtest=False,
    )
    assert not orders.empty, "orders should not be empty"
    assert "name" in orders.columns, "name column must exist"
    assert orders.iloc[0]["name"] == "トヨタ", f"expected トヨタ, got {orders.iloc[0]['name']}"

    # for_backtest=False では date/limit_price/holding_days が無いこと
    assert "date" not in orders.columns
    assert "limit_price" not in orders.columns
    assert "holding_days" not in orders.columns


def test_is_shortable_rejects_stale_snapshot():
    sh = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-01",
                "is_margin_lendable": 1,
                "short_restricted": 0,
            }
        ]
    )

    assert (
        is_shortable_asof(
            sh,
            "7203",
            "2024-01-10",
            max_age_calendar_days=4,
        )
        is False
    )


def test_is_shortable_accepts_recent_snapshot():
    sh = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-08",
                "is_margin_lendable": 1,
                "short_restricted": 0,
            }
        ]
    )

    assert (
        is_shortable_asof(
            sh,
            "7203",
            "2024-01-10",
            max_age_calendar_days=4,
        )
        is True
    )


def test_is_shortable_rejects_future_snapshot():
    sh = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-11",
                "is_margin_lendable": 1,
                "short_restricted": 0,
            }
        ]
    )

    assert (
        is_shortable_asof(
            sh,
            "7203",
            "2024-01-10",
            max_age_calendar_days=4,
        )
        is False
    )


def test_signals_to_orders_drops_unshortable_sell():
    prices = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-09",
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 100,
                "adj_close": 100,
                "volume": 1e6,
                "turnover": 1e10,
            },
            {
                "code": "7203",
                "date": "2024-01-10",
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 105,
                "adj_close": 105,
                "volume": 1e6,
                "turnover": 1e10,
            },
        ]
    )
    # 営業日カレンダーに依存するため、as_of を平日に
    sig = pd.DataFrame([{"code": "7203", "side": "SELL", "score": 1.0, "limit_price": None}])
    risk = RiskConfig(
        allow_short_without_confirmed_shortability=False,
        require_both_sides=False,
    )
    orders = signals_to_orders(
        sig,
        prices,
        as_of="2024-01-11",
        sizing_cfg={
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "market_open_unit_cap": 50,
            "require_full_adv_history": False,
            "allow_single_day_turnover_fallback": True,
        },
        risk_cfg=risk,
        shortability=None,
    )
    assert orders.empty or (orders["side"] != "SELL").all()


def test_pit_shortability_does_not_use_observation_available_after_decision():
    """判断時刻後に取得された観測を使用しない。"""
    observations = pd.DataFrame(
        [
            {
                "code": "7203",
                "effective_at": "2024-01-11T07:00:00+09:00",
                "fetched_at": "2024-01-11T09:00:00+09:00",
                "source": "test",
                "short_type": "system",
                "is_shortable": 1,
                "is_margin_lendable": 1,
                "short_restricted": 0,
                "stock_loan_fee_annual": None,
            }
        ]
    )

    # 09:00取得のデータは08:15判断時点では利用できない。
    assert (
        is_shortable_asof(
            observations,
            "7203",
            pd.Timestamp(
                "2024-01-11T08:15:00",
                tz="Asia/Tokyo",
            ),
            max_age_calendar_days=4,
        )
        is False
    )


def test_pit_shortability_uses_observation_available_before_decision():
    """判断時刻以前に利用可能だった観測を使用する。"""
    observations = pd.DataFrame(
        [
            {
                "code": "7203",
                "effective_at": "2024-01-11T07:00:00+09:00",
                "fetched_at": "2024-01-11T07:05:00+09:00",
                "source": "test",
                "short_type": "system",
                "is_shortable": 1,
                "is_margin_lendable": 1,
                "short_restricted": 0,
                "stock_loan_fee_annual": None,
            }
        ]
    )

    assert (
        is_shortable_asof(
            observations,
            "7203",
            pd.Timestamp(
                "2024-01-11T08:15:00",
                tz="Asia/Tokyo",
            ),
            max_age_calendar_days=4,
        )
        is True
    )
