"""リスク選択とreject理由のテスト。"""

from __future__ import annotations

import pandas as pd

from jp_signal.risk import (
    RiskConfig,
    select_orders_with_reasons,
)


def make_risk(
    *,
    max_orders: int = 10,
    max_net: float = 1_000_000,
    require_both_sides: bool = True,
) -> RiskConfig:
    return RiskConfig(
        max_orders_per_day=max_orders,
        max_gross_exposure_yen=10_000_000,
        max_single_name_exposure_yen=2_000_000,
        max_long_exposure_yen=5_000_000,
        max_short_exposure_yen=5_000_000,
        max_net_exposure_yen=max_net,
        require_both_sides=require_both_sides,
        allow_short_without_confirmed_shortability=False,
    )


def test_rejected_order_has_reason():
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "SELL",
                "value_yen": 1_000_000,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = select_orders_with_reasons(
        orders,
        make_risk(require_both_sides=False),
    )

    assert result.selected.empty
    assert result.rejected.iloc[0]["reason"] == "NOT_SHORTABLE"


def test_require_both_sides_records_reason():
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = select_orders_with_reasons(
        orders,
        make_risk(require_both_sides=True),
    )

    assert result.selected.empty
    assert "REQUIRE_BOTH_SIDES" in set(result.rejected["reason"])


def test_order_limit_does_not_consume_all_slots_with_buys():
    orders = pd.DataFrame(
        [
            {
                "code": "1001",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 10.0,
                "shortable": False,
            },
            {
                "code": "1002",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 9.0,
                "shortable": False,
            },
            {
                "code": "2001",
                "side": "SELL",
                "value_yen": 1_000_000,
                "score": 8.0,
                "shortable": True,
            },
        ]
    )

    result = select_orders_with_reasons(
        orders,
        make_risk(
            max_orders=2,
            max_net=0,
            require_both_sides=True,
        ),
    )

    assert len(result.selected) == 2
    assert set(result.selected["side"]) == {"BUY", "SELL"}


def test_duplicate_code_is_rejected():
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "value_yen": 1_000_000,
                "score": 2.0,
                "shortable": False,
            },
            {
                "code": "7203",
                "side": "SELL",
                "value_yen": 1_000_000,
                "score": 1.0,
                "shortable": True,
            },
        ]
    )

    result = select_orders_with_reasons(
        orders,
        make_risk(require_both_sides=False),
    )

    assert len(result.selected) == 1
    assert "DUPLICATE_CODE" in set(result.rejected["reason"])
